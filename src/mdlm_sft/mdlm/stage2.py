import os
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset, load_from_disk
from datasets import concatenate_datasets
from mdlm_sft.mdlm.mdlm_load_model import download_base_model
from mdlm_sft.mdlm.evaluate_score_v2 import evaluate
# ---------------------------------------------------------------------------
def train_fn(load_model_train_path, save_model_train_path, load_data_train_path, load_data_eval_path, *, round_name, extra_overrides=()):
    cmd = [
        sys.executable, "-m", "mdlm_sft.mdlm.mdlm_sft_v2",
        f"model_name_or_path={load_model_train_path}",
        f"output_dir={save_model_train_path}",
        f"train_ds_path={load_data_train_path}",
        f"eval_ds_path={load_data_eval_path}",
        f"run_name=mdlm-sft-{round_name}",
        f"hydra.run.dir=runs/{round_name}/train",
        *extra_overrides,
    ]
    print(f"[{round_name}] train: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def gen_fn(load_model_gen_path, load_data_gen_path, save_data_gen_path, *, round_name, extra_overrides=()):
    cmd = [
        sys.executable, "-m", "mdlm_sft.mdlm.mdlm_gen_v2",
        f"model_name_or_path={load_model_gen_path}",
        f"dataset_input_path={load_data_gen_path}",
        f"dataset_output_path={save_data_gen_path}",
        f"hydra.run.dir=runs/{round_name}/gen",
        *extra_overrides,
    ]
    print(f"[{round_name}] gen: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)



EXPERIMENT = {
    "_refs": {
        "MOD":   "artifacts_weights_mdlm_base_mdlm-owt_chat",
        "DATA":  "datasets_writingprompts-strat",
        "MIX":   "datasets_writingprompts-strat-mix",
        "STATS": "datasets_writingprompts-strat/stats",
        "SPLIT": "strat_train_12pct",
    },
    "dataset_hub_id": "avgJo3/writingprompts-strat",
    "load_data_eval_path": "{DATA}/strat_eval",  
    
    # ** STAGE 1 ** 
    "id_finetune": {
        "train_overrides": {
            "max_steps":                    4,
            "eval_steps":                   2,
            "logging_steps":                1,
            "per_device_train_batch_size":  2,
            "per_device_eval_batch_size":   2,
            "gradient_accumulation_steps":  1,
            "torch_compile":                False,
            "activation_offloading":        False,
            "bf16":                         True,    
            "fp16":                         False,
            "report_to":                    "none",
            "eval_strategy": "no",             
            "eval_on_start": False,
        },
        "gen_overrides": {
            "batch_size":      2,
            "response_length": 1,
            "num_steps":       1,
        },

        # MAIN ITER 
        "rounds": {                     
            "ROUND-0": {
                "load_model_train_path": "{MOD}",
                "save_model_train_path": "{MOD}-r0",
                "load_data_train_path":  "{DATA}/{SPLIT}",

                "load_model_gen_path":   "{MOD}-r0",
                "load_data_gen_path":    "{DATA}/{SPLIT}",
                "save_data_gen_path":    "{DATA}/{SPLIT}-gen-r0",
            },
            "ROUND-1": {
                "load_model_train_path": "{MOD}",
                "save_model_train_path": "{MOD}-r1",
                "load_data_train_path":  "{DATA}/{SPLIT}-gen-r0",

                "load_model_gen_path":   "{MOD}-r1",
                "load_data_gen_path":    "{DATA}/{SPLIT}",
                "save_data_gen_path":    "{DATA}/{SPLIT}-gen-r1",
            },
       },
  
       
    },        
}    

def resolve_refs(obj, refs):
    if isinstance(obj, str):  return obj.format_map(refs)
    if isinstance(obj, dict): return {k: resolve_refs(v, refs) for k, v in obj.items()}
    return obj


def mix_ds(train_ds_path: str, gen_ds_path: str, output_ds_path: str, fact: float = 1.0) -> None:
    if not 0.0 <= fact <= 1.0:
        raise ValueError(f"mix_factor must be in [0, 1], got {fact}")

    base_ds = load_from_disk(train_ds_path)
    gen_ds  = load_from_disk(gen_ds_path)

    n_gen = int(len(gen_ds) * fact)
    if n_gen > 0:
        gen_ds = gen_ds.shuffle(seed=42).select(range(n_gen))
        out_ds = concatenate_datasets([base_ds, gen_ds]).shuffle(seed=42)
    else:
        out_ds = base_ds  # fact=0 → pure gold, no-op concat

    out_ds.save_to_disk(output_ds_path)


########################
# POST-PROCESSING STEPS

def hmerge_gen_datasets(gen_paths: dict[str, str], output_path: str, *, id_col: str = "id") -> None:
    """Horizontally concat generated datasets aligned on `id_col`. Renames `completion` -> `completion-<suffix>`."""
    if not gen_paths:
        raise ValueError("gen_paths is empty.")

    aligned, ref_ids, ref_name = [], None, None
    for suffix, path in gen_paths.items():
        ds = load_from_disk(path).sort(id_col).remove_columns(["prompt", "prompt_token_count", "completion_token_count", "text_token_count"])
        ids = ds[id_col]

        if ref_ids is None:
            ref_ids, ref_name = ids, suffix
        elif ids != ref_ids:
            raise ValueError(f"{suffix}: '{id_col}' does not align with {ref_name}.")

        ds = ds.rename_column("completion", f"completion-{suffix}")
        if aligned:
            ds = ds.remove_columns([id_col])
        aligned.append(ds)

    merged = concatenate_datasets(aligned, axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(output_path)



def main() -> None:
    # os.environ["WANDB_PROJECT"] = "mdlm-sft"

    refs = EXPERIMENT.pop("_refs")
    experiment = resolve_refs(EXPERIMENT, refs)

    download_base_model()
    dataset_dir = Path(refs["DATA"])
    if not dataset_dir.exists():
        load_dataset(experiment["dataset_hub_id"]).save_to_disk(str(dataset_dir))

    # Phase 1: id_finetune rounds.
    id_ft = experiment["id_finetune"]
    train_overrides = tuple(f"{k}={v}" for k, v in id_ft.get("train_overrides", {}).items())
    gen_overrides   = tuple(f"{k}={v}" for k, v in id_ft.get("gen_overrides",   {}).items())
    GEN_KEYS = ("load_model_gen_path", "load_data_gen_path", "save_data_gen_path")

    rounds      = id_ft["rounds"]
    round_names = list(rounds)                      # preserves insertion order
    last_round  = round_names[-1]

    for round_name in round_names:
        cfg = rounds[round_name]

        # Detect gen presence: all-or-nothing on the three gen path keys.
        gen_present = [k for k in GEN_KEYS if k in cfg]
        if gen_present and len(gen_present) != len(GEN_KEYS):
            missing = set(GEN_KEYS) - set(gen_present)
            raise ValueError(
                f"{round_name}: partial gen config — missing {sorted(missing)}. "
                f"Provide all of {GEN_KEYS} or none."
            )
        has_gen = bool(gen_present)

        # Safeguard: a gen-less round is only legal as the last round, because
        # subsequent rounds chain off this round's gen output.
        if not has_gen and round_name != last_round:
            raise ValueError(
                f"{round_name}: skipping gen is only allowed for the final round "
                f"(found later round(s): {round_names[round_names.index(round_name) + 1:]})."
            )

        train_fn(
            load_model_train_path=cfg["load_model_train_path"],
            save_model_train_path=cfg["save_model_train_path"],
            load_data_train_path=cfg["load_data_train_path"],
            load_data_eval_path=experiment["load_data_eval_path"],
            round_name=round_name,
            extra_overrides=train_overrides,
        )

        if has_gen:
            gen_fn(
                load_model_gen_path=cfg["load_model_gen_path"],
                load_data_gen_path=cfg["load_data_gen_path"],
                save_data_gen_path=cfg["save_data_gen_path"],
                round_name=round_name,
                extra_overrides=gen_overrides,
            )

    ##### ID EVAL HERE 
#   # LOAD BASE MODEL - SCORE_EVAL ON ALL GEN-DS
# 
# 
    ##### POST-PROCESSING
    # POST - STATS 
    # 1. gen datasets: compute stats for each gen dataset, save to disk.
    # 2. load stats - concat into a single table, save to disk.


    # POST - DATASETS: horizontal merge of all gen datasets from all rounds, aligned on `id`.
    rounds = resolve_refs(EXPERIMENT["id_finetune"]["rounds"], refs)
    gen_paths = {
        Path(cfg["save_data_gen_path"]).name: cfg["save_data_gen_path"]
        for cfg in rounds.values()
        if "save_data_gen_path" in cfg and Path(cfg["save_data_gen_path"]).exists()
    }
    hmerge_gen_datasets(gen_paths, f"{refs['DATA']}/{refs['SPLIT']}-gen-hmerged")
    ds = load_from_disk(f"{refs['DATA']}/{refs['SPLIT']}-gen-hmerged")
    print(ds)            

    # 2DO: POST-MODELS

if __name__ == "__main__":    
    main()