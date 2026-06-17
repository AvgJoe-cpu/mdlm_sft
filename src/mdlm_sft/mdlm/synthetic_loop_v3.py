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
            "ROUND-2": {
                "load_model_train_path": "{MOD}",
                "save_model_train_path": "{MOD}-r2",
                "load_data_train_path":  "{DATA}/{SPLIT}-gen-r1",

                "load_model_gen_path":   "{MOD}-r2",
                "load_data_gen_path":    "{DATA}/{SPLIT}",
                "save_data_gen_path":    "{DATA}/{SPLIT}-gen-r2",
            },
            "ROUND-3": {
                "load_model_train_path": "{MOD}",
                "save_model_train_path": "{MOD}-r3",
                "load_data_train_path":  "{DATA}/{SPLIT}-gen-r2",
            },
       },

        # MAIN ITER 
        # Round 1-mix starts after the first pass on the original data.
        # mix must have occured here

        "rounds-mix": {
            "ROUND-1-mix": {
                "mix_factor":            1.0,
                "mix_base_path":         "{DATA}/{SPLIT}",                 
                "mix_gen_path":          "{DATA}/{SPLIT}-gen-r0",          
                "mix_output_path":       "{MIX}/{SPLIT}-mix-r0",           

                # TRAIN
                "load_model_train_path": "{MOD}",                        
                "save_model_train_path": "{MOD}-r1-mix",
                "load_data_train_path":  "{MIX}/{SPLIT}-mix-r0",           

                # GEN   
                "load_model_gen_path":   "{MOD}-r1-mix",
                "load_data_gen_path":    "{DATA}/{SPLIT}",
                "save_data_gen_path":    "{DATA}/{SPLIT}-gen-r1-mix",
            },      
            "ROUND-2-mix": {
                "mix_factor":            1.0,
                "mix_base_path":         "{DATA}/{SPLIT}",                 
                "mix_gen_path":          "{DATA}/{SPLIT}-gen-r1-mix",          
                "mix_output_path":       "{MIX}/{SPLIT}-mix-r1",           

                # TRAIN
                "load_model_train_path": "{MOD}",                        
                "save_model_train_path": "{MOD}-r2-mix",
                "load_data_train_path":  "{MIX}/{SPLIT}-mix-r1",           

                # GEN   
                "load_model_gen_path":   "{MOD}-r2-mix",
                "load_data_gen_path":    "{DATA}/{SPLIT}",
                "save_data_gen_path":    "{DATA}/{SPLIT}-gen-r2-mix",
            },      
            "ROUND-3-mix": {
                "mix_factor":            1.0,
                "mix_base_path":         "{DATA}/{SPLIT}",                 
                "mix_gen_path":          "{DATA}/{SPLIT}-gen-r2-mix",          
                "mix_output_path":       "{MIX}/{SPLIT}-mix-r2",           
                
                # TRAIN
                "load_model_train_path": "{MOD}",                        
                "save_model_train_path": "{MOD}-r3-mix",
                "load_data_train_path":  "{MIX}/{SPLIT}-mix-r2",           
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



def main() -> None:
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
        gen_present = [k for k in GEN_KEYS if k in cfg]
        if gen_present and len(gen_present) != len(GEN_KEYS):
            missing = set(GEN_KEYS) - set(gen_present)
            raise ValueError(
                f"{round_name}: partial gen config — missing {sorted(missing)}. "
                f"Provide all of {GEN_KEYS} or none."
            )
        has_gen = bool(gen_present)
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


    #
    for round_name, round_cfg in id_ft["rounds-mix"].items():
        print(f"\n[{round_name}] === MIX ROUND START ===")
        mix_ds(
            train_ds_path=round_cfg["mix_base_path"],
            gen_ds_path=round_cfg["mix_gen_path"],
            output_ds_path=round_cfg["mix_output_path"],
            fact=round_cfg.get("mix_factor", 1.0),
        )

        # Sanity: verify the trainer's input == the mix's output.
        train_path = round_cfg["load_data_train_path"]
        mix_out    = round_cfg["mix_output_path"]
        print(f"[{round_name}] trainer will load: {train_path}")
        print(f"[{round_name}] mix wrote to:     {mix_out}")
        assert train_path == mix_out, (
            f"{round_name}: load_data_train_path ({train_path}) "
            f"does not match mix_output_path ({mix_out}); "
            f"trainer would not see mixed data."
        )

        # Sanity: verify the file exists and matches the size we just wrote.
        loaded = load_from_disk(train_path)
        print(f"[{round_name}] trainer-side |dataset| = {len(loaded)}")

if __name__ == "__main__":    
    main()