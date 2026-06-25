import gc
import hashlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import create_bucket, list_bucket_tree, sync_bucket

from mdlm_sft.mdlm.mdlm_load_model import download_base_model, download_mdlm_cot_checkpoint

# ---------------------------------------------------------------------------
def train_fn(load_model_train_path, save_model_train_path,
             load_data_train_path, load_data_eval_path,
             *, round_name, extra_overrides=()):
    def _dashify(s: str) -> str:
        return s if s.startswith("--") else f"--{s}"

    cmd = [
        sys.executable, "-m", "mdlm_sft.mdlm.mdlm_sft_v3",
        f"--model_name_or_path={load_model_train_path}",
        f"--output_dir={save_model_train_path}",
        f"--train_ds_path={load_data_train_path}",
        f"--eval_ds_path={load_data_eval_path}",
        f"--run_name=mdlm-sft-{round_name}",
        *(_dashify(s) for s in extra_overrides),
    ]
    print(f"[{round_name}] train: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)  


EXPERIMENT = {
    "_refs": {
        "MOD":   "artifacts_weights_mdlm_base_mdlm-owt_chat",
        "DATA":  "datasets_dailydialog-strat",
        "MIX":   "datasets_dailydialog-strat-mix",
        "SPLIT": "strat_train_12pct",
    },
    "reasoning_model": False,  # whether to use the reasoning-capable model variant (chat template + cot pretraining)
    "dataset_hub_id": "avgJo3/dailydialog-strat",
    "load_data_eval_path": "{DATA}/strat_eval",

    "bucket": {
        "namespace": "avgJo3",                 
        "name":      "mdlm-sft-artifacts",     
        "private":   True,
    },

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
            "batch_size":      4,
            "response_length": 1,
            "num_steps":       1,
        },

        # MAIN ITER
        "rounds": {
            "ROUND-0": {
                "load_model_train_path": "{MOD}",
                "save_model_train_path": "{MOD}-r0",
                "load_data_train_path":  "{DATA}/{SPLIT}",
            },
        },       
    },        
}    

def resolve_refs(obj, refs):
    if isinstance(obj, str):  return obj.format_map(refs)
    if isinstance(obj, dict): return {k: resolve_refs(v, refs) for k, v in obj.items()}
    return obj

def main() -> None:
    experiment = resolve_refs(EXPERIMENT, (refs := EXPERIMENT.pop("_refs")))

    if experiment["reasoning_model"]:
        download_mdlm_cot_checkpoint()
    else:
        download_base_model()

    dataset_dir = Path(refs["DATA"])
    if not dataset_dir.exists():
        load_dataset(experiment["dataset_hub_id"]).save_to_disk(str(dataset_dir))

    id_ft = experiment["id_finetune"]
    train_overrides = tuple(f"{k}={v}" for k, v in id_ft.get("train_overrides", {}).items())
    gen_overrides   = tuple(f"{k}={v}" for k, v in id_ft.get("gen_overrides",   {}).items())
    GEN_KEYS = ("load_model_gen_path", "load_data_gen_path", "save_data_gen_path")

    # =====================================================================
    # (1) BASELINE ROUNDS
    # =====================================================================
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

        del cfg, gen_present, has_gen   # NOT round_name (the for-loop rebinds it)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
        except ImportError:
            pass

    del round_name, round_names, last_round

if __name__ == "__main__":
    main()