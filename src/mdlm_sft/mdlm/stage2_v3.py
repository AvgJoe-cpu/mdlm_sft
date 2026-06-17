import os
import subprocess
import sys
from pathlib import Path
import os
import sys
import shutil
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset, load_from_disk
from datasets import concatenate_datasets
from mdlm_sft.mdlm.mdlm_load_model import download_base_model, download_mdlm_cot_checkpoint
from mdlm_sft.mdlm.evaluate_score_v2 import evaluate

from huggingface_hub import create_bucket, sync_bucket, list_bucket_tree

# ---------------------------------------------------------------------------
def train_fn(load_model_train_path, save_model_train_path, load_data_train_path, load_data_eval_path, *, round_name, extra_overrides=()):
    cmd = [
        sys.executable, "-m", "mdlm_sft.mdlm.mdlm_sft_v2",
        f"model_name_or_path={load_model_train_path}",
        f"output_dir={save_model_train_path}",
        f"train_ds_path={load_data_train_path}",
        f"eval_ds_path={load_data_eval_path}",
        f"run_name=mdlm-sft-{round_name}",
        f"hydra.run.dir=exp/{round_name}/train",
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
        f"hydra.run.dir=exp/{round_name}/gen",
        *extra_overrides,
    ]
    print(f"[{round_name}] gen: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)



EXPERIMENT = {
    "_refs": {
        "MOD":   "artifacts_weights_mdlm_cot_s3",
        "DATA":  "datasets_writingprompts-strat",
        "MIX":   "datasets_writingprompts-strat-mix",
        "STATS": "datasets_writingprompts-strat/stats",
        "SPLIT": "strat_train_12pct",
    },
    "reasoning_model": True,  # whether to use the reasoning-capable model variant (chat template + cot pretraining)
    "dataset_hub_id": "avgJo3/writingprompts-strat",
    "load_data_eval_path": "{DATA}/strat_eval",

    "bucket": {
        "namespace": "avgJo3",                 
        "name":      "mdlm-sft-artifacts",     
        "private":   True,
    },

    "id_finetune": {
        "train_overrides": {
            "per_device_train_batch_size":  64,
            "per_device_eval_batch_size":   64,
            "gradient_accumulation_steps":  10,
            "eval_steps":                   100,
            "activation_offloading":        False,
            "lr_scheduler_type":            "cosine",
            "torch_compile":                True,
            "warmup_ratio":                 0.05,
            "weight_decay":                 0.03,
            "learning_rate":                5e-4,
            "adam_beta1":                   0.88,
            "adam_beta2":                   0.98,
            "max_grad_norm":                1.0,
            "dataloader_num_workers":       16,
            "dataloader_prefetch_factor":   16,
            "max_steps":                    600,
            "num_train_epochs":             9999,
            "report_to":                   "wandb",
        },
        "gen_overrides": {
            "batch_size":      64,
            "response_length": 128,
            "num_steps":       128,
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
    },
}


def resolve_refs(obj, refs):
    if isinstance(obj, str):  return obj.format_map(refs)
    if isinstance(obj, dict): return {k: resolve_refs(v, refs) for k, v in obj.items()}
    return obj


def _unique_run_id() -> str:
    """SHA-256 over (date, datetime-with-microseconds) -> short hex prefix."""
    now = datetime.now(timezone.utc)
    payload = f"{now.date().isoformat()}|{now.isoformat(timespec='microseconds')}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{now.strftime('%Y%m%dT%H%M%S')}-{digest[:16]}"


def _stage_artifacts(experiment: dict, staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    rounds = experiment["id_finetune"]["rounds"]
    GEN_KEY = "save_data_gen_path"

    manifest_lines = []
    for round_name, cfg in rounds.items():
        round_dir = staging_dir / round_name
        round_dir.mkdir(parents=True, exist_ok=True)

        # Model checkpoint
        model_src = Path(cfg["save_model_train_path"])
        if model_src.exists():
            model_dst = round_dir / "model"
            shutil.copytree(model_src, model_dst, dirs_exist_ok=True)
            manifest_lines.append(f"{round_name}/model <- {model_src}")
        else:
            manifest_lines.append(f"{round_name}/model MISSING ({model_src})")

        # Generated dataset (optional — last round may skip gen)
        if GEN_KEY in cfg:
            data_src = Path(cfg[GEN_KEY])
            if data_src.exists():
                data_dst = round_dir / "data"
                shutil.copytree(data_src, data_dst, dirs_exist_ok=True)
                manifest_lines.append(f"{round_name}/data  <- {data_src}")
            else:
                manifest_lines.append(f"{round_name}/data  MISSING ({data_src})")

    (staging_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")


def upload_artifacts_to_bucket(experiment: dict, *, bucket_namespace: str, bucket_name: str, private: bool = True) -> str:
    run_id = _unique_run_id()
    staging_root = Path("uploads")
    staging_dir = staging_root / run_id

    print(f"[upload] staging artifacts to {staging_dir}")
    _stage_artifacts(experiment, staging_dir)

    create_bucket(bucket_name, private=private, exist_ok=True)
    remote = f"hf://buckets/{bucket_namespace}/{bucket_name}/{run_id}"
    print(f"[upload] syncing {staging_dir} -> {remote}")
    sync_bucket(str(staging_dir), remote)

    # Quick sanity listing.
    print(f"[upload] contents of {remote}:")
    for item in list_bucket_tree(f"{bucket_namespace}/{bucket_name}", prefix=run_id, recursive=True):
        print(f"  {item.path}  ({item.size} bytes)")

    return remote


def main() -> None:
    refs = EXPERIMENT.pop("_refs")
    experiment = resolve_refs(EXPERIMENT, refs)
    
    if experiment["reasoning_model"]:
        download_mdlm_cot_checkpoint()
    else:     
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

    bucket_cfg = experiment["bucket"]
    remote = upload_artifacts_to_bucket(
        experiment,
        bucket_namespace=bucket_cfg["namespace"],
        bucket_name=bucket_cfg["name"],
        private=bucket_cfg.get("private", True),
    )
    print(f"[done] artifacts uploaded to {remote}")


if __name__ == "__main__":    
    main()