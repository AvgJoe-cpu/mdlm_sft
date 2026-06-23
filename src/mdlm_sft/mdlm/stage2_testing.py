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
        "MOD":   "artifacts_weights_mdlm_base_mdlm-owt_chat",
        "DATA":  "datasets_writingprompts-strat",
        "MIX":   "datasets_writingprompts-strat-mix",
        "STATS": "datasets_writingprompts-strat/stats",
        "SPLIT": "strat_train_12pct",
    },
    "reasoning_model": False,  # whether to use the reasoning-capable model variant (chat template + cot pretraining)
    "dataset_hub_id": "avgJo3/writingprompts-strat",
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

                "load_model_gen_path":   "{MOD}-r0",
                "load_data_gen_path":    "{DATA}/{SPLIT}",
                "save_data_gen_path":    "{DATA}/{SPLIT}-gen-r0",
            },
        },
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

                # # GEN   
                # "load_model_gen_path":   "{MOD}-r1-mix",
                # "load_data_gen_path":    "{DATA}/{SPLIT}",
                # "save_data_gen_path":    "{DATA}/{SPLIT}-gen-r1-mix",
            },      
            # "ROUND-2-mix": {
            #     "mix_factor":            1.0,
            #     "mix_base_path":         "{DATA}/{SPLIT}",                 
            #     "mix_gen_path":          "{DATA}/{SPLIT}-gen-r1-mix",          
            #     "mix_output_path":       "{MIX}/{SPLIT}-mix-r1",           

            #     # TRAIN
            #     "load_model_train_path": "{MOD}",                        
            #     "save_model_train_path": "{MOD}-r2-mix",
            #     "load_data_train_path":  "{MIX}/{SPLIT}-mix-r1",           

            #     # # GEN   
            #     "load_model_gen_path":   "{MOD}-r2-mix",
            #     "load_data_gen_path":    "{DATA}/{SPLIT}",
            #     "save_data_gen_path":    "{DATA}/{SPLIT}-gen-r2-mix",
            # },      
            # "ROUND-3-mix": {
            #     "mix_factor":            1.0,
            #     "mix_base_path":         "{DATA}/{SPLIT}",                 
            #     "mix_gen_path":          "{DATA}/{SPLIT}-gen-r2-mix",          
            #     "mix_output_path":       "{MIX}/{SPLIT}-mix-r2",           
                
            #     # TRAIN
            #     "load_model_train_path": "{MOD}",                        
            #     "save_model_train_path": "{MOD}-r3-mix",
            #     "load_data_train_path":  "{MIX}/{SPLIT}-mix-r2",           
            # },      

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


def _stage_artifacts(experiment: dict, staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    id_ft = experiment["id_finetune"]
    rounds = {**id_ft["rounds"], **id_ft.get("rounds-mix", {})}
    GEN_KEY = "save_data_gen_path"
    MIX_KEY = "mix_output_path"

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

        # Generated dataset (optional — last round of a stage may skip gen)
        if GEN_KEY in cfg:
            data_src = Path(cfg[GEN_KEY])
            if data_src.exists():
                data_dst = round_dir / "data"
                shutil.copytree(data_src, data_dst, dirs_exist_ok=True)
                manifest_lines.append(f"{round_name}/data  <- {data_src}")
            else:
                manifest_lines.append(f"{round_name}/data  MISSING ({data_src})")

        # Mixed dataset (only present on mix rounds)
        if MIX_KEY in cfg:
            mix_src = Path(cfg[MIX_KEY])
            if mix_src.exists():
                mix_dst = round_dir / "mix"
                shutil.copytree(mix_src, mix_dst, dirs_exist_ok=True)
                manifest_lines.append(f"{round_name}/mix   <- {mix_src}")
            else:
                manifest_lines.append(f"{round_name}/mix   MISSING ({mix_src})")

    (staging_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")


def upload_artifacts_to_bucket(experiment: dict, *, bucket_namespace: str, bucket_name: str, private: bool = True) -> str:
    now = datetime.now(timezone.utc)
    payload = f"{now.date().isoformat()}|{now.isoformat(timespec='microseconds')}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    run_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{digest[:16]}"

    staging_dir = Path("uploads") / run_id

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

    # =====================================================================
    # (2) MIX ROUNDS
    # =====================================================================
    rounds_mix      = id_ft["rounds-mix"]
    round_names_mix = list(rounds_mix)              # preserves insertion order
    last_round_mix  = round_names_mix[-1]

    for round_name in round_names_mix:
        cfg = rounds_mix[round_name]

        gen_present = [k for k in GEN_KEYS if k in cfg]
        if gen_present and len(gen_present) != len(GEN_KEYS):
            missing = set(GEN_KEYS) - set(gen_present)
            raise ValueError(
                f"{round_name}: partial gen config — missing {sorted(missing)}. "
                f"Provide all of {GEN_KEYS} or none."
            )
        has_gen = bool(gen_present)
        if not has_gen and round_name != last_round_mix:
            raise ValueError(
                f"{round_name}: skipping gen is only allowed for the final round "
                f"(found later round(s): {round_names_mix[round_names_mix.index(round_name) + 1:]})."
            )

        # --- MIX STEP ---
        print(f"\n[{round_name}] === MIX ROUND START ===")
        mix_ds(
            train_ds_path=cfg["mix_base_path"],
            gen_ds_path=cfg["mix_gen_path"],
            output_ds_path=cfg["mix_output_path"],
            fact=cfg.get("mix_factor", 1.0),
        )

        # Sanity: verify the trainer's input == the mix's output.
        train_path = cfg["load_data_train_path"]
        mix_out    = cfg["mix_output_path"]
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
        del loaded

        # --- TRAIN ---
        train_fn(
            load_model_train_path=cfg["load_model_train_path"],
            save_model_train_path=cfg["save_model_train_path"],
            load_data_train_path=cfg["load_data_train_path"],
            load_data_eval_path=experiment["load_data_eval_path"],
            round_name=round_name,
            extra_overrides=train_overrides,
        )
        # --- GEN (optional, only skippable on final mix round) ---
        if has_gen:
            gen_fn(
                load_model_gen_path=cfg["load_model_gen_path"],
                load_data_gen_path=cfg["load_data_gen_path"],
                save_data_gen_path=cfg["save_data_gen_path"],
                round_name=round_name,
                extra_overrides=gen_overrides,
            )

        del cfg, gen_present, has_gen, train_path, mix_out   # NOT round_name
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

    del round_name, round_names_mix, last_round_mix, rounds_mix, train_overrides, gen_overrides

    # PUSH ALL ARTIFACTS TO BUCKET
    bucket_cfg = experiment["bucket"]
    delay = 30
    for attempt in range(1, 4):
        try:
            remote = upload_artifacts_to_bucket(
                experiment,
                bucket_namespace=bucket_cfg["namespace"],
                bucket_name=bucket_cfg["name"],
                private=bucket_cfg.get("private", True),
            )
            print(f"[done] artifacts uploaded to {remote}")
            break
        except Exception as e:
            print(f"[upload] attempt {attempt}/3 failed: {type(e).__name__}: {e}")
            if attempt == 3:
                print("[upload] ERROR: giving up; artifacts intact under `uploads/`.")
                raise
            time.sleep(delay)
            delay *= 2

if __name__ == "__main__":
    main()