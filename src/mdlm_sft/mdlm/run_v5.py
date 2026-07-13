import gc
import hashlib
import pprint
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import create_bucket, list_bucket_tree, sync_bucket

from mdlm_sft.mdlm.mdlm_gen_v3   import MDLMGenerationConfig, run_inference
from mdlm_sft.mdlm.mdlm_load_model import download_base_model
from mdlm_sft.mdlm.mdlm_sft_v4   import MDLMSFTConfig, run_training


# ── scratch dir ──────────────────────────────────────────────────────────────
SCRATCH = Path(tempfile.mkdtemp(prefix="mdlm-sft-"))
print(f"[stub] scratch dir: {SCRATCH}")

BASE_DATASET_PATH = "avgJo3/tinystories-strat"
DATA              = str(SCRATCH / "data")
MODEL             = str(SCRATCH / "model")

# ── round builders ───────────────────────────────────────────────────────────
# Naming convention: {kind}_{variant-}?{action}_r{round}
#   BASE artifacts:         m_trained_r0,              d_train-generated_r0
#   ABLATION artifacts:     m_ablation-trained_r1,     d_ablation-generated_r1
#   BASE-MIX artifacts:     m_mix-trained_r1,          d_mix-{generated,mixed}_r1
#   MIX-ABLATION artifacts: m_mix-ablation-trained_r1, d_mix-ablation-{generated,mixed}_r1
# Roots (unadorned):        m_base, d_train, d_validation

def base_rounds(n_rounds: int) -> dict:
    """BASE: each round reloads m_base, trains on previous round's generation."""
    rounds = {}
    for n in range(n_rounds):
        train_input = f"{DATA}_train" if n == 0 else f"{DATA}_train-generated_r{n-1}"
        rounds[f"R{n}-train"] = {
            "model_name_or_path": f"{MODEL}_base",
            "output_dir":         f"{MODEL}_trained_r{n}",
            "train_ds_path":      train_input,
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_trained_r{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_train-generated_r{n}",
        }
    return rounds


def ablation_rounds(n_rounds: int) -> dict:
    """ABLATION: reloads previous round's checkpoint (not base). Reuses BASE's R0."""
    rounds = {}
    for n in range(1, n_rounds):
        prev_model = f"{MODEL}_trained_r0"        if n == 1 else f"{MODEL}_ablation-trained_r{n-1}"
        prev_data  = f"{DATA}_train-generated_r0" if n == 1 else f"{DATA}_ablation-generated_r{n-1}"
        rounds[f"R{n}-train"] = {
            "model_name_or_path": prev_model,
            "output_dir":         f"{MODEL}_ablation-trained_r{n}",
            "train_ds_path":      prev_data,
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_ablation-trained_r{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_ablation-generated_r{n}",
        }
    return rounds


def mix_rounds(n_rounds: int, mix_factor: float = 1.0) -> dict:
    """BASE-MIX: each round mixes previous generation with d_train,
    then trains m_base on the mix. Reuses BASE's R0 generation for R1."""
    rounds = {}
    for n in range(1, n_rounds):
        prev_gen = f"{DATA}_train-generated_r0" if n == 1 else f"{DATA}_mix-generated_r{n-1}"
        rounds[f"R{n}-mix"] = {
            "mix_factor":      mix_factor,
            "mix_base_path":   f"{DATA}_train",
            "mix_gen_path":    prev_gen,
            "mix_output_path": f"{DATA}_mix-mixed_r{n}",
        }
        rounds[f"R{n}-train"] = {
            "model_name_or_path": f"{MODEL}_base",
            "output_dir":         f"{MODEL}_mix-trained_r{n}",
            "train_ds_path":      f"{DATA}_mix-mixed_r{n}",
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_mix-trained_r{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_mix-generated_r{n}",
        }
    return rounds


def mix_ablation_rounds(n_rounds: int, mix_factor: float = 1.0) -> dict:
    """MIX-ABLATION: mix + ablation. Each round mixes previous generation with d_train,
    then trains the *previous round's checkpoint* (not m_base) on the mix.
    Reuses BASE's R0 outputs for R1's boundary case."""
    rounds = {}
    for n in range(1, n_rounds):
        prev_gen   = f"{DATA}_train-generated_r0" if n == 1 else f"{DATA}_mix-ablation-generated_r{n-1}"
        prev_model = f"{MODEL}_trained_r0"        if n == 1 else f"{MODEL}_mix-ablation-trained_r{n-1}"
        rounds[f"R{n}-mix"] = {
            "mix_factor":      mix_factor,
            "mix_base_path":   f"{DATA}_train",
            "mix_gen_path":    prev_gen,
            "mix_output_path": f"{DATA}_mix-ablation-mixed_r{n}",
        }
        rounds[f"R{n}-train"] = {
            "model_name_or_path": prev_model,
            "output_dir":         f"{MODEL}_mix-ablation-trained_r{n}",
            "train_ds_path":      f"{DATA}_mix-ablation-mixed_r{n}",
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_mix-ablation-trained_r{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_mix-ablation-generated_r{n}",
        }
    return rounds


# ── mix stage impl ───────────────────────────────────────────────────────────
def mix_ds(mix_base_path: str, mix_gen_path: str, mix_output_path: str,
           mix_factor: float = 1.0) -> None:
    if not 0.0 <= mix_factor <= 1.0:
        raise ValueError(f"mix_factor must be in [0, 1], got {mix_factor}")

    base_ds = load_from_disk(mix_base_path)
    gen_ds  = load_from_disk(mix_gen_path)

    n_gen = int(len(gen_ds) * mix_factor)
    if n_gen > 0:
        gen_ds = gen_ds.shuffle(seed=42).select(range(n_gen))
        out_ds = concatenate_datasets([base_ds, gen_ds]).shuffle(seed=42)
    else:
        out_ds = base_ds

    out_ds.save_to_disk(mix_output_path)


# ── upload ───────────────────────────────────────────────────────────────────
def upload_artifacts_to_bucket(scratch_dir: Path, *, namespace: str, bucket_name: str,
                               private: bool = True) -> str:
    """Sync entire scratch dir to a HF Storage Bucket under a unique run_id prefix."""
    now = datetime.now(timezone.utc)
    digest = hashlib.sha256(now.isoformat().encode()).hexdigest()[:16]
    run_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{digest}"

    create_bucket(bucket_name, private=private, exist_ok=True)
    remote = f"hf://buckets/{namespace}/{bucket_name}/{run_id}"
    print(f"[upload] syncing {scratch_dir} → {remote}")
    sync_bucket(str(scratch_dir), remote)

    print(f"[upload] contents:")
    for item in list_bucket_tree(f"{namespace}/{bucket_name}", prefix=run_id, recursive=True):
        print(f"  {item.path}  ({item.size} bytes)")

    return remote


# ── config ───────────────────────────────────────────────────────────────────
config = {
    # ── Shared defaults across all arms and rounds ──────────────────────
    "train_overrides": {
        # HPO winners (from Stage-B)
        "learning_rate":     1.84e-3,
        "warmup_ratio":      0.1,
        "weight_decay":      0.01,
        "time_epsilon":      1e-3,
        "lr_scheduler_type": "cosine",
        "loss_weight_type":  "scheduler",

        # Epoch-based training: 10 epochs per round.
        # (max_steps removed; num_train_epochs governs length.)
        "num_train_epochs":  10,

        # Batch / memory (production, matches HPO effective B = 320)
        "per_device_train_batch_size":  64,
        "per_device_eval_batch_size":   64,
        "gradient_accumulation_steps":  5,
        "bf16":                         True,
        "fp16":                         False,
        "torch_compile":                True,
        "activation_offloading":        False,

        # Eval / logging
        "eval_strategy":  "epoch",     # one eval per epoch, substrate-agnostic
        "eval_on_start":  True,        # baseline eval before training
        "logging_steps":  50,
        "report_to":      "wandb",

        # Save: keep the final checkpoint per round (needed for chain training).
        "save_strategy":     "no",

        # Reproducibility
        "seed":       42,
    },
    "RUNS": {
        "BASE": {
            "ROUNDS": base_rounds(n_rounds=2),
        },
        "ABLATION": {
            "ROUNDS": ablation_rounds(n_rounds=2),
        },
        "BASE-MIX": {
            "ROUNDS": mix_rounds(n_rounds=2),
        },
        "MIX-ABLATION": {
            "ROUNDS": mix_ablation_rounds(n_rounds=2),
        },
    },
}
pprint.pprint(config, sort_dicts=False, width=120)


# ── prepare roots ────────────────────────────────────────────────────────────
download_base_model(target=f"{MODEL}_base")

dsd = load_dataset(BASE_DATASET_PATH)
print(f"[stub] dataset splits: {list(dsd.keys())}")
dsd["train"].save_to_disk(f"{DATA}_train")
dsd["validation"].save_to_disk(f"{DATA}_validation")


# ── executor ─────────────────────────────────────────────────────────────────
global_ov = config["train_overrides"]

for run_name, run in config["RUNS"].items():
    run_ov = run.get("train_overrides", {})
    for stage, sc in run["ROUNDS"].items():
        print(f"\n=== {run_name} / {stage} ===")
        if stage.endswith("-train"):
            merged = {**global_ov, **run_ov}
            run_training(MDLMSFTConfig(**sc, **merged), save_last=True)
        elif stage.endswith("-inference"):
            run_inference(MDLMGenerationConfig(**sc))
        elif stage.endswith("-mix"):
            mix_ds(**sc)
        else:
            raise ValueError(f"unknown stage suffix: {stage!r}")

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass



delay = 30
for attempt in range(1, 4):
    try:
        remote = upload_artifacts_to_bucket(
            SCRATCH,
            namespace="avgJo3",
            bucket_name="mdlm-sft-artifacts",
            private=True,
        )
        print(f"[done] artifacts uploaded to {remote}")
        break
    except Exception as e:
        print(f"[upload] attempt {attempt}/3 failed: {type(e).__name__}: {e}")
        if attempt == 3:
            print(f"[upload] giving up. Local artifacts at: {SCRATCH}")
            raise
        time.sleep(delay)
        delay *= 2