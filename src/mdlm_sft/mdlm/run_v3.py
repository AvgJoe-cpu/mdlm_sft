import tempfile
from pathlib import Path

# ── scratch dir ──────────────────────────────────────────────────────────────
SCRATCH = Path(tempfile.mkdtemp(prefix="mdlm-sft-"))
print(f"[stub] scratch dir: {SCRATCH}")

BASE_DATASET_PATH = "avgJo3/tinystories-strat"
DATA              = str(SCRATCH / "data")
MODEL             = str(SCRATCH / "model")

# ── round builders ───────────────────────────────────────────────────────────
# Naming convention: {kind}_{variant-}?{action}@{round}
#   BASE artifacts:     m_trained@N,          d_train-generated@N
#   ABLATION artifacts: m_ablation-trained@N, d_ablation-generated@N
# Roots (unadorned):    m_base, d_train, d_validation

def base_rounds(n_rounds: int) -> dict:
    """BASE: each round reloads m_base, trains on previous round's generation."""
    rounds = {}
    for n in range(n_rounds):
        train_input = f"{DATA}_train" if n == 0 else f"{DATA}_train-generated@{n-1}"
        rounds[f"R{n}-train"] = {
            "model_name_or_path": f"{MODEL}_base",
            "output_dir":         f"{MODEL}_trained@{n}",
            "train_ds_path":      train_input,
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_trained@{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_train-generated@{n}",
        }
    return rounds


def ablation_rounds(n_rounds: int) -> dict:
    """ABLATION: reloads previous round's checkpoint (not base). Reuses BASE's R0."""
    rounds = {}
    for n in range(1, n_rounds + 1):
        prev_model = f"{MODEL}_trained@0"          if n == 1 else f"{MODEL}_ablation-trained@{n-1}"
        prev_data  = f"{DATA}_train-generated@0"   if n == 1 else f"{DATA}_ablation-generated@{n-1}"
        rounds[f"R{n}-train"] = {
            "model_name_or_path": prev_model,
            "output_dir":         f"{MODEL}_ablation-trained@{n}",
            "train_ds_path":      prev_data,
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_ablation-trained@{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_ablation-generated@{n}",
        }
    return rounds


def mix_rounds(n_rounds: int, mix_factor: float = 0.5) -> dict:
    """BASE-MIX: each round mixes previous generation with d_train,
    then trains m_base on the mix. Reuses BASE's R0 generation for R1."""
    rounds = {}
    for n in range(1, n_rounds + 1):
        prev_gen = f"{DATA}_train-generated@0"       if n == 1 else f"{DATA}_mix-generated@{n-1}"
        rounds[f"R{n}-mix"] = {
            "mix_factor":      mix_factor,
            "mix_base_path":   f"{DATA}_train",
            "mix_gen_path":    prev_gen,
            "mix_output_path": f"{DATA}_mix-mixed@{n}",
        }
        rounds[f"R{n}-train"] = {
            "model_name_or_path": f"{MODEL}_base",
            "output_dir":         f"{MODEL}_mix-trained@{n}",
            "train_ds_path":      f"{DATA}_mix-mixed@{n}",
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_mix-trained@{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_mix-generated@{n}",
        }
    return rounds


def mix_ablation_rounds(n_rounds: int, mix_factor: float = 0.5) -> dict:
    """MIX-ABLATION: mix + ablation combined.
    Each round mixes previous generation with d_train, then trains
    the *previous round's checkpoint* (not m_base) on the mix.
    Reuses BASE's R0 outputs for R1's boundary case."""
    rounds = {}
    for n in range(1, n_rounds + 1):
        prev_gen   = f"{DATA}_train-generated@0"           if n == 1 else f"{DATA}_mix-ablation-generated@{n-1}"
        prev_model = f"{MODEL}_trained@0"                  if n == 1 else f"{MODEL}_mix-ablation-trained@{n-1}"
        rounds[f"R{n}-mix"] = {
            "mix_factor":      mix_factor,
            "mix_base_path":   f"{DATA}_train",
            "mix_gen_path":    prev_gen,
            "mix_output_path": f"{DATA}_mix-ablation-mixed@{n}",
        }
        rounds[f"R{n}-train"] = {
            "model_name_or_path": prev_model,                              # ← ablation: reload prev, not m_base
            "output_dir":         f"{MODEL}_mix-ablation-trained@{n}",
            "train_ds_path":      f"{DATA}_mix-ablation-mixed@{n}",
            "eval_ds_path":       f"{DATA}_validation",
        }
        rounds[f"R{n}-inference"] = {
            "model_name_or_path":  f"{MODEL}_mix-ablation-trained@{n}",
            "dataset_input_path":  f"{DATA}_train",
            "dataset_output_path": f"{DATA}_mix-ablation-generated@{n}",
        }
    return rounds


# ── config ───────────────────────────────────────────────────────────────────
config = {
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
        "eval_strategy":                "no",
        "eval_on_start":                False,
        "save_strategy":                "no",
    },
    "RUNS": {
        "BASE": {
            "train_overrides": {"learning_rate": 1e-5},
            "ROUNDS": base_rounds(n_rounds=2),
        },
        "ABLATION": {
            "train_overrides": {"learning_rate": 1e-5},
            "ROUNDS": ablation_rounds(n_rounds=2),
        },
        # "BASE-MIX": {
        #     "train_overrides": {"learning_rate": 1e-5},
        #     "ROUNDS": mix_rounds(n_rounds=2),
        # },
        # "MIX-ABLATION": {
        #     "train_overrides": {"learning_rate": 1e-5},
        #     "ROUNDS": mix_ablation_rounds(n_rounds=2),
        # },
    },
}
# import pprint 
# pprint.pprint(config)

import pprint
pprint.pprint(config, sort_dicts=False, width=120)

# global_ov = config["train_overrides"]

# for run_name, run in config["RUNS"].items():
#     run_ov = run.get("train_overrides", {})
#     for stage, sc in run["ROUNDS"].items():
#         print(f"\n=== {run_name} / {stage} ===")
#         if stage.endswith("-train"):
#             stage_ov = sc.pop("train_overrides", {})   # currently no stage overrides, but harmless
#             merged   = {**global_ov, **run_ov, **stage_ov}
#             run_training(MDLMSFTConfig(**sc, **merged), save_last=True)
#         elif stage.endswith("-inference"):
#             run_inference(MDLMGenerationConfig(**sc))
#         else:
#             raise ValueError(f"unknown stage suffix: {stage!r}")