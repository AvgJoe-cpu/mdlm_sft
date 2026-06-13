import subprocess
import sys
from pathlib import Path

import subprocess
import sys
import tempfile
from pathlib import Path

from datasets import Dataset

from mdlm_sft.mdlm.mdlm_load_model_v2 import download_base_model

# ---------------------------------------------------------------------------
# Real subprocess train/gen — no stubs.
# ---------------------------------------------------------------------------
def train_fn(load_model_train_path, save_model_train_path,
             load_data_train_path, load_data_eval_path,
             *, round_name, extra_overrides=()):
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


def gen_fn(load_model_gen_path, load_data_gen_path, save_data_gen_path,
           *, round_name, extra_overrides=()):
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


# ---------------------------------------------------------------------------
# Toy data
# ---------------------------------------------------------------------------
toy_train_data = [
    {"prompt": "What is the capital of France?",      "completion": "Paris."},
    {"prompt": "What is the largest mammal?",         "completion": "The blue whale."},
    {"prompt": "Who wrote 'To Kill a Mockingbird'?",  "completion": "Harper Lee."},
    {"prompt": "What is the boiling point of water?", "completion": "100 °C."},
    {"prompt": "Who painted the Mona Lisa?",          "completion": "Leonardo da Vinci."},
]
toy_eval_data = [
    {"prompt": "What is the capital of Italy?",       "completion": "Rome."},
    {"prompt": "What is the smallest mammal?",        "completion": "The bumblebee bat."},
    {"prompt": "Who wrote '1984'?",                   "completion": "George Orwell."},
]


# ---------------------------------------------------------------------------
# Fast-iteration overrides — keeps the smoke test under a couple of minutes.
# Tune as needed.
# ---------------------------------------------------------------------------
TRAIN_FAST_OVERRIDES = (
    "max_steps=4",
    "num_train_epochs=1",
    "logging_steps=1",
    "per_device_train_batch_size=2",
    "per_device_eval_batch_size=2",
    "warmup_ratio=0.0",
    "warmup_steps=0",
    "gradient_checkpointing=false",
    "torch_compile=false",
    "activation_offloading=false",   # CUDA-only in trl, breaks construction on Mac
    "eval_strategy=no",              # smoke test: train-only; avoids fp64 eval path on MPS
    "eval_on_start=false",           # ditto
    "report_to=none",
)

GEN_FAST_OVERRIDES = (
    "batch_size=2",
    "response_length=10",
    "num_steps=4",
)


def main() -> None:
    # 1. Prepare base model (idempotent: skips re-download if files exist).
    print("[smoke] preparing base model...")
    download_base_model()
    base_model_path = Path("artifacts_weights_mdlm_base_mdlm-owt_chat").resolve()
    assert base_model_path.exists(), f"base model dir missing after prep: {base_model_path}"
    print(f"[smoke] base model ready at: {base_model_path}")

    # 2. Everything else lives in a temp dir.
    with tempfile.TemporaryDirectory(prefix="smoke-") as tmp:
        tmp = Path(tmp)
        print(f"[smoke] tmp dir: {tmp}")

        # Toy datasets.
        train_data_path = tmp / "toy_train"
        eval_data_path  = tmp / "toy_eval"
        Dataset.from_list(toy_train_data).save_to_disk(str(train_data_path))
        Dataset.from_list(toy_eval_data).save_to_disk(str(eval_data_path))

        # Round 1 paths.
        trained_r1  = tmp / "trained-r1"
        gen_data_r1 = tmp / "gen-data-r1"

        # 3. One real round.
        round_name = "ROUND-1"
        print(f"\n=== {round_name} ===")
        train_fn(
            load_model_train_path=str(base_model_path),
            save_model_train_path=str(trained_r1),
            load_data_train_path=str(train_data_path),
            load_data_eval_path=str(eval_data_path),
            round_name=round_name,
            extra_overrides=TRAIN_FAST_OVERRIDES,
        )
        gen_fn(
            load_model_gen_path=str(trained_r1),
            load_data_gen_path=str(train_data_path),
            save_data_gen_path=str(gen_data_r1),
            round_name=round_name,
            extra_overrides=GEN_FAST_OVERRIDES,
        )

        # 4. Verify the gen output is a real dataset with a 'completion' column.
        ds = Dataset.load_from_disk(str(gen_data_r1))
        print(f"\n[smoke] gen output: {len(ds)} rows, columns={ds.column_names}")
        assert "completion" in ds.column_names, "gen output missing 'completion' column"
        print("[smoke] sample completion:", ds[0]["completion"])
        print("[smoke] OK")

if __name__ == "__main__":
    main()