import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

from mdlm_sft.mdlm.mdlm_load_model_v2 import download_base_model


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
# Experiment definition.
# ---------------------------------------------------------------------------
EXPERIMENT = {
    "name": "baseline-self-distill-3rounds",
    "description": (
        "Baseline: each round trains from the base model on the previous "
        "round's gen output, then generates from the original prompts. "
        "Round 1 trains on the original training set."
    ),
    "dataset": "avgJo3/writingprompts-strat",
    "shared": {
        "load_data_eval_path": "datasets_writingprompts-strat/strat_eval",
        "inds_downstream_eval_ds_path": "datasets_writingprompts-strat/validation",
        "inds_downstream_test_ds_path": "datasets_writingprompts-strat/test",
    },
    "train_overrides": {
        "per_device_train_batch_size":  64,
        "per_device_eval_batch_size":   64,
        "gradient_accumulation_steps":  10,
        "eval_steps":                   50,
        "activation_offloading":        False,
        "lr_scheduler_type":            "cosine",
        "torch_compile":                False,
        "warmup_ratio":                 0.05,
        "weight_decay":                 0.03,
        "learning_rate":                5e-4,
        "adam_beta1":                   0.88,
        "adam_beta2":                   0.98,
        "max_grad_norm":                2.0,
        "dataloader_num_workers":       16,
        "dataloader_prefetch_factor":   16,
        "max_steps":                    300,
        "num_train_epochs":             9999,
        "report_to":                   "wandb",
    },
    
    "gen_overrides": {
        "batch_size":      32,
        "response_length": 128,
        "num_steps":       128,
    },

    "rounds": {
        "ROUND-1": {
            "load_model_train_path": "artifacts_weights_mdlm_base_mdlm-owt_chat",
            "save_model_train_path": "artifacts_weights_mdlm_base_mdlm-owt_chat-r1",
            "load_data_train_path":  "datasets_writingprompts-strat/strat_train_12pct",

            "load_model_gen_path":   "artifacts_weights_mdlm_base_mdlm-owt_chat-r1",
            "load_data_gen_path":    "datasets_writingprompts-strat/strat_train_12pct",
            "save_data_gen_path":    "datasets_writingprompts-strat/gen_train_r1",
        },
        "ROUND-2": {
            "load_model_train_path": "artifacts_weights_mdlm_base_mdlm-owt_chat",
            "save_model_train_path": "artifacts_weights_mdlm_base_mdlm-owt_chat-r2",
            "load_data_train_path":  "datasets_writingprompts-strat/gen_train_r1",   # ← chain

            "load_model_gen_path":   "artifacts_weights_mdlm_base_mdlm-owt_chat-r2",
            "load_data_gen_path":    "datasets_writingprompts-strat/strat_train_12pct",
            "save_data_gen_path":    "datasets_writingprompts-strat/gen_train_r2",
        },
        "ROUND-3": {
            "load_model_train_path": "artifacts_weights_mdlm_base_mdlm-owt_chat",
            "save_model_train_path": "artifacts_weights_mdlm_base_mdlm-owt_chat-r3",
            "load_data_train_path":  "datasets_writingprompts-strat/gen_train_r2",   # ← chain

            "load_model_gen_path":   "artifacts_weights_mdlm_base_mdlm-owt_chat-r3",
            "load_data_gen_path":    "datasets_writingprompts-strat/strat_train_12pct",
            "save_data_gen_path":    "datasets_writingprompts-strat/gen_train_r3",
        },
    },
    "ind_downstream_model_eval": {
        "ROUND-1":  "artifacts_weights_mdlm_base_mdlm-owt_chat-r1",
        "ROUND-2":  "artifacts_weights_mdlm_base_mdlm-owt_chat-r2",
        "ROUND-3":  "artifacts_weights_mdlm_base_mdlm-owt_chat-r3",
        "BASELINE": "artifacts_weights_mdlm_base_mdlm-owt_chat",
    },   
}   


# stub for eval
def eval_fn(model_path, eval_ds_path):
    print(f"Evaluating {model_path} on {eval_ds_path}")
    return {"accuracy": 0.0}  # dummy   


def main() -> None:
    download_base_model()

    dataset_dir = Path("datasets_writingprompts-strat")
    if not dataset_dir.exists():
        load_dataset(EXPERIMENT["dataset"]).save_to_disk(str(dataset_dir))

    shared = EXPERIMENT["shared"]
    for round_name, round_cfg in EXPERIMENT["rounds"].items():
        cfg = {**shared, **round_cfg}   # round-specific wins on conflict
        train_fn(
            load_model_train_path=cfg["load_model_train_path"],
            save_model_train_path=cfg["save_model_train_path"],
            load_data_train_path=cfg["load_data_train_path"],
            load_data_eval_path=cfg["load_data_eval_path"],
            round_name=round_name,
            extra_overrides=tuple(f"{k}={v}" for k, v in EXPERIMENT["train_overrides"].items()),
        )
        gen_fn(
            load_model_gen_path=cfg["load_model_gen_path"],
            load_data_gen_path=cfg["load_data_gen_path"],
            save_data_gen_path=cfg["save_data_gen_path"],
            round_name=round_name,
            extra_overrides=tuple(f"{k}={v}" for k, v in EXPERIMENT["gen_overrides"].items()),
        )

    ##     
    # eval on downstream tasks
    ind_eval_models = EXPERIMENT["ind_downstream_model_eval"]
    eval_sets = {
        "eval": shared["inds_downstream_eval_ds_path"],
        "test": shared["inds_downstream_test_ds_path"],
    }
    for model_name, model_path in ind_eval_models.items():        
        for split_name, ds_path in eval_sets.items():
            metrics = eval_fn(model_path=model_path, eval_ds_path=ds_path)
            print(f"{model_name} / {split_name}: {metrics}")
            
if __name__ == "__main__":
    main()