from datasets import load_from_disk, Dataset
import sys
import subprocess
from datasets import Dataset
# load base model 
# load benchmark dataset, Dataset 
# NOTE: this is kept separate from the other sft script, since this is the final ood loop. 
# NOTE: the cfg will also be loaded 

def train_fn(
    load_model_train_path: str,
    save_model_train_path: str,
    load_data_train_path: str,
    load_data_eval_path: str,
    *,
    round_name: str,
    extra_overrides: tuple = (),
):
    cmd = [
        sys.executable, "-m", "mdlm_sft.mdlm.mdlm_sft_v2",
        f"model_name_or_path={load_model_train_path}",
        f"output_dir={save_model_train_path}",
        f"train_ds_path={load_data_train_path}",
        f"eval_ds_path={load_data_eval_path}",
        f"run_name=mdlm-sft-{round_name}",
        f"hydra.run.dir={RUNS_ROOT}/{round_name}/train",
        *extra_overrides,
    ]
    print(f"[{round_name}] train: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def gen_fn(
    load_model_gen_path: str,
    load_data_gen_path: str,
    save_data_gen_path: str,
    *,
    round_name: str,
    extra_overrides: tuple = (),
):
    cmd = [
        sys.executable, "-m", "mdlm_sft.mdlm.mdlm_gen_v2",
        f"model_name_or_path={load_model_gen_path}",
        f"dataset_input_path={load_data_gen_path}",
        f"dataset_output_path={save_data_gen_path}",
        f"hydra.run.dir={RUNS_ROOT}/{round_name}/gen",
        *extra_overrides,
    ]
    print(f"[{round_name}] gen: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)



EXPERIMENT = {
    "description": "Prototype for stage 3: draft code for the final evaluation loop",    
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
    },
}    

id_ft = EXPERIMENT["id_finetune"]
train_overrides = tuple(f"{k}={v}" for k, v in id_ft.get("train_overrides", {}).items())
gen_overrides   = tuple(f"{k}={v}" for k, v in id_ft.get("gen_overrides",   {}).items())


upstream_model_paths = {
    "model1": "/Users/jona/Documents/mdlm_sft/artifacts_weights_mdlm_base_mdlm-owt_chat",
    "model2": "/Users/jona/Documents/mdlm_sft/artifacts_weights_mdlm_base_mdlm-owt_chat",
}

OOD_TRAIN = "datasets_gsm8k/main_train"
OOD_EVAL = "datasets_gsm8k/main_test"
RUNS_ROOT = "artifacts_stage3"

###### PART 1: TRAINING (ONLINE 1)
# tracker: per-model output_dir (consumed by downstream gen step)
sft_model_paths: dict[str, str] = {}

for name, path in upstream_model_paths.items():
    round_name = f"ood-{name}"                              # e.g. ood-model1
    save_model_train_path = f"{RUNS_ROOT}/{round_name}/checkpoints"

    train_fn(
        load_model_train_path=path,
        save_model_train_path=save_model_train_path,
        load_data_train_path=OOD_TRAIN,
        load_data_eval_path=OOD_EVAL,
        round_name=round_name,
        extra_overrides=train_overrides,
    )

    sft_model_paths[name] = save_model_train_path

print("sft_model_paths:", sft_model_paths)

####### - - PART 2: GEN (ONLINE 2)
# ONLINE (2): GENERATION
# tracker: per-model output dataset path (consumed by offline eval step)
gen_ds_paths: dict[str, str] = {}

for name, model_path in sft_model_paths.items():
    round_name = f"ood-{name}"                              # matches training
    save_data_gen_path = f"{RUNS_ROOT}/{round_name}/gen_out"

    gen_fn(
        load_model_gen_path=model_path,
        load_data_gen_path=OOD_TRAIN,                     # same ds path as training
        save_data_gen_path=save_data_gen_path,
        round_name=round_name,
        extra_overrides=gen_overrides,
    )

    gen_ds_paths[name] = save_data_gen_path

print("gen_ds_paths:", gen_ds_paths)
# from here on: offline — downstream consumes dataset paths, not model paths

# from datasets import load_from_disk
###### PART 2: EVAL (OFFLINE 1)

# def parse_fn(example, column_names: tuple[str, str] = None):
#     """BASELINE WITHOUT RE"""
#     pred_final = example[column_names[0]].split("####")[1].strip()
#     gold_final = example[column_names[1]].split("####")[1].strip()
#     correct = 1 if pred_final == gold_final else 0
#     return {"pred_final": pred_final, "gold_final": gold_final, "correct": correct}


# def eval_acc(ds: Dataset):
#     parsed_ds = ds.map(parse_fn, fn_kwargs={"column_names": ("completion", "gold")})
#     acc = sum(parsed_ds["correct"]) / len(parsed_ds["correct"])
#     return acc


# # main loop: consume gen_ds_paths from ONLINE (2)
# acc_results: dict[str, float] = {}

# for name, ds_path in gen_ds_paths.items():
#     ds = load_from_disk(ds_path)
#     acc_results[name] = eval_acc(ds)

# print("acc_results:", acc_results)