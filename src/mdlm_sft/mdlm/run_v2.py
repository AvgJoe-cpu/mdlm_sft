from datasets import load_dataset
from typing import Optional
from dataclasses import dataclass
# define config 
# model load path / save path 
# data load path 
# hyperparams 
from huggingface_hub import dataclasses

from mdlm_sft.mdlm.mdlm_gen_v3 import MDLMGenerationConfig, run_inference
from mdlm_sft.mdlm.mdlm_load_model import download_base_model
from mdlm_sft.mdlm.mdlm_sft_v4 import MDLMSFTConfig, run_training


import tempfile
from pathlib import Path

SCRATCH = Path(tempfile.mkdtemp(prefix="mdlm-sft-"))
print(f"[stub] scratch dir: {SCRATCH}")

BASE_DATASET_PATH = "avgJo3/tinystories-strat"           # hub id, unchanged
DATASET_DIR       = str(SCRATCH / "tinystories-strat")
MIX               = str(SCRATCH / "mix")
BASE_MODEL_PATH   = str(SCRATCH / "artifacts_weights_mdlm_base_mdlm-owt_chat")


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
            "ROUNDS": {
                "R0-train": {
                    "model_name_or_path": f"{MODEL}_base",
                    "output_dir":         f"{MODEL}_r0",
                    "train_ds_path":      f"{DATA}_train",
                    "eval_ds_path":       f"{DATA}_validation",
                },
                "R0-inference": {
                    "model_name_or_path":  f"{MODEL}_r0",
                    "dataset_input_path":  f"{DATA}_train",
                    "dataset_output_path": f"{DATA}_train-r0",###
                },                                              #
                                                                #
                "R1-train": {                                   #
                    "model_name_or_path": f"{MODEL}_base",      #
                    "output_dir":         f"{MODEL}_r1",   ##########
                    "train_ds_path":      f"{DATA}_train-r0", ###   #
                },                                                  #
                "R1-inference": {                                   #
                    "model_name_or_path":  f"{MODEL}_r1",  ##########
                    "dataset_input_path":  f"{DATA}_train",
                    "dataset_output_path": f"{DATA}_train-r1",###
                },                                              #
                #                                                 #
                # "R2-train": {                                   #
                #     "model_name_or_path": f"{MODEL}_base",      #
                #     "output_dir":         f"{MODEL}_r2",   ##########
                #     "train_ds_path":      f"{DATA}_train-r1", ###   #
                # },                                                  #
                # "R2-inference": {                                   #
                #     "model_name_or_path":  f"{MODEL}_r2",  ##########
                #     "dataset_input_path":  f"{DATA}_train",
                #     "dataset_output_path": f"{DATA}_train-r2",###
                # },                                              #
            },  
        },
    #     "BASE-MIX": {
    #         "train_overrides": {"learning_rate": 1e-5},
    #     "BASE-MIX": {
    #         "ROUNDS": {
    #             "R1-mix": {
    #                 "mix_gen_path":    f"{DATA}_train-r0",          
    #                 "mix_output_path": f"{DATA}_mix_r0_mixed",     #########            
    #             },                                                         #
    #             "R1-train": {                                              #
    #                 "model_name_or_path": f"{MODEL}_base",                 #
    #                 "output_dir":         f"{MODEL}_mix_r1",               #
    #                 "train_ds_path":      f"{DATA}_mix_r0_mixed",  #########
    #             },
    #             "R1-inference": {
    #                 "model_name_or_path":  f"{MODEL}_mix_r1",
    #                 "dataset_input_path":  f"{DATA}_train",
    #                 "dataset_output_path": f"{DATA}_train-r1-mix",   
    #             },

    #             "R2-mix": {
    #                 "mix_gen_path":    f"{DATA}_train-r1-mix",          
    #                 "mix_output_path": f"{DATA}_mix_r1_mixed",     #########            
    #             },                                                         #

    #             "R2-train": {                                              #
    #                 "model_name_or_path": f"{MODEL}_base",                 #
    #                 "output_dir":         f"{MODEL}_mix_r2",               #
    #                 "train_ds_path":      f"{DATA}_mix_r1_mixed",  #########
    #             },
    #             "R2-inference": {
    #                 "model_name_or_path":  f"{MODEL}_mix_r2",
    #                 "dataset_input_path":  f"{DATA}_train",
    #                 "dataset_output_path": f"{DATA}_train-r2-mix",   
    #             },
    #         },
    #     },
    #     # SELF - DISTILLATION 
    #     "ABLATION": {
    #         "train_overrides": {"learning_rate": 1e-5},         # ← run default
    #         "ROUNDS": {
    #             "R0-train": {
    #                 "model_name_or_path": f"{MODEL}_base",
    #                 "output_dir":         f"{MODEL}_r0",
    #                 "train_ds_path":      f"{DATA}_train",
    #                 "eval_ds_path":       f"{DATA}_validation",                
    #             },
    #             "R0-inference": {
    #                 "model_name_or_path":  f"{MODEL}_r0",
    #                 "dataset_input_path":  f"{DATA}_train",
    #                 "dataset_output_path": f"{DATA}_train-r0",###
    #             },
    #             "R1-train": {
    #                 "model_name_or_path": f"{MODEL}_r0",
    #                 "output_dir":         f"{MODEL}_r1-ablation",
    #                 "train_ds_path":      f"{DATA}_train-r0",
    #                 "eval_ds_path":       f"{DATA}_validation",                
    #             },
    #             "R1-inference": {
    #                 "model_name_or_path":  f"{MODEL}_r1-ablation",
    #                 "dataset_input_path":  f"{DATA}_train",
    #                 "dataset_output_path": f"{DATA}_train-r1-ablation",
    #             },
    #         },
    #     },
    #     "MIX-ABLATION": {
    #         "train_overrides": {"learning_rate": 1e-5},         # ←
    #         "ROUNDS": {
    #             "R0-train": {
    #                 "model_name_or_path": f"{MODEL}_base",
    #                 "output_dir":         f"{MODEL}_r0",
    #                 "train_ds_path":      f"{DATA}_train",
    #                 "eval_ds_path":       f"{DATA}_validation",                
    #             },
    #             "R0-inference": {
    #                 "model_name_or_path":  f"{MODEL}_r0",
    #                 "dataset_input_path":  f"{DATA}_train",
    #                 "dataset_output_path": f"{DATA}_train-r0",###
    #             },
    #             "R1-mix": {
    #                 "mix_factor":      0.5,
    #                 "mix_base_path":   f"{DATA}_train",  
    #                 "mix_gen_path":    f"{DATA}_train-r0",
    #                 "mix_output_path": f"{DATA}_train-r0-mix-ablation_mixed",
    #             },
    #             "R1-train": {
    #                 "model_name_or_path": f"{MODEL}_r0",
    #                 "output_dir":         f"{MODEL}_r1-mix-ablation",
    #                 "train_ds_path":      f"{DATA}_train-r0-mix-ablation_mixed",
    #                 "eval_ds_path":       f"{DATA}_validation",                
    #             },
    #             "R1-inference": {
    #                 "model_name_or_path":  f"{MODEL}_r1-mix-ablation",
    #                 "dataset_input_path":  f"{DATA}_train",
    #                 "dataset_output_path": f"{DATA}_train-r1-mix-ablation",
    #             },
    # },  
    }
}

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



download_base_model()
load_dataset(BASE_DATASET_PATH).save_to_disk(str(DATASET_DIR))


for run_name, run in config["RUNS"].items():
    for stage, sc in run["ROUNDS"].items():
        print(f"\n=== {run_name} / {stage} ===")
        if   stage.endswith("-train"):     run_training(MDLMSFTConfig(**sc, **config["train_overrides"]), save_last=True)
        elif stage.endswith("-inference"): run_inference(MDLMGenerationConfig(**sc))
        else: raise ValueError(f"unknown stage suffix: {stage!r}")