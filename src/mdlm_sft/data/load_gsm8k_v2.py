import gc

from datasets import DatasetDict, load_dataset
from .utils import normalize_whitespace, add_hash_id, count_tokens_fn

from pathlib import Path

# def render_thought_process(batch):
#     return {
#         "prompt": batch["question"],
#         "completion": [
#             a.replace("####", "<answer>") for a in batch["answer"]
#         ],
#     }

# def render_thought_process(batch):
#     return {
#         "completion": [
#             a.replace("<<", "<think>").replace(">>", "</think>")
#             for a in batch["completion"]
#         ],
#     }


def render_thought_process(batch):
    return {
        "prompt": batch["question"],
        "completion": [
            a.replace("####", "<answer>")
             .replace("<<", "<think>")
             .replace(">>", "</think>")
            for a in batch["answer"]
        ],
    }

def load_and_process_gsm8k(BATCH_SIZE: int = 1000, VAL_SIZE: int = 500, SPLIT_SEED: int = 42, OUTPUT_DIR=None) -> DatasetDict:
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e9)

        dd_main = load_dataset("openai/gsm8k", "main")
        dd_main = dd_main.map(render_thought_process, batched=True, batch_size=BATCH_SIZE).remove_columns(["question", "answer"])
        dd_main = (
            dd_main
            .map(normalize_whitespace, fn_kwargs={"field_name": "prompt"},                                              batched=True, batch_size=BATCH_SIZE, desc="[main] Normalizing whitespace in prompt")
            .map(normalize_whitespace, fn_kwargs={"field_name": "completion"},                                          batched=True, batch_size=BATCH_SIZE, desc="[main] Normalizing whitespace in completion")
            .map(add_hash_id,          fn_kwargs={"field_names": ("prompt", "completion")}, with_indices=True,          batched=True, batch_size=BATCH_SIZE, desc="[main] Adding hash ID")
            .map(count_tokens_fn,      fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"},                      batched=True, batch_size=BATCH_SIZE, desc="[main] Counting tokens in prompt")
            .map(count_tokens_fn,      fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"},                  batched=True, batch_size=BATCH_SIZE, desc="[main] Counting tokens in completion")
            .map(lambda x: {"text_token_count": [p + c for p, c in zip(x["prompt_token_count"], x["completion_token_count"])]}, batched=True, desc="[main] Calculating total token count")
        )

        dd_soc = load_dataset("openai/gsm8k", "socratic")
        dd_soc = dd_soc.map(render_thought_process, batched=True, batch_size=BATCH_SIZE).remove_columns(["question", "answer"])
        dd_soc = (
            dd_soc
            .map(normalize_whitespace, fn_kwargs={"field_name": "prompt"},                                              batched=True, batch_size=BATCH_SIZE, desc="[socratic] Normalizing whitespace in prompt")
            .map(normalize_whitespace, fn_kwargs={"field_name": "completion"},                                          batched=True, batch_size=BATCH_SIZE, desc="[socratic] Normalizing whitespace in completion")
            .map(add_hash_id,          fn_kwargs={"field_names": ("prompt", "completion")}, with_indices=True,          batched=True, batch_size=BATCH_SIZE, desc="[socratic] Adding hash ID")
            .map(count_tokens_fn,      fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"},                      batched=True, batch_size=BATCH_SIZE, desc="[socratic] Counting tokens in prompt")
            .map(count_tokens_fn,      fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"},                  batched=True, batch_size=BATCH_SIZE, desc="[socratic] Counting tokens in completion")
            .map(lambda x: {"text_token_count": [p + c for p, c in zip(x["prompt_token_count"], x["completion_token_count"])]}, batched=True,desc="[socratic] Calculating total token count")
        )
        del tokenizer
        gc.collect()
        
        main_split = dd_main["train"].train_test_split(
            test_size=VAL_SIZE, seed=SPLIT_SEED
        )
        soc_split  = dd_soc["train"].train_test_split(
            test_size=VAL_SIZE, seed=SPLIT_SEED
        )

        dd = DatasetDict({
            "main_train":       main_split["train"],   # ~6.9k, SFT training
            "main_validation":  main_split["test"],    # 500,   in-training eval
            "main_test":        dd_main["test"],       # 1319,  final inference

            "socratic_train":       soc_split["train"],
            "socratic_validation":  soc_split["test"],
            "socratic_test":        dd_soc["test"],
        })
        del dd_main, dd_soc, main_split, soc_split
        gc.collect()
        print(dd)

        if OUTPUT_DIR is not None:
            dd.save_to_disk(OUTPUT_DIR)
            del dd
            gc.collect()
            dd = DatasetDict.load_from_disk(OUTPUT_DIR)  # mmap'd, low RSS

        return dd

    finally:
        gc.collect()

def upload_gsm8k_to_hf():
    dd = load_and_process_gsm8k()
    dd.push_to_hub("avgJo3/gsm8k-processed", private=True)
    del dd 

    
#load_and_process_writingprompts()
if __name__ == "__main__":
    upload_gsm8k_to_hf()
