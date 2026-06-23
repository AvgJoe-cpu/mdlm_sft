import gc
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict

from .utils import normalize_whitespace, add_hash_id, count_tokens_fn, create_nested_stratified_splits_hf, _ratio_to_split_name, sent_tokenize, split_sentences_to_prompt_completion    

import datasets
datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB


def format_cot_completion_post(batch):  # C2: (labels + CoT)
    return {
        "prompt": batch["source"],
        "completion": [
            f"<think> {r} </think> {t}"
            for r, t in zip(batch["rationale"], batch["target"])
        ],
    }

def load_and_process_cot_collection(BATCH_SIZE: int = 1000, OUTPUT_DIR=None) -> DatasetDict:
    try:
        ds = load_dataset("avgJo3/Cot-collection-datasets-4.8.5", split="train")
        dataset = (
            ds
            .map(normalize_whitespace, batched=True, batch_size=BATCH_SIZE, fn_kwargs={"field_name": "source"},    desc="Normalizing whitespace in 'source'")
            .map(normalize_whitespace, batched=True, batch_size=BATCH_SIZE, fn_kwargs={"field_name": "rationale"}, desc="Normalizing whitespace in 'rationale'")
            .map(normalize_whitespace, batched=True, batch_size=BATCH_SIZE, fn_kwargs={"field_name": "target"},    desc="Normalizing whitespace in 'target'")
        ).rename_columns({"source": "prompt", "target": "completion"})
        del ds
        gc.collect()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e9)

        dataset = (
            dataset
            .map(add_hash_id,     batched=True, with_indices=True, batch_size=BATCH_SIZE, fn_kwargs={"field_names": ("prompt", "completion")},            desc="Adding unique hash IDs")
            .map(count_tokens_fn, batched=True, batch_size=BATCH_SIZE,                    fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"},     desc="Counting tokens in 'prompt'")
            .map(count_tokens_fn, batched=True, batch_size=BATCH_SIZE,                    fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"}, desc="Counting tokens in 'completion'")
            .map(count_tokens_fn, batched=True, batch_size=BATCH_SIZE,                    fn_kwargs={"tokenizer": tokenizer, "field_name": "rationale"},  desc="Counting tokens in 'rationale'")
            .map(
                lambda x: {"total_tokens": [p + c + r for p, c, r in zip(x["prompt_token_count"], x["completion_token_count"], x["rationale_token_count"])]},
                batched=False, desc="Calculating total tokens",
            )
        )
        del tokenizer
        gc.collect()

        results = create_nested_stratified_splits_hf(
            dataset=dataset,
            task_column="task",
            token_column="total_tokens",
            max_tokens=1024,
            eval_ratio=0.10,
            split_ratios=[0.25, 0.5, 1.0],
            random_seed=42,
        )
        del dataset
        gc.collect()

        # Build DatasetDict: eval + every training split
        dataset_dict: Dict[str, Dataset] = {"eval": results["eval"]}
        for ratio, split_info in sorted(results["splits"].items()):
            dataset_dict[_ratio_to_split_name(ratio)] = split_info["train"]

        dd = DatasetDict(dataset_dict)
        dd = dd.map(format_cot_completion_post, batched=True, batch_size=BATCH_SIZE, desc="Formatting CoT completions with <think> tags").remove_columns(["rationale"])    

        del results, dataset_dict
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


def upload_cot_to_hf():
    dd = load_and_process_cot_collection()
    dd.push_to_hub("avgJo3/cot-strat", private=False)
    del dd 