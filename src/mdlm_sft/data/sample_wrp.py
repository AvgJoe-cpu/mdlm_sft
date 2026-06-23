from .utils import normalize_whitespace, add_hash_id, count_tokens_fn, create_nested_stratified_splits_hf, _ratio_to_split_name

from typing import Dict, Callable, Any
import datasets
from datasets import Dataset, DatasetDict, load_dataset
import gc 
import numpy as np
import re

#datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB


def load_and_process_writingprompts(BATCH_SIZE: int = 8000, N_BINS: int = 8, OUTPUT_DIR=None) -> DatasetDict:
    try:
        dd = load_dataset("euclaise/writingprompts").rename_columns({"story": "completion"})

        from transformers import AutoTokenizer  
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e9)

        dd = (
            dd
            .map(normalize_whitespace, fn_kwargs={"field_name": "completion"}                         , batched=True, batch_size=BATCH_SIZE, desc="Normalizing whitespace in completion")
            .map(normalize_whitespace, fn_kwargs={"field_name": "prompt"}                             , batched=True, batch_size=BATCH_SIZE, desc="Normalizing whitespace in prompt")
            .map(add_hash_id, with_indices=True, fn_kwargs={"field_names": ("prompt", "completion")}  , batched=True, batch_size=BATCH_SIZE, desc="Adding hash ID")
            .map(count_tokens_fn,      fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"}     , batched=True, batch_size=BATCH_SIZE, desc="Counting tokens in prompt")
            .map(count_tokens_fn,      fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"} , batched=True, batch_size=BATCH_SIZE, desc="Counting tokens in completion")
            .map(lambda x: {"text_token_count": x["prompt_token_count"] + x["completion_token_count"]}, desc="Calculating total token count")
        )
        del tokenizer  
        gc.collect()

        filtered_ds = dd["train"].filter(lambda x: x["text_token_count"] <= 1024, desc="Filtering examples with total token count > 1024")

        token_counts = np.array(filtered_ds["text_token_count"])
        bin_edges    = np.linspace(token_counts.min(), token_counts.max(), N_BINS + 1)

        ds = filtered_ds.map(
            lambda batch: {"bin": [f"bin_{i}" for i in np.searchsorted(bin_edges[1:-1], batch["text_token_count"], side="left")]},
            batched=True,
            desc="Assigning bins based on token count",
        )
        del filtered_ds, token_counts, bin_edges
        gc.collect()

        result = create_nested_stratified_splits_hf(
            dataset=ds,
            task_column="bin",
            eval_ratio=0.1,
            split_ratios=[0.125, 0.25, 0.5, 1.0],
            random_seed=42,
            token_column=None,
            max_tokens=None,
        )

        dataset_dict: Dict[str, Dataset] = {
            "strat_eval": result["strat_eval"].remove_columns("bin"),
            "validation": dd["validation"],
            "test":       dd["test"],
        }
        for ratio, split_info in sorted(result["splits"].items()):
            dataset_dict[_ratio_to_split_name(ratio)] = split_info["train"].remove_columns("bin")
        out_dd = DatasetDict(dataset_dict)
        del ds, result, dataset_dict, dd
        gc.collect()
        
        print(out_dd)

        if OUTPUT_DIR is not None:
            out_dd.save_to_disk(OUTPUT_DIR)
            del out_dd
            gc.collect()
            out_dd = DatasetDict.load_from_disk(OUTPUT_DIR)  # mmap'd, low RSS

        return out_dd

    finally:
        gc.collect()

def upload_writingprompts_to_hf():
    dd = load_and_process_writingprompts()
    dd.push_to_hub("avgJo3/writingprompts-strat", private=False)
    del dd 

    
#load_and_process_writingprompts()
upload_writingprompts_to_hf()