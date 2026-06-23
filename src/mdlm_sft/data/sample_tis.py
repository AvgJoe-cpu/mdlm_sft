# single-column format (input: "text")
# (1) Preprocess dataset: Normalize, tokenize, count tokens, filter 
# (2) Dataset stats: Distribution of total_tokens; create bins
# (3) Stratified sampling: Sample from each bin to create balanced dataset
# (4) Process: tokenize into sentences, split into prompt/completion, save to disk
import gc   
from datasets import Dataset, ClassLabel, DatasetDict, load_dataset
from typing import List, Dict, Any, Optional        
import datasets
from transformers import AutoTokenizer
import numpy as np
import torch

from mdlm_sft.data.utils import normalize_whitespace, add_hash_id, count_tokens_fn, create_nested_stratified_splits_hf, _ratio_to_split_name, sent_tokenize, split_sentences_to_prompt_completion    


def load_and_process_tinystories(BATCH_SIZE: int = 1000, N_BINS: int = 8, OUTPUT_DIR=None):
    try:
        dd = load_dataset("roneneldan/TinyStories", )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e9)

        dd = (
            dd
            .map(normalize_whitespace, fn_kwargs={"field_name": "text"}, batched=True, batch_size=BATCH_SIZE, desc="Normalizing whitespace in text")
            .map(add_hash_id, with_indices=True, fn_kwargs={"field_names": ("text",)}, batched=True, batch_size=BATCH_SIZE, desc="Adding hash ID")
            .map(count_tokens_fn, fn_kwargs={"tokenizer": tokenizer, "field_name": "text"}, batched=True, batch_size=BATCH_SIZE, desc="Counting tokens in text")
        )
        del tokenizer  
        gc.collect()

        filtered_ds = dd["train"].filter(lambda x: x["text_token_count"] <= 1024, desc="Filtering examples with total token count > 1024")

        token_counts = np.array(filtered_ds["text_token_count"])
        bin_edges    = np.linspace(token_counts.min(), token_counts.max(), N_BINS + 1)
        ds = filtered_ds.map(
            lambda batch: {"bin": [f"bin_{i}" for i in np.searchsorted(bin_edges[1:-1], batch["text_token_count"], side="left")]},
            batched=True, batch_size=BATCH_SIZE, desc="Assigning bins based on token count",
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
        }
        for ratio, split_info in sorted(result["splits"].items()):
            dataset_dict[_ratio_to_split_name(ratio)] = split_info["train"].remove_columns("bin")
        out_dd = DatasetDict(dataset_dict)

        del ds, result, dataset_dict, dd
        gc.collect()

        # ---- Sentence-split into prompt/completion using SaT ----
        from wtpsplit import SaT
        model = SaT("sat-3l-sm")
        if torch.cuda.is_available():
            model.to("cuda")
        try:
            out_dd = (
                out_dd
                .map(sent_tokenize, fn_kwargs={"model": model, "text_field": "text"}, batched=True, batch_size=BATCH_SIZE, desc="Sentence-splitting with SaT")
                .map(split_sentences_to_prompt_completion, batched=True, desc="Splitting sentences into prompt/completion")
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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


def upload_tinystories_to_hf():
    dd = load_and_process_tinystories()
    dd.push_to_hub("avgJo3/tinystories-strat", private=False)
    del dd 

if __name__ == "__main__":
    upload_tinystories_to_hf()