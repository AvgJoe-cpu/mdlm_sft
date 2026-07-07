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

import random 
import gc
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def random_sample(dataset: Dataset, n: int, seed: int = 42) -> Dataset:
    """Uniform random sample of size n. Seeded."""
    if len(dataset) <= n:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n)
    return dataset.select(sorted(indices))

def has_enough_sentences(batch, min_sentences=2):
    return [len(s) >= min_sentences for s in batch["sentences"]]

def load_and_process_tinystories(
    train_n: int = 10_000,
    val_n: int = 1_000,
    test_n: int = 1_000,
    test_carve_fraction: float = 0.05,
    max_tokens: int = 1024,
    min_sentences: int = 2,
    oversample_factor: float = 1.10,
    tokenizer_name: str = "gpt2",
    sat_model_name: str = "sat-3l-sm",
    batch_size: int = 1000,
    seed: int = 42,
    output_dir: str | None = None,
) -> DatasetDict:
    """
    TinyStories preprocessing pipeline.

    Stages (in order):
        1. Load
        2. Carve test split from raw train (early, before processing)
        3. Normalize whitespace on `text`
        4. Add hash id (anchored to source text)
        5. Count tokens on raw `text` (cheap length signal)
        6. Length filter: drop rows > max_tokens
        7. Oversample per split (headroom for min-sentence dropout)
        8. Sentence-tokenize with SaT (expensive; now on small pool)
        9. Min-sentence filter (drop items that can't support 50/50 split)
       10. Split into prompt (first ⌈N/2⌉ sentences) and completion (rest)
       11. Trim to exact target size per split
       12. Drop `sentences`, `text`, `text_token_count` columns
       13. Recount tokens on prompt/completion (for downstream stats)

    Args:
        train_n / val_n / test_n:   target sample sizes per split.
        test_carve_fraction:        fraction of raw train reserved as test.
        max_tokens:                 drop items whose text exceeds this token count.
        min_sentences:              drop items with fewer than this many sentences.
        oversample_factor:          headroom multiplier before sentence-split filter.
        tokenizer_name / sat_model_name: model identifiers.
        batch_size:                 batch size for map/filter operations.
        seed:                       seed for sampling and split carve-off.
        output_dir:                 if provided, save/reload via disk for memory-mapping.
    """
    try:
        # --- 1. Load ---
        dd = load_dataset("roneneldan/TinyStories")

        # --- 2. Carve test split from raw train (test identity fixed early) ---
        train_test = dd["train"].train_test_split(
            test_size=test_carve_fraction, seed=seed,
        )
        dd = DatasetDict({
            "train":      train_test["train"],
            "validation": dd["validation"],
            "test":       train_test["test"],
        })
        del train_test

        # --- 3. Normalize whitespace ---
        dd = dd.map(
            normalize_whitespace, fn_kwargs={"field_name": "text"},
            batched=True, batch_size=batch_size, desc="Normalizing whitespace",
        )

        # --- 4. Hash id (source-anchored) ---
        dd = dd.map(
            add_hash_id, with_indices=True,
            fn_kwargs={"field_names": ("text",)},
            batched=True, batch_size=batch_size, desc="Adding hash ID",
        )

        # --- 5. Token count on raw text (cheap length signal) ---
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = int(1e9)
        dd = dd.map(
            count_tokens_fn,
            fn_kwargs={"tokenizer": tokenizer, "field_name": "text"},
            batched=True, batch_size=batch_size, desc="Counting tokens (text)",
        )

        # --- 6. Length filter (before expensive sentence stage) ---
        dd = dd.filter(
            lambda x: x["text_token_count"] <= max_tokens,
            desc=f"Length filter (<= {max_tokens} tokens)",
        )

        # --- 7. Oversample per split (headroom for min-sentence dropout) ---
        # Sentence tokenization is the pipeline's dominant cost; sampling before it
        # reduces work by ~2 orders of magnitude. Oversampling provides slack so the
        # subsequent min-sentence filter still yields exact target sizes.
        raw = DatasetDict({
            "train":      random_sample(dd["train"],      n=int(train_n * oversample_factor), seed=seed),
            "validation": random_sample(dd["validation"], n=int(val_n   * oversample_factor), seed=seed),
            "test":       random_sample(dd["test"],       n=int(test_n  * oversample_factor), seed=seed),
        })
        del dd
        gc.collect()

        # --- 8. Sentence tokenize with SaT ---
        from wtpsplit import SaT
        import torch
        sat = SaT(sat_model_name)
        if torch.cuda.is_available():
            sat.to("cuda")

        raw = raw.map(
            sent_tokenize, fn_kwargs={"model": sat, "text_field": "text"},
            batched=True, batch_size=batch_size, desc="Sentence tokenizing",
        )

        del sat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # --- 9. Min-sentence filter (drop items unable to support 50/50 split) ---
        raw = raw.filter(
            has_enough_sentences,
            fn_kwargs={"min_sentences": min_sentences},
            batched=True, batch_size=batch_size,
            desc=f"Filtering < {min_sentences} sentences",
        )

        # --- 10. Split into prompt (first ⌈N/2⌉ sentences) + completion (rest) ---
        raw = raw.map(
            split_sentences_to_prompt_completion,
            batched=True, batch_size=batch_size, desc="Splitting prompt/completion",
        )

        # --- 11. Trim to exact target size ---
        def trim_or_fail(ds, n, split_name):
            if len(ds) < n:
                raise RuntimeError(
                    f"{split_name}: post-filter size {len(ds)} < target {n}. "
                    f"Increase oversample_factor (current: {oversample_factor})."
                )
            return ds.select(range(n))

        out = DatasetDict({
            "train":      trim_or_fail(raw["train"],      train_n, "train"),
            "validation": trim_or_fail(raw["validation"], val_n,   "validation"),
            "test":       trim_or_fail(raw["test"],       test_n,  "test"),
        })
        del raw
        gc.collect()

        # --- 12. Drop intermediate columns ---
        out = out.remove_columns(["sentences", "text", "text_token_count"])

        # --- 13. Recount tokens on derived fields (needed for appendix stats) ---
        out = (
            out
            .map(count_tokens_fn, fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"},
                 batched=True, batch_size=batch_size, desc="Counting tokens (prompt)")
            .map(count_tokens_fn, fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"},
                 batched=True, batch_size=batch_size, desc="Counting tokens (completion)")
            .map(lambda x: {"text_token_count": x["prompt_token_count"] + x["completion_token_count"]},
                 desc="Total token count")
        )
        del tokenizer
        gc.collect()

        # --- Persist + reload for memory-mapping ---
        if output_dir is not None:
            out.save_to_disk(output_dir)
            del out
            gc.collect()
            out = DatasetDict.load_from_disk(output_dir)

        return out

    finally:
        gc.collect()

def upload_tinystories_to_hf():
    dd = load_and_process_tinystories()
    dd.push_to_hub("avgJo3/tinystories-strat", private=False)
    del dd 

if __name__ == "__main__":
    upload_tinystories_to_hf()