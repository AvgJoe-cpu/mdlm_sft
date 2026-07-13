from .utils import normalize_whitespace, add_hash_id, count_tokens_fn, create_nested_stratified_splits_hf, _ratio_to_split_name

from typing import Dict, Callable, Any
import datasets
from datasets import Dataset, DatasetDict, load_dataset
import gc 
import re

#datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB

import random
from collections import Counter, defaultdict
import re

PREFIX_RE = re.compile(r'((?:\[\s[A-Z]+\s\]\s*){1,2})(.*)')

def special_prefix(batch):
    return {"completion": [f"<answer> {c}" for c in batch["completion"]]}

def has_prefix(batch):
    return [PREFIX_RE.match(p) is not None for p in batch["prompt"]]

def parse_prefix(batch):
    labels, prompts = [], []
    for p in batch["prompt"]:
        m = PREFIX_RE.match(p)
        labels.append(m.group(1).strip())
        prompts.append(m.group(2).strip())
    return {"label": labels, "prompt": prompts}

def collapse_label(batch):
    return {
        "label": [lbl.split("]")[0] + "]" for lbl in batch["label"]],
    }


def compute_coverage_labels(dataset: Dataset, coverage: float, label_col: str = "label") -> set[str]:
    """
    Return the minimal set of labels whose combined frequency covers >= `coverage`.
    Ties broken lexicographically for reproducibility.
    """
    counts = Counter(dataset[label_col])
    total = len(dataset)
    sorted_labels = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    kept, cumulative = set(), 0
    for label, c in sorted_labels:
        kept.add(label)
        cumulative += c
        if cumulative / total >= coverage:
            break
    return kept


def stratified_sample(dataset: Dataset, n: int, label_col: str = "label", seed: int = 42) -> Dataset:
    """Proportional stratified sample of size n. Largest-remainder rounding."""
    if len(dataset) <= n:
        return dataset

    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, lbl in enumerate(dataset[label_col]):
        groups[lbl].append(i)

    total = len(dataset)
    raw = {lbl: n * len(idxs) / total for lbl, idxs in groups.items()}
    floor = {lbl: int(c) for lbl, c in raw.items()}
    remainders = sorted(((raw[lbl] - floor[lbl], lbl) for lbl in groups), reverse=True)
    leftover = n - sum(floor.values())
    for _, lbl in remainders[:leftover]:
        floor[lbl] += 1

    picked = []
    for lbl, idxs in groups.items():
        k = min(floor[lbl], len(idxs))
        picked.extend(rng.sample(idxs, k))

    return dataset.select(sorted(picked))


def random_sample(dataset: Dataset, n: int, seed: int = 42) -> Dataset:
    """Uniform random sample of size n. Seeded."""
    if len(dataset) <= n:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n)
    return dataset.select(sorted(indices))


def load_and_process_writingprompts(
    coverage_target: float = 0.99,
    max_tokens: int = 1024,
    tokenizer_name: str = "gpt2",
    batch_size: int = 8000,
    seed: int = 42,
    output_dir: str | None = None,
) -> DatasetDict:
    """
    WritingPrompts preprocessing pipeline.

    Stages (in order):
        1. Load + rename story→completion
        2. Normalize whitespace (prompt, completion)
        3. Parse prefix: extract label, drop unparseable rows, collapse multi-prefix
        4. Add hash id (from prompt + completion)
        5. Count tokens (prompt, completion, total)
        6. Filter by length (all splits, same threshold)
        7. Filter by coverage (label set from train, applied to all splits)
        8. Sampling (per split, per strategy)
    """
    try:
        # --- 1. Load ---
        dd = load_dataset("euclaise/writingprompts").rename_columns({"story": "completion"})

        # --- 2. Normalize whitespace ---
        dd = (
            dd
            .map(normalize_whitespace, fn_kwargs={"field_name": "completion"},
                 batched=True, batch_size=batch_size, desc="Normalizing whitespace (completion)")
            .map(normalize_whitespace, fn_kwargs={"field_name": "prompt"},
                 batched=True, batch_size=batch_size, desc="Normalizing whitespace (prompt)")
        )

        # --- 3. Parse prefix (cheap → run before tokenization) ---
        dd = (
            dd
            .filter(has_prefix,   batched=True, batch_size=batch_size, desc="Filtering parseable rows")
            .map(parse_prefix,    batched=True, batch_size=batch_size, desc="Extracting label")
            .map(collapse_label,  batched=True, batch_size=batch_size, desc="Collapsing multi-prefix")
        )

        # --- 4. Hash id ---
        dd = dd.map(
            add_hash_id,
            with_indices=True,
            fn_kwargs={"field_names": ("prompt", "completion")},
            batched=True, batch_size=batch_size, desc="Adding hash ID",
        )
        from transformers import AutoTokenizer  

        # --- 5. Token counts ---
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = int(1e9)
        dd = (
            dd
            .map(count_tokens_fn, fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"}, batched=True, batch_size=batch_size)
            .map(count_tokens_fn, fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"}, batched=True, batch_size=batch_size)
            .map(lambda x: {"text_token_count": x["prompt_token_count"] + x["completion_token_count"]})
        )
        del tokenizer
        gc.collect()

        # --- 6. Length filter (all splits) ---
        dd = dd.filter(
            lambda x: x["text_token_count"] <= max_tokens,
            desc=f"Length filter (<= {max_tokens} tokens)",
        )

        # --- 7. Coverage filter (label set derived from train, applied to all splits) ---
        kept_labels = compute_coverage_labels(dd["train"], coverage=coverage_target)
        dd = dd.filter(
            lambda x: x["label"] in kept_labels,
            desc=f"Coverage filter (>= {coverage_target:.0%})",
        )

          # --- 8. Sampling (per split) ---
        train_ds = stratified_sample(dd["train"],      n=10_000, seed=42)
        val_ds   = stratified_sample(dd["validation"], n=1_000,  seed=42)
        test_ds  = random_sample(dd["test"],           n=1_000,  seed=42)

        out = DatasetDict({
            "train":      train_ds,
            "validation": val_ds,
            "test":       test_ds,
        })
        out = out.map(
            special_prefix,
            batched=True, batch_size=batch_size, desc="Adding special prefix to prompt",

        )

        del train_ds, val_ds, test_ds

        # --- Persist + reload for memory-mapping ---
        if output_dir is not None:
            out.save_to_disk(output_dir)
            del out
            gc.collect()
            out = DatasetDict.load_from_disk(output_dir)

        return out
    finally:
        gc.collect()


def upload_writingprompts_to_hf():
    dd = load_and_process_writingprompts()
    dd.push_to_hub("avgJo3/writingprompts-strat", private=False)
    del dd 

    
#load_and_process_writingprompts()
if __name__ == "__main__":
    upload_writingprompts_to_hf()