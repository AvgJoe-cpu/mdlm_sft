# (1) Dataset 
# (2) Model -> ChatTemplate -> Tokenizer
import hashlib
from typing import List

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from torch.cuda import seed
from transformers import AutoTokenizer

# 1.    Load dataset
# 1.1   add IDs
# 1.2   train-test split (in distribution)
# 2.    Load tokenizer -> CountTokens -> Threshold-based filtering
# 3.    Create Training-Mix
# 3.1   Train: A: 5% warmup (no CoT, labels only), + B: 5% CoT-traces (CoT, no labels); C: 90% CoT-traces (CoT + labels)
#       A1: warmup: 2.5%: prompt-completion                                   (no CoT, labels)
#       A2: warmup: 2.5%: prompt with CoT-traces, completion with labels only (CoT, labels)
#       B1: condtt: 5%: prompt - CoT (no labels)                              (prompt, CoT are labels)
#       C1: condtt: 45%: prompt - CoT (no labels) + labels                    (prompt, CoT + labels)
#       C2: condtt: 45%: prompt - labels + CoT (no labels)                    (prompt, labels + CoT)
# 3.2   (Eval: 100% CoT-traces (CoT + labels))
#------------------------------------------------------------------------------------------------------------------------
# Memory management:
import datasets
datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB

# pre processing functions
def normalize_whitespace(batch, field_name=None):
    batch[field_name] = [" ".join(text.split()) for text in batch[field_name]]
    return batch

def count_tokens_fn(batch, tokenizer=None, field_name=None):
    token_counts = [
        len(tokenizer.encode(text, add_special_tokens=False)) 
        for text in batch[field_name]
    ]
    return {f"{field_name}_token_count": token_counts}

def add_hash_id(batch, indices, field_names=("prompt", "completion")):
    ids = []
    for i, idx in enumerate(indices):
        combined = "|".join(batch[f][i] for f in field_names)
        hash_id = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        ids.append(f"{hash_id}_{idx}")
    return {"id": ids}


# A1: (no CoT, labels)
def format_no_cot(example):
    example["prompt"] = example["source"]
    example["completion"] = f"<answer> {example['target']}"
    return example

# A2: (source + CoT, labels)
def format_cot_labels(example):
    example["prompt"] = f"{example['source']} <think> {example['rationale']} </think>"
    example["completion"] = f"<answer> {example['target']}"
    return example

# B1: (CoT, no labels)
def format_cot_only(example):
    example["prompt"] = example["source"]
    example["completion"] = f"<think> {example['rationale']} </think>"
    return example

# C1: (CoT + labels)
def format_cot_completion_pre(example):
    example["prompt"] = example["source"] 
    example["completion"] = f"<answer> {example['target']} <think> {example['rationale']} </think>"
    return example

# C2: (labels + CoT)
def format_cot_completion_post(example):
    example["prompt"] = example["source"]
    example["completion"] = f"<think> {example['rationale']} </think> <answer> {example['target']}"
    return example



def split_dataset_by_ratios(dataset: Dataset, ratios: List[float]) -> DatasetDict:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    dataset = dataset.shuffle(seed=12)

    n = len(dataset)
    sizes = [int(r * n) for r in ratios]

    remainder = n - sum(sizes)
    if remainder > 0:
        print(f"Warning: Ratios do not perfectly divide the dataset. Distributing {remainder} extra samples.")
    for i in range(remainder):
        sizes[i] += 1
    if sum(sizes) != n:
        raise ValueError(f"Sizes do not sum to dataset length: {sum(sizes)} vs {n}")

    splits = {}
    start = 0
    for i, (ratio, size) in enumerate(zip(ratios, sizes)):
        key = f"{round(ratio * 100, 4):g}%-{i}_split"
        splits[key] = dataset.select(range(start, start + size))
        start += size
    return DatasetDict(splits)


# (1) Dataset 
# (2) Model -> ChatTemplate -> Tokenizer
import hashlib
import gc
from typing import List

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

import datasets
datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB

# ── Pre-processing functions ────────────────────────────────────────────────────
def normalize_whitespace(batch, field_name=None):
    batch[field_name] = [" ".join(text.split()) for text in batch[field_name]]
    return batch

def count_tokens_fn(batch, tokenizer=None, field_name=None):
    token_counts = [
        len(tokenizer.encode(text, add_special_tokens=False)) 
        for text in batch[field_name]
    ]
    return {f"{field_name}_token_count": token_counts}

def add_hash_id(batch, indices, field_names=("prompt", "completion")):
    ids = []
    for i, idx in enumerate(indices):
        combined = "|".join(batch[f][i] for f in field_names)
        hash_id = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        ids.append(f"{hash_id}_{idx}")
    return {"id": ids}

# ── Formatting functions ────────────────────────────────────────────────────────
def format_no_cot(example):               # A1: (no CoT, labels)
    example["prompt"]     = example["source"]
    example["completion"] = f"[ANSWER] {example['target']}"
    return example

def format_cot_labels(example):           # A2: (source + CoT, labels)
    example["prompt"]     = f"{example['source']} <think> {example['rationale']} </think>"
    example["completion"] = f"[ANSWER] {example['target']}"
    return example

def format_cot_only(example):             # B1: (CoT, no labels)
    example["prompt"]     = example["source"]
    example["completion"] = f"<think> {example['rationale']} </think>"
    return example

def format_cot_completion_pre(example):   # C1: (CoT + labels)
    example["prompt"]     = example["source"]
    example["completion"] = f"[ANSWER] {example['target']} <think> {example['rationale']} </think>"
    return example

def format_cot_completion_post(example):  # C2: (labels + CoT)
    example["prompt"]     = example["source"]
    example["completion"] = f"<think> {example['rationale']} </think> [ANSWER] {example['target']}"
    return example

# ── Split utility ───────────────────────────────────────────────────────────────
def split_dataset_by_ratios(dataset: Dataset, ratios: List[float]) -> DatasetDict:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    dataset = dataset.shuffle(seed=12)
    n       = len(dataset)
    sizes   = [int(r * n) for r in ratios]

    remainder = n - sum(sizes)
    if remainder > 0:
        print(f"Warning: Ratios do not perfectly divide the dataset. Distributing {remainder} extra samples.")
    for i in range(remainder):
        sizes[i] += 1
    if sum(sizes) != n:
        raise ValueError(f"Sizes do not sum to dataset length: {sum(sizes)} vs {n}")

    splits, start = {}, 0
    for i, (ratio, size) in enumerate(zip(ratios, sizes)):
        key          = f"{round(ratio * 100, 4):g}%-{i}_split"
        splits[key]  = dataset.select(range(start, start + size))
        start       += size
    return DatasetDict(splits)


# ── Main pipeline ───────────────────────────────────────────────────────────────
try:
    # 1. Load + normalize
    raw_ds = load_dataset("avgJo3/Cot-collection-datasets-4.8.5", split="train")
    raw_ds = raw_ds.select(range(10000))
    print(raw_ds[0])

    pilot_ds = (
        raw_ds
        .map(normalize_whitespace, batched=True, fn_kwargs={"field_name": "source"},    desc="Normalizing whitespace in 'source'")
        .map(normalize_whitespace, batched=True, fn_kwargs={"field_name": "rationale"}, desc="Normalizing whitespace in 'rationale'")
        .map(normalize_whitespace, batched=True, fn_kwargs={"field_name": "target"},    desc="Normalizing whitespace in 'target'")
    )
    del raw_ds; gc.collect()

    # 2. Training-mix splits
    dd = split_dataset_by_ratios(pilot_ds, [0.025, 0.025, 0.05, 0.45, 0.45])
    del pilot_ds; gc.collect()

    _cols_to_drop = ["source", "rationale", "target", "task", "type"]
    dd["2.5%-0_split"] = dd["2.5%-0_split"].map(format_no_cot).remove_columns(_cols_to_drop)
    dd["2.5%-1_split"] = dd["2.5%-1_split"].map(format_cot_labels).remove_columns(_cols_to_drop)
    dd["5%-2_split"]   = dd["5%-2_split"].map(format_cot_only).remove_columns(_cols_to_drop)
    dd["45%-3_split"]  = dd["45%-3_split"].map(format_cot_completion_pre).remove_columns(_cols_to_drop)
    dd["45%-4_split"]  = dd["45%-4_split"].map(format_cot_completion_post).remove_columns(_cols_to_drop)

    print("Processing complete — sample outputs:")
    for key in dd:
        print(f"\n--- {key} ---")
        for i in range(2):
            print(f"Prompt: {dd[key][i]['prompt']}")
            print(f"Completion: {dd[key][i]['completion']}")
    print(dd)

    # 3. Token counting + IDs
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(10e19)

    dd = (
        dd
        .map(add_hash_id,     batched=True, with_indices=True, fn_kwargs={"field_names": ("prompt", "completion")},                   desc="Adding unique hash IDs")
        .map(count_tokens_fn, batched=True,                    fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"},            desc="Counting tokens in 'prompt'")
        .map(count_tokens_fn, batched=True,                    fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"},        desc="Counting tokens in 'completion'")
        .map(lambda x: {"total_tokens": [p + c for p, c in zip(x["prompt_token_count"], x["completion_token_count"])]}, batched=True, desc="Calculating total tokens")
    )
    del tokenizer; gc.collect()

    # 4. Flatten + shuffle + train/test split
    flat_ds = concatenate_datasets([
        dd[split].map(lambda x: {"split": [split] * len(x["prompt"])}, batched=True)
        for split in dd
    ])
    del dd; gc.collect()
    print("Flattened dataset")

    flat_ds = flat_ds.shuffle(seed=12)
    train_test_split = split_dataset_by_ratios(flat_ds, [0.2, 0.8])
    del flat_ds; gc.collect()

    print("Final train/test splits:")
    for split_name in train_test_split.keys():
        print(f"  - '{split_name}': {len(train_test_split[split_name])} samples")

    # 5. Save
    train_test_split.save_to_disk("artifacts/datasets/base/cot")
    print("Saved to disk: artifacts/datasets/base/cot")

finally:
    gc.collect()