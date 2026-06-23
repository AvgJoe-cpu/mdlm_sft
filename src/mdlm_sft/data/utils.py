import gc
import hashlib
import random
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset, concatenate_datasets
from typing import Dict, List, Optional, Any
import torch
from wtpsplit import SaT
import numpy as np
from transformers import AutoTokenizer

# 2do: move to utils
def normalize_whitespace(batch, field_name=None):
    batch[field_name] = [" ".join(text.split()) for text in batch[field_name]]
    return batch

# 2do: move to utils
def count_tokens_fn(batch: Dict[str, Any], tokenizer: Optional[Any] = None, field_name: Optional[str] = None) -> Dict[str, List[int]]:
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")
    token_counts = [
        len(tokenizer.encode(text, add_special_tokens=False))  # type: ignore
        for text in batch[field_name]
    ]
    return {f"{field_name}_token_count": token_counts}

# 2do: move to utils
def add_hash_id(batch, indices, field_names=("prompt", "completion")):
    ids = []
    for i, idx in enumerate(indices):
        combined = "|".join(batch[f][i] for f in field_names)
        hash_id = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        ids.append(f"{hash_id}_{idx}")
    return {"id": ids}

# 2do: move to utils
def create_nested_stratified_splits_hf(
    dataset: Dataset,
    task_column: str = "task_type",      # Column with task/category labels
    eval_ratio: float = 0.1,
    split_ratios: Optional[List[float]] = None,
    random_seed: int = 42,
    token_column: Optional[str] = None,  # e.g. "total_tokens" or None
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    if split_ratios is None:
        split_ratios = [0.125, 0.25, 0.5, 1.0]

    np.random.seed(random_seed)
    random.seed(random_seed)

    if token_column is not None:
        def filter_by_tokens(example: Dict[str, Any]) -> bool:
            tokens = example.get(token_column, 0)
            return tokens <= max_tokens
        
        dataset = dataset.filter(filter_by_tokens)
        print(f"After token filtering (<= {max_tokens}): {len(dataset):,} examples")

    unique_labels = sorted(dataset.unique(task_column))  # type: ignore
    dataset = dataset.cast_column(task_column, ClassLabel(names=unique_labels))

    split_ds = dataset.train_test_split(
        test_size=eval_ratio,
        stratify_by_column=task_column,
        seed=random_seed,
        shuffle=True
    )
    
    train_pool: Dataset = split_ds["train"]
    eval_ds: Dataset = split_ds["test"]
    
    print(f"Stratified eval set: {len(eval_ds):,} examples")
    print(f"Training pool: {len(train_pool):,} examples")

    unique_tasks: List[str] = sorted(train_pool.unique(task_column))
    print(f"Found {len(unique_tasks)} task types: {unique_tasks}")

    task_indices: Dict[str, List[int]] = {task: [] for task in unique_tasks}
    for idx, example in enumerate(train_pool):
        task = example[task_column]
        task_indices[task].append(idx)

    splits = {}
    sorted_ratios = sorted(split_ratios)
    
    for ratio in sorted_ratios:
        selected_indices = []
        
        for task in unique_tasks:
            group_indices = task_indices[task]
            group_size = len(group_indices)
            split_size = int(group_size * min(ratio, 1.0))
            selected_indices.extend(group_indices[:split_size])  # Nested prefix
        
        train_split = train_pool.select(selected_indices)
        
        splits[ratio] = {
            'train': train_split,
            'size': len(train_split),
            'ratio': ratio
        }
        print(f"Split {ratio:.3f}: {len(train_split):,} examples (stratified)")

    return {
        'strat_eval': eval_ds,
        'splits': splits,
        'metadata': {
            'total_after_filter': len(dataset),
            'eval_ratio': eval_ratio,
            'seed': random_seed,
            'task_column': task_column,
            'split_ratios': split_ratios
        }
    }

# 2do: move to utils - convert ratio to split name
def _ratio_to_split_name(ratio: float) -> str:
    return f"strat_train_{round(ratio * 100):d}pct"


# TIS-specific processing functions
def sent_tokenize(batch, model=None, text_field=None):
    return {
        "sentences": [
            list(model.split(text))
            for text in batch[text_field]
        ]
    }

def split_sentences_to_prompt_completion(batch):
    prompts, completions = [], []
    for sentences in batch["sentences"]:
        mid_idx = round(len(sentences) / 2)
        prompts.append(" ".join(sentences[:mid_idx]))
        completions.append(" ".join(sentences[mid_idx:]))
    return {"prompt": prompts, "completion": completions}