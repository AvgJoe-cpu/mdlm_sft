import gc
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, Tuple
import random
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset, concatenate_datasets
from typing import Dict, List, Optional, Any
from .utils import normalize_whitespace, add_hash_id, count_tokens_fn, _ratio_to_split_name, sent_tokenize, split_sentences_to_prompt_completion    
import numpy as np

import datasets
datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB


def create_nested_stratified_splits_hf(
    dataset: Dataset,
    task_column: str = "task_type",      # Column with task/category labels
    eval_ratio: float = 0.10,
    split_ratios: Optional[List[float]] = None,
    random_seed: int = 42,
    token_column: Optional[str] = None,  # e.g. "total_tokens" or None
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    if split_ratios is None:
        split_ratios = [0.25, 0.5, 1.0]

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

    def _stratified_split(ds: Dataset, test_size: float) -> Tuple[Dataset, Dataset]:
        out = ds.train_test_split(test_size=test_size, stratify_by_column=task_column, seed=random_seed, shuffle=True)
        return out["train"], out["test"]

    train_pool, eval_pool = _stratified_split(dataset, test_size=eval_ratio)

    # Test set is a random (non-stratified) half of the eval pool —
    # we treat task labels as unknown at test time.
    eval_test = eval_pool.train_test_split(test_size=0.5, seed=random_seed, shuffle=True)
    eval_ds, test_ds = eval_test["train"], eval_test["test"]

    print(f"Stratified eval set: {len(eval_ds):,} examples")
    print(f"Stratified test set: {len(test_ds):,} examples")
    print(f"Training pool: {len(train_pool):,} examples")
    
    unique_tasks: List[str] = sorted(train_pool.unique(task_column))
    print(f"Found {len(unique_tasks)} task types: {unique_tasks}")

    # Build task -> [row indices] over the training pool.
    # Fetching the column once is dramatically faster than iterating rows.
    task_indices: Dict[str, List[int]] = {task: [] for task in unique_tasks}
    for idx, task in enumerate(train_pool[task_column]):
        task_indices[task].append(idx)

    # Nested-prefix splits at each ratio (same RNG-free determinism as before:
    # we take the first `ratio * N_task` indices per task, in pool order).
    splits: Dict[float, Dict[str, Any]] = {}
    for ratio in sorted(split_ratios):
        r = min(ratio, 1.0)
        selected_indices = [
            i
            for task in unique_tasks
            for i in task_indices[task][: int(len(task_indices[task]) * r)]
        ]
        train_split = train_pool.select(selected_indices)
        splits[ratio] = {"train": train_split, "size": len(train_split), "ratio": ratio}
        print(f"Split {ratio:.3f}: {len(train_split):,} examples (stratified)")

    return {
        "strat_eval": eval_ds,
        "strat_test": test_ds,
        "splits": splits,
        "metadata": {
            "total_after_filter": len(dataset),
            "eval_ratio": eval_ratio,
            "seed": random_seed,
            "task_column": task_column,
            "split_ratios": split_ratios,
        },
    }

def format_cot_completion_pre(batch):  # C1: (CoT + labels)
    return {
        "prompt": batch["prompt"],
        "completion": [
            f"{r} {t}"
            for r, t in zip(batch["rationale"], batch["completion"])
        ],
    }

def load_and_process_cot_collection(BATCH_SIZE: int = 8000, OUTPUT_DIR=None) -> DatasetDict:
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
                batched=True, desc="Calculating total tokens",
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
            split_ratios=[0.25,0.5, 1.0],
            random_seed=42,
        )
        del dataset
        gc.collect()

        # Build DatasetDict: eval + every training split
        dataset_dict: Dict[str, Dataset] = {"eval": results["strat_eval"], "test": results["strat_test"]}
        for ratio, split_info in sorted(results["splits"].items()):
            dataset_dict[_ratio_to_split_name(ratio)] = split_info["train"]

        dd = DatasetDict(dataset_dict)
        dd = dd.map(format_cot_completion_pre, batched=True, batch_size=BATCH_SIZE, desc="Formatting CoT completions with <think> tags").remove_columns(["rationale"])    

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


if __name__ == "__main__":
    upload_cot_to_hf()