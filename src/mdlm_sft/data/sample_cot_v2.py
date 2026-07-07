import gc
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, Tuple
import random
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset, concatenate_datasets
from typing import Dict, List, Optional, Any
from .utils import normalize_whitespace, add_hash_id, count_tokens_fn, _ratio_to_split_name, sent_tokenize, split_sentences_to_prompt_completion    
import numpy as np
from collections import defaultdict
import gc
from collections import Counter
from typing import Optional

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


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



def create_nested_stratified_train_splits(
    train_pool: Dataset,
    task_column: str = "task",
    split_ratios: Optional[List[float]] = None,
    random_seed: int = 42,
) -> Dict[float, Dataset]:
    """
    Produce nested-prefix stratified training splits from a pre-carved train_pool.

    Given `train_pool` (already filtered and carved from eval/test), returns
    a dict mapping each ratio in `split_ratios` to a Dataset containing that
    fraction of items per task label. Splits are nested: for r1 < r2, the
    r1-split is a strict subset of the r2-split.

    Args:
        train_pool:     the carved training pool (post filter, post eval/test carve).
        task_column:    the label column used for per-task stratification.
        split_ratios:   fractions in (0, 1]; must include 1.0 if you want full pool.
        random_seed:    seed for reproducibility (used only for numpy/random state).

    Returns:
        Dict[ratio -> Dataset], sorted ascending by ratio.
    """
    if split_ratios is None:
        split_ratios = [0.25, 0.5, 1.0]

    if not split_ratios or any(r <= 0 or r > 1 for r in split_ratios):
        raise ValueError(f"split_ratios must be in (0, 1]; got {split_ratios}")

    np.random.seed(random_seed)
    random.seed(random_seed)

    unique_tasks: List[str] = sorted(train_pool.unique(task_column))
    print(f"Building nested splits over {len(unique_tasks)} task types")

    # Build task -> [row indices] over the training pool.
    # Fetching the column once is much faster than iterating rows.
    task_indices: Dict[str, List[int]] = {task: [] for task in unique_tasks}
    for idx, task in enumerate(train_pool[task_column]):
        task_indices[task].append(idx)

    # Nested-prefix splits: for each task, take the first ceil(ratio * N_task)
    # indices in pool order. Because pool order is already shuffled (from the
    # upstream stratified carve), the "first K" is effectively a random subset
    # of that task's items, and nesting is guaranteed by construction.
    splits: Dict[float, Dataset] = {}
    for ratio in sorted(split_ratios):
        selected_indices = [
            i
            for task in unique_tasks
            for i in task_indices[task][: int(len(task_indices[task]) * ratio)]
        ]
        splits[ratio] = train_pool.select(selected_indices)
        print(f"  ratio={ratio:.3f}: {len(splits[ratio]):,} items")

    return splits

def format_cot_completion_pre(batch):
    """Fold rationale into completion using the <answer> special token as boundary."""
    return {
        "completion": [
            f"{r} <answer> {t}"
            for r, t in zip(batch["rationale"], batch["completion"])
        ],
    }

def load_and_process_cot_collection(
    train_n: int = 1_500_000,
    val_n: int = 15_000,
    test_n: int = 15_000,
    carve_val_test_n: int = 50_000,
    split_ratios: Optional[list[float]] = None,
    max_tokens: int = 1024,
    coverage_target: float = 0.99,
    label_col: str = "task",
    tokenizer_name: str = "gpt2",
    batch_size: int = 8_000,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> DatasetDict:
    """
    CoT-Collection preprocessing pipeline (scaling-study output shape).

    Stages (in order):
        1.  Load (single source split)
        2.  Normalize whitespace (source, target, rationale)
        3.  Rename source→prompt, target→completion
        4.  Add hash id (from prompt + completion)
        5.  Count tokens (prompt, completion, rationale) + total_tokens
        6.  Length filter (total_tokens <= max_tokens)
        7.  Coverage filter (keep labels reaching cumulative coverage_target)
        8.  Drop singleton labels (defensive: ensures stratified carve is valid)
        9.  Stratified carve: train_pool vs val_test_pool (by absolute count)
       10.  Random carve: val_pool vs test_pool (task-agnostic at test time)
       11.  Sample val + test to exact target sizes
       12.  Cap train_pool at train_n via stratified_sample (definitional 100%)
       13.  Nested-prefix stratified training splits at each ratio
       14.  Assemble DatasetDict (eval + test + per-ratio train splits)
       15.  Save + reload for memory-mapping

    Output shape:
        {
            "validation":   Dataset of val_n rows,
            "test":         Dataset of test_n rows,
            "train_{r}pct": Dataset of ~(r * train_n) rows, for each r in split_ratios,
        }
    """
    if split_ratios is None:
        split_ratios = [0.25, 0.5, 1.0]

    try:
        # --- 1. Load ---
        dd = load_dataset("avgJo3/Cot-collection-datasets-4.8.5", split="train")

        # --- 2. Normalize whitespace on all text fields ---
        dd = (
            dd
            .map(normalize_whitespace, fn_kwargs={"field_name": "source"},
                 batched=True, batch_size=batch_size, desc="Normalizing whitespace (source)")
            .map(normalize_whitespace, fn_kwargs={"field_name": "target"},
                 batched=True, batch_size=batch_size, desc="Normalizing whitespace (target)")
            .map(normalize_whitespace, fn_kwargs={"field_name": "rationale"},
                 batched=True, batch_size=batch_size, desc="Normalizing whitespace (rationale)")
        )

        # --- 3. Rename to canonical schema ---
        dd = dd.rename_columns({"source": "prompt", "target": "completion"})

        # --- 4. Hash id (anchored to prompt + completion) ---
        dd = dd.map(
            add_hash_id, with_indices=True,
            fn_kwargs={"field_names": ("prompt", "completion")},
            batched=True, batch_size=batch_size, desc="Adding hash ID",
        )

        # --- 5. Token counting + total ---
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = int(1e9)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"]})

        dd = (
            dd
            .map(count_tokens_fn, fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"},
                 batched=True, batch_size=batch_size, desc="Counting tokens (prompt)")
            .map(count_tokens_fn, fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"},
                 batched=True, batch_size=batch_size, desc="Counting tokens (completion)")
            .map(count_tokens_fn, fn_kwargs={"tokenizer": tokenizer, "field_name": "rationale"},
                 batched=True, batch_size=batch_size, desc="Counting tokens (rationale)")
            .map(
                lambda x: {
                    "total_tokens": [
                        p + c + r
                        for p, c, r in zip(
                            x["prompt_token_count"],
                            x["completion_token_count"],
                            x["rationale_token_count"],
                        )
                    ]
                },
                batched=True, desc="Total token count",
            )
        )
        del tokenizer
        gc.collect()

        # --- 6. Length filter (cheap; before coverage aggregation) ---
        n_before = len(dd)
        dd = dd.filter(
            lambda x: x["total_tokens"] <= max_tokens,
            desc=f"Length filter (<= {max_tokens} tokens)",
        )
        print(f"[length] {n_before:,} -> {len(dd):,} "
              f"({len(dd) / n_before:.1%} retained)")

        # --- 7. Coverage filter on task label ---
        kept_labels = compute_coverage_labels(
            dd, label_col=label_col, coverage=coverage_target,
        )
        n_before = len(dd)
        dd = dd.filter(
            lambda x: x[label_col] in kept_labels,
            desc=f"Coverage filter ({coverage_target:.0%} target)",
        )
        print(f"[coverage] {n_before:,} -> {len(dd):,} "
              f"({len(dd) / n_before:.1%} retained), "
              f"{len(kept_labels)} labels kept")

        # --- 8. Defensive: drop singleton labels (required for stratified carve) ---
        label_counts = Counter(dd[label_col])
        singletons = {lbl for lbl, c in label_counts.items() if c < 2}
        if singletons:
            n_before = len(dd)
            dd = dd.filter(
                lambda x: x[label_col] not in singletons,
                desc=f"Dropping {len(singletons)} singleton labels",
            )
            print(f"[singleton] {n_before:,} -> {len(dd):,}, "
                  f"dropped {len(singletons)} label(s)")

        # --- Sanity check before carving ---
        required = train_n + carve_val_test_n
        if len(dd) < required:
            raise ValueError(
                f"Post-filter pool ({len(dd):,}) < required "
                f"({required:,} = {train_n:,} train + {carve_val_test_n:,} carve). "
                f"Reduce train_n, reduce carve_val_test_n, or relax coverage_target."
            )

        # --- 9. Stratified carve: train_pool vs val_test_pool ---
        # HF requires ClassLabel for stratified split
        unique_labels = sorted(dd.unique(label_col))
        dd = dd.cast_column(label_col, ClassLabel(names=unique_labels))

        first_carve = dd.train_test_split(
            test_size=carve_val_test_n,
            stratify_by_column=label_col,
            seed=seed,
            shuffle=True,
        )
        train_pool = first_carve["train"]
        val_test_pool = first_carve["test"]
        del first_carve, dd
        gc.collect()

        # --- 10. Random carve: val_pool vs test_pool (task-agnostic at test) ---
        second_carve = val_test_pool.train_test_split(
            test_size=0.5,
            seed=seed,
            shuffle=True,
        )
        val_pool = second_carve["train"]
        test_pool = second_carve["test"]
        del second_carve, val_test_pool
        gc.collect()

        print(f"[carve] train_pool={len(train_pool):,}, "
              f"val_pool={len(val_pool):,}, test_pool={len(test_pool):,}")

        # --- 11. Sample val + test to exact target sizes ---
        val_ds = stratified_sample(val_pool, n=val_n, label_col=label_col, seed=seed)
        test_ds = random_sample(test_pool, n=test_n, seed=seed)
        del val_pool, test_pool
        gc.collect()

        print(f"[sample eval/test] validation={len(val_ds):,}, test={len(test_ds):,}")

        # --- 12. Cap train_pool at train_n (definitional 100% for scaling study) ---
        n_before = len(train_pool)
        train_pool = stratified_sample(
            train_pool, n=train_n, label_col=label_col, seed=seed,
        )
        print(f"[cap] train_pool {n_before:,} -> {len(train_pool):,} "
              f"(target: {train_n:,})")

        # --- 13. Nested-prefix stratified training splits ---
        train_splits = create_nested_stratified_train_splits(
            train_pool,
            task_column=label_col,
            split_ratios=split_ratios,
            random_seed=seed,
        )
        del train_pool
        gc.collect()

        # --- 14. Assemble output DatasetDict ---
        out_dict: dict[str, Dataset] = {
            "validation": val_ds,
            "test":       test_ds,
        }
        for ratio in sorted(split_ratios):
            out_dict[_ratio_to_split_name(ratio)] = train_splits[ratio]

        out = DatasetDict(out_dict)
        out = out.map(
            format_cot_completion_pre,
            batched=True, batch_size=batch_size,
            remove_columns=["rationale"],   # drop as part of the map
            desc="Formatting CoT completions",
        )        
        del val_ds, test_ds, train_splits, out_dict
        gc.collect()

        print(f"[output] {list(out.keys())}")
        for split_name, ds in out.items():
            print(f"  {split_name}: {len(ds):,} items")

        # --- 15. Persist + reload for memory-mapping ---
        if output_dir is not None:
            out.save_to_disk(output_dir)
            del out
            gc.collect()
            out = DatasetDict.load_from_disk(output_dir)

        return out

    finally:
        gc.collect()

def upload_cot_to_hf():
    dd = load_and_process_cot_collection()
    dd.push_to_hub("avgJo3/cot-strat", private=False)
    del dd 


if __name__ == "__main__":
    upload_cot_to_hf()