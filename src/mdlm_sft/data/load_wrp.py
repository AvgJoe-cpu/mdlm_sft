from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from mdlm_sft.paths import DATASETS
import hashlib

#from .shared import add_hash_id, count_tokens # temporalily redefined here to avoid circular imports, can be cleaned up later


def normalize_whitespace(example, field_name=None):
    example[field_name] = " ".join(example[field_name].split())
    return example

def count_tokens_fn(example, tokenizer=None, field_name=None):
    token_count = len(tokenizer.encode(example[field_name], add_special_tokens=False))
    return {f"{field_name}_token_count": token_count}

def add_hash_id(example, idx, field_names=("prompt", "completion")):
    combined = "|".join(example[f] for f in field_names)
    hash_id = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return {"id": f"{hash_id}_{idx}"}


def create_nested_subdatasets(
    D: Dataset,
    sizes: list[int],
    seed: int = 42,
) -> dict[str, Dataset]:
    if len(sizes) == 0:
        raise ValueError("`sizes` must be non-empty.")
    if any(s <= 0 for s in sizes):
        raise ValueError("All sizes must be positive.")
    if any(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1)):
        raise ValueError("`sizes` must be strictly ascending.")
    if sizes[-1] > len(D):
        raise ValueError(f"largest size {sizes[-1]} exceeds |D|={len(D)}.")

    shuffled = D.shuffle(seed=seed)
    subsets: dict[str, Dataset] = {}
    for i, target_size in enumerate(sizes):
        subsets[f"D{i}"] = shuffled.select(range(target_size))
    return subsets


def verify_nested(subsets: dict[str, Dataset], sizes: list[int]) -> None:
    keys = [f"D{i}" for i in range(len(sizes))]

    for k, s in zip(keys, sizes):
        assert len(subsets[k]) == s, f"{k}: expected {s}, got {len(subsets[k])}"

    for k in keys:
        ids = subsets[k]["id"]
        assert len(set(ids)) == len(ids), f"{k}: duplicate ids found"

    for a, b in zip(keys[:-1], keys[1:]):
        ids_a = set(subsets[a]["id"])
        ids_b = set(subsets[b]["id"])
        assert ids_a.issubset(ids_b), f"{a} ⊄ {b} (nesting broken)"

    for a, b in zip(keys[:-1], keys[1:]):
        n = len(subsets[a])
        assert (
            subsets[a]["id"] == subsets[b]["id"][:n]
        ), f"{a} is not a row-order prefix of {b}"


def sample_by_token_range(
    ds: Dataset,
    token_range: tuple[int, int],
    num_samples: int,
    seed: int = 42,
) -> Dataset:
    lo, hi = token_range
    if lo >= hi:
        raise ValueError(f"token_range: lo={lo} must be < hi={hi}")
    
    pool = ds.filter(lambda x: lo < x["total_tokens"] <= hi)
    if num_samples > len(pool):
        raise ValueError(f"num_samples={num_samples} exceeds pool size={len(pool)} for range=({lo}, {hi}]")
    
    return pool.shuffle(seed=seed).select(range(num_samples))


def load_and_process_wrp(demo_size=10000, force_reprocess=False, to_hub=False):
    dataset_key = "wrp"

    base_path = DATASETS[dataset_key]["base_path"]
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"\nResolving dataset base path for '{dataset_key}':")
    print(f"Save location: {base_path}")
    print(f"Dataset: {DATASETS[dataset_key]['name']}")

    print(f"\nLoading {DATASETS[dataset_key]['name']}...")
    dd = load_dataset(DATASETS[dataset_key]["hf-path"])

    print("\nDataset structure:")
    print(dd)

    print("\nDataset splits (original):")
    for split_name in dd.keys():
        print(f"  - '{split_name}': {len(dd[split_name])} samples")

    if not force_reprocess and all((base_path / split_name).exists() for split_name in dd.keys()):
        for split_name in dd.keys():
            print(f"  - {split_name}: {base_path / split_name}")
        print("\nTo reprocess, set force_reprocess=True")
        return

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(10e19)

    # STEP 1: PROCESS EACH SPLIT
    processed_splits = {}
    for split_name in dd.keys():

        if demo_size:
            ds = dd[split_name].select(range(min(demo_size, len(dd[split_name]))))
            print(f"  Using {len(ds)} samples (demo mode)")
        else:
            ds = dd[split_name]
            print(f"  Using full split: {len(ds)} samples")

        ds = ds.rename_column("story", "completion")
        ds = ds.map(
            normalize_whitespace,
            fn_kwargs={"field_name": "prompt"},
            desc="Normalizing whitespace in 'prompt'"
        ).map(
            normalize_whitespace,
            fn_kwargs={"field_name": "completion"},
            desc="Normalizing whitespace in 'completion'"
        )
        ds = ds.map(
            count_tokens_fn,
            fn_kwargs={"tokenizer": tokenizer, "field_name": "prompt"},
        ).map(
            count_tokens_fn,
            fn_kwargs={"tokenizer": tokenizer, "field_name": "completion"},
        ).map(
            lambda x: {"total_tokens": x["prompt_token_count"] + x["completion_token_count"]}
        )
        ds = ds.map(add_hash_id, with_indices=True, desc="Adding hash IDs")

        processed_splits[split_name] = ds
        print(ds)

    # STEP 2: CREATE NESTED DS FROM PROCESSED TRAIN SPLIT
    train_ds = processed_splits["train"]
    ids = train_ds["id"]
    print(f"Total train samples: {len(ids)}")
    print(f"Unique IDs: {len(set(ids))}")
    print(f"Duplicates: {len(ids) - len(set(ids))}")
    size_pcts = [0.25, 0.5, 1.0]

    sizes = sorted(set(max(1, round(p * len(train_ds))) for p in size_pcts))
    subset_percents = [f"{p * 100:.1f}%" for p in size_pcts]

    print(f"Creating nested subsets with sizes: {sizes} ({subset_percents})")
    subsets = create_nested_subdatasets(train_ds, sizes)
    verify_nested(subsets, sizes)
    print(subsets)


    # STEP 3: STRATIFIED SAMPLING OF TEST SPLIT
    # 3a: THRESHOLD-BASED SAMPLING ON TOKEN COUNT
    train_ds = processed_splits["train"]
    test_ds = processed_splits["test"]

    threshold_splits = {
        "train_T0_0_512":    sample_by_token_range(train_ds, token_range=(0, 512),    num_samples=5000, seed=42),
        "train_T1_512_1024": sample_by_token_range(train_ds, token_range=(512, 1024), num_samples=5000, seed=42),
        "test_T0_0_512":     sample_by_token_range(test_ds,  token_range=(0, 512),    num_samples=2500, seed=42),
        "test_T1_512_1024":  sample_by_token_range(test_ds,  token_range=(512, 1024), num_samples=2500, seed=42),
    }

    # COMBINE ALL SPLITS INTO A SINGLE DATASETDICT
    from datasets import DatasetDict

    final_dd = DatasetDict({
        **processed_splits,
        **{f"nested_{k}": v for k, v in subsets.items()},
        **threshold_splits,
    })
    
    if to_hub:
        final_dd.push_to_hub(
            repo_id="avgJo3/wrp",
            private=False,
        )

        # 4. Quick sanity check
        reloaded = load_dataset("avgJo3/wrp")
        assert list(final_dd.keys()) == list(reloaded.keys())
        print("Upload & reload verified!")



if __name__ == "__main__":
    load_and_process_wrp(demo_size=1000, force_reprocess=True)


