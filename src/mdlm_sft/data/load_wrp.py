"""Load and process Writing Prompts dataset."""

from datasets import load_dataset
from transformers import AutoTokenizer
from mdlm_sft.paths import DATASETS
from .shared import add_hash_id, count_tokens


def rename_columns(example):
    """Rename 'story' to 'completion', keep 'prompt' as is."""
    return {
        "prompt": example["prompt"],
        "completion": example["story"]
    }


def load_and_process_wrp(demo_size=10000, force_reprocess=False):
    """
    Load and process Writing Prompts dataset.
    
    Args:
        demo_size: Number of samples per split (None for full dataset)
        force_reprocess: If True, reprocess even if dataset exists
    
    Returns:
        None (saves processed datasets to disk)
    """
    dataset_key = "wrp"
    
    # Resolve the base path. This script is the *producer* of the base
    # dataset, so reference the path directly from DATASETS and create it.
    # (resolve_dataset_base_path is consumer-side: it raises if the dir is
    # missing, which would make a fresh clone impossible to bootstrap.)
    base_path = DATASETS[dataset_key]["base_path"]
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"\nResolving dataset base path for '{dataset_key}':")
    print(f"Save location: {base_path}")
    print(f"Dataset: {DATASETS[dataset_key]['name']}")
    
    # Load dataset
    print(f"\nLoading {DATASETS[dataset_key]['name']}...")
    dd = load_dataset(DATASETS[dataset_key]["hf-path"])
    
    print("\nDataset structure:")
    print(dd)
    
    print("\nDataset splits (original):")
    for split_name in dd.keys():
        print(f"  - '{split_name}': {len(dd[split_name])} samples")
    
    # Check if already processed (inline check)
    if not force_reprocess and all((base_path / split_name).exists() for split_name in dd.keys()):
        print("\n" + "="*60)
        print("⚠️  Dataset already processed! Skipping...")
        print("="*60)
        print("\nExisting locations:")
        for split_name in dd.keys():
            print(f"  - {split_name}: {base_path / split_name}")
        print("\nTo reprocess, set force_reprocess=True")
        return
    
    # Load tokenizer once
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Process each split
    for split_name in dd.keys():
        print(f"\n{'='*60}")
        print(f"Processing split: {split_name}")
        print(f"{'='*60}")
        
        # Select samples (demo or full)
        if demo_size:
            ds = dd[split_name].select(range(min(demo_size, len(dd[split_name]))))
            print(f"  Using {len(ds)} samples (demo mode)")
        else:
            ds = dd[split_name]
            print(f"  Using full split: {len(ds)} samples")
        
        print(f"  Original columns: {ds.column_names}")
        
        # Show original sample
        print(f"\n  Original sample:")
        print(f"    {ds[0]}")
        
        # 1. Rename columns (story -> completion)
        print(f"\n  [1/3] Renaming columns (story -> completion)...")
        ds = ds.map(rename_columns, remove_columns=ds.column_names, desc="Renaming columns")
        print(f"  New columns: {ds.column_names}")
        
        # 2. Add hash ID
        print(f"\n  [2/3] Adding hash IDs...")
        ds = ds.map(add_hash_id, desc="Adding hash IDs")
        
        # 3. Count tokens
        print(f"  [3/3] Counting tokens...")
        ds = ds.map(
            lambda x: count_tokens(x, tokenizer),
            desc="Counting tokens"
        )
        
        # Show processed sample
        print(f"\n  Processed sample from {split_name}:")
        print(f"    - Prompt: {ds[0]['prompt'][:50]}...")
        print(f"    - Completion: {ds[0]['completion'][:50]}...")
        print(f"    - Token count: {ds[0]['token_count']}")
        print(f"    - ID: {ds[0]['id'][:16]}...")
        
        # Save this split to disk
        split_save_path = base_path / split_name
        print(f"\n  Saving {split_name} to: {split_save_path}")
        ds.save_to_disk(str(split_save_path))
        print(f"  ✅ {split_name} saved ({len(ds)} samples)!")
    
    print(f"\n{'='*60}")
    print("✅ All splits processed and saved!")
    print(f"{'='*60}")
    
    print("\nSaved locations:")
    for split_name in dd.keys():
        actual_size = min(demo_size, len(dd[split_name])) if demo_size else len(dd[split_name])
        print(f"  - {split_name}: {base_path / split_name} ({actual_size} samples)")


if __name__ == "__main__":
    load_and_process_wrp(demo_size=200000)
