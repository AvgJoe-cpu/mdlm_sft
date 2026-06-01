"""Load and process TinyStories dataset with sentence tokenization."""

from datasets import load_dataset
from transformers import AutoTokenizer
from wtpsplit import SaT
from mdlm_sft.paths import DATASETS, resolve_dataset_base_path
from .shared import add_hash_id, count_sentence_tokens

def count_sentence_tokens(example, tokenizer, sentences_field="sentences"):
    """
    Count total tokens in the sentences of an example.
    
    Args:
        example: Dataset example containing a list of sentences
        tokenizer: Hugging Face tokenizer instance
        sentences_field: Name of the field containing the list of sentences
    Returns:
        Dict

    """
    total_tokens = 0
    for sent in example.get(sentences_field, []):
        if sent.strip():  # Skip empty sentences
            total_tokens += len(tokenizer.encode(sent, add_special_tokens=False))
    return {"token_count": total_tokens}

def sent_tokenize(example, model=None, text_field="text"):
    """
    Tokenize text into sentences using wtpsplit.
    
    Args:
        example: Dataset example
        model: SaT model instance
        text_field: Name of the text field to split
    
    Returns:
        Dict with 'sentences' key containing list of sentences
    """
    text = example.get(text_field, "")
    sentences = model.split(text)
    return {"sentences": sentences}


def load_and_process_tis(demo_size=10000, force_reprocess=False):
    """
    Load and process TinyStories dataset.
    
    Args:
        demo_size: Number of samples per split (None for full dataset)
        force_reprocess: If True, reprocess even if dataset exists
    
    Returns:
        None (saves processed datasets to disk)
    """
    dataset_key = "tis"
    
    # Resolve the base path
    print(f"\nResolving dataset base path for '{dataset_key}':")
    base_path = resolve_dataset_base_path(dataset_key)
    print(f"Save location: {base_path}")
    print(f"Dataset: {DATASETS[dataset_key]['name']}")
    
    # Load dataset to get split names
    print(f"\nLoading {DATASETS[dataset_key]['name']}...")
    dd = load_dataset(DATASETS[dataset_key]["hf-path"])
    
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
    
    # Create models once (reuse across all splits)
    print("\nLoading sentence tokenizer model...")
    sat_model = SaT("sat-3l-sm")
    sat_model.half()  # Use half precision
    
    print("Loading GPT-2 tokenizer...")
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
        
        # 1. Add hash ID (before processing)
        print(f"  [1/4] Adding hash IDs...")
        ds = ds.map(add_hash_id, desc="Adding hash IDs")
        
        # 2. Add sentence tokenization
        print(f"  [2/4] Sentence tokenizing...")
        ds = ds.map(
            lambda x: sent_tokenize(x, model=sat_model),
            desc="Sentence tokenizing"
        )
        
        # 3. Add sentence count column
        print(f"  [3/4] Counting sentences...")
        ds = ds.map(
            lambda batch: {
                "sent_count": [len([s for s in sents if s.strip()]) for sents in batch["sentences"]]
            },
            batched=True,
            desc="Counting sentences"
        )
        
        # 4. Add token count
        print(f"  [4/4] Counting tokens...")
        ds = ds.map(
            lambda x: count_sentence_tokens(x, tokenizer),
            desc="Counting tokens"
        )
        
        # Show sample
        print(f"\n  Sample from {split_name}:")
        print(f"    - Sentence count: {ds[0]['sent_count']}")
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
    load_and_process_tis(demo_size=10000)