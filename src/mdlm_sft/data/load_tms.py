"""Load and process Tell Me a Story (encrypted) dataset."""

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.fernet import Fernet
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from mdlm_sft.paths import DATASETS, resolve_dataset_base_path
from .shared import add_hash_id, count_tokens


def decrypt_file(filename, skey_file, pkey_file):
    """Decrypt the encrypted JSONL file."""
    # Convert to Path objects
    filename = Path(filename)
    skey_file = Path(skey_file)
    pkey_file = Path(pkey_file)
    
    # Check if files exist
    if not filename.exists():
        raise FileNotFoundError(f"Encrypted file not found: {filename}")
    if not skey_file.exists():
        raise FileNotFoundError(f"Symmetric key file not found: {skey_file}")
    if not pkey_file.exists():
        raise FileNotFoundError(f"Private key file not found: {pkey_file}")
    
    # Load the private key
    with open(pkey_file, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )

    # Load the symmetrical key
    with open(skey_file, 'rb') as f:
        skey = f.read()

    # Load the file to decrypt
    with open(filename, 'rb') as f:
        data = f.read()

    # Decrypt the symmetrical key
    unenc_skey = private_key.decrypt(
        skey,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Decrypt the data
    f = Fernet(unenc_skey)
    decrypted = f.decrypt(data)
    
    return decrypted.decode('utf-8')


def load_and_process_tms(
    filename="data_encrypted.jsonl",
    skey_file="skey.key",
    pkey_file="private_key.pem",
    demo_size=10000,
    force_reprocess=False
):
    """
    Load and process Tell Me a Story (encrypted) dataset.
    
    Args:
        filename: Path to encrypted JSONL file
        skey_file: Path to symmetric key file
        pkey_file: Path to private key file
        demo_size: Number of samples (None for full dataset)
        force_reprocess: If True, reprocess even if dataset exists
    
    Returns:
        None (saves processed dataset to disk)
    """
    dataset_key = "tms"
    split_name = "all"  # Since there's no train/val split, use "all"
    
    # Resolve the base path
    print(f"\nResolving dataset base path for '{dataset_key}':")
    base_path = resolve_dataset_base_path(dataset_key)
    print(f"Save location: {base_path}")
    print(f"Dataset: {DATASETS[dataset_key]['name']}")
    
    # Check if already processed (inline check)
    save_path = base_path / split_name
    if not force_reprocess and save_path.exists():
        print("\n" + "="*60)
        print("⚠️  Dataset already processed! Skipping...")
        print("="*60)
        print(f"\nExisting location: {save_path}")
        print("\nTo reprocess, set force_reprocess=True")
        return
    
    print("\n" + "="*60)
    print("Step 1: Decrypting file...")
    print("="*60)
    print(f"  Encrypted file: {filename}")
    print(f"  Symmetric key: {skey_file}")
    print(f"  Private key: {pkey_file}")
    
    # Decrypt the file
    decrypted_content = decrypt_file(filename, skey_file, pkey_file)
    
    # Parse JSONL
    print("\nParsing JSONL data...")
    lines = decrypted_content.strip().split('\n')
    data = [json.loads(line) for line in lines if line.strip()]
    
    print(f"Total records: {len(data)}")
    
    # Limit samples if demo_size specified
    if demo_size:
        print(f"Using first {demo_size} records for demo")
        data = data[:demo_size]
    else:
        print(f"Using full dataset: {len(data)} records")
    
    # Create HuggingFace Dataset
    print("\nCreating HuggingFace Dataset...")
    ds = Dataset.from_list(data)
    
    print(f"Dataset created with {len(ds)} samples")
    print(f"Columns: {ds.column_names}")
    print(f"\nSample record:")
    print(ds[0])
    
    print("\n" + "="*60)
    print("Step 2: Processing dataset...")
    print("="*60)
    
    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Process the dataset
    print("\n[1/2] Adding hash IDs...")
    ds = ds.map(add_hash_id, desc="Adding hash IDs")
    
    print("[2/2] Counting tokens...")
    ds = ds.map(
        lambda x: count_tokens(x, tokenizer),
        desc="Counting tokens"
    )
    
    # Show sample
    print("\n" + "="*60)
    print("Processed sample:")
    print("="*60)
    print(f"  - Token count: {ds[0]['token_count']}")
    print(f"  - ID: {ds[0]['id'][:16]}...")
    
    # Save to disk
    print("\n" + "="*60)
    print("Step 3: Saving dataset...")
    print("="*60)
    
    print(f"Saving to: {save_path}")
    ds.save_to_disk(str(save_path))
    
    print(f"\n✅ Dataset processed and saved!")
    print(f"   Location: {save_path}")
    print(f"   Samples: {len(ds)}")
    print(f"   Columns: {ds.column_names}")


if __name__ == "__main__":
    load_and_process_tms(
        filename="data_encrypted.jsonl",
        skey_file="skey.key",
        pkey_file="private_key.pem",
        demo_size=10000
    )