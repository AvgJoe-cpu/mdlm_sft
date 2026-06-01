"""
Demo script to load and process all datasets.

Run from project root with: uv run python -m mdlm_sft.data.demo_load_ds
"""

from mdlm_sft.data import load_and_process_tis, load_and_process_tms, load_and_process_wrp


def main():
    """Load and process all datasets in demo mode."""
    
    print("="*80)
    print("DATASET PROCESSING DEMO")
    print("="*80)
    print("\nThis will process 3 datasets with 10k samples each:")
    print("  1. TinyStories (tis) - with sentence tokenization")
    print("  2. Tell Me a Story (tms) - encrypted dataset")
    print("  3. Writing Prompts (wrp) - standard processing")
    print("\n" + "="*80)
    
    # 1. Process TinyStories
    print("\n\n")
    print("█" * 80)
    print("█ 1/3: PROCESSING TINYSTORIES")
    print("█" * 80)
    try:
        load_and_process_tis(demo_size=10000, force_reprocess=False)
        print("\n✅ TinyStories processing complete!")
    except Exception as e:
        print(f"\n❌ Error processing TinyStories: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Process Tell Me a Story (encrypted)
    print("\n\n")
    print("█" * 80)
    print("█ 2/3: PROCESSING TELL ME A STORY (ENCRYPTED)")
    print("█" * 80)
    try:
        load_and_process_tms(
            filename="data_encrypted.jsonl",
            skey_file="skey.key",
            pkey_file="private_key.pem",
            demo_size=10000,
            force_reprocess=False
        )
        print("\n✅ Tell Me a Story processing complete!")
    except FileNotFoundError as e:
        print(f"\n⚠️  Skipping Tell Me a Story: Required encryption files not found")
        print(f"    Missing: {e.filename}")
        print(f"    This is expected if you don't have the encrypted dataset files")
    except Exception as e:
        print(f"\n❌ Error processing Tell Me a Story: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Process Writing Prompts
    print("\n\n")
    print("█" * 80)
    print("█ 3/3: PROCESSING WRITING PROMPTS")
    print("█" * 80)
    try:
        load_and_process_wrp(demo_size=10000, force_reprocess=False)
        print("\n✅ Writing Prompts processing complete!")
    except Exception as e:
        print(f"\n❌ Error processing Writing Prompts: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n\n")
    print("="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print("\nAll available datasets have been processed and saved.")
    print("\nTo load a processed dataset:")
    print("  from datasets import load_from_disk")
    print("  from mdlm_sft.paths import resolve_dataset_base_path")
    print("  ")
    print("  # For TinyStories or Writing Prompts (with splits):")
    print("  base_path = resolve_dataset_base_path('tis')")
    print("  train_ds = load_from_disk(str(base_path / 'train'))")
    print("  ")
    print("  # For Tell Me a Story (single dataset):")
    print("  base_path = resolve_dataset_base_path('tms')")
    print("  ds = load_from_disk(str(base_path / 'all'))")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()