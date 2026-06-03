import gc
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from wtpsplit import SaT
from mdlm_sft.paths import DATASETS, resolve_dataset_base_path
from .shared import add_hash_id, count_sentence_tokens


def count_sentence_tokens(example, tokenizer, sentences_field="sentences"):
    total_tokens = 0
    for sent in example.get(sentences_field, []):
        if sent.strip():
            total_tokens += len(tokenizer.encode(sent, add_special_tokens=False))
    return {"token_count": total_tokens}



def sent_tokenize(example, model=None, text_field="text"):
    text = example.get(text_field, "")
    sentences = model.split(text)
    return {"sentences": sentences}

def split_sentences_to_prompt_completion(batch):
    prompts, completions = [], []
    for sentences in batch["sentences"]:
        mid_idx = round(len(sentences) / 2)
        prompts.append(" ".join(sentences[:mid_idx]))
        completions.append(" ".join(sentences[mid_idx:]))
    return {"prompt": prompts, "completion": completions}


def load_and_process_tis(demo_size=10000, force_reprocess=False):
    dataset_key = "tis"
    base_path.mkdir(parents=True, exist_ok=True)
    base_path = resolve_dataset_base_path(dataset_key)

    dd = load_dataset(DATASETS[dataset_key]["hf-path"])

    if not force_reprocess and all((base_path / s).exists() for s in dd.keys()):
        print("Dataset already processed. Set force_reprocess=True to reprocess.")
        return

    sat_model = SaT("sat-3l-sm")
    sat_model.half()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    for split_name in dd.keys():
        ds = dd[split_name].select(range(min(demo_size, len(dd[split_name])))) if demo_size else dd[split_name]
        print(f"{split_name}: {len(ds)} samples")

        ds = ds.map(add_hash_id, desc="[1/5] hash IDs")
        ds = ds.map(
            lambda x: sent_tokenize(x, model=sat_model),
            desc="Sentence tokenizing"
        )
        ds = ds.map(
            lambda batch: {"sent_count": [len([s for s in sents if s.strip()]) for sents in batch["sentences"]]},
            batched=True, desc="[3/5] sentence counts"
        )
        ds = ds.map(lambda batch: count_sentence_tokens(batch, tokenizer), batched=True, desc="[4/5] token counts")
        ds = ds.map(split_sentences_to_prompt_completion, batched=True, desc="[5/5] prompt/completion split")

        ds.save_to_disk(str(base_path / split_name))
        print(f"  ✅ saved → {base_path / split_name}")

    # --- cleanup: free model/tokenizer refs and flush GPU/MPS memory ---
    del sat_model, tokenizer, dd
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("  🧹 CUDA cache cleared.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("  🧹 MPS cache cleared.")
    except ImportError:
        pass  # torch not installed; nothing to free


if __name__ == "__main__":
    load_and_process_tis(demo_size=10000, force_reprocess=True)