from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM, HfArgumentParser
import gc
from datasets import load_from_disk
import torch

from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler
from .mdlm_helpers.mdlm_sampler_sft import MinimalMDLMSampler, SFTMixinBatchedVarlen

import datasets
datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB
from typing import Optional


@dataclass
class MDLMGenerationConfig:
    # required
    model_name_or_path:  Optional[str] = None
    dataset_input_path:  Optional[str] = None
    dataset_output_path: Optional[str] = None
    # generation knobs
    response_length: int = 128
    num_steps:       int = 128
    batch_size:      int = 64


def generate_mdlm(
    batch,
    tokenizer=None,
    model=None,
    sampler=None,
    response_length=None,
    num_steps=None,
):
    messages_list = [[{"role": "user", "content": p}] for p in batch["prompt"]]

    encoded = tokenizer.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    model_dtype = next(model.parameters()).dtype
    prompt_ids = encoded["input_ids"].to(model.device)
    attn = encoded["attention_mask"].to(model.device)
    prompt_lens = attn.sum(dim=1).long()

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer has no pad_token_id"
    assert pad_id != tokenizer.mask_token_id, "pad_token_id == mask_token_id"

    with torch.autocast(device_type=str(model.device).split(":")[0], dtype=model_dtype):
        out = sampler.sample_sft(
            prompt_ids,
            prompt_lens=prompt_lens,
            pad_token_id=pad_id,
            response_length=response_length,
            num_steps=num_steps,
        )

    decoded = [
        tokenizer.decode(
            out[b, int(prompt_lens[b]) : int(prompt_lens[b]) + response_length],
            skip_special_tokens=False,
        )
        for b in range(out.shape[0])
    ]
    return {"completion": decoded}



def run_inference(cfg: MDLMGenerationConfig) -> None:
    # Enforce "required" here instead of via Hydra's MISSING sentinel.
    for name in ("model_name_or_path", "dataset_input_path", "dataset_output_path"):
        if getattr(cfg, name) is None:
            raise ValueError(f"{name} must be provided")

    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    scheduler = LinearAlphaScheduler()
    sampler = MinimalMDLMSampler(
        backbone=model, scheduler=scheduler, mask_index=tokenizer.mask_token_id,
    )
    sampler.sample_sft = SFTMixinBatchedVarlen.sample_sft.__get__(sampler, type(sampler))

    ds = load_from_disk(cfg.dataset_input_path)
    ds = ds.map(
        generate_mdlm,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer, "model": model, "sampler": sampler,
            "response_length": cfg.response_length, "num_steps": cfg.num_steps,
        },
        batch_size=cfg.batch_size,
    )

    output_path = Path(cfg.dataset_output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(output_path)

    del model, tokenizer, sampler, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = HfArgumentParser(MDLMGenerationConfig)
    (cfg,) = parser.parse_args_into_dataclasses()
    run_inference(cfg)