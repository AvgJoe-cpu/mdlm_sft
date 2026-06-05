from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
import gc
from datasets import load_from_disk
import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler
from .mdlm_helpers.mdlm_sampler_sft import MinimalMDLMSampler, SFTMixinBatchedVarlen


@dataclass
class GenerationConfig:
    response_length: int = 10
    num_steps: int = 10
    batch_size: int = 8
    dataset_input_path: str = MISSING   # must be overridden on CLI
    dataset_output_path: str = MISSING  # must be overridden on CLI
    model_name_or_path: str = MISSING   # must be overridden on CLI


cs = ConfigStore.instance()
cs.store(name="config", node=GenerationConfig)


def generate_mdlm(
    batch,
    tokenizer=None,
    model=None,
    sampler=None,
    scheduler=None,
    response_length=None,  # fix 2: was referenced as config.response_length
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
            scheduler=scheduler,
        )

    decoded = [
        tokenizer.decode(
            out[b, int(prompt_lens[b]) : int(prompt_lens[b]) + response_length],
            skip_special_tokens=True,
        )
        for b in range(out.shape[0])
    ]
    return {"completion": decoded}


@hydra.main(config_path=None, config_name="config", version_base=None)  # fix 8
def run_inference(cfg: GenerationConfig) -> None:  # fix 1: was InferenceConfig
    # fix 3, 5: cfg.pop() → cfg.<field>; positional arg to from_pretrained
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True, torch_dtype="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True
    )

    scheduler = LinearAlphaScheduler()
    sampler = MinimalMDLMSampler(
        backbone=model,
        scheduler=scheduler,
        mask_index=tokenizer.mask_token_id,
    )
    sampler.sample_sft = SFTMixinBatchedVarlen.sample_sft.__get__(sampler, type(sampler))

    # fix 4: was cfg.pop("input_dataset_path") and wrong field name
    ds = load_from_disk(cfg.dataset_input_path)

    ds = ds.map(
        generate_mdlm,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "sampler": sampler,
            "scheduler": scheduler,
            "response_length": cfg.response_length,
            "num_steps": cfg.num_steps,
        },
        batch_size=cfg.batch_size,
    )

    # fix 6: was output_dataset_path.mkdir() on a str; wrap in Path
    output_path = Path(cfg.dataset_output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(output_path)

    del model, tokenizer, sampler, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":  # fix 9: required for python -m
    run_inference()