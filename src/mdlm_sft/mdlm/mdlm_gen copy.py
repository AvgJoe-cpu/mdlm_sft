from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollator, PreTrainedTokenizerBase, DefaultDataCollator
import gc
from datasets import load_from_disk
import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING

from .mdlm_helpers.mdlm_scheduler import make_alpha_scheduler, LinearAlphaScheduler
from .mdlm_helpers.mdlm_sampler_sft import (
    MinimalMDLMSampler,
    SFTMixinBatchedVarlen,
)
@dataclass
class GenerationConfig:
    response_length: int = 10 
    num_steps: int  = 10
    batch_size: int = 8
    dataset_input_path: str = "path/to/input/dataset"
    dataset_output_path: str = "path/to/output/dataset"
    model_name_or_path: str = "path/to/model"


cs = ConfigStore.instance()
cs.store(name="config", node=GenerationConfig)    

def generate_mdlm(
    batch,
    tokenizer=None,
    model=None,
    sampler=None,
    scheduler=None,
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

    # Match model dtype
    model_dtype = next(model.parameters()).dtype
    prompt_ids = encoded["input_ids"].to(model.device)
    attn = encoded["attention_mask"].to(model.device)
    prompt_lens = attn.sum(dim=1).long()

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer has no pad_token_id"
    assert pad_id != tokenizer.mask_token_id, "pad_token_id == mask_token_id"

    with torch.autocast(device_type=str(model.device).split(':')[0], dtype=model_dtype):
        out = sampler.sample_sft(
            prompt_ids,
            prompt_lens=prompt_lens,
            pad_token_id=pad_id,
            response_length=response_length,
            num_steps=num_steps,
            scheduler=scheduler,
        )

    R = config.response_length
    decoded = [
        tokenizer.decode(
            out[b, int(prompt_lens[b]) : int(prompt_lens[b]) + R],
            skip_special_tokens=True,
        )
        for b in range(out.shape[0])
    ]
    return {"completion": decoded}


def run_inference(cfg: InferenceConfig) -> None:
    model_name_or_path = cfg.pop("model_name_or_path")
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path=model_name_or_path, trust_remote_code=True, torch_dtype="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path=model_name_or_path, trust_remote_code=True)


    scheduler =LinearAlphaScheduler()

    sampler = MinimalMDLMSampler(
        backbone=model,
        scheduler=scheduler,
        mask_index=tokenizer.mask_token_id,
    )
    sampler.sample_sft = SFTMixinBatchedVarlen.sample_sft.__get__(
        sampler, type(sampler)
    )    
    input_dataset_path = cfg.pop("input_dataset_path")
    ds = load_from_disk(input_dataset_path)
    output_dataset_path = cfg.pop("output_dataset_path")

    ds = ds.map(
        generate_mdlm,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "model": model, "sampler": sampler, "scheduler": scheduler, "response_length": cfg.response_length, "num_steps": cfg.num_steps},
        batch_size=cfg.batch_size,
    )    
    output_dataset_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(output_dataset_path)

    del model, tokenizer, sampler, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()