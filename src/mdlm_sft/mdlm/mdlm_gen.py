import gc
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING
from datasets import load_from_disk

from .mdlm_load_model import load_model_and_tokenizer
from .mdlm_config import ModelConfig, register_configs
from .mdlm_helpers.mdlm_sampler_sft import (
    MinimalMDLMSampler,
    SFTMixinBatchedVarlen,
)
from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler
from ..paths import MDLM_CONFIG_DIR, DATASET_GEN_DIR


# ============================================================ #
# Inference-local resolver (the ONLY place the gen output
# path is derived). Layout:
#   artifacts/datasets/gen/<model_type>/<dataset>/<run>/<save_name>
# ============================================================ #
def _gen_out(model_type: str, dataset_key: str, run_name: str, save_name: str) -> str:
    return str(DATASET_GEN_DIR / model_type / dataset_key / run_name / save_name)


OmegaConf.register_new_resolver("mdlm_gen_out", _gen_out, replace=True)


# Runtime-only container handed to the sampler (always built from cfg values).
@dataclass
class MDLMSamplerConfig:
    """Configuration for MDLM sampling"""
    response_length: int
    num_steps: int


# ============================================================ #
# Schema (type-checker only -- defaults come from YAML,
#          derived paths come from resolvers/interpolation)
# ============================================================ #
@dataclass
class InferenceConfig:
    """Configuration for MDLM inference"""
    # reused model block: provides model_name/tokenizer/dtype + derived
    # base_path / hf_path / tokenizer_cache_path / checkpoints_path
    model: ModelConfig = field(default_factory=ModelConfig)

    # inputs
    checkpoint_name: str = MISSING
    dataset_key: str = MISSING
    split: str = MISSING
    num_samples: int = MISSING

    # generation
    response_length: int = MISSING
    num_steps: int = MISSING
    batch_size: int = MISSING

    # output
    save_name: str = MISSING

    # derived (no new path math in code)
    model_path: str = "${.model.checkpoints_path}/${.checkpoint_name}"
    input_path: str = "${ds_base:${.dataset_key},${.split}}"
    output_path: str = "${mdlm_gen_out:mdlm,${.dataset_key},${.checkpoint_name},${.save_name}}"


def register_inf_configs() -> None:
    # Reuse shared registration: registers the `model` group + the OmegaConf
    # resolvers (mdlm_model / ds_base) at mdlm_config import time.
    register_configs()
    cs = ConfigStore.instance()
    cs.store(name="mdlm_infer_config", node=InferenceConfig)


register_inf_configs()


def generate_mdlm(
    batch,
    tokenizer=None,
    model=None,
    sampler=None,
    config: MDLMSamplerConfig = None,
):
    """Generate responses for a batch of prompts"""
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

    # Ensure sampler uses correct dtype
    with torch.autocast(device_type=str(model.device).split(':')[0], dtype=model_dtype):
        out = sampler.sample_sft(
            prompt_ids,
            prompt_lens=prompt_lens,
            pad_token_id=pad_id,
            **dataclasses.asdict(config),
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
    """Execute MDLM inference with pre-resolved configuration"""

    # Derived paths arrive as strings (OmegaConf interpolation) -> wrap.
    model_path = Path(cfg.model_path)
    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path)

    print("=" * 60)
    print("MDLM Inference")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Samples: {cfg.num_samples}")
    print(f"Response length: {cfg.response_length}")
    print(f"Steps: {cfg.num_steps}")
    print("=" * 60)

    # Always warm-load the trained checkpoint directly (weights + tokenizer
    # baked in: resized vocab, chat template, ChatML specials). model_path
    # mirrors exactly what training writes:
    #   .../checkpoints/<model_name>/<run_name>/checkpoint-N
    model, tokenizer = load_model_and_tokenizer(
        cfg.model,
        load_path=str(model_path),
        is_checkpoint=True,
    )

    model.eval()

    # Setup sampler
    sampler_config = MDLMSamplerConfig(
        response_length=cfg.response_length,
        num_steps=cfg.num_steps,
    )

    sampler = MinimalMDLMSampler(
        backbone=model,
        scheduler=LinearAlphaScheduler(),
        mask_index=tokenizer.mask_token_id,
    )
    sampler.sample_sft = SFTMixinBatchedVarlen.sample_sft.__get__(
        sampler, type(sampler)
    )

    # Load dataset
    print(f"Loading dataset from: {input_path}")
    ds = load_from_disk(str(input_path))
    ds = ds.select(range(min(cfg.num_samples, len(ds))))

    print(f"Running inference on {len(ds)} samples...")
    ds = ds.map(
        generate_mdlm,
        batched=True,
        batch_size=cfg.batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "sampler": sampler,
            "config": sampler_config,
        },
    )

    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_path}")
    ds.save_to_disk(str(output_path))

    print("✓ Inference complete!")

    # Cleanup
    del model, tokenizer, sampler, sampler_config, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path=str(MDLM_CONFIG_DIR), config_name="mdlm_inf_config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("MDLM Inference Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    run_cfg: InferenceConfig = OmegaConf.to_object(cfg)
    run_inference(run_cfg)


if __name__ == "__main__":
    main()
