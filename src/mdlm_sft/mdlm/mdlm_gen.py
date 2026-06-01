import gc
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk

from .mdlm_load_model import load_model_and_tokenizer
from .mdlm_config import MDLMModelConfig
from .mdlm_helpers.mdlm_sampler_sft import (
    MinimalMDLMSampler,
    SFTMixinBatchedVarlen,
)
from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler  # Add this line
from ..paths import MDLM_CONFIG_DIR, MDLM_MODELS, DATASET_BASE_DIR


@dataclass
class MDLMSamplerConfig:
    """Configuration for MDLM sampling"""
    response_length: int = 256
    num_steps: int = 100


@dataclass
class InferenceConfig:
    """Configuration for MDLM inference"""
    # Model settings
    model_name: str = "mdlm-owt"
    checkpoint_name: str = "toy_run"  # Name of checkpoint to load
    
    # Dataset settings
    dataset_key: str = "wrp"
    split: str = "test"  # Which split to run inference on
    num_samples: int = 10
    
    # Generation settings
    response_length: int = 256
    num_steps: int = 100
    batch_size: int = 4
    
    # Output
    save_name: str = "inference_results"
    
    def __post_init__(self):
        """Resolve paths"""
        model_info = MDLM_MODELS[self.model_name]
        
        # Model path
        self.model_path = model_info["checkpoints_path"] / self.checkpoint_name
        
        # Input dataset path
        self.input_path = DATASET_BASE_DIR / self.dataset_key / self.split
        
        # Output path
        self.output_path = model_info["checkpoints_path"] / self.checkpoint_name / "inference" / self.save_name


def generate_mdlm(
    batch,
    tokenizer=None,
    model=None,
    sampler=None,
    config: MDLMSamplerConfig = MDLMSamplerConfig(),
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
    return {"gen": decoded}


def run_inference(cfg: InferenceConfig) -> None:
    """Execute MDLM inference"""
    
    print("=" * 60)
    print("MDLM Inference")
    print("=" * 60)
    print(f"Model: {cfg.model_path}")
    print(f"Input: {cfg.input_path}")
    print(f"Output: {cfg.output_path}")
    print(f"Samples: {cfg.num_samples}")
    print(f"Response length: {cfg.response_length}")
    print(f"Steps: {cfg.num_steps}")
    print("=" * 60)
    
    # Load model
    model_cfg = MDLMModelConfig(model_name=cfg.model_name)
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    
    # Load checkpoint if specified
    if cfg.checkpoint_name:
        checkpoint_path = cfg.model_path / "pytorch_model.bin"
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=model.device)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, using base model")
    
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
    print(f"Loading dataset from: {cfg.input_path}")
    ds = load_from_disk(str(cfg.input_path))
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
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {cfg.output_path}")
    ds.save_to_disk(str(cfg.output_path))
    
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
    
    # Convert to structured config
    inference_cfg = InferenceConfig(
        model_name=cfg.model_name,
        checkpoint_name=cfg.checkpoint_name,
        dataset_key=cfg.dataset_key,
        split=cfg.split,
        num_samples=cfg.num_samples,
        response_length=cfg.response_length,
        num_steps=cfg.num_steps,
        batch_size=cfg.batch_size,
        save_name=cfg.save_name,
    )
    
    run_inference(inference_cfg)


if __name__ == "__main__":
    main()