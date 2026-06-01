import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import GenerationConfig
from datasets import load_from_disk

from .ar_config import register_configs, InferenceRunConfig, ModelConfig, DatasetConfig, InferenceConfig
from .ar_load_model import load_model_and_tokenizer
from ..paths import AR_CONFIG_DIR


register_configs()


def generate_ar(batch, tokenizer=None, model=None, gen_config=None):
    """Generate text for a batch of prompts"""
    messages_list = [
        [{"role": "user", "content": prompt}] for prompt in batch["prompt"]
    ]
    formatted_texts = tokenizer.apply_chat_template(
        messages_list, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        generation_config=gen_config,
    )

    results = tokenizer.batch_decode(
        [
            generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
            for i in range(len(generated_ids))
        ],
        skip_special_tokens=True,
    )

    del model_inputs, generated_ids, formatted_texts, messages_list
    return {"story": results}


def run_inference(cfg: InferenceRunConfig) -> None:
    """Execute inference with pre-resolved configuration"""
    
    # Load dataset
    print(f"Loading dataset from: {cfg.dataset.data_load_path}")
    ds = load_from_disk(str(cfg.dataset.data_load_path))
    ds = ds.select(range(cfg.dataset.num_samples))

    # Load model
    print(f"Loading model: {cfg.model_load_path}")
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    tokenizer.padding_side = "left"

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=cfg.inference.max_new_tokens,
        num_beams=cfg.inference.num_beams,
        do_sample=cfg.inference.do_sample,
        use_cache=cfg.inference.use_cache,
        temperature=cfg.inference.temperature,
        num_return_sequences=cfg.inference.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    print(f"Running inference on {cfg.dataset.num_samples} samples...")
    ds = ds.map(
        generate_ar,
        batched=True,
        batch_size=cfg.inference.batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "gen_config": gen_config,
        },
    )
    
    print(f"Saving results to: {cfg.data_save_path}")
    ds.save_to_disk(str(cfg.data_save_path))
    
    print("✓ Inference complete")
    torch.cuda.empty_cache()
    del model, tokenizer, gen_config, ds


@hydra.main(version_base=None, config_path=str(AR_CONFIG_DIR), config_name="ar_inf_config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("Inference Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Convert DictConfig to structured config
    model_cfg = ModelConfig(
        model_name=cfg.model.model_name,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
        for_training=cfg.model.for_training,
    )
    
    dataset_cfg = DatasetConfig(
        dataset_key=cfg.dataset.dataset_key,
        num_samples=cfg.dataset.num_samples,
    )
    
    inference_cfg = InferenceConfig(
        max_new_tokens=cfg.inference.max_new_tokens,
        num_beams=cfg.inference.num_beams,
        do_sample=cfg.inference.do_sample,
        use_cache=cfg.inference.use_cache,
        temperature=cfg.inference.temperature,
        num_return_sequences=cfg.inference.num_return_sequences,
        batch_size=cfg.inference.batch_size,
        output_suffix=cfg.inference.output_suffix,
    )
    
    infer_cfg = InferenceRunConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        inference=inference_cfg,
        checkpoint_name=cfg.checkpoint_name,
        seed=cfg.seed,
    )
    
    # Run inference
    run_inference(infer_cfg)


if __name__ == "__main__":
    main()