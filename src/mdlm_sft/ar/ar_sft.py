import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
import wandb

from .ar_config import register_configs, RunConfig, ModelConfig, DatasetConfig, TrainingConfig
from .ar_load_model import load_model_and_tokenizer
from ..paths import AR_CONFIG_DIR


register_configs()


def format_to_messages(example):
    """Format dataset examples to messages format for SFT"""
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


def run_training(cfg: RunConfig) -> None:
    """Execute training run with pre-resolved configuration"""
    
    # Load dataset
    print(f"Loading dataset from: {cfg.dataset.data_load_path}")
    train_ds = load_from_disk(str(cfg.dataset.data_load_path))

    print(f"Selecting {cfg.dataset.num_samples} samples and preprocessing...")
    train_ds = train_ds.select(range(cfg.dataset.num_samples))
    train_dataset = train_ds.map(format_to_messages).remove_columns(train_ds.column_names)

    # Load model
    print(f"Loading model: {cfg.model_load_path}")
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    # Initialize wandb (only if report_to includes wandb)
    if "wandb" in cfg.training.report_to:
        wandb.init(
            project="ar-model-training",  # Change this to your project name
            name=cfg.run_name or f"{cfg.dataset.dataset_key}_{cfg.model.model_name}_sft",
            config={
                "model_name": cfg.model.model_name,
                "model_dtype": cfg.model.dtype,
                "dataset": cfg.dataset.dataset_key,
                "num_samples": cfg.dataset.num_samples,
                "learning_rate": cfg.training.learning_rate,
                "batch_size": cfg.training.batch_size,
                "num_epochs": cfg.training.num_epochs,
                "optim": cfg.training.optim,
                "bf16": cfg.training.bf16,
                "use_liger_kernel": cfg.training.use_liger_kernel,
                "checkpoint_name": cfg.checkpoint_name,
            },
            dir=str(cfg.wandb_log_dir),  # Log to the configured wandb directory
        )

    # Setup training args
    training_args = SFTConfig(
        output_dir=str(cfg.model_save_path),
        logging_dir=str(cfg.wandb_log_dir),  # Updated to use wandb_log_dir
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        bf16=cfg.training.bf16,
        optim=cfg.training.optim,
        use_liger_kernel=cfg.training.use_liger_kernel,
        logging_steps=cfg.training.logging_steps,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        report_to=cfg.training.report_to,
        assistant_only_loss=cfg.training.assistant_only_loss,
        remove_unused_columns=cfg.training.remove_unused_columns,
        push_to_hub=cfg.training.push_to_hub,
        run_name=cfg.run_name or f"{cfg.dataset.dataset_key}_{cfg.model.model_name}_sft",  # Add run name
    )

    print(f"Starting training with {cfg.dataset.num_samples} samples...")
    print(f"Saving to: {cfg.model_save_path}")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    print(f"✓ Training complete. Model saved to: {cfg.model_save_path}")
    
    # Finish wandb run
    if "wandb" in cfg.training.report_to:
        wandb.finish()
    
    # Cleanup
    torch.cuda.empty_cache()
    del model, tokenizer, training_args, trainer, train_dataset


@hydra.main(version_base=None, config_path=str(AR_CONFIG_DIR), config_name="ar_sft_config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("Training Configuration")
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
    
    training_cfg = TrainingConfig(
        num_epochs=cfg.training.num_epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        bf16=cfg.training.bf16,
        optim=cfg.training.optim,
        use_liger_kernel=cfg.training.use_liger_kernel,
        logging_steps=cfg.training.logging_steps,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        report_to=cfg.training.report_to,
        assistant_only_loss=cfg.training.assistant_only_loss,
        remove_unused_columns=cfg.training.remove_unused_columns,
        push_to_hub=cfg.training.push_to_hub,
    )
    
    run_cfg = RunConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        training=training_cfg,
        checkpoint_name=cfg.checkpoint_name if hasattr(cfg, 'checkpoint_name') else None,
        run_name=cfg.run_name if hasattr(cfg, 'run_name') else None,
        seed=cfg.seed,
    )
    
    # Run training
    run_training(run_cfg)


if __name__ == "__main__":
    main()