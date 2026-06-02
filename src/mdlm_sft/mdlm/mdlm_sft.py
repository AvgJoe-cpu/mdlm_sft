import gc
import os
from functools import partial

import torch
from datasets import load_from_disk
import hydra
from omegaconf import DictConfig, OmegaConf

from .mdlm_config import register_configs, TrainConfig
from .mdlm_helpers.mdlm_scheduler import make_alpha_scheduler
from .mdlm_helpers.mdlm_trainer_sft import (
    MDLMConfig,
    MDLMSFTTrainer,
    NLLPPLMetricComputer,
    SFTCollator,
)
from .mdlm_load_model import load_model_and_tokenizer
from ..paths import MDLM_CONFIG_DIR

register_configs()


def format_to_messages(example):
    """Format dataset examples to messages format for SFT"""
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


def run_training(cfg: TrainConfig) -> None:
    """Execute MDLM training run with pre-resolved configuration"""

    # When checkpoint_name is set we warm-start from a self-contained trained
    # checkpoint (weights + tokenizer); otherwise we build the base model.
    # Either way this is a *fresh* run (new optimizer/scheduler/dataset).
    print(f"Loading model from: {cfg.model_load_path}")
    model, tokenizer = load_model_and_tokenizer(
        cfg.model,
        load_path=cfg.model_load_path,
        is_checkpoint=bool(cfg.checkpoint_name),
    )

    def _sft_map_fn(example, tokenizer, max_length):
        enc = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            max_length=max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]
        labels = [tok if m == 1 else -100 for tok, m in zip(input_ids, assistant_mask)]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "assistant_mask": assistant_mask,
        }

    sft_map_fn = partial(_sft_map_fn, tokenizer=tokenizer, max_length=cfg.dataset.max_length)

    print(f"Loading training dataset from: {cfg.dataset.train_data_load_path}")
    train_ds = load_from_disk(str(cfg.dataset.train_data_load_path))
    if cfg.dataset.num_train_samples and cfg.dataset.num_train_samples > 0:
        train_ds = train_ds.select(range(min(cfg.dataset.num_train_samples, len(train_ds))))

    train_ds = train_ds.map(format_to_messages).map(sft_map_fn)
    train_ds = train_ds.select_columns(["input_ids", "labels", "assistant_mask"])

    print(f"Loading test dataset from: {cfg.dataset.test_data_load_path}")
    test_ds = load_from_disk(str(cfg.dataset.test_data_load_path))
    if cfg.dataset.num_test_samples and cfg.dataset.num_test_samples > 0:
        test_ds = test_ds.select(range(min(cfg.dataset.num_test_samples, len(test_ds))))
        
    test_ds = test_ds.map(format_to_messages).map(sft_map_fn)
    test_ds = test_ds.select_columns(["input_ids", "labels", "assistant_mask"])

    scheduler = make_alpha_scheduler(cfg.training.scheduler)
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)

    args = MDLMConfig(
        push_to_hub=False,
        output_dir=str(cfg.model_save_path),
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.grad_clip,
        seed=cfg.training.seed,
        adam_beta1=cfg.training.adam_beta1,
        adam_beta2=cfg.training.adam_beta2,
        time_epsilon=cfg.training.time_epsilon,
        loss_weight_type=cfg.training.loss_weight_type,
        dataloader_num_workers=cfg.training.num_workers,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        report_to=cfg.training.report_to,
        batch_eval_metrics=cfg.training.batch_eval_metrics,
        remove_unused_columns=cfg.training.remove_unused_columns,
        bf16=cfg.training.bf16,
    )

    metric_computer = NLLPPLMetricComputer()

    print(f"Starting training with {cfg.dataset.num_train_samples} samples...")
    print(f"Saving to: {cfg.model_save_path}")

    trainer = MDLMSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=collator,
        scheduler=scheduler,
        compute_metrics=metric_computer,
    )

    trainer.train()
    print(f"\u2713 Training complete. Model saved to: {cfg.model_save_path}")

    del trainer, args, collator, scheduler, train_ds, test_ds, model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@hydra.main(version_base=None, config_path=str(MDLM_CONFIG_DIR), config_name="mdlm_sft_config")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())  # Hydra changes cwd

    print("=" * 60)
    print("MDLM Training Configuration")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # W&B sweeps pass hyperparameters as Hydra-style CLI overrides
    # (key=value, via ${args_no_hyphens} in the sweep command), so they are
    # already composed into `cfg` here -- no manual wandb.config bridging
    # needed. The HF Trainer (report_to=wandb) handles wandb.init/logging.
    run_cfg: TrainConfig = OmegaConf.to_object(cfg)

    run_training(run_cfg)


if __name__ == "__main__":
    main()
