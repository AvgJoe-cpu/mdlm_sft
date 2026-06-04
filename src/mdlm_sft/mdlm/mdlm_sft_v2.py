from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Any, Tuple
import math 
import torch
import torch.nn.functional as F

import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollator, PreTrainedTokenizerBase, DefaultDataCollator
from trl import SFTConfig, SFTTrainer, trainer

from .mdlm_helpers.mdlm_scheduler import make_alpha_scheduler, LinearAlphaScheduler


class CustomForwardSFTTrainer(SFTTrainer):
    """
    Skeleton for subclassing SFTTrainer when your model's `forward(...)`
    signature or return structure differs from the standard autoregressive
    transformer contract.
    """

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: Optional[Any] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        compute_loss_func: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        callbacks: Optional[list] = None,
        optimizers: Tuple = (None, None),
        optimizer_cls_and_kwargs: Optional[Tuple] = None,
        preprocess_logits_for_metrics: Optional[Any] = None,
        peft_config: Optional[Any] = None,
        formatting_func: Optional[Any] = None,

        scheduler: Optional[Any] = None,
        time_epsilon: float = 1e-5,
        loss_weight_type: str = "scheduler",
    ):
        # AMENDABLE: Store extra state before calling super().

        # MUST: Pass **all** standard arguments through to super().__init__().
        # CANNOT AMEND: Skipping this call.  The parent sets up:
        #   - Dataset formatting / chat-template application / packing
        #   - PEFT wrapping (LoraConfig, etc.)
        #   - Data collator default (DataCollatorForLanguageModeling or VLM variant)
        #   - Processor / tokenizer setup
        #   - The entire HF Trainer accelerator / distributed / checkpointing state machine
        self.scheduler = scheduler
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        self._eval_nll_sum    = 0.0
        self._eval_masked_sum = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        input_ids        = inputs["input_ids"]
        labels           = inputs["labels"]
        attention_mask   = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        maskable_mask = labels != -100

        # 1. timesteps
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=input_ids.device)
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. masking
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # 3. forward
        inputs["use_cache"] = False
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)

        # 4. weights
        loss_weights = (
            self.scheduler.weight(t).unsqueeze(1)
            if self.loss_weight_type == "scheduler"
            else 1.0
        )

        # 5. weighted CE
        assert (input_ids[maskable_mask] == labels[maskable_mask]).all(), \
            "Mismatch between input_ids and labels at valid positions"
        token_nll = F.cross_entropy(
            outputs.logits.transpose(1, 2), input_ids, reduction="none",
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

        # 6. loss
        local_loss = token_nll.sum() / maskable_mask.sum().clamp_min(1)
        if num_items_in_batch is not None:
            loss = local_loss * (maskable_mask.sum().clamp_min(1) / num_items_in_batch)
        else:
            loss = local_loss


        # ── metrics ──────────────────────────────────────────────────────────────
        with torch.no_grad():
            if mode == "train":
                if attention_mask is not None:
                    num_tokens_in_batch = self.accelerator.gather_for_metrics(
                        attention_mask.sum()
                    ).sum().item()
                else:
                    local_count = torch.tensor(l, device=input_ids.device)
                    num_tokens_in_batch = self.accelerator.gather_for_metrics(
                        local_count
                    ).sum().item()

                self._total_train_tokens += num_tokens_in_batch
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

            predictions       = outputs.logits.argmax(dim=-1)
            log_probs         = torch.log_softmax(outputs.logits, dim=-1)
            per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

            correct_tokens = ((predictions == input_ids) & masked_mask).sum()
            entropy_sum    = (per_token_entropy * masked_mask).sum()
            total_tokens   = masked_mask.sum()

            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens).sum()
            entropy_sum    = self.accelerator.gather_for_metrics(entropy_sum).sum()
            total_tokens   = self.accelerator.gather_for_metrics(total_tokens).sum()

            accuracy = (correct_tokens / total_tokens).item() if total_tokens > 0 else 0.0
            entropy  = (entropy_sum    / total_tokens).item() if total_tokens > 0 else 0.0

            self._metrics[mode]["mean_token_accuracy"].append(accuracy)
            self._metrics[mode]["entropy"].append(entropy)

            if mode == "eval":
                token_nll_unweighted = F.cross_entropy(
                    outputs.logits.transpose(1, 2), input_ids, reduction="none"
                ) * masked_mask.float()

                batch_nll  = self.accelerator.gather_for_metrics(token_nll_unweighted.sum()).sum().item()
                batch_mask = self.accelerator.gather_for_metrics(masked_mask.sum()).sum().item()

                self._eval_nll_sum    += batch_nll
                self._eval_masked_sum += batch_mask

        return (loss, outputs) if return_outputs else loss


    def evaluate(self, *args, **kwargs):
        self._eval_nll_sum    = 0.0
        self._eval_masked_sum = 0.0
        result = super().evaluate(*args, **kwargs)
        # log() has now read the accumulators — clear so they don't bleed into train logs
        self._eval_nll_sum    = 0.0
        self._eval_masked_sum = 0.0
        return result

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == "eval" and self._eval_masked_sum > 0:
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
            mean_nll = self._eval_nll_sum / self._eval_masked_sum
            metrics["eval_nll"] = mean_nll
            metrics["eval_bpd"] = mean_nll / math.log(2)
            metrics["eval_ppl"] = math.exp(mean_nll)
        elif mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()



@dataclass
class TrainingConfig:
    # ── Experiment identity ───────────────────────────────────────────────────
    output_dir: str = "output"
    model_name_or_path: str = "bert-base-uncased"
    run_name: Optional[str] = None
    seed: int = 42

    resume_from_checkpoint: Optional[str] = None  # path to checkpoint dir, or "latest"

    # ── Data ─────────────────────────────────────────────────────────────────
    max_length: int = 1024
    shuffle_dataset: bool = True
    dataset_num_proc: Optional[int] = None
    train_ds_path: str = "train_ds"
    eval_ds_path: str = "eval_ds"

    # ── Optimization ─────────────────────────────────────────────────────────
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "linear"  # "cosine" | "linear" | "constant" | ...
    warmup_ratio: Optional[float] = None
    warmup_steps: int = 0              # overridden by warmup_ratio if set
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # ── Training loop ────────────────────────────────────────────────────────
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 5.0
    max_steps: int = -1                # overrides num_train_epochs if > 0
    bf16: bool = True

    # ── Eval ─────────────────────────────────────────────────────────────────
    per_device_eval_batch_size: int =1
    eval_strategy: str = "steps"       # "steps" | "epoch" | "no"
    eval_steps: int = 100
    eval_on_start: bool = True
    #metric_for_best_model: str = "eval_nll"
    #greater_is_better: bool = False
    #load_best_model_at_end: bool = True

    # ── Logging ──────────────────────────────────────────────────────────────
    logging_steps: int = 25
    report_to: str = "wandb"
    project: Optional[str] = None      # only relevant if report_to != "none"

    # ── Memory & performance ─────────────────────────────────────────────────
    gradient_checkpointing: bool = False
    activation_offloading: bool = True
    torch_compile: bool = True
    use_liger_kernel: bool = True
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 8
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True


cs = ConfigStore.instance()
cs.store(name="config", node=TrainingConfig)

def format_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }

def run_training(cfg: TrainingConfig) -> None:

    def _sft_map_fn(example, tokenizer=None, max_length=None):
        enc = tokenizer.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True, max_length=max_length, truncation=True)
        
        input_ids      = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]
        attention_mask = enc["attention_mask"]
        labels = [tok if m == 1 else -100 for tok, m in zip(input_ids, assistant_mask)]        
        return {"input_ids": input_ids, "labels": labels, "assistant_masks": assistant_mask, "attention_mask": attention_mask}    
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model_name_or_path = cfg_dict.pop("model_name_or_path")
    resume_from_checkpoint = cfg_dict.pop("resume_from_checkpoint", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="auto")
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="auto")

    train_ds_path, eval_ds_path = cfg_dict.pop("train_ds_path"), cfg_dict.pop("eval_ds_path")
    train_ds = load_from_disk(train_ds_path).map(format_to_messages)
    eval_ds  = load_from_disk(eval_ds_path).map(format_to_messages)

    train_ds = train_ds.map(_sft_map_fn, fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length})
    eval_ds  = eval_ds.map(_sft_map_fn, fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length})

    scheduler = LinearAlphaScheduler()
    args = SFTConfig(
        **cfg_dict,
        logging_first_step=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = CustomForwardSFTTrainer(
        scheduler=scheduler,
        time_epsilon=1e-3,
        loss_weight_type="uniform",
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer
    )

    try:
        trainer.train()
    finally:
        # Delete stale references to free memory promptly
        del trainer
        del model
        del tokenizer
        del scheduler
        del args
        del train_ds
        del eval_ds
        del cfg_dict
        del resume_from_checkpoint

        # Garbage collect and release CUDA memory
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
@hydra.main(version_base=None, config_name="config")
def main(cfg: TrainingConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()