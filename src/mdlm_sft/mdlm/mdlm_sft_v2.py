from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, Union
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

    ### CHANGE (stability #3): token-weighted accumulators (Σcorrect/Σentropy/Σtokens).
    @staticmethod
    def _zero_sums() -> dict[str, float]:
        return {"correct": 0.0, "entropy": 0.0, "tokens": 0.0}

    ### FAITHFUL: reproducible eval noise. In eval we seed a per-batch generator so the
    ### SAME (t, mask) pattern is replayed at every checkpoint -> the high-variance 1/t
    ### weight becomes COMMON-MODE noise, so eval_nll DIFFERENCES across checkpoints are
    ### signal (the run-to-run jitter you flagged disappears). Train stays unseeded.
    ### `stream` decorrelates the t-draw (0) from the mask-draw (1); process_index
    ### decorrelates DDP ranks. Estimand/expectation are unchanged — only the RNG is fixed.
    def _eval_rand(self, shape, device, mode, stream: int):
        if mode != "eval" or not self.deterministic_eval:
            return torch.rand(*shape, device=device)
        seed = (self._eval_seed
                + 1_000_003 * self._eval_step
                + 1009 * stream
                + 97 * self.accelerator.process_index)
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.rand(*shape, device=device, generator=g)

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
        time_epsilon: float = 1e-3,
        loss_weight_type: str = "scheduler",
        deterministic_eval: bool = True,   ### FAITHFUL: reproducible eval noise (see _eval_rand).
        eval_seed: int = 0,                ### FAITHFUL: base seed for eval noise.
    ):
        self.scheduler = scheduler
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type
        self.deterministic_eval = deterministic_eval   ### FAITHFUL
        self._eval_seed = eval_seed                     ### FAITHFUL
        self._eval_step = 0                             ### FAITHFUL: per-eval batch counter (reset in evaluate()).
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

        ### FAITHFUL: own scalar accumulators (NO torchmetrics). `_eval_token_sum` counts
        ### ALL assistant tokens (the dimension), not the masked subset -> bits/assistant-token.
        self._eval_nll_sum   = 0.0   # Σ over eval set of  w(t)·CE   (weighted NELBO numerator)
        self._eval_token_sum = 0.0   # Σ over eval set of  maskable-token count   (denominator)
        self._metric_sums = {"train": self._zero_sums(), "eval": self._zero_sums()}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        input_ids        = inputs["input_ids"]
        labels           = inputs["labels"]
        attention_mask   = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        maskable_mask = labels != -100
        n_maskable    = maskable_mask.sum().clamp_min(1)  ### CHANGE (P0): cache once; reused below.

        # 1. timesteps
        ### FAITHFUL: route both noise draws through _eval_rand (seeded in eval, raw in train).
        t = self.time_epsilon + (1 - self.time_epsilon) * self._eval_rand((b,), input_ids.device, mode, stream=0)
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. masking
        masked_mask = (
            self._eval_rand((b, l), input_ids.device, mode, stream=1) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # 3. forward
        ### CHANGE (P0): removed dead `inputs["use_cache"] = False` (inputs never passed to model()).
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)

        ### CHANGE (P0): collapse [B,L,V] -> [N,V] on the masked subset, then drop `outputs`.
        ### fp32 keeps the softmax/CE reductions stable. In eval (no_grad via prediction_step)
        ### `del outputs` frees [B,L,V] immediately; in train the graph holds it until backward.
        masked_logits  = outputs.logits[masked_mask].float()   # [N, V]
        masked_targets = input_ids[masked_mask]                # [N]
        del outputs

        # 4. weights (TRAINING gradient only)
        ### CHANGE (P0): gathered on the masked subset -> [N], no [B,L] temporary.
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]  # [N]
        else:
            loss_weights = 1.0

        # 5. CE (single pass on [N, V]; reused for loss AND eval NELBO)
        assert (input_ids[maskable_mask] == labels[maskable_mask]).all(), \
            "Mismatch between input_ids and labels at valid positions"
        ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")  # [N]

        # 6. loss
        local_loss = (ce * loss_weights).sum() / n_maskable
        if num_items_in_batch is not None:
            loss = local_loss * (n_maskable / num_items_in_batch)
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

            ### CHANGE (P0): accuracy + entropy on [N,V] only (diagnostic, masked-token denom).
            log_probs         = torch.log_softmax(masked_logits, dim=-1)         # [N, V]
            per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)       # [N]
            correct           = (masked_logits.argmax(dim=-1) == masked_targets) # [N]
            del log_probs, masked_logits

            n_masked       = masked_mask.sum()
            n_masked_g     = self.accelerator.gather_for_metrics(n_masked).sum()
            correct_tokens = self.accelerator.gather_for_metrics(correct.sum()).sum()
            entropy_sum    = self.accelerator.gather_for_metrics(per_token_entropy.sum()).sum()

            self._metric_sums[mode]["correct"] += correct_tokens.item()
            self._metric_sums[mode]["entropy"] += entropy_sum.item()
            self._metric_sums[mode]["tokens"]  += n_masked_g.item()

            ### FAITHFUL: EVAL-ONLY continuous-time NELBO per assistant token (MDLM-faithful).
            ###   numerator  = Σ_masked w(t)·CE   with w(t) = -α'/(1-α)  (=1/t for linear)
            ###   denominator= Σ maskable          (ALL assistant tokens, masked or not)
            ### Always weighted, regardless of loss_weight_type (eval must be schedule-faithful;
            ### the training gradient's weighting is a separate, orthogonal choice).
            if mode == "eval":
                w = self.scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]   # [N]
                batch_nll  = (w.double() * ce.double()).sum()                        ### FAITHFUL: fp64 sum.
                batch_toks = maskable_mask.sum()
                batch_nll  = self.accelerator.gather_for_metrics(batch_nll).sum().item()
                batch_toks = self.accelerator.gather_for_metrics(batch_toks).sum().item()
                self._eval_nll_sum   += batch_nll      ### FAITHFUL: own accumulator (was self.valid_nll).
                self._eval_token_sum += batch_toks     ### FAITHFUL: maskable denominator.
                self._eval_step      += 1              ### FAITHFUL: advance reproducible-noise counter.

        ### CHANGE (P0): never return [B,L,V] logits to the eval loop.
        return (loss, {}) if return_outputs else loss

    ### CHANGE (P1): loss-only prediction step; never retain/concatenate per-batch logits.
    ### Wrapping compute_loss in no_grad here is also what lets `del outputs` actually free
    ### the [B,L,V] activation during eval. autocast matches train (bf16).
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def evaluate(self, *args, **kwargs):
        ### FAITHFUL: reset NELBO accumulators + reproducible-noise counter before the loop.
        self._eval_nll_sum   = 0.0
        self._eval_token_sum = 0.0
        self._eval_step      = 0
        self._metric_sums["eval"] = self._zero_sums()
        result = super().evaluate(*args, **kwargs)
        # log() has consumed the accumulators — clear so they don't bleed into train logs.
        self._eval_nll_sum   = 0.0
        self._eval_token_sum = 0.0
        self._metric_sums["eval"] = self._zero_sums()
        return result

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        ### CHANGE (stability #3): token-weighted accuracy/entropy (Σ/Σ).
        ### FIX: emit ONLY when tokens were accumulated. NO `else 0.0` fallback — the
        ### end-of-training summary log() fires in eval mode with empty accumulators
        ### (tok == 0); any 0.0 written here is plotted as a phantom drop on the
        ### eval_entropy / eval_mean_token_accuracy charts. Mirrors the `_eval_token_sum > 0`
        ### guard used for nll/bpd/ppl below.
        sums = self._metric_sums[mode]
        tok  = sums["tokens"]
        if tok > 0:
            metrics["mean_token_accuracy"] = sums["correct"] / tok
            metrics["entropy"]             = sums["entropy"] / tok

        ### FAITHFUL: weighted NELBO -> bpd/ppl, read from the OWN accumulators (renamed denom).
        if mode == "eval" and self._eval_token_sum > 0:
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
            mean_nll = self._eval_nll_sum / self._eval_token_sum   # Σ w·CE / Σ maskable
            metrics["eval_nll"] = mean_nll
            metrics["eval_bpd"] = mean_nll / math.log(2)
            ### CHANGE (stability #2): clamp exponent so early-training nll can't overflow ppl to +inf.
            metrics["eval_ppl"] = math.exp(min(mean_nll, 30.0))
        elif mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()
        self._metric_sums[mode] = self._zero_sums()

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
    lr_scheduler_type: str = "cosine"  # "cosine" | "linear" | "constant" | ...#
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None  # e.g. {"num_cycles": 2}

    warmup_ratio: Optional[float] = 0.03
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
    metric_for_best_model: str = "eval_nll"
    greater_is_better: bool = False
    load_best_model_at_end: bool = False

    # ── Logging ──────────────────────────────────────────────────────────────
    logging_steps: int = 25
    report_to: str = "wandb"
    project: Optional[str] = "CoT-chat"      # only relevant if report_to != "none"

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

DatasetInput = Union[str, Dataset]

def run_training(
    cfg: TrainingConfig,
    train_ds: Optional[DatasetInput] = None,
    eval_ds:  Optional[DatasetInput] = None,
) -> None:

    def _sft_map_fn(example, tokenizer=None, max_length=None):
        enc = tokenizer.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True, max_length=max_length, truncation=True)
        
        input_ids      = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]
        attention_mask = enc["attention_mask"]
        labels = [tok if m == 1 else -100 for tok, m in zip(input_ids, assistant_mask)]        
        return {"input_ids": input_ids, "labels": labels, "assistant_masks": assistant_mask, "attention_mask": attention_mask}    
    
    cfg_dict               = OmegaConf.to_container(cfg, resolve=True)
    model_name_or_path     = cfg_dict.pop("model_name_or_path")
    resume_from_checkpoint = cfg_dict.pop("resume_from_checkpoint", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="auto")
    model     = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="auto")

    # Always pop both to prevent leaking into SFTConfig
    train_src = train_ds if train_ds is not None else cfg_dict.pop("train_ds_path")
    eval_src  = eval_ds  if eval_ds  is not None else cfg_dict.pop("eval_ds_path")

    if not all(isinstance(x, (str, Dataset)) for x in (train_src, eval_src)):
        raise TypeError(f"Expected str or Dataset, got {type(train_src).__name__!r} / {type(eval_src).__name__!r}")
    if type(train_src) is not type(eval_src):
        raise TypeError(f"Types must match: {type(train_src).__name__!r} vs {type(eval_src).__name__!r}")

    _load = lambda x: (load_from_disk(x, keep_in_memory=True) if isinstance(x, str) else x)
    train_dataset = _load(train_src).map(format_to_messages).map(_sft_map_fn, fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length})
    eval_dataset  = _load(eval_src).map(format_to_messages).map(_sft_map_fn,  fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length})

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
        loss_weight_type="scheduler",
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        # Delete stale references to free memory promptly
        del trainer, model, tokenizer, scheduler, args, train_dataset, eval_dataset, cfg_dict, resume_from_checkpoint
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