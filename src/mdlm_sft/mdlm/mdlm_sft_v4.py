from __future__ import annotations
from transformers import HfArgumentParser

import dataclasses
import gc
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

import torch
import torch.nn.functional as F
import datasets
from datasets import Dataset, load_from_disk
from transformers import AutoModelForMaskedLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler

datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB
log = logging.getLogger(__name__)


@dataclass
class MDLMSFTConfig(SFTConfig):
    # ── MDLM-trainer fields (live on `args`, swept via Hydra) ────────────────
    time_epsilon:                  float         = 1e-3
    loss_weight_type:              str           = "scheduler"   # "scheduler" | "uniform"
    deterministic_eval:            bool          = True
    eval_seed:                     int           = 0

    # ── Non-SFTConfig plumbing (popped before trainer construction) ──────────
    model_name_or_path:            str           = "bert-base-uncased"
    train_ds_path:                 Optional[str] = None
    eval_ds_path:                  Optional[str] = None
    resume_from_checkpoint:        Optional[str] = None
    max_length:                    int           = 1024

    # ── Run identity ─────────────────────────────────────────────────────────
    output_dir:                    str           = "outputs"

    # ── Eval & logging ───────────────────────────────────────────────────────
    report_to:                     str           = "wandb"
    logging_steps:                 int           = 100
    eval_strategy:                 str           = "steps"
    eval_steps:                    int           = 100
    eval_on_start:                 bool          = True
    metric_for_best_model:         str           = "eval_nll"
    greater_is_better:             bool          = False
    load_best_model_at_end:        bool          = False

    # ── Memory & performance ─────────────────────────────────────────────────
    bf16:                          bool          = True
    gradient_checkpointing:        bool          = False
    activation_offloading:         bool          = False
    torch_compile:                 bool          = True
    use_liger_kernel:              bool          = False
    dataloader_num_workers:        int           = 16
    dataloader_prefetch_factor:    int           = 16
    dataloader_pin_memory:         bool          = True
    dataloader_persistent_workers: bool          = True
    use_cpu:                       bool          = False


class CustomForwardSFTTrainer(SFTTrainer):
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
    def _eval_rand(self, shape: tuple[int, ...] | torch.Size, device: torch.device | str, mode: str, stream: int) -> torch.Tensor:
        if mode != "eval" or not self.deterministic_eval:
            return torch.rand(*shape, device=device)
        seed = (self._eval_seed
                + 1_000_003 * self._eval_step
                + 1009 * stream
                + 97 * self.accelerator.process_index)
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.rand(*shape, device=device, generator=g)    

    def __init__(
        self,
        args: MDLMSFTConfig,                 # ← typed to our subclass
        alpha_scheduler: Optional[Any] = None,   # stays a constructor kwarg: it's an instance, not config
        *pargs,
        **kwargs,
    ):
        # Custom fields set BEFORE super().__init__, read straight from args.
        self.alpha_scheduler    = alpha_scheduler or LinearAlphaScheduler()
        self.time_epsilon       = args.time_epsilon
        self.loss_weight_type   = args.loss_weight_type
        self.deterministic_eval = args.deterministic_eval
        self._eval_seed         = args.eval_seed
        self._eval_step         = 0

        super().__init__(args=args, *pargs, **kwargs)

        self._eval_nll_sum   = 0.0
        self._eval_token_sum = 0.0
        self._metric_sums    = {"train": self._zero_sums(), "eval": self._zero_sums()}    

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs=False, num_items_in_batch=None) -> torch.Tensor | tuple[torch.Tensor, dict]:
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
            p_mask = 1.0 - self.alpha_scheduler(t).unsqueeze(1).expand(b, l)

            # 2. masking
            masked_mask = (
                self._eval_rand((b, l), input_ids.device, mode, stream=1) < p_mask
            ) & maskable_mask
            noised_input_ids = torch.where(
                masked_mask, self.processing_class.mask_token_id, input_ids
            ) # type: ignore

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
                loss_weights = self.alpha_scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]  # [N]
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
                n_masked_g     = self.accelerator.gather_for_metrics(n_masked).sum().item()
                correct_tokens = self.accelerator.gather_for_metrics(correct.sum()).sum().item()
                entropy_sum    = self.accelerator.gather_for_metrics(per_token_entropy.sum()).sum().item()

                self._metric_sums[mode]["correct"] += correct_tokens
                self._metric_sums[mode]["entropy"] += entropy_sum
                self._metric_sums[mode]["tokens"]  += n_masked_g
                ### FAITHFUL: EVAL-ONLY continuous-time NELBO per assistant token (MDLM-faithful).
                ###   numerator  = Σ_masked w(t)·CE   with w(t) = -α'/(1-α)  (=1/t for linear)
                ###   denominator= Σ maskable          (ALL assistant tokens, masked or not)
                ### Always weighted, regardless of loss_weight_type (eval must be schedule-faithful;
                ### the training gradient's weighting is a separate, orthogonal choice).
                if mode == "eval":
                    w = self.alpha_scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]   # [N]
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




def format_to_messages(batch):
    prompts = batch["prompt"]
    completions = batch["completion"]
    return {"messages": [[{"role": "user", "content": p}, {"role": "assistant", "content": c}] for p, c in zip(prompts, completions)]}


def _sft_map_fn(batch, tokenizer=None, max_length=None):
    enc = tokenizer.apply_chat_template(
        batch["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        max_length=max_length,
        truncation=True,
    )
    input_ids       = enc["input_ids"]
    assistant_masks = enc["assistant_masks"]
    attention_mask  = enc["attention_mask"]

    # Vectorized -100 masking per example (lengths differ, so do it per row with numpy)
    labels = [
        np.where(np.asarray(m, dtype=bool), np.asarray(t), -100).tolist()
        for t, m in zip(input_ids, assistant_masks)
    ]
    return {
        "input_ids":       input_ids,
        "labels":          labels,
        "assistant_masks": assistant_masks,
        "attention_mask":  attention_mask,
    }

def run_training(cfg: MDLMSFTConfig, save_last: bool= True) -> None:

    resume_from_checkpoint = cfg.resume_from_checkpoint
    max_length             = cfg.max_length

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True, device_map="auto")

    train_ds = load_from_disk(path=cfg.train_ds_path, keep_in_memory=True)
    train_dataset = (
        train_ds
        .map(format_to_messages, batched=True, num_proc=4)
        .map(_sft_map_fn, batched=True, batch_size=8000, num_proc=4,
             fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})
    ).remove_columns(["prompt", "completion", "messages"])
    del train_ds

    eval_ds = load_from_disk(path=cfg.eval_ds_path, keep_in_memory=True)
    eval_dataset = (
        eval_ds
        .map(format_to_messages, batched=True, num_proc=4)
        .map(_sft_map_fn, batched=True, batch_size=8000, num_proc=4,
             fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})
    ).remove_columns(["prompt", "completion", "messages"])
    del eval_ds
    gc.collect()

    model   = AutoModelForMaskedLM.from_pretrained(cfg.model_name_or_path, trust_remote_code=True, device_map="auto")
    trainer = CustomForwardSFTTrainer(args=cfg, alpha_scheduler=LinearAlphaScheduler(), model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, processing_class=tokenizer)

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        log.info("Training finished")
        if save_last: trainer.save_model(cfg.output_dir)
    finally:
        try:
            for attr in ("optimizer", "lr_scheduler", "model_wrapped", "model", "train_dataset", "eval_dataset", "callback_handler"):
                if hasattr(trainer, attr): setattr(trainer, attr, None)
        except Exception: pass
        try:
            import wandb         # End any active W&B run so the next sweep trial starts clean
            if wandb.run is not None: wandb.finish()
        except Exception: pass

        del trainer, model, tokenizer, train_dataset, eval_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = HfArgumentParser(MDLMSFTConfig)
    (cfg,) = parser.parse_args_into_dataclasses()
    log.info("Training config:\n%s", cfg.to_json_string())

    try:
        run_training(cfg)
    finally:
        for h in list(log.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                log.removeHandler(h)
        except Exception: pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
