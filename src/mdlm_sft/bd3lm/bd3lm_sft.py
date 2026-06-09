import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from trl import SFTConfig, SFTTrainer

from mdlm_sft.mdlm.mdlm_helpers.mdlm_scheduler import BaseAlphaScheduler, LinearAlphaScheduler


@dataclass
class AppendEOSBlockWrapper(CollatorWrapper):
    """Right-pad every example with EOS so its length is a multiple of `block_size`.

    Padded positions are labeled with `eos_token_id` (NOT -100), so the model is
    trained to emit EOS on the tail — this is what makes the shared, per-batch
    block mask sound (padding does not need a per-sample attention mask).
    """

    block_size: int = 32

    def before(self, features):
        for ex in features:
            ids  = ex["input_ids"]
            labs = ex["labels"]
            assert isinstance(ids, list) and isinstance(labs, list)

            L = len(ids)
            target = (L + self.block_size - 1) // self.block_size * self.block_size
            pad_len = target - L
            if pad_len > 0:
                ex["input_ids"] = ids  + [self.tokenizer.eos_token_id] * pad_len
                ex["labels"]    = labs + [self.tokenizer.eos_token_id] * pad_len
        return features


# ======================================================================================
# Block-diffusion attention mask (redundant copy of the original)
# ======================================================================================
def _create_bd3lm_attention_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """FlexAttention `mask_mod` for the [x_t ; x_0] concatenation (length 2n).

    Combines three sub-masks:
      - M_BD  : block-diagonal self-attention within noised blocks (x_t<->x_t)
      - M_OBC : offset block-causal cross-attention (x_t attends to earlier x_0)
      - M_BC  : block-causal attention to update x_0 (x_0<->x_0)
    """
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    block_q  = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal      = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    block_causal        = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    return block_diagonal | offset_block_causal | block_causal


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class BD3LMSFTConfig(SFTConfig):
    """SFTConfig + block-diffusion knobs."""

    block_size: int = 32
    time_epsilon: float = 1e-3                 # t ∈ [eps, 1); avoids degenerate t→0
    loss_weight_type: str = "scheduler"        # "scheduler" | "uniform"
    deterministic_eval: bool = True            # reproducible per-batch eval noise
    eval_seed: int = 0


# ======================================================================================
# Trainer
# ======================================================================================
class BD3LMSFTTrainer(SFTTrainer):

    # ----- token-weighted accumulators (Σcorrect / Σentropy / Σtokens) -----
    @staticmethod
    def _zero_sums() -> dict[str, float]:
        return {"correct": 0.0, "entropy": 0.0, "tokens": 0.0}

    # ----- reproducible eval noise -----
    ### FAITHFUL: in eval we seed a per-batch generator so the SAME (t, mask) pattern is
    ### replayed at every checkpoint -> the high-variance 1/t weight becomes COMMON-MODE
    ### noise, so eval_nll DIFFERENCES across checkpoints are signal. Train stays unseeded.
    ### `stream` decorrelates the t-draw (0) from the mask-draw (1); process_index
    ### decorrelates DDP ranks. Estimand/expectation unchanged — only the RNG is fixed.
    def _eval_rand(self, shape, device, mode, stream: int):
        if mode != "eval" or not self.deterministic_eval:
            return torch.rand(*shape, device=device)
        seed = (
            self._eval_seed
            + 1_000_003 * self._eval_step
            + 1009 * stream
            + 97 * self.accelerator.process_index
        )
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.rand(*shape, device=device, generator=g)

    # ----- FlexAttention block mask (built fresh per forward) -----
    def _build_block_mask(self, l: int, device):
        # Lazy import: flex is only required when this trainer actually runs.
        from torch.nn.attention.flex_attention import create_block_mask

        return create_block_mask(
            partial(_create_bd3lm_attention_mask, block_size=self.block_size, n=l),
            B=None,
            H=None,
            Q_LEN=l * 2,
            KV_LEN=l * 2,
            device=device,
        )

    # ----- init -----
    def __init__(
        self,
        args: BD3LMSFTConfig | None = None,
        scheduler: BaseAlphaScheduler | None = None,
        *pargs,
        **kwargs,
    ):
        if not (0.0 < args.time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.block_size  = args.block_size
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.deterministic_eval = args.deterministic_eval   ### FAITHFUL
        self._eval_seed = args.eval_seed                     ### FAITHFUL
        self._eval_step = 0                                  ### FAITHFUL: per-eval batch counter.

        super().__init__(args=args, *pargs, **kwargs)

        ### FAITHFUL: own scalar accumulators (NO torchmetrics). `_eval_token_sum` counts
        ### ALL assistant tokens (the dimension), not the masked subset -> bits/assistant-token.
        self._eval_nll_sum   = 0.0     # Σ w(t)·CE over eval set (weighted NELBO numerator)
        self._eval_token_sum = 0.0   # Σ maskable-token count   (denominator)
        self._metric_sums = {"train": self._zero_sums(), "eval": self._zero_sums()}

    # ----- loss -----
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        input_ids = inputs["input_ids"]
        labels    = inputs["labels"]
        # NOTE: the incoming padding `attention_mask` is intentionally dropped — BD3LM
        # overrides attention entirely with the block mask, and AppendEOSBlockWrapper has
        # already turned padding into supervised EOS tokens.

        b, l = input_ids.shape
        maskable_mask = labels != -100
        n_maskable = maskable_mask.sum().clamp_min(1)

        # 1. timesteps  (seeded in eval via _eval_rand, raw rand in train)
        t = self.time_epsilon + (1 - self.time_epsilon) * self._eval_rand(
            (b,), input_ids.device, mode, stream=0
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. masking  (positions with label == -100 are never masked)
        masked_mask = (
            self._eval_rand((b, l), input_ids.device, mode, stream=1) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(masked_mask, self.processing_class.mask_token_id, input_ids)

        # 3. block-diffusion forward: feed [x_t ; x_0] with the block mask.
        #    concat_input_ids: [b, 2l] — first l noisy (x_t), last l clean (x_0).
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)
        block_mask = self._build_block_mask(l, input_ids.device)  # FlexAttention BlockMask
        base_pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)  # [b, 2l]

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=block_mask,
            position_ids=concat_position_ids,
        )

        ### Only the x_t half supervises. Slice -> [b, l, V], then gather the masked subset
        ### -> [N, V] (fp32 for stable softmax/CE) and free the [b, 2l, V] activation in eval.
        logits_xt      = outputs.logits[:, :l]                  # [b, l, V]
        masked_logits  = logits_xt[masked_mask].float()     # [N, V]
        masked_targets = input_ids[masked_mask]            # [N]
        del outputs, logits_xt

        # 4. training weights (gathered on the masked subset -> [N])
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]
        else:
            loss_weights = 1.0

        # 5. CE (single pass on [N, V]; reused for loss AND eval NELBO)
        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"
        ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")  # [N]

        # 6. loss  (token-normalized; rescaled to the global token count for grad-accum/DDP)
        local_loss = (ce * loss_weights).sum() / n_maskable
        if num_items_in_batch is not None:
            loss = local_loss * (n_maskable / num_items_in_batch)
        else:
            loss = local_loss

        # ── metrics ──────────────────────────────────────────────────────────────────
        with torch.no_grad():
            if mode == "train":
                if (am := inputs.get("attention_mask", None)) is not None:
                    num_tokens_in_batch = (
                        self.accelerator.gather_for_metrics(am.sum()).sum().item()
                    )
                else:
                    local_count = torch.tensor(b * l, device=input_ids.device)
                    num_tokens_in_batch = (
                        self.accelerator.gather_for_metrics(local_count).sum().item()
                    )
                self._total_train_tokens += num_tokens_in_batch
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

            ### accuracy + entropy on [N, V] only (diagnostic; masked-token denominator).
            log_probs = torch.log_softmax(masked_logits, dim=-1)            # [N, V]
            per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # [N]
            correct = masked_logits.argmax(dim=-1) == masked_targets        # [N]
            del log_probs, masked_logits

            n_masked = masked_mask.sum()
            n_masked_g = self.accelerator.gather_for_metrics(n_masked).sum()
            correct_tokens = self.accelerator.gather_for_metrics(correct.sum()).sum()
            entropy_sum = self.accelerator.gather_for_metrics(per_token_entropy.sum()).sum()

            self._metric_sums[mode]["correct"] += correct_tokens.item()
            self._metric_sums[mode]["entropy"] += entropy_sum.item()
            self._metric_sums[mode]["tokens"]  += n_masked_g.item()

            ###   EVAL-ONLY continuous-time NELBO per assistant token (MDLM-faithful).
            ###   numerator   = Σ_masked w(t)·CE   with w(t) = -α'/(1-α)  (=1/t for linear)
            ###   denominator = Σ maskable          (ALL assistant tokens, masked or not)
            ### Always weighted, regardless of loss_weight_type.
            if mode == "eval":
                w = self.scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]  # [N]
                batch_nll = (w.double() * ce.double()).sum()                         # fp64 sum
                batch_toks = maskable_mask.sum()
                batch_nll = self.accelerator.gather_for_metrics(batch_nll).sum().item()
                batch_toks = self.accelerator.gather_for_metrics(batch_toks).sum().item()
                self._eval_nll_sum += batch_nll
                self._eval_token_sum += batch_toks
                self._eval_step += 1

        return (loss, {}) if return_outputs else loss

    # ----- loss-only prediction step (never retains [B, L, V] logits) -----
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def evaluate(self, *args, **kwargs):
        ## reset NELBO accumulators + reproducible-noise counter before the loop.
        self._eval_nll_sum = 0.0
        self._eval_token_sum = 0.0
        self._eval_step = 0
        self._metric_sums["eval"] = self._zero_sums()
        result = super().evaluate(*args, **kwargs)
        # log() has consumed the accumulators — clear so they don't bleed into train logs.
        self._eval_nll_sum = 0.0
        self._eval_token_sum = 0.0
        self._metric_sums["eval"] = self._zero_sums()
        return result

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        ### token-weighted accuracy/entropy (Σ/Σ). Emit ONLY when tokens were accumulated;
        ### no `else 0.0` fallback (the end-of-training eval-mode log() fires with empty
        ### accumulators and would otherwise plot a phantom drop).
        sums = self._metric_sums[mode]
        tok = sums["tokens"]
        if tok > 0:
            metrics["mean_token_accuracy"] = sums["correct"] / tok
            metrics["entropy"] = sums["entropy"] / tok

        ### weighted NELBO -> bpd/ppl from the OWN accumulators.
        if mode == "eval" and self._eval_token_sum > 0:
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
            mean_nll = self._eval_nll_sum / self._eval_token_sum   # Σ w·CE / Σ maskable
            metrics["eval_nll"] = mean_nll
            metrics["eval_bpd"] = mean_nll / math.log(2)
            metrics["eval_ppl"] = math.exp(min(mean_nll, 30.0))    # clamp to avoid +inf early
        elif mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()
        self._metric_sums[mode] = self._zero_sums()