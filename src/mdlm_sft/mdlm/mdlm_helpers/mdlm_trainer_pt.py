# REWRITE OF THE ORIGINAL TRAINER FOUND HERE
# NOW MORE COMPATIBLE WITH THE TF TRAINER INTERFACE

import math
import weakref
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorWithPadding, EvalPrediction, Trainer,
                          TrainingArguments)

from mdlm_sft.mdlm.mdlm_helpers.mdlm_scheduler import CosineAlphaScheduler


## TRAINING ARGS ESSENTIALLY UNCHANGED - ARGS ARE CHECKED HERE RATHER THAN IN THE TRAINER
@dataclass
class MDLMConfig(TrainingArguments):
    time_epsilon: float = 0.001
    loss_weight_type: str = "uniform"

    batch_eval_metrics: bool = True
    output_dir: str = "mdlm_output"

    def __post_init__(self):
        super().__post_init__()

        if not (0.0 < self.time_epsilon < 1.0):
            raise ValueError(f"time_epsilon must be in (0, 1), got {self.time_epsilon}")
        if self.loss_weight_type not in ("scheduler", "uniform"):
            raise ValueError(
                f"loss_weight_type must be 'scheduler' or 'uniform', "
                f"got {self.loss_weight_type!r}"
            )
        if not self.batch_eval_metrics:
            raise ValueError(
                "MDLMConfig requires batch_eval_metrics=True for per-token "
                "NLL accumulation."
            )


# ------------------------------------------------------------------
# REWRITE OF THE METRICS
class NLLPPLMetricComputer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_nll = 0.0
        self.sum_w = 0.0

    def __call__(
        self, eval_pred: EvalPrediction, compute_result: bool
    ) -> Dict[str, float]:
        token_nll = np.asarray(eval_pred.predictions.cpu(), dtype=np.float64)
        weight = np.asarray(eval_pred.label_ids.cpu(), dtype=np.float64)

        self.sum_nll += float(token_nll.sum())
        self.sum_w += float(weight.sum())

        if not compute_result:
            return {}

        mean_nll = self.sum_nll / max(self.sum_w, 1.0)
        ppl = math.exp(mean_nll)
        self.reset()
        return {"nll": mean_nll, "ppl": ppl}


#####################################################################################
# Trainer
#####################################################################################


class MDLMTrainer(Trainer):
    def __init__(
        self,
        *args,
        scheduler: Optional[BaseAlphaScheduler] = None,
        **kwargs,
    ):
        """
        Open signature, pass-through to Trainer. The only MDLM-specific
        injection is `scheduler`, kept keyword-only so it never collides
        with future positional args added by `transformers.Trainer`.
        """
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

        # --- MDLM-specific state ---
        cfg: MDLMConfig = self.args  # populated by super().__init__
        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = cfg.time_epsilon
        self.loss_weight_type = cfg.loss_weight_type

        tok = self.processing_class
        if tok is None:
            raise ValueError("MDLMTrainer requires a tokenizer via `processing_class`.")
        if getattr(tok, "padding_side", None) != "right":
            raise ValueError(f"padding_side must be 'right', got {tok.padding_side!r}.")
        if getattr(tok, "mask_token_id", None) is None:
            raise ValueError("Tokenizer must define `mask_token_id`.")

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, outputs, _, _ = self._mdlm_forward(model, inputs)
        if return_outputs:
            return loss, outputs
        return loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, _, token_nll, maskable_mask = self._mdlm_forward(model, inputs)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        predictions = token_nll.detach().contiguous()
        label_ids = maskable_mask.to(predictions.dtype).detach().contiguous()
        return (loss.detach(), predictions, label_ids)

    # NOTE: PREDICT IS NEVER CALLED FROM THE TRAINER - MDLM NEEDS A SAMPLER
    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "MDLMTrainer does not support predict() because -100 padding "
            "corrupts per-token float metrics. Use evaluate() instead."
        )

    def _mdlm_forward(self, model, inputs):
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100

        # 1. timesteps
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. masking
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # 3. forward
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)

        # 4. weights
        loss_weights = (
            self.scheduler.weight(t).unsqueeze(1)
            if self.loss_weight_type == "scheduler"
            else 1.0
        )

        # 5. weighted CE
        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"
        token_nll = F.cross_entropy(
            outputs.logits.transpose(1, 2),
            input_ids,
            reduction="none",
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

        # 6. canonical token-mean
        loss = token_nll.sum() / maskable_mask.sum().clamp_min(1)
        ##loss = token_nll.sum() / masked_mask.sum().clamp_min(1)  # Normalize by masked tokens only

        return loss, outputs, token_nll, maskable_mask

