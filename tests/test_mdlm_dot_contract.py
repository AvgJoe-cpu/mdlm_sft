"""
Contract tests for MDLMDoTTrainer.

Verifies the three guarantees of the DoT-aware MDLM trainer subclass:

  1. No-op when src_mask absent       → loss bit-identical to parent
  2. No-op when contract holds        → loss bit-identical to parent
  3. Loud failure when contract broken → AssertionError w/ diagnostic

Run:
    python -m tests.test_mdlm_dot_contract
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from mdlm_sft.mdlm.mdlm_sft_v2 import CustomForwardSFTTrainer
from mdlm_sft.mdlm.mdlm_cot import MDLMDoTTrainer 


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — minimal stubs to drive compute_loss in eval mode.
# We deliberately AVOID calling SFTTrainer.__init__ (object.__new__) so the
# test doesn't depend on accelerator/dataset/optimizer scaffolding.
# ─────────────────────────────────────────────────────────────────────────────

class _TinyMLM(nn.Module):
    """Replaces AutoModelForMaskedLM. Returns SimpleNamespace(logits=[B,L,V])."""
    def __init__(self, vocab_size: int, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head  = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


class _StubScheduler:
    """Linear schedule  α(t) = 1 - t  ⇒  w(t) = -α'/(1-α) = 1/t.
    Matches LinearAlphaScheduler's contract; standalone for test isolation."""
    def __call__(self, t):  return 1.0 - t
    def weight(self, t):    return 1.0 / t


class _StubAccelerator:
    """Single-process accelerator: identity gather, rank 0."""
    process_index = 0
    @staticmethod
    def gather_for_metrics(x): return x


def _build_trainer_skeleton(cls, *, model, tokenizer, scheduler):
    """Construct `cls` without running SFTTrainer.__init__.

    Populates ONLY the attributes compute_loss / _eval_rand currently read.
    If the parent starts reading a new attribute, we'll AttributeError loudly —
    which is the correct failure mode for a contract test."""
    t = object.__new__(cls)
    t.model               = model
    t.processing_class    = tokenizer
    t.scheduler           = scheduler
    t.time_epsilon        = 1e-3
    t.loss_weight_type    = "scheduler"
    t.deterministic_eval  = True              # seeded noise → reproducible loss
    t._eval_seed          = 0
    t._eval_step          = 0
    t._eval_nll_sum       = 0.0
    t._eval_token_sum     = 0.0
    t._metrics            = {"train": {}, "eval": {}}
    t._metric_sums        = {"train": {"correct": 0., "entropy": 0., "tokens": 0.},
                             "eval":  {"correct": 0., "entropy": 0., "tokens": 0.}}
    t._total_train_tokens = 0
    t.accelerator         = _StubAccelerator()
    return t


def _make_batch(*, batch_size, seq_len, vocab_size, src_len, pad_at_end, device):
    """Build a batch with the DoT contract PRE-satisfied:
       labels == -100 EXACTLY where (src_mask | ~attention_mask).

    Layout (per row):
       [   src tokens   |     target tokens     |   pad    ]
       |<-- src_len --->|<-- seq_len - src_len  |--pad_at_end-->
                            - pad_at_end -->|
    """
    input_ids      = torch.randint(low=10, high=vocab_size,
                                   size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    if pad_at_end > 0:
        attention_mask[:, -pad_at_end:] = 0
    src_mask       = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    src_mask[:, :src_len] = True
    labels         = input_ids.clone()
    labels[src_mask | ~attention_mask.bool()] = -100
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "src_mask":       src_mask,
    }


def _compute_loss(trainer, inputs) -> float:
    """One compute_loss call with fully reset state.

    Why reset: in eval mode the parent mutates _eval_step (drives reproducible
    noise) and the NELBO/metric accumulators. Resetting before each call makes
    every call an independent, reproducible event."""
    trainer._eval_step      = 0
    trainer._eval_nll_sum   = 0.0
    trainer._eval_token_sum = 0.0
    trainer._metrics        = {"train": {}, "eval": {}}
    trainer._metric_sums    = {"train": {"correct": 0., "entropy": 0., "tokens": 0.},
                               "eval":  {"correct": 0., "entropy": 0., "tokens": 0.}}
    with torch.no_grad():
        # dict(inputs): defensive copy in case anything mutates the dict.
        return trainer.compute_loss(trainer.model, dict(inputs)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point.
# ─────────────────────────────────────────────────────────────────────────────

def run_dot_contract_tests(
    *,
    vocab_size:    int  = 64,
    seq_len:       int  = 24,
    batch_size:    int  = 4,
    src_len:       int  = 6,
    pad_at_end:    int  = 3,
    mask_token_id: int  = 4,
    seed:          int  = 1234,
    device:        str  = "cpu",
    verbose:       bool = True,
) -> dict:
    """Run all three contract scenarios.

    Returns a dict of recorded losses / messages keyed by scenario.
    Raises AssertionError on the first scenario that fails."""

    torch.manual_seed(seed)

    # Shared fixtures. The SAME model instance is used by both trainers so
    # weights match bit-for-bit on each forward pass.
    model     = _TinyMLM(vocab_size=vocab_size).to(device).eval()
    tokenizer = SimpleNamespace(mask_token_id=mask_token_id)
    scheduler = _StubScheduler()
    batch     = _make_batch(batch_size=batch_size, seq_len=seq_len,
                            vocab_size=vocab_size, src_len=src_len,
                            pad_at_end=pad_at_end, device=device)
    base_inputs = {k: v for k, v in batch.items() if k != "src_mask"}

    parent = _build_trainer_skeleton(CustomForwardSFTTrainer,
                                     model=model, tokenizer=tokenizer, scheduler=scheduler)
    child  = _build_trainer_skeleton(MDLMDoTTrainer,
                                     model=model, tokenizer=tokenizer, scheduler=scheduler)

    results: dict[str, Any] = {}

    # ── 1. src_mask absent ─────────────────────────────────────────────────
    # Child's `if "src_mask" in inputs:` branch is skipped entirely → the
    # super().compute_loss() call sees the SAME inputs the parent does.
    parent_loss = _compute_loss(parent, base_inputs)
    child_loss  = _compute_loss(child,  base_inputs)
    assert parent_loss == child_loss, (
        f"[Scenario 1] FAIL  parent={parent_loss!r}  child={child_loss!r}"
    )
    if verbose:
        print(f"[Scenario 1] PASS  src_mask absent           loss = {parent_loss:.10f}")
    results["scenario_1_no_src_mask"] = parent_loss

    # ── 2. contract holds ──────────────────────────────────────────────────
    # Child's assertion passes (dot_maskable == labels_maskable by construction),
    # strips src_mask, delegates. Result MUST match a parent fed the same inputs
    # without src_mask — because labels are identical and the parent never
    # looked at src_mask anyway.
    parent_loss = _compute_loss(parent, base_inputs)
    child_loss  = _compute_loss(child,  batch)
    assert parent_loss == child_loss, (
        f"[Scenario 2] FAIL  parent={parent_loss!r}  child={child_loss!r}"
    )
    if verbose:
        print(f"[Scenario 2] PASS  contract holds            loss = {parent_loss:.10f}")
    results["scenario_2_contract_holds"] = parent_loss

    # ── 3. contract broken ─────────────────────────────────────────────────
    # Flip src_mask[0,0] True → False. That position now satisfies
    #   dot_maskable     = attn & ~src = 1 & True  = True
    #   labels_maskable  = labels != -100 = -100 != -100 = False
    # → falls into the "in (attn & ~src) but labels==-100 (under-trained)"
    # branch of the diagnostic. We assert that exact substring appears, which
    # also proves the diagnostic correctly identifies WHICH side disagreed.
    broken_src_mask = batch["src_mask"].clone()
    broken_src_mask[0, 0] = False
    broken_inputs = {**base_inputs, "src_mask": broken_src_mask}

    raised = False
    try:
        _compute_loss(child, broken_inputs)
    except AssertionError as e:
        raised = True
        msg = str(e)
        required = [
            "DoT contract violated",
            "in (attn & ~src) but labels==-100 (under-trained):",
        ]
        missing = [s for s in required if s not in msg]
        assert not missing, (
            f"[Scenario 3] FAIL  diagnostic missing fragments: {missing}\n"
            f"  full message:\n{msg}"
        )
        if verbose:
            print("[Scenario 3] PASS  contract broken          "
                  "AssertionError raised with diagnostic")
            print(f"             excerpt: {msg.splitlines()[0]!r}")
        results["scenario_3_contract_broken_msg"] = msg

    assert raised, "[Scenario 3] FAIL  expected AssertionError was not raised"
    return results


if __name__ == "__main__":
    run_dot_contract_tests()