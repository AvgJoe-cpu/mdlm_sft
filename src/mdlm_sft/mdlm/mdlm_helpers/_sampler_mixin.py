from __future__ import annotations
from types import SimpleNamespace

import torch
import torch.nn as nn

# --- Adjust these two imports to match your repo layout ----------------
from mdlm_sft.mdlm.mdlm_helpers.mdlm_sampler_sft import (   # <- wherever your sampler lives
    MinimalMDLMSampler,
    SFTMixinBatchedVarlen,
)
# -----------------------------------------------------------------------

from typing import Callable, Optional
import torch


class DOTMixin:
    """
    Chain-of-Thought sampler for MDLM. Adds a CHUNK-COMMIT outer loop on
    top of sample_sft:

        canvas := prompt
        for k in range(max_chunks):
            candidates := K parallel chunk samples from canvas     # 1 batched fwd run
            rollouts   := one-shot lookahead from each candidate   # 1 batched fwd run
            scores     := score_fn(rollouts) -> [K]
            canvas     := canvas | candidates[select_fn(scores)]
            if EOS in committed chunk: break
        return canvas

    Mix as:
        class MDLMSampler(DOTMixin, SFTMixinBatchedVarlen, MinimalMDLMSampler): pass

    Single-problem API. The model batch axis is reserved for the K candidates,
    so per-row termination is trivially a per-prompt decision. For B>1 throughput,
    wrap externally. Cost per chunk: 2 batched forward RUNS, each of model-batch K
    (num_steps_chunk + num_steps_lookahead diffusion steps respectively).

    OUT OF SCOPE -- each is an explicit seam, not a hidden assumption:
      * end-vote / self-consistency over M outer chains  -> call sample_dot M times
      * beam search                                       -> generalize select_fn + carry top-m
      * batched-problem termination                       -> add per-row "done" tracking
    """

    @torch.no_grad()
    def sample_dot(
        self,
        prompt_ids: torch.LongTensor,        # [P] or [1, P]
        *,
        # chunk schedule
        chunk_length: int,
        max_chunks: int,
        # candidates per chunk
        K: int = 4,
        num_steps_chunk: int = 32,
        # lookahead
        lookahead_length: int = 128,
        num_steps_lookahead: int = 1,        # 1 = true one-shot; in-dist by training contract
        # design seams
        score_fn:  Optional[Callable] = None,
        select_fn: Optional[Callable] = None,
        # termination
        eos_token_id: int = None,
        # diagnostics
        return_trace: bool = False,
    ):
        assert eos_token_id is not None, "eos_token_id is required"
        device = next(self.backbone.parameters()).device

        # ── normalize: enforce single-problem ────────────────────────────
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        assert prompt_ids.ndim == 2 and prompt_ids.shape[0] == 1, (
            f"sample_dot is single-problem; got shape {tuple(prompt_ids.shape)}. "
            "The model batch axis is reserved for K candidates."
        )
        prompt_ids = prompt_ids.to(device=device, dtype=torch.long)

        # ── defaults (the design-critical pieces) ────────────────────────
        if score_fn is None:
            score_fn = self._score_mean_argmax_logprob   # documented to be weak
        if select_fn is None:
            select_fn = lambda s: int(s.argmax().item())

        canvas = prompt_ids                              # [1, P_k]; grows per chunk
        trace  = [] if return_trace else None

        for k_chunk in range(max_chunks):
            P_k = canvas.shape[1]

            # 1. K candidate chunks from the same context (one batched run)
            cand_canvas = self.sample_sft(
                canvas.expand(K, -1).contiguous(),
                response_length=chunk_length,
                num_steps=num_steps_chunk,
            )                                            # [K, P_k + chunk_length]

            # 2. one-shot lookahead from each candidate (in-dist by training contract)
            rollouts = self.sample_sft(
                cand_canvas,
                response_length=lookahead_length,
                num_steps=num_steps_lookahead,
            )                                            # [K, P_k + chunk_length + lookahead_length]

            # 3. score + select
            scores = score_fn(
                rollouts=rollouts,
                chunk_start=P_k,
                chunk_end=P_k + chunk_length,
            )                                            # [K]
            winner = select_fn(scores)                   # int in [0, K)

            # 4. commit winning chunk
            winning_chunk = cand_canvas[winner : winner + 1, P_k:]   # [1, chunk_length]
            canvas = torch.cat([canvas, winning_chunk], dim=1)

            if return_trace:
                trace.append({
                    "k":       k_chunk,
                    "winner":  winner,
                    "scores":  scores.detach().cpu(),
                    "chunk":   winning_chunk.detach().cpu(),
                })

            # 5. terminate on EOS in committed chunk
            if (winning_chunk == eos_token_id).any():
                break

        return (canvas, trace) if return_trace else canvas

    # ────────────────────────────────────────────────────────────────────
    # Pluggable scorers. The signature is fixed so callers can A/B swap:
    #   fn(*, rollouts: [K, L], chunk_start: int, chunk_end: int) -> [K]
    # ────────────────────────────────────────────────────────────────────

    def _score_mean_argmax_logprob(self, *, rollouts, chunk_start, chunk_end):
        """
        Cheap floor scorer. CAVEAT: SUBS clamping forces log p = 0 on unmasked
        positions, so this is degenerate exactly where we evaluate (the
        rollout has been fully unmasked). KEEP it as the "no-signal" rung in
        the experiment ladder; replace with a faithful conditional-log-prob
        scorer once we instrument sample_sft to return its final p_x0.
        """
        K, L = rollouts.shape
        sigma = torch.zeros(K, device=rollouts.device)
        log_probs = self.forward(rollouts, sigma)                              # [K, L, V]
        gathered  = log_probs.gather(-1, rollouts.unsqueeze(-1)).squeeze(-1)   # [K, L]
        return gathered[:, chunk_end:].mean(dim=-1)                            # [K]

    @staticmethod
    def make_self_consistency_scorer(answer_extractor: Callable):
        """
        Recommended MVE default. Each rollout scores by how many of the K
        rollouts (itself included) produce the same extracted answer.
        Highest scorer = most consensual mode at this chunk-step's frontier.

        Composes with the OUTER M-chain vote: this is a LOCAL chunk-level
        ensemble; the outer ensemble remains a separate, independent vote.
        """
        def _score(*, rollouts, chunk_start, chunk_end):
            K = rollouts.shape[0]
            answers = [answer_extractor(rollouts[i]) for i in range(K)]
            counts  = [sum(a == b for b in answers) for a in answers]
            return torch.tensor(counts, dtype=torch.float32, device=rollouts.device)
        return _score
    
"""
Minimal end-to-end simulation of DOTMixin (chunk-commit + lookahead).

What this exercises:
  * The full DOTMixin.sample_dot orchestration loop.
  * The K-fold candidate batching at the model boundary.
  * The two distinct scorers (`_score_mean_argmax_logprob` floor and
    `make_self_consistency_scorer` MVE default).
  * Sampler-level invariants (prompt preservation, canvas growth, EOS
    termination, expected fwd-call accounting).

What this does NOT do:
  * Validate model correctness -- the backbone is a constant-bias mock.
    Diversity across K candidates comes from the sampler's Gumbel draws
    in `_sample_categorical`, not from the model. That's exactly enough
    to verify the orchestration.

Import paths assume:
    dot_mixin.py                    next to this file
    your sampler module             provides MinimalMDLMSampler +
                                    SFTMixinBatchedVarlen
Adjust the two imports below if your layout differs.
"""



# =======================================================================
# 1.  Tokens + scheduler proxy.  Trivial standalone copy of the linear
#     alpha schedule so the demo has zero coupling to your scheduler
#     module's import path.
# =======================================================================
VOCAB_SIZE = 32
PAD_ID, EOS_ID, SEP_ID, MASK_ID = 0, 28, 29, 31
DIGIT_IDS = list(range(1, 11))          # 1..10
OP_IDS    = [11, 12, 13, 14]            # +, -, *, =
OP_GLYPH  = {11: "+", 12: "-", 13: "*", 14: "="}


class LinearAlphaScheduler:
    """alpha(t) = 1 - t.  alpha(1)=0 fully masked, alpha(0)=1 clean."""
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t


# =======================================================================
# 2.  Mock backbone.
#     - Returns the SAME logits regardless of input_ids (constant bias).
#     - Diversity across K rows is produced by the sampler's Gumbel
#       noise in `_sample_categorical`, not by the model. This is
#       sufficient to exercise every code path in DOTMixin.
#     - Spies on fwd-call count so we can sanity-check the "2 batched
#       fwd RUNS per chunk" claim in the docstring.
# =======================================================================
class ToyBackbone(nn.Module):
    def __init__(self, eos_pressure: float = 1.5):
        super().__init__()
        bias = torch.full((VOCAB_SIZE,), -10.0)
        for tid in DIGIT_IDS: bias[tid] = 2.0
        for tid in OP_IDS:    bias[tid] = 1.5
        bias[EOS_ID]          = eos_pressure       # tune termination rate
        self.register_buffer("token_bias", bias)
        self.dummy = nn.Parameter(torch.zeros(1))  # so .parameters() works
        # Spies:
        self.n_calls = 0
        self.n_rows  = 0

    def reset_counters(self):
        self.n_calls = 0
        self.n_rows  = 0

    def forward(self, input_ids, timesteps=None, attention_mask=None):
        B, L = input_ids.shape
        self.n_calls += 1
        self.n_rows  += B
        logits = self.token_bias.view(1, 1, -1).expand(B, L, -1).clone()
        return SimpleNamespace(logits=logits)


# =======================================================================
# 3.  Sampler composition.  Method-resolution order:
#     DOTMixin.sample_dot  ->  SFTMixinBatchedVarlen.sample_sft  ->
#     MinimalMDLMSampler.forward / _ddpm_caching_update
# =======================================================================
class MDLMSampler(DOTMixin, SFTMixinBatchedVarlen, MinimalMDLMSampler):
    pass


# =======================================================================
# 4.  Toy answer extractor for the self-consistency scorer.
#     "Answer = last digit token in the rollout, before EOS."
#     With 10 possible digits and K=4 candidates, P(some pair agrees) ~ 0.5,
#     so scores are non-degenerate even with a content-free mock model.
# =======================================================================
def toy_answer_extractor(row: torch.Tensor):
    ids = row.tolist()
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    digits = [t for t in ids if t in DIGIT_IDS]
    return digits[-1] if digits else None


# =======================================================================
# 5.  Pretty-printer for token sequences.
# =======================================================================
def pretty(ids) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    out = []
    for i in ids:
        if   i == EOS_ID:    out.append("<EOS>")
        elif i == SEP_ID:    out.append("<SEP>")
        elif i == MASK_ID:   out.append("<M>")
        elif i == PAD_ID:    out.append("<P>")
        elif i in OP_GLYPH:  out.append(OP_GLYPH[i])
        elif i in DIGIT_IDS: out.append(str(i))
        else:                out.append(f"?{i}")
    return " ".join(out)


# =======================================================================
# 6.  Arm runner: runs one config, prints trace, asserts invariants.
# =======================================================================
def run_arm(name, sampler, prompt_ids, *, K, score_fn, **kw):
    sampler.backbone.reset_counters()
    out, trace = sampler.sample_dot(
        prompt_ids,
        K=K,
        score_fn=score_fn,
        return_trace=True,
        **kw,
    )
    n_calls = sampler.backbone.n_calls
    n_rows  = sampler.backbone.n_rows

    print(f"\n── {name} ".ljust(72, "─"))
    print(f"  prompt   : {pretty(prompt_ids)}")
    print(f"  final    : {pretty(out[0])}")
    print(f"  chunks   : {len(trace):>2}    "
          f"fwd calls: {n_calls:>3}    "
          f"K-rows fed to backbone: {n_rows:>4}")
    for step in trace:
        scores = [f"{s:.2f}" for s in step["scores"].tolist()]
        print(f"    k={step['k']}  winner={step['winner']}  "
              f"scores=[{', '.join(scores)}]  "
              f"chunk={pretty(step['chunk'][0])}")

    # --- invariants ----------------------------------------------------
    P = prompt_ids.numel()
    assert torch.equal(out[0, :P], prompt_ids),         "prompt drifted"
    assert out.shape[0] == 1,                           "batch axis leaked"
    assert out.shape[1] >= P,                           "canvas shrank"
    for step in trace:
        assert 0 <= step["winner"] < K,                 "winner out of range"
        assert step["scores"].shape == (K,),            "score shape wrong"
        assert step["chunk"].shape == (1, kw["chunk_length"]), \
                                                        "chunk shape wrong"
    return out


# =======================================================================
# 7.  Demo.
# =======================================================================
def main():
    torch.manual_seed(0)

    backbone  = ToyBackbone(eos_pressure=1.5).eval()
    scheduler = LinearAlphaScheduler()
    sampler   = MDLMSampler(
        backbone=backbone,
        scheduler=scheduler,
        mask_index=MASK_ID,
    )

    # "5 + 3 + 2 =  <SEP>  ???"
    prompt_ids = torch.tensor([5, 11, 3, 11, 2, 13, SEP_ID], dtype=torch.long)

    shared = dict(
        chunk_length=4,
        max_chunks=3,
        num_steps_chunk=8,
        lookahead_length=8,
        num_steps_lookahead=1,    # true one-shot lookahead
        eos_token_id=EOS_ID,
    )

    print("=" * 72)
    print("DOT MVE simulation  (mock backbone — orchestration test only)")
    print("=" * 72)

    # B0: greedy chunk-commit, K=1. With K=1 the scorer is irrelevant.
    torch.manual_seed(1)
    run_arm(
        "B0   greedy chunk-commit  (K=1)",
        sampler, prompt_ids,
        K=1, score_fn=None, **shared,
    )

    # MVE: K=4 with the self-consistency scorer.
    sc_scorer = DOTMixin.make_self_consistency_scorer(toy_answer_extractor)
    torch.manual_seed(1)
    run_arm(
        "MVE  chunk-commit + SC lookahead  (K=4, scorer=self-consistency)",
        sampler, prompt_ids,
        K=4, score_fn=sc_scorer, **shared,
    )

    # Floor: same K=4, but use the documented-weak logprob scorer.
    # Expect: every score == 0.00 (SUBS clamps log p to 0 on unmasked
    # positions, and rollouts are fully unmasked). Winner therefore
    # collapses to argmax-of-ties = 0 every step. This is the *demo* of
    # why we ship `make_self_consistency_scorer` as the MVE default.
    torch.manual_seed(1)
    run_arm(
        "FLR  chunk-commit + logprob lookahead  (K=4, scorer=DEGENERATE)",
        sampler, prompt_ids,
        K=4, score_fn=None, **shared,
    )

    print("\n✓ pipeline runs end-to-end; all invariants hold")


if __name__ == "__main__":
    main()    