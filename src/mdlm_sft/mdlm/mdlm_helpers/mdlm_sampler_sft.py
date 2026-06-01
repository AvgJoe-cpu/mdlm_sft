from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Union

import torch


def _sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@dataclass
class MDLMSamplerConfig:
    response_length: int = 64
    num_steps: int = 64
    eps: float = 1e-5
    noise_removal: bool = True
    early_exit: bool = True


class MinimalMDLMSampler:
    def __init__(
        self,
        backbone,
        scheduler,
        mask_index,
        time_conditioning=False,
        neg_infinity=-1_000_000.0,
    ):
        self.backbone = backbone
        self.scheduler = scheduler
        self.mask_index = mask_index
        self.time_conditioning = time_conditioning
        self.neg_infinity = neg_infinity
        self._attention_mask = (
            None  # [VARLEN] set by sample_sft; None = attend everywhere
        )

    def forward(self, x, sigma):
        # [VARLEN] pick up the per-call mask if one was stashed; else attend everywhere
        attention_mask = self._attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(x, dtype=torch.long)
        logits = self.backbone(
            input_ids=x,
            timesteps=sigma,
            attention_mask=attention_mask,
        ).logits
        return self._subs_parameterization(logits, x)

    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = xt != self.mask_index
        logits[unmasked] = self.neg_infinity
        logits[unmasked, xt[unmasked]] = 0
        return logits

    # ── single reverse step ───────────────────────────────────────────────────
    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1

        move_chance_t = (1 - self.scheduler.alpha(t))[:, None, None]
        move_chance_s = (1 - self.scheduler.alpha(t - dt))[:, None, None]

        if p_x0 is None:
            sigma = torch.zeros(x.shape[0], device=x.device)
            p_x0 = self.forward(x, sigma).exp()  # log-probs → probs

        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    # ── outer loop ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, batch_size, seq_len, num_steps=10, eps=1e-5, noise_removal=True):
        device = next(self.backbone.parameters()).device

        # prior: all masks
        x = torch.full(
            (batch_size, seq_len), self.mask_index, dtype=torch.long, device=device
        )

        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(batch_size, 1, device=device)
            p_x0_cache, x_next = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
            if not torch.allclose(x_next, x):  # cache invalid when canvas changes
                p_x0_cache = None
            x = x_next

        if noise_removal:
            sigma = torch.zeros(batch_size, device=device)
            x = self.forward(x, sigma).argmax(dim=-1)

        return x


class SFTMixin:
    """
    Mix into MinimalMDLMSampler:

        class MDLMSampler(SFTMixin, MinimalMDLMSampler):
            pass

    or simply assign:

        MinimalMDLMSampler.sample_sft = SFTMixin.sample_sft
    """

    @torch.no_grad()
    def sample_sft(
        self,
        prompt_ids: torch.LongTensor,  # shape [P] or [1, P]
        response_length: int,
        num_steps: int = 512,
        eps: float = 1e-5,
        noise_removal: bool = True,
    ) -> torch.LongTensor:
        """
        Returns a tensor of shape [1, P + response_length] where the first P
        tokens are bit-identical to `prompt_ids` and the remaining tokens are
        sampled from the diffusion reverse process.
        """
        device = next(self.backbone.parameters()).device

        # --- normalize prompt shape -----------------------------------------
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        assert prompt_ids.ndim == 2 and prompt_ids.shape[0] == 1, (
            "sample_sft handles one prompt per call; got "
            f"shape {tuple(prompt_ids.shape)}"
        )
        prompt_ids = prompt_ids.to(device=device, dtype=torch.long)

        # Guard the SUBS-carry-over invariant: a prompt token that happens to
        # equal mask_index would be (mis)treated as a noised slot.
        assert (prompt_ids != self.mask_index).all(), (
            "prompt contains mask_index; SUBS clamping would treat those "
            "positions as noised."
        )

        P = prompt_ids.shape[1]
        L = P + response_length

        # --- initial canvas: [prompt | MASK ... MASK] -----------------------
        response_init = torch.full(
            (1, response_length),
            self.mask_index,
            dtype=torch.long,
            device=device,
        )
        x = torch.cat([prompt_ids, response_init], dim=1)  # [1, L]

        # --- reverse loop (identical structure to .sample) ------------------
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(1, 1, device=device)
            p_x0_cache, x_next = self._ddpm_caching_update(
                x,
                t,
                dt,
                p_x0=p_x0_cache,
            )
            if not torch.equal(x_next, x):
                p_x0_cache = None
            x = x_next

        if noise_removal:
            sigma = torch.zeros(1, device=device)
            x = self.forward(x, sigma).argmax(dim=-1)

        return x


# ----------------------------------------------------------------------------
# Portions adapted from kuleshov-group/mdlm and kuleshov-group/bd3lms,
# both licensed under the Apache License, Version 2.0.
#   - MDLM:   https://github.com/kuleshov-group/mdlm        (Copyright 2024 Cornell University)
#   - BD3LMs: https://github.com/kuleshov-group/bd3lms
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# Specifically borrowed:
#   * the "invalidate cache also when time_conditioning=True" rule from
#     mdlm/diffusion.py::Diffusion._sample (commit c112c52, lines ~682-685)
#   * the "block fully un-masked -> stop forwarding" early-exit idea from
#     bd3lms-family _ddpm_caching_update_ (MBD3LM fork, lines ~1030-1031)
# ----------------------------------------------------------------------------


class SFTMixinBatched:
    """
    Mix into MinimalMDLMSampler:

        class MDLMSampler(SFTMixinBatched, MinimalMDLMSampler):
            pass

    or simply assign:

        MinimalMDLMSampler.sample_sft = SFTMixinBatched.sample_sft
    """

    @torch.no_grad()
    def sample_sft(
        self,
        prompt_ids: torch.LongTensor,  # [P] or [B, P]
        response_length: int,
        num_steps: int = 512,
        eps: float = 1e-5,
        noise_removal: bool = True,
        early_exit: bool = True,  # NEW: BD3LM-style done-check
        return_nfes: bool = False,  # NEW: report NFEs used (cache-hits omitted)
    ) -> Union[torch.LongTensor, Tuple[torch.LongTensor, int]]:
        """
        Batched SFT-style sampling with cache logic borrowed from
        kuleshov-group/mdlm and kuleshov-group/bd3lms (Apache-2.0).

        Accepts:
            prompt_ids of shape [P]    -> treated as B=1
            prompt_ids of shape [B, P] -> batched; all prompts share length P

        Returns (default):  LongTensor [B, P + response_length]
        Returns if return_nfes=True:  (samples, nfes_used)

        Cache logic (vs. the previous version):
          1. Cache invalidation also fires when self.time_conditioning is True,
             matching the upstream MDLM outer-loop rule.
          2. Once every row's RESPONSE region has no MASK tokens left, we
             early-exit the reverse loop -- further calls would be no-ops
             since _ddpm_caching_update gates _x with copy_flag.
          3. Optional prompt-prefix KV warmup via self._prompt_kv_cache_fn,
             called once before the loop. No-op when the hook is absent;
             this is the seam through which BD3LM-style block-KV caching
             can be plugged in by a future backbone without further sampler
             changes.
        """
        device = next(self.backbone.parameters()).device

        # --- normalize prompt shape -----------------------------------------
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        assert prompt_ids.ndim == 2, (
            f"prompt_ids must be 1-D [P] or 2-D [B, P]; got shape "
            f"{tuple(prompt_ids.shape)}"
        )
        prompt_ids = prompt_ids.to(device=device, dtype=torch.long)

        B, P = prompt_ids.shape
        L = P + response_length

        # SUBS carry-over guard -- a prompt token equal to mask_index would
        # be treated by SUBS as a noised slot.
        assert (prompt_ids != self.mask_index).all(), (
            "prompt batch contains mask_index in at least one row; SUBS "
            "clamping would treat those positions as noised."
        )

        # --- initial canvas [B, L] : [prompt | MASK ... MASK] ---------------
        response_init = torch.full(
            (B, response_length),
            self.mask_index,
            dtype=torch.long,
            device=device,
        )
        x = torch.cat([prompt_ids, response_init], dim=1)

        # --- optional: prompt-KV warmup (BD3LM-style hook) ------------------
        # If the user wires a callable in (e.g. one that calls
        # backbone(prompt_ids, ..., store_kv=True)), we invoke it once. The
        # parent forward() must then know how to consume that state; this
        # mixin does not assume any particular contract.
        kv_warmup: Optional[Callable] = getattr(self, "_prompt_kv_cache_fn", None)
        if kv_warmup is not None:
            kv_warmup(prompt_ids)

        # --- reverse loop ---------------------------------------------------
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        # Matches MDLM outer-loop semantics (default False on this sampler).
        time_conditioning = bool(getattr(self, "time_conditioning", False))

        nfes = 0  # count actual forward calls (a cache hit is NOT an NFE)

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(B, 1, device=device)
            cache_was_hit = p_x0_cache is not None
            p_x0_cache, x_next = self._ddpm_caching_update(
                x,
                t,
                dt,
                p_x0=p_x0_cache,
            )
            if not cache_was_hit:
                nfes += 1

            # Borrowed from MDLM upstream (Apache-2.0):
            #     if (not torch.allclose(x_next, x) or self.time_conditioning):
            #         p_x0_cache = None
            # We use torch.equal because x is integer-typed; semantically
            # identical here, and avoids the float-tolerance footgun.
            if (not torch.equal(x_next, x)) or time_conditioning:
                p_x0_cache = None
            x = x_next

            # Borrowed from BD3LM family (Apache-2.0):
            # "all tokens in the active block are sampled" -> stop forwarding.
            # In SFT the active block is the response region x[:, P:].
            if early_exit and (x[:, P:] != self.mask_index).all():
                break

        if noise_removal:
            sigma = torch.zeros(B, device=device)
            x = self.forward(x, sigma).argmax(dim=-1)
            nfes += 1

        return (x, nfes) if return_nfes else x


class SFTMixinBatchedVarlen:
    @torch.no_grad()
    def sample_sft(
        self,
        # [VARLEN] prompt_ids may now be:
        #   - 1-D LongTensor [P]                              (B=1, old)
        #   - 2-D LongTensor [B, P_max]   (+ prompt_lens)     (rectangular, padded)
        #   - list/tuple of 1-D LongTensors with possibly different lengths
        prompt_ids,  # type relaxed: see above
        response_length: int,
        num_steps: int = 512,
        eps: float = 1e-5,
        noise_removal: bool = True,
        early_exit: bool = True,
        return_nfes: bool = False,
        prompt_lens: Optional[torch.LongTensor] = None,  # [VARLEN] NEW
        pad_token_id: Optional[int] = None,  # [VARLEN] NEW (required iff lengths vary)
    ) -> Union[torch.LongTensor, Tuple[torch.LongTensor, int]]:
        """
        Batched SFT-style sampling with variable-length prompts.

        Layout of the internal canvas, shape [B, P_max + R]:

            row b :  [ prompt_b  | MASK ... MASK | PAD ... PAD ]
                       length P_b      length R      length P_max - P_b

        Per-row response is x[b, P_b : P_b + R].  Prompt and trailing-pad
        columns are non-mask, so _ddpm_caching_update's copy_flag freezes
        them automatically.  Only response slots ever contain mask_index,
        so the global predicate (x == mask_index).any() == False is the
        correct multi-row done check.

        Inputs
        ------
        prompt_ids : 1-D Tensor [P], 2-D Tensor [B, P_max], or list of 1-D Tensors.
        prompt_lens : optional [B] Long; required to disambiguate a 2-D input
            whose rows are already padded.  If omitted on a 2-D input, all
            rows are assumed to have length P_max (back-compat fast path).
        pad_token_id : int, required iff the effective prompt lengths vary.
            MUST be != mask_index and SHOULD be a token your backbone is
            comfortable seeing in unattended positions (see caveat below).
        """
        device = next(self.backbone.parameters()).device
        MASK = self.mask_index

        # ================================================================
        # [VARLEN] -- normalize input into (prompt_ids_2d [B, P_max], prompt_lens [B])
        # ================================================================
        if isinstance(prompt_ids, (list, tuple)):
            # list of 1-D tensors, possibly ragged
            rows = [p.to(device=device, dtype=torch.long).flatten() for p in prompt_ids]
            B = len(rows)
            assert B > 0, "empty prompt list"
            prompt_lens = torch.tensor(
                [r.numel() for r in rows], dtype=torch.long, device=device
            )
            P_max = int(prompt_lens.max().item())
            # right-pad with mask_index temporarily; we'll overwrite these
            # positions below when we build the canvas.
            prompt_ids_2d = torch.full(
                (B, P_max), MASK, dtype=torch.long, device=device
            )
            for b, r in enumerate(rows):
                prompt_ids_2d[b, : r.numel()] = r
        else:
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            assert prompt_ids.ndim == 2, (
                f"prompt_ids must be 1-D [P], 2-D [B, P_max], or a list; got "
                f"shape {tuple(prompt_ids.shape)}"
            )
            prompt_ids_2d = prompt_ids.to(device=device, dtype=torch.long)
            B, P_max = prompt_ids_2d.shape
            if prompt_lens is None:
                # back-compat: assume rectangular -> every row has length P_max
                prompt_lens = torch.full((B,), P_max, dtype=torch.long, device=device)
            else:
                prompt_lens = prompt_lens.to(device=device, dtype=torch.long)
                assert prompt_lens.shape == (
                    B,
                ), f"prompt_lens must be shape [{B}]; got {tuple(prompt_lens.shape)}"
                assert (prompt_lens >= 0).all() and (
                    prompt_lens <= P_max
                ).all(), "prompt_lens out of range"

        # ================================================================
        # [VARLEN] -- decide whether we need a pad token
        # ================================================================
        varlen = bool((prompt_lens != prompt_lens[0]).any().item())
        if varlen:
            assert pad_token_id is not None, (
                "prompts have different lengths; pad_token_id is required so "
                "trailing-pad columns can be frozen as non-mask."
            )
            assert pad_token_id != MASK, (
                "pad_token_id must differ from mask_index, otherwise SUBS "
                "would treat pad slots as response."
            )

        R = response_length
        L = P_max + R

        # ================================================================
        # [VARLEN] -- per-row prompt-region mask, used by the SUBS guard
        # and to place the response slots correctly.
        # is_prompt[b, j] == True  iff  j < prompt_lens[b]
        # ================================================================
        col = torch.arange(L, device=device)[None, :]  # [1, L]
        is_prompt = col < prompt_lens[:, None]  # [B, L]
        # response window: [P_b, P_b + R)
        is_response = (col >= prompt_lens[:, None]) & (
            col < prompt_lens[:, None] + R
        )  # [B, L]
        is_pad = ~(is_prompt | is_response)  # [B, L]

        # ================================================================
        # [VARLEN] -- relaxed SUBS guard: only check ACTUAL prompt
        # positions for mask_index (per row), not padded columns.
        # ================================================================
        if varlen:
            real_prompt_vals = prompt_ids_2d[is_prompt[:, :P_max]]
        else:
            real_prompt_vals = prompt_ids_2d
        assert (real_prompt_vals != MASK).all(), (
            "prompt batch contains mask_index in at least one real prompt "
            "position; SUBS clamping would treat those positions as noised."
        )

        # ================================================================
        # [VARLEN] -- build the canvas [B, L].
        # Start everything as mask_index, then overwrite prompt + pad cols.
        # Net effect: ONLY response slots hold mask_index.
        # ================================================================
        x = torch.full((B, L), MASK, dtype=torch.long, device=device)
        # place prompts in their per-row slots
        prompt_cols = is_prompt[:, :P_max]  # [B, P_max]
        x[:, :P_max][prompt_cols] = prompt_ids_2d[prompt_cols]
        # place pad sentinel in trailing columns
        if varlen:
            x[is_pad] = pad_token_id

        # Invariant: every mask_index in x is a response slot. Cheap check.
        assert torch.equal((x == MASK), is_response), (
            "internal canvas invariant violated (mask_index appears outside "
            "response window)"
        )

        # ================================================================
        # [VARLEN] -- optional prompt-KV warmup hook is unchanged in spirit
        # but now receives the rectangular [B, P_max] tensor; backends that
        # care about ragged prefixes can read prompt_lens off self.
        # ================================================================
        self._last_prompt_lens = prompt_lens

        # ================================================================
        # [VARLEN] -- build the per-call attention mask and stash it on
        # self so MinimalMDLMSampler.forward (and therefore
        # _ddpm_caching_update) picks it up automatically. try/finally
        # guarantees we restore prior state on any exit path.
        # ================================================================
        attn_mask = (~is_pad).to(torch.long)  # [B, L]; 1=attend, 0=pad
        prev_attention_mask = self._attention_mask  # save (may be None)
        self._attention_mask = attn_mask
        try:
            # -------- prompt-KV warmup (may call self.forward internally)
            kv_warmup: Optional[Callable] = getattr(self, "_prompt_kv_cache_fn", None)
            if kv_warmup is not None:
                kv_warmup(prompt_ids_2d)

            # -------- reverse loop ------------------------------------------------
            timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
            dt = (1 - eps) / num_steps
            p_x0_cache = None
            time_conditioning = bool(getattr(self, "time_conditioning", False))
            nfes = 0

            for i in range(num_steps):
                t = timesteps[i] * torch.ones(B, 1, device=device)
                cache_was_hit = p_x0_cache is not None
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x,
                    t,
                    dt,
                    p_x0=p_x0_cache,
                )
                if not cache_was_hit:
                    nfes += 1

                if (not torch.equal(x_next, x)) or time_conditioning:
                    p_x0_cache = None
                x = x_next

                # [VARLEN] -- per-row done check (unchanged: invariant guarantees
                # masks only exist in response slots).
                if early_exit and not (x == MASK).any():
                    break

            # -------- noise removal -------------------------------------------
            if noise_removal:
                sigma = torch.zeros(B, device=device)
                # [VARLEN] argmax over the whole canvas is safe: SUBS clamps all
                # non-mask positions (prompt AND pad) to their current id, so
                # argmax can only change response slots.
                x = self.forward(x, sigma).argmax(dim=-1)
                nfes += 1

            return (x, nfes) if return_nfes else x
        finally:
            # [VARLEN] restore previous mask (None by default) so that other
            # entry points -- e.g. MinimalMDLMSampler.sample() -- see the
            # pristine "attend everywhere" default again.
            self._attention_mask = prev_attention_mask


# ============================================================================
# Tests
# ============================================================================
def _build_varlen_sampler(vocab_size=64, mask_index=63):
    import torch.nn as nn

    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 32)
            self.proj = nn.Linear(32, vocab_size)
            # spy slots — populated by .forward when called
            self.last_attention_mask = None
            self.last_input_ids = None

        def forward(self, input_ids, timesteps=None, attention_mask=None):
            self.last_attention_mask = attention_mask
            self.last_input_ids = input_ids
            h = self.emb(input_ids)
            return SimpleNamespace(logits=self.proj(h))

    sampler = MinimalMDLMSampler(  # noqa: F821
        backbone=ToyBackbone().eval(),
        scheduler=LinearAlphaScheduler(),  # noqa: F821
        mask_index=mask_index,
    )
    sampler.sample_sft = SFTMixinBatchedVarlen.sample_sft.__get__(
        sampler, type(sampler)
    )
    return sampler, vocab_size, mask_index


def _check_row(out_row, prompt, R, MASK, PAD, P_max):
    """One row-level invariant bundle, reused everywhere."""
    Pb = prompt.numel()
    assert torch.equal(out_row[:Pb], prompt), "prompt drift"
    assert (out_row[Pb : Pb + R] != MASK).all(), "MASK left in response"
    if Pb + R < P_max + R:  # trailing pad region
        assert (out_row[Pb + R :] == PAD).all(), "pad clobbered"


# ===========================================================================
# Run
# ===========================================================================
def run_varlen_tests():
    torch.manual_seed(0)
    sampler, V, MASK = _build_varlen_sampler()
    PAD = 0
    R = 8

    # -----------------------------------------------------------------------
    # 1. Equal-length back-compat: rectangular [B, P], no prompt_lens, no pad.
    #    Must NOT require pad_token_id (varlen=False short-circuit).
    # -----------------------------------------------------------------------
    eq = torch.tensor([[1, 5, 9, 12, 7, 3], [2, 4, 8, 11, 6, 9]], dtype=torch.long)
    out_eq = sampler.sample_sft(eq, response_length=R, num_steps=32)
    B_eq, P_eq = eq.shape
    assert out_eq.shape == (B_eq, P_eq + R)
    for b in range(B_eq):
        _check_row(out_eq[b], eq[b], R, MASK, PAD=PAD, P_max=P_eq)
    # no attention mask should leak across the call boundary
    assert sampler._attention_mask is None

    # -----------------------------------------------------------------------
    # 2. List API + 2-D+lens API: equivalence under same seed.
    # -----------------------------------------------------------------------
    ragged = [
        torch.tensor([1, 5, 9, 12, 7, 3], dtype=torch.long),
        torch.tensor([2, 4, 8, 11], dtype=torch.long),
        torch.tensor([7], dtype=torch.long),
    ]
    B = len(ragged)
    P_max = max(p.numel() for p in ragged)
    lens = torch.tensor([p.numel() for p in ragged], dtype=torch.long)
    padded = torch.full((B, P_max), PAD, dtype=torch.long)
    for b, p in enumerate(ragged):
        padded[b, : p.numel()] = p

    torch.manual_seed(0)
    out_list = sampler.sample_sft(
        ragged, response_length=R, num_steps=64, pad_token_id=PAD
    )
    torch.manual_seed(0)
    out_2d = sampler.sample_sft(
        padded, response_length=R, num_steps=64, prompt_lens=lens, pad_token_id=PAD
    )
    assert torch.equal(out_list, out_2d), "list vs (2-D+lens) APIs disagree"
    assert out_list.shape == (B, P_max + R)
    for b, p in enumerate(ragged):
        _check_row(out_list[b], p, R, MASK, PAD, P_max)

    # -----------------------------------------------------------------------
    # 3. Missing pad_token_id under varlen must be rejected.
    # -----------------------------------------------------------------------
    try:
        sampler.sample_sft(ragged, response_length=R, num_steps=4)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing pad_token_id should have raised")

    # -----------------------------------------------------------------------
    # 4. pad_token_id == mask_index must be rejected.
    # -----------------------------------------------------------------------
    try:
        sampler.sample_sft(ragged, response_length=R, num_steps=4, pad_token_id=MASK)
    except AssertionError:
        pass
    else:
        raise AssertionError("pad==MASK should have raised")

    # -----------------------------------------------------------------------
    # 5. Relaxed SUBS guard: mask_index in a REAL prompt slot must fail,
    #    but having mask_index appear *inside the padded trailing area* of
    #    a 2-D input is irrelevant because we never look there.
    # -----------------------------------------------------------------------
    bad = padded.clone()
    bad[0, 2] = MASK  # real prompt slot of row 0
    try:
        sampler.sample_sft(
            bad, response_length=R, num_steps=4, prompt_lens=lens, pad_token_id=PAD
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("MASK in real prompt slot must raise")

    # mask_index *outside* row 1's real prompt (within its padding) is OK:
    sneaky = padded.clone()
    sneaky[1, lens[1] :] = MASK  # all pad cols of row 1
    # don't even need a fresh seed — just confirm it doesn't raise:
    sampler.sample_sft(
        sneaky, response_length=R, num_steps=4, prompt_lens=lens, pad_token_id=PAD
    )

    # -----------------------------------------------------------------------
    # 6. Attention mask wiring: stash + shape + values + restoration.
    #    We spy on the backbone, which now records the mask it received.
    # -----------------------------------------------------------------------
    assert sampler._attention_mask is None
    out = sampler.sample_sft(
        ragged, response_length=R, num_steps=2, pad_token_id=PAD, noise_removal=False
    )
    m = sampler.backbone.last_attention_mask
    assert m is not None, "backbone never received an attention_mask"
    assert m.shape == (B, P_max + R), m.shape
    # per row: 1 across [0, Pb+R), 0 across [Pb+R, P_max+R)
    for b, p in enumerate(ragged):
        Pb = p.numel()
        assert (m[b, : Pb + R] == 1).all(), f"row {b}: attended region wrong"
        assert (m[b, Pb + R :] == 0).all(), f"row {b}: pad region not masked out"
    assert sampler._attention_mask is None, "mask leaked after normal return"

    # 6b. Mask must be restored on EXCEPTION too.
    sampler._attention_mask = "SENTINEL"  # pretend an outer caller had set it
    try:
        sampler.sample_sft(
            ragged, response_length=R, num_steps=4
        )  # missing pad -> raises
    except AssertionError:
        pass
    assert (
        sampler._attention_mask == "SENTINEL"
    ), "exception path failed to restore prior _attention_mask"
    sampler._attention_mask = None  # clean up

    # 6c. Equal-length call must STILL set a mask (all-ones) so the backbone
    #     sees a consistent API. We allow either "all-ones" or "None"; we just
    #     pin the current behavior so a future refactor is intentional.
    sampler.sample_sft(eq, response_length=R, num_steps=2, noise_removal=False)
    m_eq = sampler.backbone.last_attention_mask
    assert m_eq is not None and m_eq.shape == (B_eq, P_eq + R)
    assert (m_eq == 1).all(), "equal-length call should mask nothing"

    # -----------------------------------------------------------------------
    # 7. Pad-respecting freezing: with noise_removal=False AND only a few
    #    steps, the trailing pad columns must STILL be exactly PAD because
    #    _ddpm_caching_update freezes any non-mask token.
    # -----------------------------------------------------------------------
    out_short = sampler.sample_sft(
        ragged,
        response_length=R,
        num_steps=3,
        pad_token_id=PAD,
        noise_removal=False,
    )
    for b, p in enumerate(ragged):
        Pb = p.numel()
        if Pb + R < P_max + R:
            assert (
                out_short[b, Pb + R :] == PAD
            ).all(), f"row {b}: pad column drifted without noise_removal"
        # prompt must also still be intact
        assert torch.equal(out_short[b, :Pb], p)

    # -----------------------------------------------------------------------
    # 8. Early-exit: under varlen the predicate is `(x == MASK).any() == False`.
    #    Confirm it (a) fires, (b) does not change the final sample vs.
    #    early_exit=False at the same seed, (c) yields fewer NFEs.
    # -----------------------------------------------------------------------
    torch.manual_seed(0)
    a, n_on = sampler.sample_sft(
        ragged,
        response_length=R,
        num_steps=512,
        pad_token_id=PAD,
        early_exit=True,
        return_nfes=True,
    )
    torch.manual_seed(0)
    b_, n_off = sampler.sample_sft(
        ragged,
        response_length=R,
        num_steps=512,
        pad_token_id=PAD,
        early_exit=False,
        return_nfes=True,
    )
    assert torch.equal(a, b_), "early_exit changed the varlen sample"
    assert n_on <= n_off and n_on < 512, (n_on, n_off)
    print(f"  [info] varlen NFEs early_exit on/off: {n_on}/{n_off}")

    # -----------------------------------------------------------------------
    # 9. Single-row varlen edge case: B=1 list of one tensor must behave
    #    identically to passing the 1-D tensor directly under no-varlen.
    # -----------------------------------------------------------------------
    p1 = torch.tensor([3, 1, 4, 1, 5, 9], dtype=torch.long)
    torch.manual_seed(0)
    out_1d = sampler.sample_sft(p1, response_length=R, num_steps=16)
    torch.manual_seed(0)
    out_l1 = sampler.sample_sft([p1], response_length=R, num_steps=16, pad_token_id=PAD)
    assert torch.equal(out_1d, out_l1), "B=1 list path diverges from 1-D path"

    # -----------------------------------------------------------------------
    # 10. Empty-prompt row: P_b == 0 is legal (response_length tokens of
    #     unconditional generation in row b). The relaxed SUBS guard must
    #     accept it; the canvas should still satisfy the mask invariant.
    # -----------------------------------------------------------------------
    with_empty = [
        torch.tensor([7, 8, 9], dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    ]
    out_e = sampler.sample_sft(
        with_empty, response_length=R, num_steps=8, pad_token_id=PAD
    )
    assert out_e.shape == (2, max(3, 0) + R)
    assert torch.equal(out_e[0, :3], with_empty[0])
    assert (out_e[0, 3 : 3 + R] != MASK).all()
    assert (out_e[1, :R] != MASK).all()  # row 1 is all response

    print("all VARLEN SFT sampler tests OK")


