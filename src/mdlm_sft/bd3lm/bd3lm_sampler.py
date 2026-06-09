import copy
import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


# ======================================================================================
# Module-level kernel (the irreducible BD3LM ops, shared by every tier)
# ======================================================================================
def _prepare_for_sampling(
    x: torch.Tensor,
    block_size: int,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-causal bidirectional attention mask + RoPE position_ids.

    Blocks are defined in PHYSICAL coordinates (pos // block_size), shared across
    the batch. A query at block q attends to every key in blocks <= q (causal at
    the block level, bidirectional within a block).

    pad_token_id:
      - None  -> no padding; every position is a valid query/key.
      - int   -> positions equal to it are excluded as both query and key, and
                 skipped by the RoPE position counter (so the prefix's logical
                 positions stay contiguous regardless of left-padding).

    Returns:
        attn_mask    : [B, 1, T, T] bool  (True = attend)
        position_ids : [B, T]       long
    """
    B, T = x.shape
    device = x.device

    valid = torch.ones_like(x, dtype=torch.bool) if pad_token_id is None else (x != pad_token_id)

    # logical (pad-skipping) positions for RoPE
    pos_raw = torch.cumsum(valid.to(torch.long), dim=-1)          # 1-based count of valid tokens
    logical_pos = pos_raw - 1                                     # 0-based
    position_ids = torch.where(valid, logical_pos, torch.zeros_like(logical_pos)).to(
        device=device, dtype=torch.long
    )

    # physical block ids; padding -> -1 ("no block")
    pos = torch.arange(T, device=device)
    block_ids = torch.div(pos, block_size, rounding_mode="floor").view(1, T).expand(B, -1)
    block_ids = torch.where(valid, block_ids, torch.full_like(block_ids, -1))

    bid_q = block_ids.view(B, 1, T, 1)
    bid_k = block_ids.view(B, 1, 1, T)
    attn_mask = (bid_k <= bid_q) & (bid_q >= 0) & (bid_k >= 0)
    return attn_mask, position_ids


def _diffusion_step_block(
    logits: torch.Tensor,           # [B, L, V]
    x_block: torch.Tensor,          # [B, L]
    mask_block: torch.Tensor,       # [B, L] bool
    num_transfer_step: torch.Tensor,  # [B]
    temperature: float,
    remasking: str,
) -> torch.Tensor:
    """One confidence-transfer diffusion step over a single block slice."""
    B, L, _ = logits.shape
    device = logits.device
    if not mask_block.any():
        return x_block

    x0 = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)  # [B, L]

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    elif remasking == "random":
        x0_p = torch.rand((B, L), device=device)
    else:
        raise NotImplementedError(remasking)

    # only masked positions may change; rank candidates by confidence
    x0 = torch.where(mask_block, x0, x_block)
    confidence = torch.where(mask_block, x0_p, torch.full_like(x0_p, -float("inf")))

    transfer = torch.zeros_like(x0, dtype=torch.bool)
    for j in range(B):
        k = int(num_transfer_step[j].item())
        if k <= 0:
            continue
        valid_count = int((confidence[j] > -float("inf")).sum().item())
        if valid_count == 0:
            continue
        k = min(k, valid_count)
        _, sel = torch.topk(confidence[j], k)
        transfer[j, sel] = True

    x_new = x_block.clone()
    x_new[transfer] = x0[transfer]
    return x_new


# ======================================================================================
# Tier 0 — Minimal: unconditional, full-recompute block denoiser
# ======================================================================================
class MinimalBD3LMSampler:
    def __init__(
        self,
        backbone,
        scheduler: Optional[BaseAlphaScheduler] = None,
        mask_index: int = None,
        block_size: int = 32,
        pad_index: Optional[int] = None,
    ):
        assert mask_index is not None, "mask_index is required"
        self.backbone = backbone
        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.mask_index = mask_index
        self.block_size = block_size
        self.pad_index = pad_index  # only needed once padding/prompts enter (mixins)

    # Single seam every tier funnels the model through. Returns the FULL output
    # object (BD3LM tier-3 needs .past_key_values; lower tiers use only .logits).
    def forward(self, x, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False):
        return self.backbone(
            input_ids=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_blocks: int,
        steps_per_block: Optional[int] = None,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        stochastic_transfer: bool = False,
        return_dict: bool = False,
    ):
        """Unconditional generation of `num_blocks` whole blocks. No prompt, no
        cache, no CFG: every step re-forwards the prefix-so-far and slices out the
        active block's logits."""
        device = next(self.backbone.parameters()).device
        bs = self.block_size
        steps_per_block = steps_per_block if steps_per_block is not None else bs
        T = num_blocks * bs

        x = torch.full((batch_size, T), self.mask_index, dtype=torch.long, device=device)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            start, stop = b * bs, (b + 1) * bs
            block_mask_index = x[:, start:stop] == self.mask_index
            num_transfer = get_num_transfer_tokens(
                block_mask_index, steps_per_block, self.scheduler, stochastic_transfer
            )
            for s in range(num_transfer.size(1)):
                x_block = x[:, start:stop]
                mask_block = x_block == self.mask_index
                if not mask_block.any():
                    break
                # full recompute over [0, stop); active block attends to all prior blocks
                attn, pos = _prepare_for_sampling(x[:, :stop], bs, self.pad_index)
                logits = self.forward(x[:, :stop], attention_mask=attn, position_ids=pos).logits
                x[:, start:stop] = _diffusion_step_block(
                    logits[:, start:stop], x_block, mask_block,
                    num_transfer[:, s], temperature, remasking,
                )
                if histories is not None:
                    histories.append(x.clone())

        return {"sequences": x, "histories": histories} if return_dict else x


# ======================================================================================
# Tier 1 — single-prompt SFT conditioning
# ======================================================================================
class SFTMixin:
    @torch.no_grad()
    def sample_sft(
        self,
        prompt_ids: torch.LongTensor,   # [P] or [1, P]
        response_length: int,
        steps_per_block: Optional[int] = None,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        stochastic_transfer: bool = False,
    ) -> torch.LongTensor:
        """Returns [1, P + response_length] with the prompt bit-identical at the
        front. The prompt is left-padded to a block multiple internally (clean
        whole response blocks); the pad is stripped on return."""
        device = next(self.backbone.parameters()).device
        bs = self.block_size
        steps_per_block = steps_per_block if steps_per_block is not None else bs
        assert self.pad_index is not None, "SFT requires pad_index for block alignment"

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        assert prompt_ids.ndim == 2 and prompt_ids.shape[0] == 1
        prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
        assert (prompt_ids != self.mask_index).all(), "prompt contains mask_index"

        P = prompt_ids.shape[1]
        padded_P = ((P + bs - 1) // bs) * bs
        offset = padded_P - P                     # left pad; offset + P == padded_P

        x = torch.full((1, padded_P), self.pad_index, dtype=torch.long, device=device)
        x[0, offset:padded_P] = prompt_ids[0]

        num_blocks = (response_length + bs - 1) // bs
        for b in range(num_blocks):
            cur = min(bs, response_length - b * bs)
            x = torch.cat([x, torch.full((1, cur), self.mask_index, dtype=torch.long, device=device)], dim=1)
            start, stop = padded_P + b * bs, padded_P + b * bs + cur

            num_transfer = get_num_transfer_tokens(
                x[:, start:stop] == self.mask_index,
                steps_per_block, self.scheduler, stochastic_transfer,
            )
            for s in range(num_transfer.size(1)):
                x_block = x[:, start:stop]
                mask_block = x_block == self.mask_index
                if not mask_block.any():
                    break
                attn, pos = _prepare_for_sampling(x, bs, self.pad_index)
                logits = self.forward(x, attention_mask=attn, position_ids=pos).logits
                x[:, start:stop] = _diffusion_step_block(
                    logits[:, start:stop], x_block, mask_block,
                    num_transfer[:, s], temperature, remasking,
                )

        # strip left pad: prompt and response are contiguous from `offset` onward
        return x[:, offset:offset + P + response_length]


# ======================================================================================
# Tier 2 — batched + variable-length prompts + EOS early-exit + NFE accounting
# ======================================================================================
class SFTMixinBatched:
    @torch.no_grad()
    def sample_sft(
        self,
        prompt_ids,                       # [P] | [B, P] | list of 1-D tensors (ragged ok)
        response_length: int,
        steps_per_block: Optional[int] = None,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        stochastic_transfer: bool = False,
        eos_id: Optional[int] = None,
        early_exit: bool = True,
        return_nfes: bool = False,
    ) -> Union[torch.LongTensor, Tuple[torch.LongTensor, int]]:
        """Batched SFT. Prompts are LEFT-padded to a common block-multiple width
        `padded_P`; the returned canvas is [B, padded_P + R] (per-row prompt
        offsets are stashed on self._last_prompt_offsets for slicing)."""
        device = next(self.backbone.parameters()).device
        bs = self.block_size
        steps_per_block = steps_per_block if steps_per_block is not None else bs
        assert self.pad_index is not None, "batched SFT requires pad_index"

        # --- normalize to a list of 1-D prompt rows ---
        if isinstance(prompt_ids, (list, tuple)):
            rows = [p.to(device=device, dtype=torch.long).flatten() for p in prompt_ids]
        elif prompt_ids.ndim == 1:
            rows = [prompt_ids.to(device=device, dtype=torch.long)]
        else:
            rows = [r.to(device=device, dtype=torch.long) for r in prompt_ids]
        B = len(rows)
        for r in rows:
            assert (r != self.mask_index).all(), "a prompt row contains mask_index"

        prompt_lens = torch.tensor([r.numel() for r in rows], dtype=torch.long, device=device)
        padded_P = int(((int(prompt_lens.max()) + bs - 1) // bs) * bs)
        offsets = padded_P - prompt_lens                     # per-row left pad
        self._last_prompt_offsets = offsets

        x = torch.full((B, padded_P), self.pad_index, dtype=torch.long, device=device)
        for b, r in enumerate(rows):
            x[b, int(offsets[b]):padded_P] = r

        done = torch.zeros(B, dtype=torch.bool, device=device)
        num_blocks = (response_length + bs - 1) // bs
        nfes = 0

        for blk in range(num_blocks):
            if early_exit and bool(done.all()):
                break
            cur = min(bs, response_length - blk * bs)
            x = torch.cat([x, torch.full((B, cur), self.mask_index, dtype=torch.long, device=device)], dim=1)
            start, stop = padded_P + blk * bs, padded_P + blk * bs + cur

            num_transfer = get_num_transfer_tokens(
                x[:, start:stop] == self.mask_index,
                steps_per_block, self.scheduler, stochastic_transfer,
            )
            for s in range(num_transfer.size(1)):
                x_block = x[:, start:stop]
                mask_block = x_block == self.mask_index
                if not mask_block.any():
                    break
                attn, pos = _prepare_for_sampling(x, bs, self.pad_index)
                logits = self.forward(x, attention_mask=attn, position_ids=pos).logits
                nfes += 1
                x[:, start:stop] = _diffusion_step_block(
                    logits[:, start:stop], x_block, mask_block,
                    num_transfer[:, s], temperature, remasking,
                )

            if eos_id is not None:
                done |= (x[:, start:stop] == eos_id).any(dim=1)

        return (x, nfes) if return_nfes else x


# ======================================================================================
# Tier 3 — prefix KV-cache + classifier-free guidance + AR right-shift
#           (the full sophistication of the original monolithic sampler)
# ======================================================================================
class SFTMixinBatchedCached:
    @torch.no_grad()
    def sample_sft(
        self,
        prompt_ids,
        response_length: int,
        steps_per_block: Optional[int] = None,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        stochastic_transfer: bool = False,
        eos_id: Optional[int] = None,
        early_exit: bool = True,
        cfg_scale: float = 0.0,
        cfg_keep_tokens: Optional[list] = None,
        right_shift_logits: bool = False,
        return_nfes: bool = False,
    ) -> Union[torch.LongTensor, Tuple[torch.LongTensor, int]]:
        """Caches the prefix (prompt + finished blocks) once per block, then runs
        the inner diffusion steps over only the active block against that cache.
        Adds CFG (mask the given tokens in the uncond branch) and an AR-style
        cross-block right-shift of the logits."""
        device = next(self.backbone.parameters()).device
        bs = self.block_size
        steps_per_block = steps_per_block if steps_per_block is not None else bs
        assert self.pad_index is not None, "cached SFT requires pad_index"

        if isinstance(prompt_ids, (list, tuple)):
            rows = [p.to(device=device, dtype=torch.long).flatten() for p in prompt_ids]
        elif prompt_ids.ndim == 1:
            rows = [prompt_ids.to(device=device, dtype=torch.long)]
        else:
            rows = [r.to(device=device, dtype=torch.long) for r in prompt_ids]
        B = len(rows)
        prompt_lens = torch.tensor([r.numel() for r in rows], dtype=torch.long, device=device)
        padded_P = int(((int(prompt_lens.max()) + bs - 1) // bs) * bs)
        offsets = padded_P - prompt_lens
        self._last_prompt_offsets = offsets

        x = torch.full((B, padded_P), self.pad_index, dtype=torch.long, device=device)
        for b, r in enumerate(rows):
            x[b, int(offsets[b]):padded_P] = r

        # tokens "given" at the start -> masked in the unconditional CFG branch
        unmasked_index = (x != self.mask_index) & (x != self.pad_index)
        if cfg_keep_tokens:
            keep = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=device))
            unmasked_index &= ~keep

        done = torch.zeros(B, dtype=torch.bool, device=device)
        num_blocks = (response_length + bs - 1) // bs
        nfes = 0

        for blk in range(num_blocks):
            if early_exit and bool(done.all()):
                break
            T_prefix = x.shape[1]
            cur = min(bs, response_length - blk * bs)
            if cur <= 0:
                break

            # ---- 1. cache the prefix (cond, and uncond if CFG) ----
            prefix_attn, prefix_pos = _prepare_for_sampling(x, bs, self.pad_index)
            out = self.forward(x, attention_mask=prefix_attn, position_ids=prefix_pos, use_cache=True)
            cond_past = out.past_key_values
            cond_last = out.logits[:, -1:, :]
            nfes += 1
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[unmasked_index] = self.mask_index
                out_u = self.forward(un_x, attention_mask=prefix_attn, position_ids=prefix_pos, use_cache=True)
                uncond_past, uncond_last = out_u.past_key_values, out_u.logits[:, -1:, :]
                nfes += 1
            else:
                uncond_past = uncond_last = None

            # ---- 2. append the active block ----
            x = torch.cat([x, torch.full((B, cur), self.mask_index, dtype=torch.long, device=device)], dim=1)
            unmasked_index = torch.cat(
                [unmasked_index, torch.zeros((B, cur), dtype=torch.bool, device=device)], dim=1
            )
            start, stop = T_prefix, T_prefix + cur

            full_attn, full_pos = _prepare_for_sampling(x, bs, self.pad_index)
            attn_block = full_attn[:, :, start:stop, :]    # queries = block, keys = whole canvas
            pos_block = full_pos[:, start:stop]

            num_transfer = get_num_transfer_tokens(
                x[:, start:stop] == self.mask_index,
                steps_per_block, self.scheduler, stochastic_transfer,
            )

            # ---- 3. inner diffusion loop against the cached prefix ----
            for s in range(num_transfer.size(1)):
                x_block = x[:, start:stop]
                mask_block = x_block == self.mask_index
                if not mask_block.any():
                    break

                logits_block = self.forward(
                    x_block, attention_mask=attn_block, position_ids=pos_block,
                    past_key_values=copy.deepcopy(cond_past), use_cache=False,
                ).logits
                nfes += 1
                if cfg_scale > 0.0:
                    un_logits = self.forward(
                        x_block, attention_mask=attn_block, position_ids=pos_block,
                        past_key_values=copy.deepcopy(uncond_past), use_cache=False,
                    ).logits
                    nfes += 1
                    logits_block = un_logits + (cfg_scale + 1.0) * (logits_block - un_logits)

                if right_shift_logits:
                    prefix_last = (
                        uncond_last + (cfg_scale + 1.0) * (cond_last - uncond_last)
                        if cfg_scale > 0.0 else cond_last
                    )
                    shifted = torch.empty_like(logits_block)
                    shifted[:, 0:1, :] = prefix_last
                    shifted[:, 1:, :] = logits_block[:, :-1, :]
                    logits_block = shifted

                x[:, start:stop] = _diffusion_step_block(
                    logits_block, x_block, mask_block,
                    num_transfer[:, s], temperature, remasking,
                )

            if eos_id is not None:
                done |= (x[:, start:stop] == eos_id).any(dim=1)

        return (x, nfes) if return_nfes else x


# ======================================================================================
# Composed production sampler (most-advanced tier)
# ======================================================================================
class BD3LMSampler(SFTMixinBatchedCached, MinimalBD3LMSampler):
    """Full BD3LM sampler: unconditional `.sample()` + cached/CFG `.sample_sft()`."""
    pass