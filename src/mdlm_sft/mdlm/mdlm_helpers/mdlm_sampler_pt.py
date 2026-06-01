# THIS SAMPLER IS AN ADAPTATION FROM: https://github.com/kuleshov-group/mdlm
# SPECIFICALLY: (https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py)

import torch
from transformers import AutoModelForMaskedLM

from src.mdlm.mdlm_helpers.mdlm_scheduler import (BaseAlphaScheduler,
                                                  CosineAlphaScheduler,
                                                  LinearAlphaScheduler)


# ── minimal sampler ─────────────────────────────────────────────────────────
def _sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


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

    # ── forward: raw HF call → subs log-probs ─────────────────────────────────
    def forward(self, x, sigma):
        # sigma is always zeros (time_conditioning=False)
        logits = self.backbone(
            input_ids=x,
            timesteps=sigma,  # model zeros this internally anyway
        ).logits  # [B, L, V]
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

