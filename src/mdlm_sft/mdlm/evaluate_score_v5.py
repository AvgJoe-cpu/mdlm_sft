from __future__ import annotations

import json
import logging
import math
import contextlib
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, HfArgumentParser
from accelerate import PartialState
from typing import Any, Optional, Union
from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler

log = logging.getLogger(__name__)



@dataclass
class MDLMEvalConfig:
    model_name_or_path: Union[str, Path] = "bert-base-uncased"
    dataset_path: Optional[Union[str, Path]] = None
    save_summary_path: str = "eval_outputs"
    max_length: int = 1024
    per_device_eval_batch_size: int = 8
    dataloader_num_workers: int = 0
    bf16: bool = True
    num_mc_samples: int = 1
    time_epsilon: float = 1e-3
    eval_seed: int = 0


def _eval_rand(shape, device, *, seed: int, step: int, stream: int) -> torch.Tensor:
    """Deterministic eval RNG. stream=0: timestep, stream=1: token-mask."""
    gen = torch.Generator(device=device).manual_seed(
        int(seed + 1_000_003 * step + 1009 * stream)
    )
    return torch.rand(*shape, device=device, generator=gen)


class _PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = max(len(r["input_ids"]) for r in rows)
        def stack(key, pad):
            return torch.tensor(
                [r[key] + [pad] * (n - len(r[key])) for r in rows], dtype=torch.long
            )
        return {
            "input_ids": stack("input_ids", self.pad_id),
            "labels": stack("labels", -100),
            "attention_mask": stack("attention_mask", 0),
            "id": [r["id"] for r in rows],
            "prompt": [r["prompt"] for r in rows],
        }


def _nll_metrics(nll_sum: float, denom: float) -> dict[str, Any]:
    if denom <= 0:
        return {"nll": None, "bpd": None, "ppl": None, "ppl_was_clamped": False}
    mean = nll_sum / denom
    return {
        "nll": mean,
        "bpd": mean / math.log(2),
        "ppl": math.exp(min(mean, 30.0)),
        "ppl_was_clamped": mean > 30.0,
    }


def _mc_stats(values: list[float]):
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return None, None, None
    mean = float(finite.mean())
    if finite.size < 2:
        return mean, None, None
    std = float(finite.std(ddof=1))
    return mean, std, std / math.sqrt(finite.size)


@torch.no_grad()
def compute_mdlm_ppl(
    loader, *, model, tokenizer, scheduler, device: torch.device,
    bf16: bool, num_mc_samples: int, time_epsilon: float, eval_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Continuous-time MDLM variational NLL estimate.
        NLL = sum w(t)*CE over sampled masked tokens / sum over all maskable tokens
    For linear alpha(t)=1-t: w(t)=1/t. "ppl" is a variational upper bound, not
    an exact autoregressive likelihood. Sampling matches the trainer exactly.
    """
    assert num_mc_samples >= 1, f"num_mc_samples must be >= 1; got {num_mc_samples}"
    assert 0.0 < time_epsilon < 1.0, f"time_epsilon out of range: {time_epsilon}"
    assert tokenizer.mask_token_id is not None, "tokenizer.mask_token_id required"

    if bf16:
        assert device.type in {"cuda", "cpu", "xpu"}, f"bf16 unsupported on {device.type}"
        amp = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        amp = contextlib.nullcontext

    # Per-example slots keyed by dataset position (avoids duplicate-id merges).
    slots: dict[int, dict[str, Any]] = {}
    totals = {"nll": 0.0, "denom": 0.0, "correct": 0.0, "entropy": 0.0, "masked": 0.0}
    mc_nll_estimates, mc_ex_counts, mc_denoms, mc_masked = [], [], [], []
    step = 0
    n_examples: Optional[int] = None

    was_training = model.training
    model.eval()
    try:
        for mc in range(num_mc_samples):
            offset = 0
            mc_nll_sum = mc_denom = mc_masked_ct = 0.0
            for batch in tqdm(loader, desc=f"PPL (MC {mc+1}/{num_mc_samples})", leave=False):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                ids, prompts = batch["id"], batch["prompt"]
                B, L = input_ids.shape
                assert len(ids) == B and len(prompts) == B

                maskable = labels != -100
                assert not torch.any(maskable & attention_mask.eq(0)), \
                    "maskable positions must have attention_mask=1"
                assert torch.equal(input_ids[maskable], labels[maskable]), \
                    "input_ids must equal labels at maskable positions"

                per_ex_tokens = maskable.sum(dim=1).tolist()
                batch_maskable = float(maskable.sum().item())

                # Allocate/verify slots.
                batch_slots = []
                for i, tc in enumerate(per_ex_tokens):
                    pos = offset + i
                    slot = slots.get(pos)
                    if slot is None:
                        assert mc == 0, f"loader ordering changed at pos {pos}"
                        slot = {"id": ids[i], "prompt": prompts[i],
                                "n_maskable_tokens": int(tc), "nll_sum": 0.0,
                                "nll_denominator": 0.0, "correct_sum": 0.0,
                                "entropy_sum": 0.0, "masked_token_count": 0.0}
                        slots[pos] = slot
                    else:
                        assert slot["n_maskable_tokens"] == int(tc), \
                            f"maskable-token count changed at pos {pos}"
                    slot["nll_denominator"] += float(tc)
                    batch_slots.append(slot)
                offset += B
                totals["denom"] += batch_maskable
                mc_denom += batch_maskable

                if batch_maskable == 0:
                    step += 1
                    continue

                t = time_epsilon + (1.0 - time_epsilon) * _eval_rand(
                    (B,), device, seed=eval_seed, step=step, stream=0)
                alpha_t = scheduler(t)
                assert alpha_t.shape == t.shape, "scheduler(t) must be per-batch"
                p_mask = (1.0 - alpha_t).unsqueeze(1).expand(B, L)
                assert torch.isfinite(p_mask).all() and (p_mask >= 0).all() and (p_mask <= 1).all(), \
                    "scheduler produced invalid p_mask"

                mmask = (_eval_rand((B, L), device, seed=eval_seed, step=step, stream=1)
                         < p_mask) & maskable
                n_masked = int(mmask.sum().item())
                if n_masked == 0:
                    step += 1
                    continue

                noised = input_ids.masked_fill(mmask, tokenizer.mask_token_id)
                with amp():
                    logits = model(input_ids=noised, attention_mask=attention_mask).logits

                mlogits = logits[mmask].float()
                mtargets = input_ids[mmask]
                ce = F.cross_entropy(mlogits, mtargets, reduction="none")
                weights = scheduler.weight(t).unsqueeze(1).expand(B, L)[mmask]
                assert weights.shape == ce.shape and torch.isfinite(weights).all()

                weighted_ce = weights.double() * ce.double()
                log_probs = torch.log_softmax(mlogits, dim=-1)
                entropy_tok = -(log_probs.exp() * log_probs).sum(dim=-1).double()
                correct = (mlogits.argmax(dim=-1) == mtargets).double()

                rows = mmask.nonzero(as_tuple=False)[:, 0]
                # Aggregate 4 metrics per row in a single scatter.
                stacked = torch.stack((weighted_ce, correct, entropy_tok,
                                       torch.ones_like(weighted_ce)), dim=1)
                per_row = torch.zeros(B, 4, device=device, dtype=torch.float64)
                per_row.scatter_add_(0, rows.unsqueeze(1).expand(-1, 4), stacked)
                per_row_list = per_row.cpu().tolist()

                for slot, (rn, rc, re_, rm) in zip(batch_slots, per_row_list):
                    slot["nll_sum"] += rn
                    slot["correct_sum"] += rc
                    slot["entropy_sum"] += re_
                    slot["masked_token_count"] += rm

                bsum = per_row.sum(dim=0).tolist()
                totals["nll"] += bsum[0]; totals["correct"] += bsum[1]
                totals["entropy"] += bsum[2]; totals["masked"] += bsum[3]
                mc_nll_sum += bsum[0]; mc_masked_ct += bsum[3]
                step += 1

            if n_examples is None:
                n_examples = offset
            else:
                assert offset == n_examples, "loader length changed across MC passes"
            mc_ex_counts.append(offset)
            mc_denoms.append(mc_denom)
            mc_masked.append(mc_masked_ct)
            mc_nll_estimates.append(mc_nll_sum / mc_denom if mc_denom > 0 else float("nan"))
    finally:
        if was_training:
            model.train()

    per_example = []
    for pos in sorted(slots):
        s = slots[pos]
        mc_ct = s["masked_token_count"]
        denom = s["nll_denominator"]
        per_example.append({
            "id": s["id"], "prompt": s["prompt"],
            **_nll_metrics(s["nll_sum"], denom),
            "mean_token_accuracy": (s["correct_sum"] / mc_ct) if mc_ct > 0 else None,
            "entropy": (s["entropy_sum"] / mc_ct) if mc_ct > 0 else None,
            "n_maskable_tokens": s["n_maskable_tokens"],
            "n_maskable_token_evaluations": int(denom),
            "n_masked_token_evaluations": int(mc_ct),
            "observed_mask_rate": (mc_ct / denom) if denom > 0 else None,
            "n_mc_samples": num_mc_samples,
        })

    mc_mean, mc_std, mc_se = _mc_stats(mc_nll_estimates)
    denom = totals["denom"]
    masked = totals["masked"]
    corpus_nll = _nll_metrics(totals["nll"], denom)
    corpus = {
        **corpus_nll,
        "mean_token_accuracy": (totals["correct"] / masked) if masked > 0 else None,
        "entropy": (totals["entropy"] / masked) if masked > 0 else None,
        "nelbo_per_token": corpus_nll["nll"],
        "variational_ppl_upper_bound": corpus_nll["ppl"],
        "observed_mask_rate": (masked / denom) if denom > 0 else None,
        "n_examples": n_examples or 0,
        "n_maskable_tokens": int(round(denom / num_mc_samples)),
        "n_maskable_token_evaluations": int(denom),
        "n_masked_token_evaluations": int(masked),
        "n_mc_samples": num_mc_samples,
        "mc_nll_mean": mc_mean,
        "mc_nll_std": mc_std,
        "mc_nll_standard_error": mc_se,
        "mc_nll_estimates": [v if math.isfinite(v) else None for v in mc_nll_estimates],
        "mc_example_counts": mc_ex_counts,
        "mc_maskable_token_counts": [int(v) for v in mc_denoms],
        "mc_masked_token_counts": [int(v) for v in mc_masked],
    }
    return corpus, per_example




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


def run_evaluation(cfg: MDLMEvalConfig) -> None:
    # ── 1. Validate config early ─────────────────────────────────────────────
    assert cfg.dataset_path is not None, "cfg.dataset_path is required"

    # ── 2. Tokenizer ─────────────────────────────────────────────────────────
    # tokenizer = AutoTokenizer.from_pretrained(
    #     cfg.model_name_or_path, trust_remote_code=True
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        Path(cfg.model_name_or_path), trust_remote_code=True
    )    
    assert tokenizer.pad_token_id is not None, "tokenizer has no pad_token_id"
    assert tokenizer.mask_token_id is not None, "tokenizer has no mask_token_id"  # ← required by compute_mdlm_ppl

    # ── 3. Model (load in target dtype directly to save host RAM) ────────────
    device = PartialState().device
    if cfg.bf16 and device.type != "cuda":
        log.warning("bf16 requested but device is %s; falling back to fp32.", device.type)
        cfg.bf16 = False
    model_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    # model = AutoModelForMaskedLM.from_pretrained(
    #     cfg.model_name_or_path,
    #     trust_remote_code=True,
    #     torch_dtype=model_dtype,
    # ).to(device).eval()


# line 324
    model = AutoModelForMaskedLM.from_pretrained(
        Path(cfg.model_name_or_path),
        trust_remote_code=True,
        torch_dtype=model_dtype,
    ).to(device).eval()
    scheduler = LinearAlphaScheduler()

    # ── 4. Dataset: handle DatasetDict, drop unused columns ──────────────────
    from datasets import DatasetDict  # local import; cheap
    raw = load_from_disk(cfg.dataset_path)
    if isinstance(raw, DatasetDict):
        split = "test" if "test" in raw else ("validation" if "validation" in raw else next(iter(raw)))
        log.info("Loaded DatasetDict; using split %r", split)
        raw = raw[split]

    keep = {"id", "prompt", "completion"}
    raw = raw.remove_columns([c for c in raw.column_names if c not in keep])

    ds = raw.map(format_to_messages, batched=True, num_proc=4).map(
        _sft_map_fn,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length},
    )
    ds.set_format(type=None, columns=["id", "prompt", "input_ids", "labels", "attention_mask"])

    # ── 5. Loader ────────────────────────────────────────────────────────────
    loader = DataLoader(
        ds,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,                          # required: MC passes need stable ordering
        collate_fn=_PadCollator(tokenizer.pad_token_id),
        num_workers=cfg.dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── 6. Run eval + persist ────────────────────────────────────────────────
    try:
        corpus, per_ex = compute_mdlm_ppl(
            loader,
            model=model,
            tokenizer=tokenizer,
            scheduler=scheduler,
            device=device,
            bf16=cfg.bf16,
            num_mc_samples=cfg.num_mc_samples,
            time_epsilon=cfg.time_epsilon,
            eval_seed=cfg.eval_seed,
        )

        out_dir = Path(cfg.save_summary_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{Path(cfg.model_name_or_path).name}_{Path(cfg.dataset_path).name}_eval"
        corpus_path      = out_dir / f"{prefix}_corpus.json"
        per_example_path = out_dir / f"{prefix}_per_example.jsonl"

        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=2)
            f.write("\n")

        with open(per_example_path, "w") as f:
            for row in per_ex:
                f.write(json.dumps(row) + "\n")

        log.info("Wrote corpus metrics → %s", corpus_path)
        log.info("Wrote per-example metrics → %s", per_example_path)
        log.info("Corpus:\n%s", json.dumps(corpus, indent=2))
    finally:
        del model, tokenizer, loader, ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = HfArgumentParser(MDLMEvalConfig)

    # Support: `python -m ... config.json` or `... config.yaml` in addition to CLI flags.
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        path = Path(sys.argv[1]).resolve()
        if path.suffix == ".json":
            (cfg,) = parser.parse_json_file(json_file=str(path))
        else:
            (cfg,) = parser.parse_yaml_file(yaml_file=str(path))
    else:
        (cfg,) = parser.parse_args_into_dataclasses()

    assert cfg.dataset_path is not None, "--dataset_path is required"

    log.info("MDLM eval config:\n%s", json.dumps(cfg.__dict__, indent=2, default=str))
    run_evaluation(cfg)