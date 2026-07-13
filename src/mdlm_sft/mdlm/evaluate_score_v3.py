from __future__ import annotations
import numpy as np 
import json
import logging
import math
import contextlib
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, HfArgumentParser
from accelerate import PartialState

from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler

log = logging.getLogger(__name__)


@dataclass
class MDLMEvalConfig:
    # ── Paths ────────────────────────────────────────────────────────────────
    model_name_or_path: str = "bert-base-uncased"
    gold_path:          Optional[str] = None
    save_summary_path:  str = "eval_outputs"

    # ── Tokenization / batching ──────────────────────────────────────────────
    max_length:                 int  = 1024
    per_device_eval_batch_size: int  = 64
    dataloader_num_workers:     int  = 8

    # ── Precision ────────────────────────────────────────────────────────────
    bf16: bool = True

    # ── MDLM eval knobs (mirror MDLMSFTConfig where applicable) ──────────────
    num_mc_samples: int   = 1
    time_epsilon:   float = 1e-3
    eval_seed:      int   = 0


def _eval_rand(shape, device, *, seed, step, stream):
    """Same recipe as the trainer's _eval_rand, minus DDP rank (single-process eval)."""
    s = seed + 1_000_003 * step + 1009 * stream
    g = torch.Generator(device=device).manual_seed(int(s))
    return torch.rand(*shape, device=device, generator=g)


class _PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        L = max(len(r["input_ids"]) for r in rows)

        def pad(seq: list[int], value: int) -> list[int]:
            return seq + [value] * (L - len(seq))

        return {
            "input_ids":      torch.tensor([pad(r["input_ids"],      self.pad_id) for r in rows], dtype=torch.long),
            "labels":         torch.tensor([pad(r["labels"],         -100)        for r in rows], dtype=torch.long),
            "attention_mask": torch.tensor([pad(r["attention_mask"], 0)           for r in rows], dtype=torch.long),
            "id":             [r["id"]     for r in rows],
            "prompt":         [r["prompt"] for r in rows],
        }

@torch.no_grad()
def compute_mdlm_ppl(
    loader,
    *,
    model,
    tokenizer,
    scheduler,
    device: torch.device,
    bf16: bool,
    num_mc_samples: int,
    time_epsilon: float,
    eval_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if device.type in ("cuda", "cpu", "xpu"):
        autocast_dtype = torch.bfloat16 if bf16 else torch.float32
        amp_ctx = lambda: torch.autocast(device_type=device.type, dtype=autocast_dtype)
    else:
        amp_ctx = contextlib.nullcontext

    acc: dict[Any, dict[str, Any]] = {}

    def _slot(ex_idx, prompt):
        s = acc.get(ex_idx)
        if s is None:
            s = {"prompt": prompt,
                 "nll": 0.0, "toks": 0.0,
                 "correct": 0.0, "masked_toks": 0.0}
            acc[ex_idx] = s
        return s

    total_nll, total_toks = 0.0, 0.0
    total_correct, total_masked_toks = 0.0, 0.0

    step_counter = 0
    for mc in range(num_mc_samples):
        for batch in tqdm(loader, desc=f"PPL (MC {mc+1}/{num_mc_samples})", leave=False):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_ids      = batch["id"]
            batch_prompts  = batch["prompt"]

            b, l = input_ids.shape
            maskable_mask = labels != -100
            if maskable_mask.sum() == 0:
                step_counter += 1
                continue

            t = time_epsilon + (1 - time_epsilon) * _eval_rand(
                (b,), device, seed=eval_seed, step=step_counter, stream=0)
            p_mask = 1.0 - scheduler(t).unsqueeze(1).expand(b, l)
            masked_mask = (_eval_rand((b, l), device, seed=eval_seed,
                                      step=step_counter, stream=1) < p_mask) & maskable_mask
            if masked_mask.sum() == 0:
                step_counter += 1
                continue

            noised_input_ids = input_ids.masked_fill(masked_mask, tokenizer.mask_token_id)
            with amp_ctx():
                outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)

            masked_logits  = outputs.logits[masked_mask].float()
            masked_targets = input_ids[masked_mask]
            del outputs

            ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")
            w  = scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]
            correct = (masked_logits.argmax(dim=-1) == masked_targets)
            del masked_logits

            which_in_batch = masked_mask.nonzero(as_tuple=False)[:, 0].tolist()
            contrib_nll = (w.float() * ce.float()).detach().cpu().double()
            nll_cpu = contrib_nll.tolist()
            cor_cpu = correct.detach().cpu().long().tolist()

            for k_in_batch, c_nll, c_cor in zip(which_in_batch, nll_cpu, cor_cpu):
                s = _slot(batch_ids[k_in_batch], batch_prompts[k_in_batch])
                s["nll"]         += c_nll
                s["correct"]     += c_cor
                s["masked_toks"] += 1.0

            ex_tok_counts = maskable_mask.sum(dim=1).tolist()
            for k, cnt in enumerate(ex_tok_counts):
                _slot(batch_ids[k], batch_prompts[k])["toks"] += float(cnt)

            total_nll         += contrib_nll.sum().item()
            total_toks        += float(maskable_mask.sum().item())
            total_correct     += float(correct.sum().item())
            total_masked_toks += float(masked_mask.sum().item())
            step_counter += 1

    per_example = []
    for ex_idx, s in acc.items():
        nll_d = ({"nll": s["nll"] / s["toks"],
                  "bpd": (s["nll"] / s["toks"]) / math.log(2),
                  "ppl": math.exp(min(s["nll"] / s["toks"], 30.0))}
                 if s["toks"] > 0 else {"nll": None, "bpd": None, "ppl": None})
        diag_d = ({"mean_token_accuracy": s["correct"] / s["masked_toks"]}
                  if s["masked_toks"] > 0 else {"mean_token_accuracy": None})
        per_example.append({"id": ex_idx, "prompt": s["prompt"], **nll_d, **diag_d})

    mean_nll = total_nll / total_toks if total_toks > 0 else float("nan")
    corpus = {
        "nll":                 mean_nll,
        "bpd":                 mean_nll / math.log(2),
        "ppl":                 math.exp(min(mean_nll, 30.0)),
        "mean_token_accuracy": (total_correct / total_masked_toks) if total_masked_toks > 0 else float("nan"),
        "n_examples":          len(acc),
        "n_mc_samples":        num_mc_samples,
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    assert tokenizer.pad_token_id is not None, "tokenizer has no pad_token_id"

    model     = AutoModelForMaskedLM.from_pretrained(cfg.model_name_or_path, trust_remote_code=True).eval()
    scheduler = LinearAlphaScheduler()
    keep = {"id", "prompt", "completion"}
    ds = (
        load_from_disk(cfg.gold_path)
        .map(format_to_messages, batched=True, num_proc=4)
        .map(
            _sft_map_fn,
            batched=True,
            num_proc=4,
            fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length},
        )
    )
    ds.set_format(type=None, columns=["id", "prompt", "input_ids", "labels", "attention_mask"])

    loader = DataLoader(
        ds,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=_PadCollator(tokenizer.pad_token_id),
        num_workers=cfg.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = PartialState().device
    model_dtype = torch.bfloat16 if (cfg.bf16 and device.type == "cuda") else torch.float32
    model.to(device=device, dtype=model_dtype)

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

        prefix = f"{Path(cfg.model_name_or_path).name}_{Path(cfg.gold_path).name}_eval"
        corpus_path      = out_dir / f"{prefix}_corpus.json"
        per_example_path = out_dir / f"{prefix}_per_example.jsonl"

        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=2)
            f.write("\n")

        with open(per_example_path, "w") as f:
            for row in per_ex:
                f.write(json.dumps(row) + "\n")

        log.info("Evaluation finished. Corpus metrics: %s", json.dumps(corpus, indent=2))
    finally:
        del model, tokenizer, loader, ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = HfArgumentParser(MDLMEvalConfig)
    (cfg,) = parser.parse_args_into_dataclasses()
    log.info("Evaluation config:\n%s", json.dumps(cfg.__dict__, indent=2, default=str))

    try:
        run_evaluation(cfg)
    finally:
        try:
            for h in list(log.handlers):
                if isinstance(h, logging.FileHandler):
                    h.close()
                    log.removeHandler(h)
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()