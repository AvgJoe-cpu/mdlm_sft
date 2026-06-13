"""
Evaluation script for MDLM-based SFT. Computes PPL via the continuous-time NELBO estimator
python -m your_pkg.evaluate \
    gold_ds_path=/path/to/test_with_gold \
    gen_ds_path=/path/to/test_with_generations \
    model_name_or_path=/path/to/checkpoint \
    output_json=/path/to/metrics.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler


# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# @dataclass
# class EvalConfig:
#     # data
#     gold_ds_path: str = MISSING          # dataset with (prompt, completion=GOLD)
#     gen_ds_path: str = MISSING           # dataset with (prompt, completion=GENERATION)
#     output_json: str = MISSING

#     # model / tokenizer (only needed for PPL)
#     model_name_or_path: str = MISSING

#     # MDLM NELBO estimator
#     max_length: int = 1024
#     batch_size: int = 4
#     num_mc_samples: int = 8              # MC averages per example to cut 1/t variance
#     time_epsilon: float = 1e-3
#     eval_seed: int = 0                   # mirrors trainer's deterministic eval
#     loss_weight_type: str = "scheduler"  # keep as "scheduler" to match trainer's eval

#     # BERTScore
#     bertscore_model: str = "roberta-large"
#     bertscore_lang: str = "en"
#     bertscore_batch_size: int = 32
#     bertscore_rescale_with_baseline: bool = True

#     # housekeeping
#     per_example_out: Optional[str] = None   # optional CSV with per-example metrics
#     device: str = "cuda"                    # "cuda" | "cpu"
#     bf16: bool = True
#     limit: Optional[int] = None             # eval on first N examples (debug)


# cs = ConfigStore.instance()
# cs.store(name="config", node=EvalConfig)


# ---------------------------------------------------------------------------
# PPL via MDLM continuous-time NELBO
# ---------------------------------------------------------------------------
# This mirrors CustomForwardSFTTrainer.compute_loss in EVAL mode:
#   numerator   = Σ_masked  w(t) · CE         with w(t) from scheduler.weight(t)
#   denominator = Σ assistant tokens (maskable)        --> bits/assistant-token
# Plus deterministic per-batch RNG so PPL is reproducible across runs.
# We Monte-Carlo average over `num_mc_samples` (t, mask) draws per example to reduce
# the high variance from the 1/t weight (which is what the trainer accepts batch-to-batch
# but we can afford to reduce here since eval is offline).



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PPLConfig:
    # data
    gold_ds_path: str = MISSING
    output_json: str = MISSING
    per_example_out: Optional[str] = None

    # model / tokenizer
    model_name_or_path: str = MISSING

    # MDLM NELBO estimator
    max_length: int = 1024
    batch_size: int = 4
    num_mc_samples: int = 8
    time_epsilon: float = 1e-3
    eval_seed: int = 0
    loss_weight_type: str = "scheduler"   # kept for parity with trainer

    # housekeeping
    device: str = "cuda"
    bf16: bool = True
    limit: Optional[int] = None           # eval first N examples (debug)


cs = ConfigStore.instance()
cs.store(name="config", node=PPLConfig)

def _format_to_messages(example):
    return {
        "messages": [
            {"role": "user",      "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


def _sft_tokenize(example, tokenizer, max_length):
    enc = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        max_length=max_length,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    assistant_mask = enc["assistant_masks"]
    attention_mask = enc["attention_mask"]
    labels = [tok if m == 1 else -100 for tok, m in zip(input_ids, assistant_mask)]
    return {
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": attention_mask,
    }


def _pad_batch(batch, pad_id):
    """Pad a list of dicts (input_ids/labels/attention_mask) to the max length in batch."""
    L = max(len(x["input_ids"]) for x in batch)

    def _pad(seq, value):
        return seq + [value] * (L - len(seq))

    input_ids      = torch.tensor([_pad(x["input_ids"],      pad_id) for x in batch], dtype=torch.long)
    labels         = torch.tensor([_pad(x["labels"],         -100)   for x in batch], dtype=torch.long)
    attention_mask = torch.tensor([_pad(x["attention_mask"], 0)      for x in batch], dtype=torch.long)
    return input_ids, labels, attention_mask


def _eval_rand(shape, device, *, seed, step, stream):
    """Same recipe as the trainer's _eval_rand, minus DDP rank (single-process eval)."""
    s = seed + 1_000_003 * step + 1009 * stream
    g = torch.Generator(device=device).manual_seed(int(s))
    return torch.rand(*shape, device=device, generator=g)


@torch.no_grad()
def compute_mdlm_ppl(
    gold_ds: Dataset,
    *,
    model,
    tokenizer,
    scheduler,
    cfg: PPLConfig,
):
    """Returns (corpus_metrics_dict, per_example_list) — per-example NELBO/PPL/entropy/acc."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    # 1. tokenize
    ds = gold_ds.map(_format_to_messages)
    ds = ds.map(
        _sft_tokenize,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length},
        remove_columns=[c for c in ds.column_names if c not in ("prompt", "completion")],
    )

    n = len(ds)
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer has no pad_token_id"

    # NLL/PPL accumulators (weighted by w(t); denom = Σ maskable)
    per_ex_nll  = [0.0] * n
    per_ex_toks = [0.0] * n   # maskable count per example × MC samples
    total_nll, total_toks = 0.0, 0.0

    # Diagnostic accumulators (unweighted; denom = Σ masked subset)
    per_ex_correct       = [0.0] * n
    per_ex_entropy_sum   = [0.0] * n
    per_ex_masked_toks   = [0.0] * n
    total_correct, total_entropy_sum, total_masked_toks = 0.0, 0.0, 0.0

    # 2. MC loop
    step_counter = 0
    for mc in range(cfg.num_mc_samples):
        for start in tqdm(
            range(0, n, cfg.batch_size),
            desc=f"PPL (MC {mc+1}/{cfg.num_mc_samples})",
            leave=False,
        ):
            batch_rows = [ds[i] for i in range(start, min(start + cfg.batch_size, n))]
            input_ids, labels, attention_mask = _pad_batch(batch_rows, pad_id=pad_id)
            input_ids      = input_ids.to(device)
            labels         = labels.to(device)
            attention_mask = attention_mask.to(device)

            b, l = input_ids.shape
            maskable_mask = labels != -100
            if maskable_mask.sum() == 0:
                step_counter += 1
                continue

            # deterministic noise (matches trainer)
            t = cfg.time_epsilon + (1 - cfg.time_epsilon) * _eval_rand(
                (b,), device, seed=cfg.eval_seed, step=step_counter, stream=0
            )
            p_mask = 1.0 - scheduler(t).unsqueeze(1).expand(b, l)
            masked_mask = (
                _eval_rand((b, l), device, seed=cfg.eval_seed, step=step_counter, stream=1)
                < p_mask
            ) & maskable_mask

            if masked_mask.sum() == 0:
                step_counter += 1
                continue

            noised_input_ids = torch.where(
                masked_mask, tokenizer.mask_token_id, input_ids
            )

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)

            masked_logits  = outputs.logits[masked_mask].float()   # [N, V]
            masked_targets = input_ids[masked_mask]                # [N]
            del outputs

            ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")  # [N]
            w  = scheduler.weight(t).unsqueeze(1).expand(b, l)[masked_mask]        # [N]

            # ── diagnostics on masked subset (unweighted) ──────────────────────
            log_probs         = torch.log_softmax(masked_logits, dim=-1)           # [N, V]
            per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)         # [N]
            correct           = (masked_logits.argmax(dim=-1) == masked_targets)   # [N] bool
            del log_probs, masked_logits

            # bookkeeping: index each masked token back to its example
            which_ex    = masked_mask.nonzero(as_tuple=False)[:, 0]                # [N]
            contrib_nll = (w.double() * ce.double())                               # [N]
            ent_cpu     = per_token_entropy.double().tolist()
            cor_cpu     = correct.long().tolist()
            nll_cpu     = contrib_nll.tolist()
            ex_ids      = (start + which_ex).tolist()

            for ex_id, c_nll, c_ent, c_cor in zip(ex_ids, nll_cpu, ent_cpu, cor_cpu):
                per_ex_nll[ex_id]         += c_nll
                per_ex_entropy_sum[ex_id] += c_ent
                per_ex_correct[ex_id]     += c_cor
                per_ex_masked_toks[ex_id] += 1.0

            # maskable denom for NLL/PPL (counted once per MC sample)
            ex_tok_counts = maskable_mask.sum(dim=1).tolist()
            for k, cnt in enumerate(ex_tok_counts):
                per_ex_toks[start + k] += float(cnt)

            # corpus running sums
            total_nll         += contrib_nll.sum().item()
            total_toks        += float(maskable_mask.sum().item())
            total_correct     += float(correct.sum().item())
            total_entropy_sum += float(per_token_entropy.sum().item())
            total_masked_toks += float(masked_mask.sum().item())

            step_counter += 1

    # 3. per-example metrics
    per_example = []
    for i in range(n):
        # NLL/PPL: weighted, denom = Σ maskable
        if per_ex_toks[i] > 0:
            nll_i = per_ex_nll[i] / per_ex_toks[i]
            nll_d = {
                "nll": nll_i,
                "bpd": nll_i / math.log(2),
                "ppl": math.exp(min(nll_i, 30.0)),
            }
        else:
            nll_d = {"nll": None, "bpd": None, "ppl": None}

        # entropy / accuracy: unweighted, denom = Σ masked subset
        if per_ex_masked_toks[i] > 0:
            diag_d = {
                "entropy":             per_ex_entropy_sum[i] / per_ex_masked_toks[i],
                "mean_token_accuracy": per_ex_correct[i]     / per_ex_masked_toks[i],
            }
        else:
            diag_d = {"entropy": None, "mean_token_accuracy": None}

        per_example.append({"idx": i, "prompt": ds[i]["prompt"], **nll_d, **diag_d})

    # 4. corpus metrics
    mean_nll = total_nll / total_toks if total_toks > 0 else float("nan")
    corpus = {
        "nll":                 mean_nll,
        "bpd":                 mean_nll / math.log(2),
        "ppl":                 math.exp(min(mean_nll, 30.0)),
        "entropy":             (total_entropy_sum / total_masked_toks) if total_masked_toks > 0 else float("nan"),
        "mean_token_accuracy": (total_correct     / total_masked_toks) if total_masked_toks > 0 else float("nan"),
        "n_examples":          n,
        "n_mc_samples":        cfg.num_mc_samples,
    }
    return corpus, per_example
# # ---------------------------------------------------------------------------
# # BERTScore
# # ---------------------------------------------------------------------------
# def compute_bertscore(
#     *,
#     candidates: list[str],
#     references: list[str],
#     cfg: EvalConfig,
# ):
#     """Returns (corpus_metrics_dict, per_example_list)."""
#     # Lazy import so PPL-only runs don't need the package.
#     from bert_score import score as bertscore_score

#     P, R, F1 = bertscore_score(
#         cands=candidates,
#         refs=references,
#         model_type=cfg.bertscore_model,
#         lang=cfg.bertscore_lang,
#         batch_size=cfg.bertscore_batch_size,
#         rescale_with_baseline=cfg.bertscore_rescale_with_baseline,
#         verbose=False,
#     )
#     per_example = [
#         {"idx": i, "bertscore_p": p.item(), "bertscore_r": r.item(), "bertscore_f1": f.item()}
#         for i, (p, r, f) in enumerate(zip(P, R, F1))
#     ]
#     corpus = {
#         "bertscore_p":  P.mean().item(),
#         "bertscore_r":  R.mean().item(),
#         "bertscore_f1": F1.mean().item(),
#     }
#     return corpus, per_example


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: EvalConfig) -> None:
    # ---- load datasets ------------------------------------------------------
    gold_ds = load_from_disk(cfg.gold_ds_path)
    # gen_ds  = load_from_disk(cfg.gen_ds_path)

    # if cfg.limit is not None:
    #     gold_ds = gold_ds.select(range(min(cfg.limit, len(gold_ds))))

    # Align on prompt so PPL and BERTScore are over the SAME examples.
    # prompts, golds, gens = _align_on_prompt(gold_ds, gen_ds)
    # aligned_gold = Dataset.from_dict({"prompt": prompts, "completion": golds})

    # ---- PPL ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    ).to(cfg.device if torch.cuda.is_available() else "cpu").eval()

    scheduler = LinearAlphaScheduler()
    ppl_corpus, ppl_per_ex = compute_mdlm_ppl(
        aligned_gold,
        model=model,
        tokenizer=tokenizer,
        scheduler=scheduler,
        cfg=cfg,
    )

    # Free model before loading BERTScore's encoder.
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # # ---- BERTScore ----------------------------------------------------------
    # bs_corpus, bs_per_ex = compute_bertscore(
    #     candidates=gens,
    #     references=golds,
    #     cfg=cfg,
    # )    

    out_json = Path(cfg.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(corpus, indent=2))

    if cfg.per_example_out is not None:
        out_csv = Path(cfg.per_example_out)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_example[0].keys()))
            writer.writeheader()
            writer.writerows(per_example)

# if __name__ == "__main__":
#     main()