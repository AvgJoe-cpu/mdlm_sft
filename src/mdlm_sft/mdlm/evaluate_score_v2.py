import json
import math
from pathlib import Path
from typing import Any
from dataclasses import dataclass
import hydra
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchgen import model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler, make_alpha_scheduler


from accelerate import PartialState

def _eval_rand(shape, device, *, seed, step, stream):
    """Same recipe as the trainer's _eval_rand, minus DDP rank (single-process eval)."""
    s = seed + 1_000_003 * step + 1009 * stream
    g = torch.Generator(device=device).manual_seed(int(s))
    return torch.rand(*shape, device=device, generator=g)


import logging
log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    model_name_or_path: str = "/Users/jona/Documents/mdlm_sft/artifacts_weights_mdlm_base_mdlm-owt_chat"
    gold_path:          str = "/Users/jona/Documents/mdlm_sft/datasets_writingprompts-strat/strat_train_12pct"
    save_summary_path:  str = "/Users/jona/Documents/mdlm_sft/strat_eval_summary"
    max_length:                 int   = 100
    per_device_eval_batch_size: int   = 8
    dataloader_num_workers:     int   = 0
    bf16:                       bool  = True
    num_mc_samples:             int   = 1
    time_epsilon:               float = 1e-3
    eval_seed:                  int   = 42

cs = ConfigStore.instance()
cs.store(name="eval", node=EvalConfig)
# ---------------------------------------------------------------------------
# Collator (unchanged).
# ---------------------------------------------------------------------------
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
            "id":            [r["id"]    for r in rows],
            "prompt":         [r["prompt"] for r in rows],
        }

def format_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


import contextlib

@torch.no_grad()
def compute_mdlm_ppl(
#At num_mc_samples=1 this matches MDLMTrainer.evaluate() exactly. Higher values reduce MC variance at K× compute cost.

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
    # Autocast is supported on cuda/cpu/xpu but not mps. Fall back to a no-op
    # context manager elsewhere so the eval loop stays device-agnostic.
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
                 "correct": 0.0, "entropy_sum": 0.0, "masked_toks": 0.0}
            acc[ex_idx] = s
        return s

    total_nll, total_toks = 0.0, 0.0
    total_correct, total_entropy_sum, total_masked_toks = 0.0, 0.0, 0.0

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
            log_probs         = torch.log_softmax(masked_logits, dim=-1)
            per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            correct           = (masked_logits.argmax(dim=-1) == masked_targets)
            del log_probs, masked_logits

            which_in_batch = masked_mask.nonzero(as_tuple=False)[:, 0].tolist()
            contrib_nll = (w.float() * ce.float()).detach().cpu().double()
            nll_cpu = contrib_nll.tolist()
            ent_cpu = per_token_entropy.detach().cpu().double().tolist()
            cor_cpu = correct.detach().cpu().long().tolist()

            for k_in_batch, c_nll, c_ent, c_cor in zip(which_in_batch, nll_cpu, ent_cpu, cor_cpu):
                s = _slot(batch_ids[k_in_batch], batch_prompts[k_in_batch])
                s["nll"]         += c_nll
                s["entropy_sum"] += c_ent
                s["correct"]     += c_cor
                s["masked_toks"] += 1.0

            ex_tok_counts = maskable_mask.sum(dim=1).tolist()
            for k, cnt in enumerate(ex_tok_counts):
                _slot(batch_ids[k], batch_prompts[k])["toks"] += float(cnt)

            total_nll         += contrib_nll.sum().item()
            total_toks        += float(maskable_mask.sum().item())
            total_correct     += float(correct.sum().item())
            total_entropy_sum += float(per_token_entropy.sum().item())   # still on device, fp32 — fine, .item() returns Python float
            total_masked_toks += float(masked_mask.sum().item())
            step_counter += 1
            
    per_example = []
    for ex_idx, s in acc.items():
        nll_d = ({"nll": s["nll"] / s["toks"],
                  "bpd": (s["nll"] / s["toks"]) / math.log(2),
                  "ppl": math.exp(min(s["nll"] / s["toks"], 30.0))}
                 if s["toks"] > 0 else {"nll": None, "bpd": None, "ppl": None})
        diag_d = ({"entropy":             s["entropy_sum"] / s["masked_toks"],
                   "mean_token_accuracy": s["correct"]     / s["masked_toks"]}
                  if s["masked_toks"] > 0 else {"entropy": None, "mean_token_accuracy": None})
        per_example.append({"id": ex_idx, "prompt": s["prompt"], **nll_d, **diag_d})

    mean_nll = total_nll / total_toks if total_toks > 0 else float("nan")
    corpus = {
        "nll":                 mean_nll,
        "bpd":                 mean_nll / math.log(2),
        "ppl":                 math.exp(min(mean_nll, 30.0)),
        "entropy":             (total_entropy_sum / total_masked_toks) if total_masked_toks > 0 else float("nan"),
        "mean_token_accuracy": (total_correct     / total_masked_toks) if total_masked_toks > 0 else float("nan"),
        "n_examples":          len(acc),
        "n_mc_samples":        num_mc_samples,
    }
    return corpus, per_example



# ---------------------------------------------------------------------------
def evaluate(cfg) -> None:
    def _sft_map_fn(example, tokenizer=None, max_length=None):
        enc = tokenizer.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True, max_length=max_length, truncation=True)
        
        input_ids      = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]
        attention_mask = enc["attention_mask"]
        labels = [tok if m == 1 else -100 for tok, m in zip(input_ids, assistant_mask)]        
        return {"input_ids": input_ids, "labels": labels, "assistant_masks": assistant_mask, "attention_mask": attention_mask}    
        
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    assert tokenizer.pad_token_id is not None, "tokenizer has no pad_token_id"

    model     = AutoModelForMaskedLM.from_pretrained(cfg.model_name_or_path, trust_remote_code=True).eval()

    scheduler = make_alpha_scheduler("LinearalphaScheduler")

    ds = load_from_disk(cfg.gold_path)
    ds = ds.map(format_to_messages)
    keep = {"id", "prompt", "completion"}

    ds = ds.map(
        _sft_map_fn,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length},
        remove_columns=[c for c in ds.column_names if c not in keep],
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
    if cfg.bf16 and device.type == "cuda":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32
    model.to(device=device, dtype=model_dtype)

    corpus, per_ex = compute_mdlm_ppl(
        loader,
        model=model,
        tokenizer=tokenizer,
        scheduler=scheduler,
        device=device,                       # ← add
        bf16=cfg.bf16,
        num_mc_samples=cfg.num_mc_samples,
        time_epsilon=cfg.time_epsilon,
        eval_seed=cfg.eval_seed,
    )

    out_dir = Path(cfg.save_summary_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{cfg.model_name_or_path.split('/')[-1]}_{cfg.gold_path.split('/')[-1]}_eval"
    corpus_path      = out_dir / f"{prefix}_corpus.json"
    per_example_path = out_dir / f"{prefix}_per_example.jsonl"

    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2)
        f.write("\n")

    with open(per_example_path, "w") as f:
        for row in per_ex:
            f.write(json.dumps(row) + "\n")


@hydra.main(version_base=None, config_path=None, config_name="eval")
def main(cfg: EvalConfig) -> None:
    log.info("Evaluation config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    evaluate(cfg)

if __name__ == "__main__":
    main()