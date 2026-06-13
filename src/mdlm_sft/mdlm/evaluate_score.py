import torch
from torch.utils.data import DataLoader
from typing import Any
import math
import torch.nn.functional as F
from tqdm import tqdm


class _PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        L = max(len(r["input_ids"]) for r in rows)
        def pad(seq: list[int], value: int) -> list[int]: return seq + [value] * (L - len(seq))
        return {
            "input_ids":      torch.tensor([pad(r["input_ids"],      self.pad_id)  for r in rows], dtype=torch.long),
            "labels":         torch.tensor([pad(r["labels"],         -100)         for r in rows], dtype=torch.long),
            "attention_mask": torch.tensor([pad(r["attention_mask"], 0)            for r in rows], dtype=torch.long),
            "idx":            [r["idx"]    for r in rows],
            "prompt":         [r["prompt"] for r in rows],
        }
    


@torch.no_grad()
def compute_mdlm_ppl(loader, *, model, tokenizer, scheduler, cfg) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    # idx -> accumulator dict (created on first sight)
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
    for mc in range(cfg.num_mc_samples):
        for batch in tqdm(loader, desc=f"PPL (MC {mc+1}/{cfg.num_mc_samples})", leave=False):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_idxs     = batch["idx"]
            batch_prompts  = batch["prompt"]

            b, l = input_ids.shape
            maskable_mask = labels != -100
            if maskable_mask.sum() == 0:
                step_counter += 1
                continue

            t = cfg.time_epsilon + (1 - cfg.time_epsilon) * _eval_rand(
                (b,), device, seed=cfg.eval_seed, step=step_counter, stream=0)
            p_mask = 1.0 - scheduler(t).unsqueeze(1).expand(b, l)
            masked_mask = (_eval_rand((b, l), device, seed=cfg.eval_seed,
                                      step=step_counter, stream=1) < p_mask) & maskable_mask
            if masked_mask.sum() == 0:
                step_counter += 1
                continue

            noised_input_ids = torch.where(masked_mask, tokenizer.mask_token_id, input_ids)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
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
            contrib_nll = (w.double() * ce.double())
            nll_cpu = contrib_nll.tolist()
            ent_cpu = per_token_entropy.double().tolist()
            cor_cpu = correct.long().tolist()

            for k_in_batch, c_nll, c_ent, c_cor in zip(which_in_batch, nll_cpu, ent_cpu, cor_cpu):
                s = _slot(batch_idxs[k_in_batch], batch_prompts[k_in_batch])
                s["nll"]         += c_nll
                s["entropy_sum"] += c_ent
                s["correct"]     += c_cor
                s["masked_toks"] += 1.0

            ex_tok_counts = maskable_mask.sum(dim=1).tolist()
            for k, cnt in enumerate(ex_tok_counts):
                _slot(batch_idxs[k], batch_prompts[k])["toks"] += float(cnt)

            total_nll         += contrib_nll.sum().item()
            total_toks        += float(maskable_mask.sum().item())
            total_correct     += float(correct.sum().item())
            total_entropy_sum += float(per_token_entropy.sum().item())
            total_masked_toks += float(masked_mask.sum().item())
            step_counter += 1

    # per-example rows, straight from acc
    per_example = []
    for ex_idx, s in acc.items():
        nll_d = ({"nll": s["nll"] / s["toks"],
                  "bpd": (s["nll"] / s["toks"]) / math.log(2),
                  "ppl": math.exp(min(s["nll"] / s["toks"], 30.0))}
                 if s["toks"] > 0 else {"nll": None, "bpd": None, "ppl": None})
        diag_d = ({"entropy":             s["entropy_sum"] / s["masked_toks"],
                   "mean_token_accuracy": s["correct"]     / s["masked_toks"]}
                  if s["masked_toks"] > 0 else {"entropy": None, "mean_token_accuracy": None})
        per_example.append({"idx": ex_idx, "prompt": s["prompt"], **nll_d, **diag_d})

    mean_nll = total_nll / total_toks if total_toks > 0 else float("nan")
    corpus = {
        "nll":                 mean_nll,
        "bpd":                 mean_nll / math.log(2),
        "ppl":                 math.exp(min(mean_nll, 30.0)),
        "entropy":             (total_entropy_sum / total_masked_toks) if total_masked_toks > 0 else float("nan"),
        "mean_token_accuracy": (total_correct     / total_masked_toks) if total_masked_toks > 0 else float("nan"),
        "n_examples":          len(acc),
        "n_mc_samples":        cfg.num_mc_samples,
    }
    return corpus, per_example


def main():
    ds = load_from_disk(cfg.gold_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, device_map="auto").eval()

    ds = gold_ds.map(_format_to_messages)
    keep = {"idx", "prompt", "completion"}
    ds = ds.map(
        _sft_tokenize,
        fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length},
        remove_columns=[c for c in ds.column_names if c not in keep],
    )
    ds.set_format(type=None, columns=["idx", "prompt", "input_ids", "labels", "attention_mask"])

    assert tokenizer.pad_token_id is not None, "tokenizer has no pad_token_id"
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=_PadCollator(tokenizer.pad_token_id),
        num_workers=cfg.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # ---- score --------------------------------------------------------------
    corpus, per_example = compute_mdlm_ppl(
        loader, model=model, tokenizer=tokenizer, scheduler=scheduler, cfg=cfg,
    )