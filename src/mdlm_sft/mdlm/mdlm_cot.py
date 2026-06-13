"""
Chain-of-Thought trainer for MDLM.

Key innovation: Teacher forcing at the reasoning-step level.
- Training data: question + previous_steps → current_step
- Only masks target tokens (current step), preserves context (question + previous steps)
- Enables iterative multi-step reasoning during inference

Based on "Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models"
(Ye et al., 2024) - https://arxiv.org/abs/2402.07754
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Any, Tuple, List
import math
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


from mdlm_sft.mdlm.mdlm_sft_v2 import CustomForwardSFTTrainer
from mdlm_sft.mdlm.mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler


"""
MDLM trainer with explicit Diffusion-of-Thoughts (DoT) data contract.

Subclass of `CustomForwardSFTTrainer` that adds ONE thing: enforcement of the
DoT src_mask contract at the trainer boundary. All MDLM math (corruption,
schedule-weighted CE, NELBO/bpd/ppl accumulation, reproducible eval noise,
token-weighted metrics, prediction_step, evaluate-reset) is inherited unchanged.

DoT contract (see HKUNLP/diffusion-of-thoughts train.py L309):
    nll_loss_mask = attn_mask & (~src_mask).squeeze(-1)

In MDLM-on-SFTTrainer terms this means:
    maskable positions  ==  (attention_mask & ~src_mask)
                        ==  (labels != -100)        # collator-provided

Both sides MUST agree. If they don't, the collator is mislabeled and silent
training corruption would follow — we fail loudly instead.
"""

from typing import Any, Optional

import torch



class MDLMDoTTrainer(CustomForwardSFTTrainer):
    # ------------------------------------------------------------------
    # Step 1 — shell.
    # ------------------------------------------------------------------
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Step 2 — compute_loss override.
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        if "src_mask" in inputs:
            self._assert_dot_contract(inputs)
            # Parent's model(**...) call only consumes input_ids/attention_mask,
            # but strip src_mask defensively so no downstream code is surprised
            # by an unexpected kwarg if the parent ever changes.
            inputs = {k: v for k, v in inputs.items() if k != "src_mask"}

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    # ------------------------------------------------------------------
    # Contract check.
    # ------------------------------------------------------------------
    @staticmethod
    def _assert_dot_contract(inputs: dict) -> None:
        labels         = inputs["labels"]
        src_mask       = inputs["src_mask"].bool()
        attention_mask = inputs.get("attention_mask", None)

        # Shape sanity — cheap, catches collator bugs early.
        if src_mask.shape != labels.shape:
            raise AssertionError(
                f"DoT contract: src_mask shape {tuple(src_mask.shape)} "
                f"!= labels shape {tuple(labels.shape)}. "
                f"Check collator output."
            )

        # Derive the DoT-faithful maskable mask.
        if attention_mask is not None:
            dot_maskable = attention_mask.bool() & ~src_mask
        else:
            dot_maskable = ~src_mask

        labels_maskable = labels != -100

        if torch.equal(dot_maskable, labels_maskable):
            return

        # Detailed diagnostic — split the disagreement so the user knows
        # which side of the contract is wrong.
        n_total           = dot_maskable.numel()
        n_mismatch        = (dot_maskable != labels_maskable).sum().item()
        n_train_not_lbl   = (dot_maskable & ~labels_maskable).sum().item()
        n_lbl_not_train   = (~dot_maskable & labels_maskable).sum().item()
        raise AssertionError(
            "DoT contract violated: (attention_mask & ~src_mask) disagrees "
            "with (labels != -100).\n"
            f"  positions total:                                  {n_total}\n"
            f"  positions disagreeing:                            {n_mismatch}\n"
            f"  in (attn & ~src) but labels==-100 (under-trained): {n_train_not_lbl}\n"
            f"  in labels!=-100 but src or pad (over-trained):     {n_lbl_not_train}\n"
            "Fix: ensure the collator sets labels=-100 on EXACTLY "
            "(src_mask | ~attention_mask) positions."
        )


#### COLLATE FN 
# raw row                tokenized row                       batch dict
# ─────────              ─────────────                       ──────────
# {                      {                                   {
#   "src": str,    ──►     "src_ids": List[int],     ──►       "input_ids":      LongTensor [B, L],
#   "tgt": List[str]       "tgt_ids_list": List[List[int]]     "attention_mask": BoolTensor [B, L],
# }                      }                                     "src_mask":       BoolTensor [B, L],
#                                                              "labels":         LongTensor [B, L],
#                                                            }
#    (dataset)        Dataset.map(tokenize_fn)            cot_collate_fn(batch)        


_DEFAULT_RNG = random.Random()

def cot_collate_fn(
    batch: list[dict],
    *,
    sep_id: int,
    pad_id: int,
    eos_id: int,
    cot: bool = True,
    seq_len: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> dict[str, torch.Tensor]:
    rng = rng if rng is not None else _DEFAULT_RNG

    sequences: list[torch.Tensor] = []
    src_lens:  list[int] = []

    for row in batch:
        # Defensive copies — never mutate caller-owned lists.
        prompt_ids:     list[int]       = list(row["prompt_ids"])
        chunks_ids:     list[list[int]] = [list(c) for c in row["rationale_chunks_ids"]]
        completion_ids: list[int]       = list(row["completion_ids"])

        assert len(completion_ids) > 0, (
            "DoT collator: completion_ids is empty — filter such rows in data prep."
        )
        n_chunks = len(chunks_ids)

        # ── Pick stage k ∈ [0, n_chunks]. k == n_chunks means "predict completion".
        if cot:
            k = rng.randint(0, n_chunks)   # inclusive on both ends
        else:
            k = n_chunks                   # always predict completion

        # ── Build src = prompt + chunks[:k] ──────────────────────────────────
        src_ids: list[int] = list(prompt_ids)
        for prev in chunks_ids[:k]:
            src_ids.extend(prev)

        # ── Pick tgt + EOS rule ──────────────────────────────────────────────
        if k < n_chunks:
            # Predicting an intermediate reasoning step. No EOS — the rationale
            # isn't done, and inference will continue with another step.
            tgt_ids = chunks_ids[k]
            append_eos = False
        else:
            # Predicting the final answer. EOS marks "rationale + answer done."
            # In no-cot mode we additionally pull all rationale chunks into tgt.
            if cot:
                tgt_ids = list(completion_ids)
            else:
                tgt_ids = [tok for c in chunks_ids for tok in c] + completion_ids
            append_eos = True

        if append_eos:
            tgt_ids = tgt_ids + [eos_id]

        # ── seq_len truncation: tgt is sacred, src is left-trimmed, +1 for SEP ─
        if seq_len is not None:
            tgt_ids = tgt_ids[: seq_len - 1]
            src_budget = seq_len - len(tgt_ids) - 1
            if src_budget <= 0:
                src_ids = []
            elif src_budget < len(src_ids):
                src_ids = src_ids[-src_budget:]

        seq = src_ids + [sep_id] + tgt_ids
        sequences.append(torch.tensor(seq, dtype=torch.int64))
        src_lens.append(len(src_ids) + 1)   # +1: SEP belongs to src side

    # ── Pad to BATCH max ────────────────────────────────────────────────────
    input_ids = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
    B, L = input_ids.shape

    pos        = torch.arange(L).unsqueeze(0).expand(B, -1)
    lengths    = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    src_lens_t = torch.tensor(src_lens,                    dtype=torch.long)

    attention_mask = pos < lengths.unsqueeze(1)
    src_mask       = pos < src_lens_t.unsqueeze(1)

    labels = input_ids.clone()
    labels[src_mask | ~attention_mask] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "src_mask":       src_mask,
        "labels":         labels,
    }


def _dot_map_fn(example: dict[str, str | list[str]], tokenizer: PreTrainedTokenizer) -> dict[str, list[int] | list[list[int]]]:
    prompt_ids        = tokenizer(example["prompt"],     add_special_tokens=False).input_ids
    completion_ids    = tokenizer(example["completion"], add_special_tokens=False).input_ids
    chunks_ids        = [tokenizer(c, add_special_tokens=False).input_ids
                         for c in example["rationale_chunks"]]
    return {
        "prompt_ids":           prompt_ids,
        "rationale_chunks_ids": chunks_ids,
        "completion_ids":       completion_ids,
    }

# ─── end-to-end smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    from datasets import Dataset, Features, Sequence, Value

    # ── 1. Tiny vocabulary + mock tokenizer ─────────────────────────────────
    _WORDS = [
        "[PAD]", "[EOS]", "[SEP]",
        "What", "is", "2", "3", "5", "+", "?",
        "First", "note", "that", "Second", "add", "the",
        "numbers", "together", "answer", "So", "result", ".",
    ]
    _tok2id = {w: i for i, w in enumerate(_WORDS)}

    PAD_ID = _tok2id["[PAD]"]   # 0
    EOS_ID = _tok2id["[EOS]"]   # 1
    SEP_ID = _tok2id["[SEP]"]   # 2

    class _MockTokenizer:
        """Whitespace-split tokenizer; unknown words map to PAD (0)."""
        class _Enc:
            def __init__(self, ids: list[int]) -> None:
                self.input_ids = ids

        def __call__(self, text: str, add_special_tokens: bool = False) -> "_Enc":
            return self._Enc([_tok2id.get(w, PAD_ID) for w in text.split()])

    tokenizer = _MockTokenizer()

    # ── 2. Synthetic raw rows ────────────────────────────────────────────────
    raw_rows = [
        {   # two reasoning steps — full CoT path
            "prompt": "What is 2 + 3 ?",
            "rationale_chunks": [
                "First note that 2 + 3",
                "Second add the numbers together .",
            ],
            "completion": "So the answer is 5 .",
        },
        {   # one reasoning step
            "prompt": "What is 5 + 2 ?",
            "rationale_chunks": [
                "First note that 5 + 2",
            ],
            "completion": "the result is 5 .",
        },
        {   # zero reasoning steps — edge case: predict completion directly
            "prompt": "What is 3 + 2 ?",
            "rationale_chunks": [],
            "completion": "the answer is 5 .",
        },
    ]

    # ── 3. Dataset.from_list + tokenize ─────────────────────────────────────
    # Declare explicit Arrow features so nested Sequence(Sequence(...)) is
    # round-tripped correctly even when inner lists differ in length.
    tok_features = Features({
        "prompt_ids":           Sequence(Value("int64")),
        "rationale_chunks_ids": Sequence(Sequence(Value("int64"))),
        "completion_ids":       Sequence(Value("int64")),
    })

    ds = Dataset.from_list(raw_rows)
    ds = ds.map(
        lambda ex: _dot_map_fn(ex, tokenizer),
        features=tok_features,
        remove_columns=["prompt", "rationale_chunks", "completion"],
    )
    ds = ds.with_format("python")   # return Python lists, not Arrow scalars

    print("── Tokenised dataset ────────────────────────────────────────────")
    for i in range(len(ds)):
        row = ds[i]
        print(
            f"  row {i} | prompt_ids={row['prompt_ids']}"
            f" | n_chunks={len(row['rationale_chunks_ids'])}"
            f" | completion_ids={row['completion_ids']}"
        )

    # ── 4. Run the collator ──────────────────────────────────────────────────
    rng   = random.Random(42)           # fixed seed → reproducible stage k
    batch = [ds[i] for i in range(len(ds))]

    out = cot_collate_fn(
        batch,
        sep_id  = SEP_ID,
        pad_id  = PAD_ID,
        eos_id  = EOS_ID,
        cot     = True,
        seq_len = None,
        rng     = rng,
    )

    B, L = out["input_ids"].shape
    print(f"\n── Batch: {B} rows × {L} positions ─────────────────────────────")
    for key, tensor in out.items():
        print(f"  {key:15s}: shape={tuple(tensor.shape)}  dtype={tensor.dtype}")

    print("\n── Per-row trainable-token breakdown ────────────────────────────")
    for i in range(B):
        n_train = (out["labels"][i] != -100).sum().item()
        n_seq   = out["attention_mask"][i].sum().item()
        n_src   = out["src_mask"][i].sum().item()
        print(
            f"  row {i}: src={n_src:2d}  target={n_train:2d}  "
            f"pad={L - n_seq:2d}  total_seq={n_seq:2d}/{L}"
        )

    # ── 5. Correctness assertions ────────────────────────────────────────────
    src_or_pad = out["src_mask"] | ~out["attention_mask"]
    target_pos = out["attention_mask"] & ~out["src_mask"]

    # A: labels == -100 on every src/pad position
    assert (out["labels"][src_or_pad] == -100).all(), (
        "FAIL A: labels should be -100 on src/pad positions"
    )

    # B: labels hold real token ids on every target position
    assert (out["labels"][target_pos] != -100).all(), (
        "FAIL B: labels should hold real token ids on target positions"
    )

    # C: every row has at least one trainable token
    for i in range(B):
        n = (out["labels"][i] != -100).sum().item()
        assert n > 0, f"FAIL C: row {i} has zero trainable tokens"

    # D: full DoT contract (mirrors MDLMDoTTrainer._assert_dot_contract)
    #    This is the exact check the trainer runs at training time.
    MDLMDoTTrainer._assert_dot_contract({
        "labels":         out["labels"],
        "src_mask":       out["src_mask"],
        "attention_mask": out["attention_mask"],
    })

    print("\n✓  All assertions passed — collator satisfies the DoT contract.")