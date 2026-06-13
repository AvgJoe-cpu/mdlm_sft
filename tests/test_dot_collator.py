"""
Contract tests for cot_collate_fn under the A1 schema.

Scenarios:
  1. Shapes & dtypes
  2. DoT contract: labels==-100 iff (src_mask | ~attention_mask)
  3. SEP placement: token at src_lens[i]-1 is sep_id
  4. No-mutation + RNG determinism
  5. cot k=0           → src = prompt, tgt = chunks[0],     no EOS
  6. cot k=mid         → src = prompt + chunks[:mid], tgt = chunks[mid], no EOS
  7. cot k=N           → src = prompt + all chunks, tgt = completion + EOS
  8. cot=False         → src = prompt, tgt = all chunks + completion + EOS
  9. Empty chunks list → only k=0 reachable; tgt = completion + EOS
 10. Empty completion  → assertion error (loud failure)
 11. seq_len trunc: tgt preserved, src left-trimmed, total == seq_len
 12. Padding: pad cells hold pad_id and have attention_mask == False
 13. datasets.Dataset.from_list rows accepted

Run:
    python -m tests.test_cot_collator
"""
from __future__ import annotations

import random
import torch
from datasets import Dataset

from mdlm_sft.mdlm.cot_collator import cot_collate_fn


SEP_ID, PAD_ID, EOS_ID = 1, 2, 3


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _row(prompt_ids, chunks_ids_list, completion_ids):
    return {
        "prompt_ids":           prompt_ids,
        "rationale_chunks_ids": chunks_ids_list,
        "completion_ids":       completion_ids,
    }

def _dummy_batch():
    """3 rows; varying chunk counts INCLUDING zero-chunk case."""
    return [
        _row([10, 11],     [[20, 21], [22, 23]],          [50, 51]),
        _row([12, 13, 14], [[30, 31, 32], [33, 34], [35]],[52, 53, 54]),
        _row([15],         [],                            [55]),       # no rationale
    ]

class _ForcedRNG:
    """Pin k to a specific value for testing."""
    def __init__(self, k): self.k = k
    def randint(self, a, b):
        assert a <= self.k <= b, f"_ForcedRNG: k={self.k} out of [{a},{b}]"
        return self.k


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def _s1_shapes():
    out = cot_collate_fn(_dummy_batch(), sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         rng=random.Random(0))
    B = 3; L = out["input_ids"].shape[1]
    for k, want in (("input_ids", torch.int64), ("labels", torch.int64),
                    ("attention_mask", torch.bool), ("src_mask", torch.bool)):
        assert out[k].shape == (B, L) and out[k].dtype == want, k
    print(f"[1] PASS  shapes [{B},{L}] + dtypes correct")

def _s2_dot_contract():
    out = cot_collate_fn(_dummy_batch(), sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         rng=random.Random(0))
    nontrain = out["src_mask"] | ~out["attention_mask"]
    want = torch.where(nontrain, torch.full_like(out["input_ids"], -100), out["input_ids"])
    assert torch.equal(out["labels"], want)
    print(f"[2] PASS  labels == -100 iff (src_mask | ~attention_mask)")

def _s3_sep_placement():
    out = cot_collate_fn(_dummy_batch(), sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         rng=random.Random(0))
    src_lens = out["src_mask"].sum(dim=1).tolist()
    for i, sl in enumerate(src_lens):
        assert out["input_ids"][i, sl - 1].item() == SEP_ID
    print(f"[3] PASS  SEP at src_lens[i]-1 for all rows")

def _s4_no_mutation_and_determinism():
    batch = _dummy_batch()
    snap  = [(list(r["prompt_ids"]),
              [list(c) for c in r["rationale_chunks_ids"]],
              list(r["completion_ids"])) for r in batch]
    a = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID, rng=random.Random(7))
    b = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID, rng=random.Random(7))
    for k in a: assert torch.equal(a[k], b[k]), k
    for i, (p, cs, comp) in enumerate(snap):
        assert batch[i]["prompt_ids"]           == p
        assert batch[i]["rationale_chunks_ids"] == cs
        assert batch[i]["completion_ids"]       == comp
    print(f"[4] PASS  no mutation + RNG determinism")

def _s5_k_zero():
    batch = [_row([10, 11], [[20, 21], [22, 23], [24]], [50, 51])]
    out = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         rng=_ForcedRNG(0))
    want = torch.tensor([[10, 11, SEP_ID, 20, 21]], dtype=torch.int64)  # no EOS (k<N)
    assert torch.equal(out["input_ids"], want), out["input_ids"].tolist()
    print(f"[5] PASS  cot k=0 → tgt=chunks[0], no EOS")

def _s6_k_mid():
    batch = [_row([10, 11], [[20, 21], [22, 23], [24]], [50, 51])]
    out = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         rng=_ForcedRNG(1))
    want = torch.tensor([[10, 11, 20, 21, SEP_ID, 22, 23]], dtype=torch.int64)
    assert torch.equal(out["input_ids"], want), out["input_ids"].tolist()
    print(f"[6] PASS  cot k=1 → src includes chunks[0], tgt=chunks[1], no EOS")

def _s7_k_completion():
    batch = [_row([10, 11], [[20, 21], [22, 23], [24]], [50, 51])]
    out = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         rng=_ForcedRNG(3))   # k = N = 3 → completion
    want = torch.tensor([[10, 11, 20, 21, 22, 23, 24, SEP_ID, 50, 51, EOS_ID]],
                        dtype=torch.int64)
    assert torch.equal(out["input_ids"], want), out["input_ids"].tolist()
    print(f"[7] PASS  cot k=N → tgt=completion+EOS")

def _s8_no_cot():
    batch = [_row([10, 11], [[20, 21], [22, 23], [24]], [50, 51])]
    out = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         cot=False, rng=random.Random(0))
    want = torch.tensor([[10, 11, SEP_ID, 20, 21, 22, 23, 24, 50, 51, EOS_ID]],
                        dtype=torch.int64)
    assert torch.equal(out["input_ids"], want), out["input_ids"].tolist()
    print(f"[8] PASS  no-cot → tgt=concat(chunks)+completion+EOS")

def _s9_empty_chunks():
    batch = [_row([10, 11], [], [50, 51])]
    out = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         rng=_ForcedRNG(0))   # only valid value
    want = torch.tensor([[10, 11, SEP_ID, 50, 51, EOS_ID]], dtype=torch.int64)
    assert torch.equal(out["input_ids"], want), out["input_ids"].tolist()
    print(f"[9] PASS  empty chunks → tgt=completion+EOS")

def _s10_empty_completion_raises():
    batch = [_row([10], [[20]], [])]
    raised = False
    try:
        cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                       rng=random.Random(0))
    except AssertionError as e:
        raised = True
        assert "completion_ids is empty" in str(e)
    assert raised
    print(f"[10] PASS  empty completion → AssertionError raised")

def _s11_seq_len_trunc():
    long_prompt = list(range(100, 120))
    batch = [_row(long_prompt, [[200]], [201, 202])]
    out = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         seq_len=10, rng=_ForcedRNG(1))   # k=N → tgt=completion+EOS = [201,202,EOS] (3 toks)
    # budget: tgt=3, SEP=1, src budget=6. src = prompt + chunks[:1] = 100..119 + [200] (21 toks).
    # left-trim to last 6 → [115,116,117,118,119,200]
    want = torch.tensor([[115, 116, 117, 118, 119, 200, SEP_ID, 201, 202, EOS_ID]],
                        dtype=torch.int64)
    assert out["input_ids"].shape[1] == 10
    assert torch.equal(out["input_ids"], want), out["input_ids"].tolist()
    print(f"[11] PASS  seq_len=10: tgt preserved, src left-trimmed")

def _s12_padding():
    out = cot_collate_fn(_dummy_batch(), sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                         rng=random.Random(0))
    pad_pos = ~out["attention_mask"]
    if pad_pos.any():
        toks = out["input_ids"][pad_pos]
        assert (toks == PAD_ID).all()
        print(f"[12] PASS  {pad_pos.sum().item()} pad cells, all = PAD_ID")
    else:
        print(f"[12] N/A   no padding needed")

def _s13_from_list():
    ds    = Dataset.from_list(_dummy_batch())
    batch = [ds[i] for i in range(len(ds))]
    out   = cot_collate_fn(batch, sep_id=SEP_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                           rng=random.Random(0))
    assert out["input_ids"].shape[0] == len(batch)
    print(f"[13] PASS  Dataset.from_list rows accepted")


def run_cot_collator_tests():
    _s1_shapes()
    _s2_dot_contract()
    _s3_sep_placement()
    _s4_no_mutation_and_determinism()
    _s5_k_zero()
    _s6_k_mid()
    _s7_k_completion()
    _s8_no_cot()
    _s9_empty_chunks()
    _s10_empty_completion_raises()
    _s11_seq_len_trunc()
    _s12_padding()
    _s13_from_list()
    print("\nAll cot_collate_fn (A1 schema) contract tests passed ✓")


if __name__ == "__main__":
    run_cot_collator_tests()