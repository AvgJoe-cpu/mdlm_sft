"""
Minimal, faithful re-implementation of Swayamdipta et al. (2020),
"Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics".

Per-(example, checkpoint) signal: mean_token_accuracy ∈ [0, 1]
  confidence  := mean over checkpoints
  variability := stdev over checkpoints
  correctness := fraction of checkpoints with signal > tau   (default tau=0.5)

Per-sample-ness is preserved end-to-end:
  - compute_cartography returns one row per idx.
  - cartography.parquet contains every example.
  - plot_cartography draws every example by default; subsampling is opt-in
    (rendering concession only, identical to the paper's "25K shown for clarity").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SIGNAL_COL = "mean_token_accuracy"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "idx" not in df.columns or SIGNAL_COL not in df.columns:
        raise ValueError(f"{path} must have 'idx' and {SIGNAL_COL!r}; got {list(df.columns)}")
    return df[["idx", SIGNAL_COL]].rename(columns={SIGNAL_COL: "signal"})


def _check_seeds_match(run_dirs: list[Path]) -> None:
    seeds = []
    for d in run_dirs:
        manifest = d / "manifest.json"
        if not manifest.exists():
            print(f"[warn] {manifest} missing; cannot verify eval_seed.")
            continue
        cfg = json.loads(manifest.read_text()).get("config", {})
        seeds.append(cfg.get("eval_seed"))
    seeds = [s for s in seeds if s is not None]
    if len(set(seeds)) > 1:
        raise RuntimeError(
            f"eval_seed differs across runs ({seeds}); cartography requires the same "
            f"deterministic-eval seed so MC noise cancels across checkpoints."
        )


def compute_cartography(
    per_example_csvs: list[Path],
    *,
    correctness_tau: float = 0.5,
) -> pd.DataFrame:
    """One row per idx: (idx, confidence, variability, correctness). No aggregation across examples."""
    if len(per_example_csvs) < 2:
        raise ValueError("cartography needs at least 2 checkpoints.")

    long = pd.concat(
        [_load_one(p).assign(ckpt=k) for k, p in enumerate(per_example_csvs)],
        ignore_index=True,
    ).dropna(subset=["signal"])

    K = len(per_example_csvs)
    counts = long.groupby("idx")["signal"].size()
    long = long[long["idx"].isin(counts[counts == K].index)]

    return (
        long.groupby("idx")["signal"]
        .agg(
            confidence="mean",
            variability="std",
            correctness=lambda s: float((s > correctness_tau).mean()),
        )
        .reset_index()
    )


def plot_cartography(
    cart: pd.DataFrame,
    out_png: Path,
    *,
    title: str = "Data Map",
    plot_subsample: int | None = None,   # viz-only; None = plot every example
) -> None:
    """Renders Fig. 1 of the paper. Each dot is one example."""
    df = (
        cart if plot_subsample is None or len(cart) <= plot_subsample
        else cart.sample(plot_subsample, random_state=0)
    )

    sns.set_theme(style="whitegrid", context="paper")

    g = sns.JointGrid(data=df, x="variability", y="confidence", height=6, ratio=4)
    g.plot_joint(
        sns.scatterplot, hue=df["correctness"],
        palette="RdYlBu", s=12, alpha=0.7, edgecolor=None, legend="brief",
    )
    g.plot_marginals(sns.histplot, bins=40, color="0.4")

    g.ax_joint.set_xlim(0.0, max(0.5, df["variability"].max() * 1.05))
    g.ax_joint.set_ylim(0.0, 1.0)

    g.ax_joint.text(0.02, 0.97, "easy-to-learn", fontsize=9, va="top",
                    transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.02, 0.03, "hard-to-learn", fontsize=9, va="bottom",
                    transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.97, 0.50, "ambiguous", fontsize=9, va="center", ha="right",
                    transform=g.ax_joint.transAxes)

    g.figure.suptitle(title, y=1.02)
    g.figure.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(g.figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_example_csv", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--title", default="Data Map")
    p.add_argument("--correctness_tau", type=float, default=0.5)
    p.add_argument("--require_same_seed", action="store_true")
    p.add_argument("--plot_subsample", type=int, default=None,
                   help="Optional viz-only cap on plotted points (parquet always has all).")
    args = p.parse_args()

    csvs = [Path(x) for x in args.per_example_csv]
    if args.require_same_seed:
        _check_seeds_match([c.parent for c in csvs])

    cart = compute_cartography(csvs, correctness_tau=args.correctness_tau)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cart.to_parquet(out_dir / "cartography.parquet", index=False)
    plot_cartography(cart, out_dir / "cartography.png", title=args.title,
                     plot_subsample=args.plot_subsample)
    print(f"wrote {out_dir/'cartography.parquet'} ({len(cart)} examples) "
          f"and {out_dir/'cartography.png'} ({len(csvs)} checkpoints)")


# ---------------------------------------------------------------------------
# Self-contained demo
# ---------------------------------------------------------------------------
def _demo():
    """
    Fabricate 3 checkpoints with 4 hand-crafted example archetypes so the
    three regions (plus a 'drifting' group) are visually obvious:

      idx 000..199  easy-to-learn :  signal ~0.95 at every checkpoint      (high μ, low σ)
      idx 200..299  hard-to-learn :  signal ~0.05 at every checkpoint      (low μ, low σ)
      idx 300..399  ambiguous     :  signal swings 0.15 / 0.85 / 0.20      (mid μ, high σ)
      idx 400..499  improving     :  signal drifts 0.20 → 0.50 → 0.80      (mid μ, mid σ)
    """
    import json
    import tempfile
    import numpy as np

    rng = np.random.default_rng(0)

    def make_csv(out_csv: Path, k: int) -> None:
        n_easy, n_hard, n_amb, n_imp = 200, 100, 100, 100
        n = n_easy + n_hard + n_amb + n_imp

        easy_centers = [0.95, 0.95, 0.95]
        hard_centers = [0.05, 0.05, 0.05]
        amb_centers  = [0.15, 0.85, 0.20]
        imp_centers  = [0.20, 0.50, 0.80]

        easy = rng.normal(easy_centers[k], 0.02, n_easy).clip(0, 1)
        hard = rng.normal(hard_centers[k], 0.02, n_hard).clip(0, 1)
        amb  = rng.normal(amb_centers[k],  0.08, n_amb ).clip(0, 1)
        imp  = rng.normal(imp_centers[k],  0.05, n_imp ).clip(0, 1)
        signal = np.concatenate([easy, hard, amb, imp])

        df = pd.DataFrame({
            "idx":                 np.arange(n),
            "prompt":              [f"prompt_{i}" for i in range(n)],
            "nll":                 -np.log(signal.clip(1e-3)),
            "bpd":                 -np.log2(signal.clip(1e-3)),
            "ppl":                 1.0 / signal.clip(1e-3),
            "entropy":             rng.uniform(1.0, 4.0, n),
            "mean_token_accuracy": signal,
        })
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dirs = [tmp / f"ckpt_{k}" for k in range(3)]
        csvs = []
        for k, run_dir in enumerate(run_dirs):
            csv = run_dir / "per_example.csv"
            make_csv(csv, k)
            (run_dir / "manifest.json").write_text(
                json.dumps({"config": {"eval_seed": 42, "num_mc_samples": 8}})
            )
            csvs.append(csv)

        _check_seeds_match(run_dirs)
        cart = compute_cartography(csvs, correctness_tau=0.5)

        # Per-archetype assertions: each lands where the construction predicts.
        easy = cart[cart["idx"] < 200]
        hard = cart[(cart["idx"] >= 200) & (cart["idx"] < 300)]
        amb  = cart[(cart["idx"] >= 300) & (cart["idx"] < 400)]
        imp  = cart[(cart["idx"] >= 400) & (cart["idx"] < 500)]

        assert easy["confidence"].mean() > 0.90, easy["confidence"].mean()
        assert easy["variability"].mean() < 0.05, easy["variability"].mean()

        assert hard["confidence"].mean() < 0.10, hard["confidence"].mean()
        assert hard["variability"].mean() < 0.05, hard["variability"].mean()

        assert amb["variability"].mean() > 0.25, amb["variability"].mean()

        assert 0.40 < imp["confidence"].mean() < 0.60, imp["confidence"].mean()
        assert imp["variability"].mean() > 0.20, imp["variability"].mean()

        # Render to a stable, inspectable location.
        out_dir = Path("eval_runs/_demo_cartography")
        out_dir.mkdir(parents=True, exist_ok=True)
        cart.to_parquet(out_dir / "cartography.parquet", index=False)
        plot_cartography(cart, out_dir / "cartography.png",
                         title="Demo: 4 archetypes across 3 checkpoints")
        print(f"[demo] all archetype assertions passed")
        print(f"[demo] wrote {out_dir/'cartography.parquet'} ({len(cart)} examples) "
              f"and {out_dir/'cartography.png'}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        sys.argv.remove("--demo")
        _demo()
    else:
        main()