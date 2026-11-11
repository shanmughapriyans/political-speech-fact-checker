"""
evaluate.py  --  Universal evaluation harness for Module 2 classifiers
=======================================================================
One function to rule them all.  Every model -- baseline TF-IDF,
fine-tuned DeBERTa, ensemble -- plugs into ``evaluate()`` and produces
the same metric dict so comparisons are perfectly fair.

Interface contract
------------------
A *scorer* is any callable with the signature::

    scorer(texts: list[str]) -> list[float]   # returns P(CFS) in [0, 1]

The harness is intentionally stateless: it never calls ``fit()``,
never loads files, never trains.  It only calls ``scorer(texts)``
and crunches the numbers.

Outputs
-------
``evaluate()`` returns a dict with keys:

    precision, recall, f1          -- at the chosen threshold
    best_f1_threshold              -- argmax F1 across the PR curve
    best_f1_precision              -- precision at best-F1 threshold
    best_f1_recall                 -- recall at best-F1 threshold
    target_precision_threshold     -- lowest threshold where precision
                                      >= target_precision (default 0.70)
    target_precision_recall        -- recall achieved there
    auc_pr                         -- area under the PR curve
    support_pos, support_neg       -- class sizes
    pr_curve_path                  -- path to the saved PNG (or None)

Usage
-----
    from evaluate import evaluate, load_dataset
    from baseline_classifier import predict_score, load_model

    builder, clf = load_model()
    scorer = lambda texts: [predict_score(t, builder, clf) for t in texts]

    df_test  = load_dataset("groundtruth")
    results  = evaluate(scorer, df_test, split_name="Baseline / Test",
                        save_plot=True)
    print(results)

CLI
---
    python evaluate.py --split test       # baseline on groundtruth
    python evaluate.py --split val        # baseline on 2016 debates
    python evaluate.py --split train      # sanity-check
    python evaluate.py --all              # all three splits in one pass
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")   # no display needed; saves to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _to_binary(verdict_series: pd.Series) -> pd.Series:
    return (verdict_series == 1).astype(int)


_SPLIT_YEAR_MAP = {
    "train": (None, "2012"),
    "val":   ("2016", "2016"),
    "test":  None,               # groundtruth.csv
}


def load_dataset(split: str) -> pd.DataFrame:
    """Load a named split from the standard data files.

    Parameters
    ----------
    split : {'train', 'val', 'test'}
        - ``'train'``  -- crowdsourced debates up to and including 2012
        - ``'val'``    -- crowdsourced 2016 debates
        - ``'test'``   -- groundtruth.csv (expert labels, held-out)

    Returns
    -------
    pd.DataFrame with columns: Text, Sentiment, label, File_id
    """
    import re

    def _year(fid: str) -> str:
        m = re.search(r"\b(\d{4})\b", str(fid))
        return m.group(1) if m else "0000"

    if split == "test":
        df = pd.read_csv(DATA_DIR / "groundtruth.csv")
        df["label"] = _to_binary(df["Verdict"])
        return df

    cs = pd.read_csv(DATA_DIR / "crowdsourced.csv")
    cs["year"]  = cs["File_id"].apply(_year)
    cs["label"] = _to_binary(cs["Verdict"])

    lo, hi = _SPLIT_YEAR_MAP[split]
    if lo and hi:
        mask = (cs["year"] >= lo) & (cs["year"] <= hi)
    elif hi:
        mask = cs["year"] <= hi
    else:
        mask = pd.Series([True] * len(cs))

    return cs[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    scorer: Callable[[list[str]], list[float]],
    dataset: pd.DataFrame,
    split_name: str = "Evaluation",
    threshold: float = 0.5,
    target_precision: float = 0.70,
    batch_size: int = 256,
    save_plot: bool = True,
    plot_tag: str = "",
    verbose: bool = True,
) -> dict:
    """Evaluate a scorer on a dataset and return a metrics dict.

    Parameters
    ----------
    scorer : callable
        ``scorer(texts: list[str]) -> list[float]``
        Returns P(CFS) in [0, 1] for each text.  Can be a lambda wrapping
        any model -- sklearn, HuggingFace, rule-based, etc.
    dataset : pd.DataFrame
        Must have columns ``'Text'`` and ``'label'`` (0=Non-CFS, 1=CFS).
    split_name : str
        Human-readable name for print output and plot titles.
    threshold : float
        Decision threshold for the primary metrics row.  Default 0.5.
    target_precision : float
        Find the lowest threshold that achieves at least this precision,
        then report the recall at that point.  Useful for "high-precision"
        operating points where false alarms are costly.
    batch_size : int
        Texts per ``scorer`` call.  Matters for transformer scorers.
    save_plot : bool
        If True, save a precision-recall curve PNG to plots/.
    plot_tag : str
        Optional string appended to the plot filename for disambiguation.
    verbose : bool
        If True, print a formatted report to stdout.

    Returns
    -------
    dict
        See module docstring for key list.
    """
    texts  = dataset["Text"].tolist()
    y_true = dataset["label"].values

    # ---- run scorer in batches ----
    t0     = time.monotonic()
    scores = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        scores.extend(scorer(batch))
    elapsed = time.monotonic() - t0

    scores  = np.array(scores, dtype=float)
    y_pred  = (scores >= threshold).astype(int)

    # ---- primary metrics at chosen threshold ----
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    # ---- PR curve ----
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, scores)
    # pr_prec / pr_rec have len = len(thresholds) + 1; align them
    pr_f1 = np.where(
        (pr_prec[:-1] + pr_rec[:-1]) > 0,
        2 * pr_prec[:-1] * pr_rec[:-1] / (pr_prec[:-1] + pr_rec[:-1]),
        0.0,
    )
    auc_pr = auc(pr_rec, pr_prec)

    # ---- best-F1 threshold ----
    best_f1_idx       = int(np.argmax(pr_f1))
    best_f1_thresh    = float(pr_thresh[best_f1_idx])
    best_f1_prec      = float(pr_prec[best_f1_idx])
    best_f1_rec       = float(pr_rec[best_f1_idx])
    best_f1_val       = float(pr_f1[best_f1_idx])

    # ---- target-precision threshold ----
    high_prec_mask = pr_prec[:-1] >= target_precision
    if high_prec_mask.any():
        # Among thresholds achieving target_precision, pick highest recall
        cand_recall = np.where(high_prec_mask, pr_rec[:-1], -1.0)
        tp_idx      = int(np.argmax(cand_recall))
        tp_thresh   = float(pr_thresh[tp_idx])
        tp_rec      = float(pr_rec[tp_idx])
        tp_prec     = float(pr_prec[tp_idx])
    else:
        tp_thresh = 1.0
        tp_rec    = 0.0
        tp_prec   = 0.0

    # ---- support ----
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    # ---- confusion matrix values ----
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    # ---- plot ----
    plot_path = None
    if save_plot:
        plot_path = _save_pr_curve(
            pr_prec, pr_rec, pr_thresh,
            auc_pr=auc_pr,
            threshold=threshold,
            best_f1_thresh=best_f1_thresh,
            tp_thresh=tp_thresh,
            split_name=split_name,
            plot_tag=plot_tag,
        )

    result = {
        # primary metrics
        "split":              split_name,
        "threshold":          threshold,
        "precision":          round(prec, 4),
        "recall":             round(rec, 4),
        "f1":                 round(f1, 4),
        # best-F1 operating point
        "best_f1_threshold":  round(best_f1_thresh, 4),
        "best_f1_precision":  round(best_f1_prec, 4),
        "best_f1_recall":     round(best_f1_rec, 4),
        "best_f1":            round(best_f1_val, 4),
        # high-precision operating point
        "target_precision":              target_precision,
        "target_precision_threshold":    round(tp_thresh, 4),
        "target_precision_achieved":     round(tp_prec, 4),
        "target_precision_recall":       round(tp_rec, 4),
        # curve summary
        "auc_pr":             round(float(auc_pr), 4),
        # dataset info
        "n_samples":          len(y_true),
        "support_pos":        n_pos,
        "support_neg":        n_neg,
        # confusion matrix
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        # perf
        "scorer_seconds":     round(elapsed, 2),
        "ms_per_sample":      round(elapsed / max(len(texts), 1) * 1000, 2),
        # artefact
        "pr_curve_path":      str(plot_path) if plot_path else None,
    }

    if verbose:
        _print_report(result)

    return result


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _print_report(r: dict) -> None:
    """Print a formatted evaluation report to stdout."""
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {r['split']}")
    print(sep)
    print(f"  Samples  : {r['n_samples']:,}  "
          f"(CFS={r['support_pos']:,} / Non-CFS={r['support_neg']:,})")
    print()
    print(f"  -- At threshold = {r['threshold']:.3f} --")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1 (CFS)  : {r['f1']:.4f}")
    print()
    print(f"  -- Best-F1 operating point --")
    print(f"  Threshold : {r['best_f1_threshold']:.4f}")
    print(f"  Precision : {r['best_f1_precision']:.4f}")
    print(f"  Recall    : {r['best_f1_recall']:.4f}")
    print(f"  F1 (CFS)  : {r['best_f1']:.4f}")
    print()
    print(f"  -- High-precision operating point "
          f"(target precision >= {r['target_precision']:.2f}) --")
    print(f"  Threshold : {r['target_precision_threshold']:.4f}")
    print(f"  Precision : {r['target_precision_achieved']:.4f}")
    print(f"  Recall    : {r['target_precision_recall']:.4f}")
    print()
    print(f"  AUC-PR    : {r['auc_pr']:.4f}")
    print()
    print(f"  Confusion matrix (threshold={r['threshold']:.2f}):")
    print(f"                 Pred Non-CFS   Pred CFS")
    print(f"  True Non-CFS      {r['tn']:>7d}    {r['fp']:>7d}")
    print(f"  True CFS          {r['fn']:>7d}    {r['tp']:>7d}")
    print()
    print(f"  Scored {r['n_samples']:,} samples in {r['scorer_seconds']:.2f}s "
          f"({r['ms_per_sample']:.1f} ms/sample)")
    if r["pr_curve_path"]:
        print(f"  PR curve : {r['pr_curve_path']}")
    print(sep)


# ---------------------------------------------------------------------------
# PR curve plot
# ---------------------------------------------------------------------------

def _save_pr_curve(
    pr_prec, pr_rec, pr_thresh,
    auc_pr: float,
    threshold: float,
    best_f1_thresh: float,
    tp_thresh: float,
    split_name: str,
    plot_tag: str,
) -> Path:
    """Save a precision-recall curve PNG and return its path."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Main curve
    ax.plot(pr_rec, pr_prec, lw=2, color="#4C72B0",
            label=f"PR curve  (AUC={auc_pr:.3f})")

    # Annotate the default threshold point
    _mark_threshold(ax, pr_prec, pr_rec, pr_thresh, threshold,
                    color="#DD8452", label=f"threshold={threshold:.2f}")

    # Annotate best-F1 threshold
    _mark_threshold(ax, pr_prec, pr_rec, pr_thresh, best_f1_thresh,
                    color="#55A868", label=f"best-F1 @ {best_f1_thresh:.2f}")

    # Annotate high-precision threshold
    if tp_thresh < 1.0:
        _mark_threshold(ax, pr_prec, pr_rec, pr_thresh, tp_thresh,
                        color="#C44E52", label=f"high-prec @ {tp_thresh:.2f}")

    # ISO-F1 curves (light grey)
    for iso_f1 in [0.3, 0.5, 0.7]:
        x = np.linspace(0.01, 1.0, 200)
        y = iso_f1 * x / (2 * x - iso_f1 + 1e-9)
        mask = (y >= 0) & (y <= 1)
        ax.plot(x[mask], y[mask], "--", lw=0.8, color="grey", alpha=0.5)
        ax.text(x[mask][-1] + 0.01, y[mask][-1],
                f"F1={iso_f1}", fontsize=7, color="grey", va="center")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve\n{split_name}", fontsize=13)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    tag = ("_" + plot_tag.replace(" ", "_")) if plot_tag else ""
    safe_name = split_name.lower().replace(" ", "_").replace("/", "_")
    plot_path = PLOTS_DIR / f"pr_curve_{safe_name}{tag}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def _mark_threshold(ax, pr_prec, pr_rec, pr_thresh, thresh, color, label):
    """Place a dot on the PR curve at a specific threshold."""
    if len(pr_thresh) == 0:
        return
    idx = int(np.searchsorted(pr_thresh, thresh, side="left"))
    idx = min(idx, len(pr_thresh) - 1)
    ax.plot(pr_rec[idx], pr_prec[idx], "o", color=color,
            markersize=8, label=label, zorder=5)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare(results: list[dict], metric: str = "f1") -> pd.DataFrame:
    """Pretty-print a side-by-side comparison table of multiple evaluate() results.

    Parameters
    ----------
    results : list[dict]
        Each element is the return value of ``evaluate()``.
    metric : str
        Primary sort column.  Default ``'f1'``.

    Returns
    -------
    pd.DataFrame
        Comparison table (also printed to stdout).
    """
    rows = []
    for r in results:
        rows.append({
            "Model / Split":    r["split"],
            "Threshold":        r["threshold"],
            "Precision":        r["precision"],
            "Recall":           r["recall"],
            "F1 (CFS)":         r["f1"],
            "Best-F1":          r["best_f1"],
            "Best-F1 thresh":   r["best_f1_threshold"],
            "AUC-PR":           r["auc_pr"],
            "ms/sample":        r["ms_per_sample"],
        })
    col_map = {"f1": "F1 (CFS)", "precision": "Precision",
               "recall": "Recall", "auc_pr": "AUC-PR"}
    sort_col = col_map.get(metric, metric)
    df = pd.DataFrame(rows).sort_values(sort_col, ascending=False)
    print("\n" + df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluation harness")
    p.add_argument("--split", choices=["train", "val", "test"], default="test",
                   help="Which dataset split to evaluate on (default: test)")
    p.add_argument("--all",   action="store_true",
                   help="Evaluate on all three splits and print comparison table")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold (default 0.5)")
    p.add_argument("--target-precision", type=float, default=0.70,
                   help="Target precision for high-precision op point (default 0.70)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip saving the PR curve PNG")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load baseline scorer
    from baseline_classifier import load_model, predict_score

    print("Loading baseline model ...")
    builder, clf = load_model()

    def scorer(texts: list[str]) -> list[float]:
        return [predict_score(t, builder, clf) for t in texts]

    splits = ["train", "val", "test"] if args.all else [args.split]
    all_results = []

    for split in splits:
        print(f"\nLoading dataset: {split!r} ...")
        df = load_dataset(split)
        tag = "baseline"
        result = evaluate(
            scorer=scorer,
            dataset=df,
            split_name=f"Baseline / {split.capitalize()}",
            threshold=args.threshold,
            target_precision=args.target_precision,
            save_plot=not args.no_plot,
            plot_tag=tag,
        )
        all_results.append(result)

    if args.all and len(all_results) > 1:
        print("\n\n=== Comparison table ===")
        compare(all_results)
