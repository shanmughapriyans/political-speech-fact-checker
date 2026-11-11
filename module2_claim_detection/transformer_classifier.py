"""
transformer_classifier.py  --  Fine-tune distilroberta-base for CFS detection
==============================================================================
Trains a DistilRoBERTa sequence classifier on the ClaimBuster debates corpus
and produces a scorer that plugs directly into evaluate.py.

Architecture choices
--------------------
  Model     : distilroberta-base (82 M params, 6 layers)
              Fast enough for CPU experimentation; upgrade to roberta-base
              or deberta-v3-small with --model for better accuracy.
  Labels    : 2-class binary (CFS=1 / Non-CFS=0)
  Loss      : Weighted cross-entropy (handles ~3:1 class imbalance)
  Max length: 128 tokens for sentence-only; 192 with --add-context
  Epochs    : 3 (typical sweet-spot; watch val F1 per epoch)
  LR        : 2e-5 with linear warmup over 6% of training steps

GPU vs CPU
----------
The script auto-detects device and adjusts batch size / gradient accumulation
so effective batch size stays near 32 regardless:
  GPU  : per_device_train_batch_size=16, grad_accum=2
  CPU  : per_device_train_batch_size=4,  grad_accum=8   (slow but works)

Run on Colab
------------
See the comment block at the bottom of this file for the Colab setup
commands (one cell to paste and run).

Outputs
-------
  models/transformer/          -- best HuggingFace checkpoint
  models/transformer_config.json -- training metadata + val results
  plots/pr_curve_*_transformer.png  (via evaluate.py)

Usage
-----
  # Train on local machine / Colab GPU:
  python transformer_classifier.py

  # Larger model:
  python transformer_classifier.py --model roberta-base

  # Add preceding sentence as context:
  python transformer_classifier.py --add-context

  # Evaluate a saved checkpoint against the harness:
  python transformer_classifier.py --eval-only

  # Score a single sentence:
  python transformer_classifier.py --predict "We consume 50% of cocaine."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
TRANSFORMER_DIR = MODELS_DIR / "transformer"
CONFIG_PATH = MODELS_DIR / "transformer_config.json"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL   = "distilroberta-base"
MAX_LEN_SENT    = 128    # sentence only
MAX_LEN_CONTEXT = 192    # sentence + preceding context
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-5
WARMUP_RATIO    = 0.06
WEIGHT_DECAY    = 0.01
TRAIN_CUTOFF    = "2012"
VAL_YEAR        = "2016"


# ---------------------------------------------------------------------------
# Data helpers (mirrors evaluate.py / baseline_classifier.py)
# ---------------------------------------------------------------------------

def _to_binary(verdict_series):
    return (verdict_series == 1).astype(int)


def _load_split(split: str):
    import pandas as pd, re

    def year(fid):
        m = re.search(r"\b(\d{4})\b", str(fid))
        return m.group(1) if m else "0000"

    if split == "test":
        df = pd.read_csv(DATA_DIR / "groundtruth.csv")
        df["label"] = _to_binary(df["Verdict"])
        return df

    cs = pd.read_csv(DATA_DIR / "crowdsourced.csv")
    cs["year"]  = cs["File_id"].apply(year)
    cs["label"] = _to_binary(cs["Verdict"])

    if split == "train":
        return cs[cs["year"] <= TRAIN_CUTOFF].reset_index(drop=True)
    return cs[cs["year"].str.startswith(VAL_YEAR)].reset_index(drop=True)


def _build_texts(df, add_context: bool) -> list[str]:
    """Build input strings, optionally prepending the previous sentence as context."""
    if not add_context or "context" not in df.columns:
        return df["Text"].tolist()
    texts = []
    for _, row in df.iterrows():
        ctx = str(row.get("context", "") or "").strip()
        sent = str(row["Text"]).strip()
        texts.append(f"{ctx} {sent}".strip() if ctx else sent)
    return texts


# ---------------------------------------------------------------------------
# HuggingFace Dataset wrapper
# ---------------------------------------------------------------------------

class ClaimDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Weighted-loss Trainer
# ---------------------------------------------------------------------------

class WeightedTrainer:
    """Thin wrapper: returns a HuggingFace Trainer subclass with weighted loss."""

    @staticmethod
    def build(class_weights: torch.Tensor, **trainer_kwargs):
        from transformers import Trainer

        weights = class_weights  # captured in closure

        class _WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss_fct = CrossEntropyLoss(
                    weight=weights.to(outputs.logits.device)
                )
                loss = loss_fct(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        return _WeightedTrainer(**trainer_kwargs)


# ---------------------------------------------------------------------------
# Compute metrics (called by Trainer each epoch)
# ---------------------------------------------------------------------------

def _make_compute_metrics(threshold: float = 0.5):
    from sklearn.metrics import f1_score, precision_score, recall_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        preds  = (probs >= threshold).astype(int)
        return {
            "f1":        round(f1_score(labels, preds, zero_division=0), 4),
            "precision": round(precision_score(labels, preds, zero_division=0), 4),
            "recall":    round(recall_score(labels, preds, zero_division=0), 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_name: str = DEFAULT_MODEL,
    add_context: bool = False,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> None:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device.upper()}")
    print(f"Model  : {model_name}")
    print(f"Context: {'yes' if add_context else 'no'}")
    print()

    # ---- data ----
    print("Loading data ...")
    df_train = _load_split("train")
    df_val   = _load_split("val")
    print(f"  train={len(df_train):,}  val={len(df_val):,}")

    max_len = MAX_LEN_CONTEXT if add_context else MAX_LEN_SENT

    # ---- tokenise ----
    print(f"Tokenising (max_len={max_len}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts = _build_texts(df_train, add_context)
    val_texts   = _build_texts(df_val,   add_context)

    enc_train = tokenizer(train_texts, truncation=True, padding=True,
                          max_length=max_len, return_tensors="pt")
    enc_val   = tokenizer(val_texts,   truncation=True, padding=True,
                          max_length=max_len, return_tensors="pt")

    train_ds = ClaimDataset(enc_train, df_train["label"].tolist())
    val_ds   = ClaimDataset(enc_val,   df_val["label"].tolist())

    # ---- class weights ----
    n_neg = (df_train["label"] == 0).sum()
    n_pos = (df_train["label"] == 1).sum()
    w_neg = (n_neg + n_pos) / (2.0 * n_neg)
    w_pos = (n_neg + n_pos) / (2.0 * n_pos)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
    print(f"Class weights: Non-CFS={w_neg:.3f}  CFS={w_pos:.3f}")

    # ---- model ----
    print(f"Loading {model_name} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # ---- batch size + grad accum (maintain ~32 effective batch) ----
    if device == "cuda":
        per_device_bs = 16
        grad_accum    = 2
    else:
        per_device_bs = 4
        grad_accum    = 8
        print("  [CPU mode] Training will be slow. Consider Colab for GPU.")

    total_steps   = (len(train_ds) // (per_device_bs * grad_accum)) * num_epochs
    warmup_steps  = max(1, int(total_steps * WARMUP_RATIO))

    training_args = TrainingArguments(
        output_dir                  = str(TRANSFORMER_DIR),
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = per_device_bs,
        per_device_eval_batch_size  = 32,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = learning_rate,
        warmup_steps                = warmup_steps,
        weight_decay                = WEIGHT_DECAY,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_steps               = 50,
        save_total_limit            = 2,
        report_to                   = "none",
        no_cuda                     = (device == "cpu"),
        dataloader_num_workers      = 0,   # avoids Windows multiprocessing issues
    )

    trainer = WeightedTrainer.build(
        class_weights    = class_weights,
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        compute_metrics  = _make_compute_metrics(),
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\nTraining for up to {num_epochs} epochs "
          f"(early-stop patience=2 on val F1) ...")
    t0 = time.monotonic()
    trainer.train()
    elapsed = time.monotonic() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min.")

    # ---- save best checkpoint + metadata ----
    trainer.save_model(str(TRANSFORMER_DIR))
    tokenizer.save_pretrained(str(TRANSFORMER_DIR))

    meta = {
        "model_name":    model_name,
        "add_context":   add_context,
        "max_len":       max_len,
        "num_epochs":    num_epochs,
        "learning_rate": learning_rate,
        "device":        device,
        "train_seconds": round(elapsed, 1),
        "val_history":   [
            {k: v for k, v in log.items() if "eval" in k or "epoch" in k}
            for log in trainer.state.log_history
            if "eval_f1" in log
        ],
    }
    CONFIG_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Checkpoint saved : {TRANSFORMER_DIR}")
    print(f"Config saved     : {CONFIG_PATH}")

    # ---- auto-evaluate on test split ----
    print("\nRunning evaluate.py harness on test split ...")
    _evaluate_saved(add_context=add_context)


# ---------------------------------------------------------------------------
# Inference / scorer
# ---------------------------------------------------------------------------

def load_transformer_scorer(
    checkpoint_dir: str | Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
):
    """Return a scorer callable compatible with evaluate.py.

    Parameters
    ----------
    checkpoint_dir : path to the saved HF checkpoint.
                     Defaults to models/transformer/.
    batch_size : int
        Inference batch size.
    device : str | None
        "cuda", "cpu", or None (auto-detect).

    Returns
    -------
    callable : texts: list[str] -> list[float]  (P(CFS) in [0,1])
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cp_dir = Path(checkpoint_dir or TRANSFORMER_DIR)
    if not cp_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {cp_dir}. "
            "Run:  python transformer_classifier.py"
        )

    # Load metadata for max_len / add_context
    meta = {}
    if CONFIG_PATH.exists():
        meta = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    max_len     = meta.get("max_len", MAX_LEN_SENT)
    add_context = meta.get("add_context", False)

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading transformer scorer from {cp_dir}  [device={_device}]")

    tokenizer = AutoTokenizer.from_pretrained(str(cp_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(cp_dir))
    model.eval()
    model.to(_device)

    def scorer(texts: list[str]) -> list[float]:
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = tokenizer(
                batch, truncation=True, padding=True,
                max_length=max_len, return_tensors="pt"
            )
            enc = {k: v.to(_device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
        return all_probs

    return scorer


# ---------------------------------------------------------------------------
# Evaluate saved checkpoint through the harness
# ---------------------------------------------------------------------------

def _evaluate_saved(add_context: bool = False) -> None:
    from evaluate import evaluate, load_dataset, compare
    from baseline_classifier import load_model, predict_score

    scorer_t = load_transformer_scorer()

    all_results = []
    for split in ["val", "test"]:
        df = load_dataset(split)
        r  = evaluate(
            scorer     = scorer_t,
            dataset    = df,
            split_name = f"Transformer / {split.capitalize()}",
            save_plot  = True,
            plot_tag   = "transformer",
        )
        all_results.append(r)

    # Also pull baseline results for comparison
    print("\nLoading baseline for side-by-side comparison ...")
    builder, clf = load_model()
    baseline_scorer = lambda texts: [predict_score(t, builder, clf) for t in texts]
    for split in ["val", "test"]:
        df = load_dataset(split)
        r  = evaluate(
            scorer     = baseline_scorer,
            dataset    = df,
            split_name = f"Baseline / {split.capitalize()}",
            save_plot  = False,
            verbose    = False,
        )
        all_results.append(r)

    print("\n=== Model comparison ===")
    compare(all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune distilroberta-base for CFS detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transformer_classifier.py                    # train with defaults
  python transformer_classifier.py --model roberta-base
  python transformer_classifier.py --add-context      # sentence + context
  python transformer_classifier.py --eval-only        # skip training
  python transformer_classifier.py --predict "GDP grew by 2.1 percent."
        """,
    )
    p.add_argument("--model",        default=DEFAULT_MODEL,
                   help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    p.add_argument("--add-context",  action="store_true",
                   help="Prepend preceding sentence as context")
    p.add_argument("--epochs",       type=int, default=NUM_EPOCHS)
    p.add_argument("--lr",           type=float, default=LEARNING_RATE)
    p.add_argument("--eval-only",    action="store_true",
                   help="Skip training; evaluate existing checkpoint")
    p.add_argument("--predict",      metavar="TEXT",
                   help="Score a single sentence (requires saved checkpoint)")
    p.add_argument("--threshold",    type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.predict:
        scorer = load_transformer_scorer()
        score  = scorer([args.predict])[0]
        label  = "CFS" if score >= args.threshold else "Non-CFS"
        print(f"Text   : {args.predict!r}")
        print(f"Score  : {score:.4f}")
        print(f"Label  : {label}  (threshold={args.threshold})")

    elif args.eval_only:
        _evaluate_saved()

    else:
        train(
            model_name  = args.model,
            add_context = args.add_context,
            num_epochs  = args.epochs,
            learning_rate = args.lr,
        )


# ---------------------------------------------------------------------------
# ==========================================================================
# COLAB SETUP — paste this into a Colab cell and run
# ==========================================================================
#
# !pip install transformers datasets accelerate scikit-learn pysbd -q
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# import shutil, os
# PROJECT = "/content/drive/MyDrive/fact_verification"   # adjust path
# os.makedirs(f"{PROJECT}/module2_claim_detection/data", exist_ok=True)
#
# # Upload crowdsourced.csv and groundtruth.csv to the data/ folder, then:
# os.chdir(f"{PROJECT}/module2_claim_detection")
#
# !python transformer_classifier.py --model distilroberta-base --epochs 3
#
# # After training, download models/transformer/ back to your laptop.
# # Or keep it on Drive and point load_transformer_scorer() at the Drive path.
# ==========================================================================
