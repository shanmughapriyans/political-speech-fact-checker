"""
baseline_classifier.py  --  Module 2 TF-IDF + Logistic Regression baseline
============================================================================
Mirrors the original ClaimBuster approach: classical handcrafted features
plus a linear model.  This gives a concrete F1 number to beat when the
transformer classifier is introduced.

Dataset (ClaimBuster debates corpus)
-------------------------------------
  data/crowdsourced.csv   -- 22,501 sentences, crowdsourced labels
  data/groundtruth.csv    --  1,032 sentences, expert labels

Verdict labels
  -1  Not claim-worthy (NCS) -- opinions, chit-chat, questions
   0  Uncertain / borderline
   1  Check-worthy Factual Statement (CFS)  <-- what we want to detect

Label strategy: 2-class (binary)
  CFS = 1    (Verdict == 1)
  Non-CFS = 0  (Verdict == -1 or 0)

Split strategy: by debate year (not random row shuffle)
  Train      -- debates up to and including 2012
  Validation -- 2016 debates  (unseen debates, same era)
  Test       -- groundtruth.csv (expert-labelled, held out entirely)

This is the "hard" split a review panel respects -- it tests whether
the model generalises to new debates, not just new sentences from
debates it has already partially seen.

Features
--------
  1. TF-IDF word unigrams + bigrams  (max 50k features)
  2. TF-IDF character 3-5 grams      (max 30k features)
  3. Hand-crafted scalar features (ClaimBuster paper):
       - sentence length (token count)
       - contains a number or percentage
       - contains a named entity (approximated by capital-word ratio)
       - pre-computed sentiment score (already in dataset)

Model
-----
  LogisticRegression with class_weight='balanced' to handle the ~3:1
  imbalance between non-CFS and CFS.

Outputs
-------
  models/baseline_vectorizer.joblib   -- fitted TF-IDF transformer
  models/baseline_classifier.joblib   -- fitted LogisticRegression
  baseline_results.txt                -- F1, precision, recall, confusion matrix

Run
---
  python baseline_classifier.py
  python baseline_classifier.py --predict "We consume 50% of the world's cocaine."
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CROWDSOURCED_CSV = DATA_DIR / "crowdsourced.csv"
GROUNDTRUTH_CSV  = DATA_DIR / "groundtruth.csv"

VECTORIZER_PATH  = MODELS_DIR / "baseline_vectorizer.joblib"
CLASSIFIER_PATH  = MODELS_DIR / "baseline_classifier.joblib"
RESULTS_PATH     = Path(__file__).parent / "baseline_results.txt"

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
# 2-class: CFS (1) vs Non-CFS (0)
def _to_binary(verdict_series: pd.Series) -> pd.Series:
    return (verdict_series == 1).astype(int)


# ---------------------------------------------------------------------------
# Hand-crafted features  (ClaimBuster paper Section 3.2)
# ---------------------------------------------------------------------------
_NUMBER_RE   = re.compile(r'\b\d[\d,\.]*%?\b|\b\d+\s*percent\b', re.I)
_CAPITAL_RE  = re.compile(r'\b[A-Z][a-z]+\b')


def _handcrafted(texts: pd.Series, sentiment: pd.Series) -> np.ndarray:
    """Return an (N, 4) float matrix of hand-crafted features.

    Features
    --------
    0  token_count        -- normalised sentence length
    1  has_number         -- 1 if text contains a digit/percentage
    2  capital_word_ratio -- proxy for named-entity density
    3  sentiment          -- pre-computed sentiment score
    """
    rows = []
    for text, sent in zip(texts, sentiment):
        tokens = text.split()
        n_tok  = len(tokens)
        has_num = float(bool(_NUMBER_RE.search(text)))
        cap_ratio = len(_CAPITAL_RE.findall(text)) / max(n_tok, 1)
        rows.append([
            min(n_tok / 50.0, 1.0),   # normalise to [0, 1]
            has_num,
            cap_ratio,
            float(sent) if pd.notna(sent) else 0.0,
        ])
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading + splitting
# ---------------------------------------------------------------------------
TRAIN_CUTOFF = "2012"   # debates up to and including 2012
VAL_YEAR     = "2016"   # unseen debates (same crowdsourced set)


def _extract_year(file_id: str) -> str:
    """Extract the 4-digit year from a filename like '2012-10-03.txt'."""
    m = re.search(r'\b(\d{4})\b', str(file_id))
    return m.group(1) if m else "0000"


def load_data():
    """Load, merge where appropriate, and return split DataFrames."""
    cs = pd.read_csv(CROWDSOURCED_CSV)
    gt = pd.read_csv(GROUNDTRUTH_CSV)

    cs["year"] = cs["File_id"].apply(_extract_year)
    gt["year"] = gt["File_id"].apply(_extract_year)

    # Binary label
    cs["label"] = _to_binary(cs["Verdict"])
    gt["label"] = _to_binary(gt["Verdict"])

    # Temporal split on crowdsourced data
    train_mask = cs["year"] <= TRAIN_CUTOFF
    val_mask   = cs["year"].str.startswith(VAL_YEAR)

    train = cs[train_mask].reset_index(drop=True)
    val   = cs[val_mask].reset_index(drop=True)
    test  = gt.reset_index(drop=True)

    print(f"Train : {len(train):>6,} rows  "
          f"(CFS={train['label'].sum():,} / {(~train['label'].astype(bool)).sum():,} non-CFS)")
    print(f"Val   : {len(val):>6,} rows  "
          f"(CFS={val['label'].sum():,} / {(~val['label'].astype(bool)).sum():,} non-CFS)")
    print(f"Test  : {len(test):>6,} rows  "
          f"(CFS={test['label'].sum():,} / {(~test['label'].astype(bool)).sum():,} non-CFS)")
    print()

    return train, val, test


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Combines TF-IDF (word + char) and hand-crafted features."""

    def __init__(self):
        self.word_tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.char_tfidf = TfidfVectorizer(
            ngram_range=(3, 5),
            max_features=30_000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="char_wb",
        )
        self.scaler = MaxAbsScaler()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        word_X = self.word_tfidf.fit_transform(df["Text"])
        char_X = self.char_tfidf.fit_transform(df["Text"])
        hand_X = _handcrafted(df["Text"], df["Sentiment"])
        hand_sp = csr_matrix(hand_X)
        combined = hstack([word_X, char_X, hand_sp])
        combined = self.scaler.fit_transform(combined)
        self._fitted = True
        return combined

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform()")
        word_X = self.word_tfidf.transform(df["Text"])
        char_X = self.char_tfidf.transform(df["Text"])
        hand_X = _handcrafted(df["Text"], df["Sentiment"])
        hand_sp = csr_matrix(hand_X)
        combined = hstack([word_X, char_X, hand_sp])
        return self.scaler.transform(combined)

    def transform_text(self, text: str, sentiment: float = 0.0) -> csr_matrix:
        """Transform a single raw text string for inference."""
        df = pd.DataFrame({"Text": [text], "Sentiment": [sentiment]})
        return self.transform(df)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(y_true, y_pred, split_name: str, label_names=("Non-CFS", "CFS")) -> dict:
    print(f"\n{'='*60}")
    print(f"  {split_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"             Pred Non-CFS  Pred CFS")
    print(f"  True Non-CFS   {cm[0,0]:>7d}  {cm[0,1]:>7d}")
    print(f"  True CFS       {cm[1,0]:>7d}  {cm[1,1]:>7d}")

    return {
        "split":     split_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "cm":        cm,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate() -> tuple[FeatureBuilder, LogisticRegression]:
    print("Loading data ...")
    train, val, test = load_data()

    print("Building features ...")
    builder = FeatureBuilder()
    X_train = builder.fit_transform(train)
    X_val   = builder.transform(val)
    X_test  = builder.transform(test)

    y_train = train["label"].values
    y_val   = val["label"].values
    y_test  = test["label"].values

    print(f"Feature matrix shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print()

    print("Training LogisticRegression (class_weight=balanced) ...")
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    results = []
    results.append(_evaluate(y_train, clf.predict(X_train), "TRAIN"))
    results.append(_evaluate(y_val,   clf.predict(X_val),   "VALIDATION (2016 debates)"))
    results.append(_evaluate(y_test,  clf.predict(X_test),  "TEST (groundtruth expert labels)"))

    # Save summary
    lines = ["Baseline Classifier Results\n", "=" * 60 + "\n"]
    lines.append(f"Model  : LogisticRegression  C=1.0  class_weight=balanced\n")
    lines.append(f"Split  : temporal by debate year\n")
    lines.append(f"Train  : up to {TRAIN_CUTOFF}   Val: {VAL_YEAR}   Test: groundtruth.csv\n\n")
    for r in results:
        lines.append(f"{r['split']}\n")
        lines.append(f"  Precision: {r['precision']:.4f}\n")
        lines.append(f"  Recall   : {r['recall']:.4f}\n")
        lines.append(f"  F1 (CFS) : {r['f1']:.4f}\n")
        lines.append(f"  Confusion matrix:\n{r['cm']}\n\n")

    RESULTS_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Save model artefacts
    joblib.dump(builder,   VECTORIZER_PATH)
    joblib.dump(clf,       CLASSIFIER_PATH)
    print(f"Vectorizer saved : {VECTORIZER_PATH}")
    print(f"Classifier saved : {CLASSIFIER_PATH}")

    return builder, clf


# ---------------------------------------------------------------------------
# Inference helper (for server.py / pipeline)
# ---------------------------------------------------------------------------

def load_model() -> tuple["FeatureBuilder", LogisticRegression]:
    """Load saved model artefacts for inference."""
    if not VECTORIZER_PATH.exists() or not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run:  python baseline_classifier.py"
        )
    # joblib pickled FeatureBuilder as '__main__.FeatureBuilder' when
    # baseline_classifier.py was run directly.  We must inject the class
    # into sys.modules['__main__'] so pickle can find it regardless of
    # which script is the current entry point.
    import sys as _sys
    _sys.modules.setdefault("__main__", _sys.modules[__name__])
    _m = _sys.modules["__main__"]
    if not hasattr(_m, "FeatureBuilder"):
        setattr(_m, "FeatureBuilder", FeatureBuilder)
    builder = joblib.load(VECTORIZER_PATH)
    clf     = joblib.load(CLASSIFIER_PATH)
    return builder, clf


def predict_score(text: str, builder: FeatureBuilder = None,
                  clf: LogisticRegression = None) -> float:
    """Return P(CFS) for a single sentence string.

    Parameters
    ----------
    text : str
        The sentence to score.
    builder, clf : optional
        Pre-loaded model objects.  Loaded from disk if not provided.

    Returns
    -------
    float
        P(CFS) in [0.0, 1.0].
    """
    if builder is None or clf is None:
        builder, clf = load_model()
    X = builder.transform_text(text)
    prob = clf.predict_proba(X)[0]
    cfs_idx = list(clf.classes_).index(1)
    return float(prob[cfs_idx])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train or run the baseline CFS classifier")
    p.add_argument("--predict", metavar="TEXT",
                   help="Score a single sentence and exit (requires trained model)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for --predict (default 0.5)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.predict:
        score = predict_score(args.predict)
        label = "CFS" if score >= args.threshold else "Non-CFS"
        print(f"Text   : {args.predict!r}")
        print(f"Score  : {score:.4f}")
        print(f"Label  : {label}  (threshold={args.threshold})")
    else:
        train_and_evaluate()
