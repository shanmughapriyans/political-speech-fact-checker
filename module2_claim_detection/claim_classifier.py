"""
claim_classifier.py  --  Unified inference wrapper for Module 2
===============================================================
Every downstream component (pipeline, server, tests) calls exactly
one method::

    classifier.score(text, context="") -> float   # P(CFS) in [0, 1]

The concrete implementation (baseline TF-IDF or fine-tuned transformer)
is decided at construction time.  Everything downstream is oblivious to
which model is running, so you can swap them with zero other changes.

Classes
-------
ClaimClassifier          -- abstract base (protocol definition)
BaselineClaimClassifier  -- wraps TF-IDF + LogisticRegression
TransformerClaimClassifier -- wraps a saved HuggingFace checkpoint
ScriptedClaimClassifier  -- dependency-injection fake for unit tests
                            (same pattern as ScriptedWhisperModel in Module 1)

Factory
-------
::

    from claim_classifier import build_classifier

    clf = build_classifier("baseline")     # TF-IDF + LR
    clf = build_classifier("transformer")  # HuggingFace checkpoint
    clf = build_classifier("scripted", scores=[0.9, 0.1, 0.8])

Latency
-------
Each call to ``.score()`` records wall-clock time.  Call
``.latency_stats()`` to get mean / p95 / p99 latency in milliseconds.
This number goes directly into the end-to-end latency budget in the
final report.

Checkpoint (run this file directly)
------------------------------------
::

    python claim_classifier.py              # latency benchmark, both models
    python claim_classifier.py --selftest   # all unit tests with Scripted fake
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ClaimClassifier(ABC):
    """Abstract interface every classifier implementation must satisfy."""

    @abstractmethod
    def score(self, text: str, context: str = "") -> float:
        """Return P(CFS) in [0.0, 1.0] for a single sentence.

        Parameters
        ----------
        text : str
            The sentence to score.
        context : str
            Optional preceding sentence(s) as context.  Implementations
            that do not use context must silently ignore it.
        """

    def score_batch(
        self,
        texts: List[str],
        contexts: Optional[List[str]] = None,
    ) -> List[float]:
        """Score a list of sentences.  Default: loop over ``score()``.

        Subclasses may override with a vectorised implementation.
        """
        ctxs = contexts or [""] * len(texts)
        return [self.score(t, c) for t, c in zip(texts, ctxs)]

    # ------------------------------------------------------------------
    # Latency tracking
    # ------------------------------------------------------------------

    def _init_latency(self) -> None:
        self._latency_ms: list[float] = []

    def _record(self, elapsed_s: float) -> None:
        if hasattr(self, "_latency_ms"):
            self._latency_ms.append(elapsed_s * 1000.0)

    def latency_stats(self) -> dict:
        """Return latency statistics (ms) over all ``.score()`` calls so far."""
        if not getattr(self, "_latency_ms", []):
            return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0,
                    "p95_ms": 0.0, "p99_ms": 0.0}
        arr = np.array(self._latency_ms)
        return {
            "n":       len(arr),
            "mean_ms": round(float(arr.mean()), 2),
            "p50_ms":  round(float(np.percentile(arr, 50)), 2),
            "p95_ms":  round(float(np.percentile(arr, 95)), 2),
            "p99_ms":  round(float(np.percentile(arr, 99)), 2),
        }

    def reset_latency(self) -> None:
        """Clear latency history."""
        self._latency_ms = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Baseline: TF-IDF + LogisticRegression
# ---------------------------------------------------------------------------

class BaselineClaimClassifier(ClaimClassifier):
    """Wraps the trained TF-IDF + LogisticRegression model.

    The model is loaded **once** at construction time from the joblib
    artefacts produced by ``baseline_classifier.py``.

    Parameters
    ----------
    model_dir : str | None
        Directory containing ``baseline_vectorizer.joblib`` and
        ``baseline_classifier.joblib``.  Defaults to ``models/``.
    """

    def __init__(self, model_dir: str | None = None) -> None:
        from pathlib import Path
        import baseline_classifier as bc

        self._bc = bc

        if model_dir:
            from pathlib import Path as _P
            bc.VECTORIZER_PATH = _P(model_dir) / "baseline_vectorizer.joblib"
            bc.CLASSIFIER_PATH = _P(model_dir) / "baseline_classifier.joblib"

        print("Loading baseline model ...")
        t0 = time.monotonic()
        self._builder, self._clf = bc.load_model()
        elapsed = time.monotonic() - t0
        print(f"  Baseline model loaded in {elapsed:.2f}s")
        self._init_latency()

    def score(self, text: str, context: str = "") -> float:
        t0  = time.monotonic()
        val = self._bc.predict_score(text, self._builder, self._clf)
        self._record(time.monotonic() - t0)
        return val

    def score_batch(
        self,
        texts: List[str],
        contexts: Optional[List[str]] = None,
    ) -> List[float]:
        """Vectorised batch scoring (one TF-IDF transform for the whole batch)."""
        import pandas as pd
        t0  = time.monotonic()
        ctxs = contexts or [""] * len(texts)
        df = pd.DataFrame({"Text": texts, "Sentiment": [0.0] * len(texts)})
        X  = self._builder.transform(df)
        probs = self._clf.predict_proba(X)
        cfs_idx = list(self._clf.classes_).index(1)
        results = [float(p[cfs_idx]) for p in probs]
        per_sample = (time.monotonic() - t0) / max(len(texts), 1)
        for _ in texts:
            self._record(per_sample)
        return results

    def __repr__(self) -> str:
        return "BaselineClaimClassifier(TF-IDF + LogisticRegression)"


# ---------------------------------------------------------------------------
# Transformer: HuggingFace checkpoint
# ---------------------------------------------------------------------------

class TransformerClaimClassifier(ClaimClassifier):
    """Wraps a saved HuggingFace sequence-classification checkpoint.

    Parameters
    ----------
    checkpoint_dir : str | None
        Path to the saved checkpoint.  Defaults to ``models/transformer/``.
    device : str | None
        ``"cuda"``, ``"cpu"``, or None (auto-detect).
    batch_size : int
        Sentences per forward pass in ``score_batch()``.
    add_context : bool | None
        If True, prepend ``context`` to ``text`` with a separator.
        If None, read from ``models/transformer_config.json``.
    """

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        add_context: bool | None = None,
    ) -> None:
        import json
        from pathlib import Path

        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        cp_dir = Path(checkpoint_dir or "models/transformer")
        if not cp_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint at {cp_dir}. "
                "Run: python transformer_classifier.py"
            )

        # Load training metadata
        cfg_path = Path("models/transformer_config.json")
        meta = {}
        if cfg_path.exists():
            meta = json.loads(cfg_path.read_text(encoding="utf-8"))

        self._max_len     = meta.get("max_len", 128)
        self._add_context = (
            add_context if add_context is not None
            else meta.get("add_context", False)
        )
        self._batch_size  = batch_size

        self._device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading transformer from {cp_dir}  [device={self._device}] ...")
        t0 = time.monotonic()
        self._tokenizer = AutoTokenizer.from_pretrained(str(cp_dir))
        self._model     = AutoModelForSequenceClassification.from_pretrained(
            str(cp_dir)
        )
        self._model.eval()
        self._model.to(self._device)
        elapsed = time.monotonic() - t0
        print(f"  Transformer loaded in {elapsed:.2f}s")
        self._init_latency()

    def _build_input(self, text: str, context: str) -> str:
        if self._add_context and context:
            return f"{context.strip()} {text.strip()}"
        return text.strip()

    def score(self, text: str, context: str = "") -> float:
        import torch
        inp = self._build_input(text, context)
        t0  = time.monotonic()
        enc = self._tokenizer(
            inp, truncation=True, padding=True,
            max_length=self._max_len, return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        self._record(time.monotonic() - t0)
        return float(prob)

    def score_batch(
        self,
        texts: List[str],
        contexts: Optional[List[str]] = None,
    ) -> List[float]:
        import torch
        ctxs    = contexts or [""] * len(texts)
        inputs  = [self._build_input(t, c) for t, c in zip(texts, ctxs)]
        results = []
        for i in range(0, len(inputs), self._batch_size):
            batch = inputs[i : i + self._batch_size]
            t0    = time.monotonic()
            enc   = self._tokenizer(
                batch, truncation=True, padding=True,
                max_length=self._max_len, return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            per_sample = (time.monotonic() - t0) / max(len(batch), 1)
            for p in probs:
                results.append(float(p))
                self._record(per_sample)
        return results

    def __repr__(self) -> str:
        return (
            f"TransformerClaimClassifier("
            f"device={self._device}, max_len={self._max_len}, "
            f"context={self._add_context})"
        )


# ---------------------------------------------------------------------------
# Scripted fake  (dependency-injection for unit tests)
# ---------------------------------------------------------------------------

class ScriptedClaimClassifier(ClaimClassifier):
    """Returns pre-programmed scores in sequence.

    Mirrors the ``ScriptedWhisperModel`` pattern from Module 1.
    Useful for unit tests that need deterministic classifier behaviour
    without loading any model.

    Parameters
    ----------
    scores : list[float] | Iterator[float]
        Scores to return, in order.  When exhausted, raises StopIteration
        unless ``loop=True``.
    loop : bool
        If True, cycle through ``scores`` indefinitely.

    Example
    -------
    ::

        fake = ScriptedClaimClassifier(scores=[0.9, 0.1, 0.8], loop=True)
        assert fake.score("any text") == 0.9
        assert fake.score("any text") == 0.1
        assert fake.score("any text") == 0.8
        assert fake.score("any text") == 0.9   # loops
    """

    def __init__(
        self,
        scores: list[float] | None = None,
        loop: bool = False,
    ) -> None:
        self._scores = list(scores or [0.5])
        self._loop   = loop
        self._idx    = 0
        self._init_latency()

    def score(self, text: str, context: str = "") -> float:
        if self._idx >= len(self._scores):
            if self._loop:
                self._idx = 0
            else:
                raise StopIteration(
                    "ScriptedClaimClassifier: score list exhausted"
                )
        val = self._scores[self._idx]
        self._idx += 1
        self._record(0.0)   # zero latency for the fake
        return val

    def reset(self) -> None:
        """Rewind the score sequence to the beginning."""
        self._idx = 0

    def __repr__(self) -> str:
        return f"ScriptedClaimClassifier(scores={self._scores}, loop={self._loop})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_classifier(
    backend: str = "baseline",
    **kwargs,
) -> ClaimClassifier:
    """Instantiate and return a ClaimClassifier.

    Parameters
    ----------
    backend : {'baseline', 'transformer', 'scripted'}
    **kwargs
        Passed to the concrete constructor.

    Returns
    -------
    ClaimClassifier
    """
    backends = {
        "baseline":    BaselineClaimClassifier,
        "transformer": TransformerClaimClassifier,
        "scripted":    ScriptedClaimClassifier,
    }
    if backend not in backends:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from: {list(backends)}"
        )
    return backends[backend](**kwargs)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_selftest() -> None:
    print("=== ClaimClassifier self-test ===\n")

    # -- 1. ScriptedClaimClassifier basic behaviour --
    print("-- Test 1: ScriptedClaimClassifier basic sequence --")
    fake = ScriptedClaimClassifier(scores=[0.9, 0.1, 0.8])
    assert fake.score("anything") == 0.9
    assert fake.score("anything") == 0.1
    assert fake.score("anything") == 0.8
    try:
        fake.score("anything")
        assert False, "Should have raised StopIteration"
    except StopIteration:
        pass
    print("   [PASS] sequence exhaustion -> StopIteration")

    # -- 2. loop=True --
    print("-- Test 2: ScriptedClaimClassifier loop=True --")
    fake = ScriptedClaimClassifier(scores=[0.7, 0.3], loop=True)
    vals = [fake.score("x") for _ in range(6)]
    assert vals == [0.7, 0.3, 0.7, 0.3, 0.7, 0.3], vals
    print(f"   [PASS] looped values: {vals}")

    # -- 3. score_batch uses score() --
    print("-- Test 3: score_batch --")
    fake = ScriptedClaimClassifier(scores=[0.6, 0.4, 0.9], loop=False)
    batch = fake.score_batch(["a", "b", "c"])
    assert batch == [0.6, 0.4, 0.9], batch
    print(f"   [PASS] batch result: {batch}")

    # -- 4. reset() --
    print("-- Test 4: reset() --")
    fake = ScriptedClaimClassifier(scores=[0.5, 0.5])
    fake.score("x")
    fake.reset()
    assert fake.score("x") == 0.5
    print("   [PASS] reset works")

    # -- 5. latency_stats() --
    print("-- Test 5: latency_stats() --")
    fake = ScriptedClaimClassifier(scores=[0.5] * 10, loop=True)
    for _ in range(10):
        fake.score("x")
    stats = fake.latency_stats()
    assert stats["n"] == 10
    assert stats["mean_ms"] == 0.0   # Scripted records 0 ms
    print(f"   [PASS] latency_stats: {stats}")

    # -- 6. build_classifier factory --
    print("-- Test 6: build_classifier('scripted') --")
    clf = build_classifier("scripted", scores=[0.88], loop=True)
    assert isinstance(clf, ScriptedClaimClassifier)
    assert clf.score("test") == 0.88
    print("   [PASS] factory returns correct type")

    # -- 7. Plugs into evaluate.py scorer protocol --
    print("-- Test 7: scorer protocol compatible with evaluate.py --")
    fake = ScriptedClaimClassifier(scores=[0.9, 0.1, 0.8, 0.3], loop=True)
    scorer = lambda texts: fake.score_batch(texts)
    result = scorer(["claim A", "not claim", "claim B", "opinion"])
    assert len(result) == 4
    assert all(isinstance(s, float) for s in result)
    print(f"   [PASS] scorer protocol: {[round(s,2) for s in result]}")

    print("\n[PASS] All self-tests passed.")


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def _run_benchmark() -> None:
    print("\n=== ClaimClassifier latency benchmark ===\n")

    test_sentences = [
        "The unemployment rate fell to 3.5 percent last month.",
        "I think that is a really nice idea.",
        "The U.S. exported 2.3 billion dollars worth of goods.",
        "We should all try to be kinder to each other.",
        "GDP grew by 2.1 percent in the third quarter.",
    ] * 20   # 100 sentences

    # Baseline benchmark
    try:
        print("Baseline classifier:")
        clf = BaselineClaimClassifier()
        clf.reset_latency()
        scores = clf.score_batch(test_sentences)
        stats  = clf.latency_stats()
        print(f"  {len(test_sentences)} sentences scored")
        print(f"  mean={stats['mean_ms']:.2f}ms  "
              f"p95={stats['p95_ms']:.2f}ms  "
              f"p99={stats['p99_ms']:.2f}ms")
        print(f"  Sample scores: {[round(s, 3) for s in scores[:5]]}")
    except Exception as e:
        print(f"  [SKIP] Baseline not available: {e}")

    print()

    # Transformer benchmark (only if checkpoint exists)
    from pathlib import Path
    if Path("models/transformer").exists():
        try:
            print("Transformer classifier:")
            clf = TransformerClaimClassifier()
            clf.reset_latency()
            scores = clf.score_batch(test_sentences)
            stats  = clf.latency_stats()
            print(f"  {len(test_sentences)} sentences scored")
            print(f"  mean={stats['mean_ms']:.2f}ms  "
                  f"p95={stats['p95_ms']:.2f}ms  "
                  f"p99={stats['p99_ms']:.2f}ms")
            print(f"  Sample scores: {[round(s, 3) for s in scores[:5]]}")
        except Exception as e:
            print(f"  [SKIP] Transformer not available: {e}")
    else:
        print("Transformer classifier: [SKIP] no checkpoint at models/transformer/")
        print("  Run: python transformer_classifier.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="ClaimClassifier inference wrapper")
    p.add_argument("--selftest",  action="store_true",
                   help="Run all unit tests with ScriptedClaimClassifier")
    p.add_argument("--benchmark", action="store_true",
                   help="Run latency benchmark (default if no flag given)")
    p.add_argument("--backend",   default="baseline",
                   choices=["baseline", "transformer"],
                   help="Which model to benchmark")
    p.add_argument("--predict",   metavar="TEXT",
                   help="Score a single sentence")
    p.add_argument("--context",   default="",
                   help="Preceding sentence for --predict")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    if args.selftest:
        _run_selftest()
    elif args.predict:
        clf   = build_classifier(args.backend)
        score = clf.score(args.predict, args.context)
        label = "CFS" if score >= args.threshold else "Non-CFS"
        print(f"Text    : {args.predict!r}")
        print(f"Context : {args.context!r}")
        print(f"Score   : {score:.4f}")
        print(f"Label   : {label}  (threshold={args.threshold})")
        print(f"Latency : {clf.latency_stats()['mean_ms']:.2f} ms")
    else:
        _run_selftest()
        _run_benchmark()
