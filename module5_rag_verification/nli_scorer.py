"""
nli_scorer.py  —  RoBERTa-MNLI entailment scoring
==================================================
Wraps facebook/bart-large-mnli (or cross-encoder/nli-deberta-v3-small
for a lighter model) to score whether a hypothesis (the claim or answer)
is entailed by a premise (the evidence passage).

Returns three probabilities: entailment, neutral, contradiction.
The VerdictLabel mapping is:
  entailment   → SUPPORTED  (entailment_score = p_entail)
  contradiction → REFUTED    (entailment_score = p_contradict, inverted)
  neutral      → NEI

Offline / no-model mode
-----------------------
If transformers is not installed or the model can't load, falls back to
a keyword-overlap heuristic (good enough for demos, bad for production).

Usage::

    scorer = NLIScorer()
    result = scorer.score(premise="ONS says unemployment is 3.8%.",
                          hypothesis="Unemployment fell to 3.8 percent.")
    # result = {"entailment": 0.91, "neutral": 0.07, "contradiction": 0.02}
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict

log = logging.getLogger("module5.nli")

_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"   # ~180MB; accurate & fast


class NLIScorer:
    """
    NLI-based entailment scorer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        "cpu" or "cuda".  Auto-detects GPU if "auto".
    """

    def __init__(
        self,
        model_name: str = _NLI_MODEL,
        device: str = "auto",
    ) -> None:
        self._pipeline = None
        self._model_name = model_name

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            if device == "auto":
                _device = 0 if torch.cuda.is_available() else -1
            else:
                _device = 0 if device == "cuda" else -1

            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=model_name,
                device=_device,
                hypothesis_template="{}",
            )
            log.info("NLI model loaded: %s (device=%s)", model_name, _device)
        except Exception as exc:
            log.warning("NLI model unavailable (%s) — using keyword heuristic", exc)

    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Score entailment of hypothesis given premise.

        Returns
        -------
        dict with keys: entailment, neutral, contradiction (all in [0,1],
        sum to 1.0).
        """
        if self._pipeline is not None:
            return self._score_model(premise, hypothesis)
        return self._score_heuristic(premise, hypothesis)

    def score_batch(
        self, premises: list[str], hypothesis: str
    ) -> list[Dict[str, float]]:
        """Score one hypothesis against multiple premises."""
        return [self.score(p, hypothesis) for p in premises]

    def best_label(self, scores: Dict[str, float]) -> str:
        """Return the label with the highest score."""
        return max(scores, key=lambda k: scores[k])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _score_model(self, premise: str, hypothesis: str) -> Dict[str, float]:
        try:
            # Use premise as sequence, hypothesis as candidate label
            result = self._pipeline(
                premise,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template=f"This text implies that {hypothesis}" + ". {{}}",
                multi_label=False,
            )
            scores = dict(zip(result["labels"], result["scores"]))
            return {
                "entailment":    round(scores.get("entailment", 0.0), 4),
                "neutral":       round(scores.get("neutral", 0.0), 4),
                "contradiction": round(scores.get("contradiction", 0.0), 4),
            }
        except Exception as exc:
            log.warning("NLI inference failed: %s", exc)
            return self._score_heuristic(premise, hypothesis)

    def _score_heuristic(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Keyword overlap heuristic — no model needed.
        Gives a rough entailment signal for offline demos.
        """
        def _tokens(t: str) -> set[str]:
            return set(re.sub(r"[^\w\s]", " ", t.lower()).split())

        p_tokens = _tokens(premise)
        h_tokens = _tokens(hypothesis)
        # Remove stop words
        stops = {"the", "a", "an", "is", "was", "are", "were", "in", "of",
                 "to", "and", "or", "that", "it", "this", "for", "at", "by"}
        p_tokens -= stops
        h_tokens -= stops

        if not h_tokens:
            return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

        overlap = len(p_tokens & h_tokens) / len(h_tokens)

        # Check for explicit negation
        negation_words = {"not", "no", "never", "fell", "dropped", "declined",
                          "decreased", "lower", "below"}
        p_neg = bool(p_tokens & negation_words)
        h_neg = bool(h_tokens & negation_words)
        contradiction_boost = 0.3 if p_neg != h_neg else 0.0

        entailment = min(0.9, overlap * 1.2)
        contradiction = min(0.8, contradiction_boost + (1 - overlap) * 0.3)
        neutral = max(0.0, 1.0 - entailment - contradiction)

        total = entailment + neutral + contradiction
        return {
            "entailment":    round(entailment / total, 4),
            "neutral":       round(neutral / total, 4),
            "contradiction": round(contradiction / total, 4),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== NLIScorer self-test ===\n")
    scorer = NLIScorer()

    pairs = [
        ("UK unemployment rate fell to 3.8% in January 2024, per ONS.",
         "Unemployment has fallen to 3.8 percent, the lowest in a decade.",
         "entailment"),
        ("Global CO2 emissions rose 1.1% in 2023, a new record.",
         "Carbon emissions fell last year.",
         "contradiction"),
        ("The government announced a new housing policy.",
         "Unemployment is at a 10-year low.",
         "neutral"),
    ]

    all_pass = True
    for premise, hypothesis, expected in pairs:
        scores = scorer.score(premise, hypothesis)
        predicted = max(scores, key=lambda k: scores[k])
        status = "PASS" if predicted == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] expected={expected}, got={predicted}")
        print(f"         scores={scores}")
        print(f"         P: {premise[:60]!r}")
        print(f"         H: {hypothesis[:60]!r}")
        print()

    print("[PASS] All NLI tests passed." if all_pass else "[WARN] Some tests failed (heuristic mode is approximate).")

