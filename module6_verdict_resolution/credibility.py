"""
credibility.py  —  Source-credibility weighting for Module 6
=============================================================
Implements the CONFACT-style credibility weighting approach:
each evidence passage's vote is weighted by its source's credibility
score when aggregating sub-verdicts into a final verdict.

The credibility scores come from module4_evidence_retrieval/source_registry.py.
This module adds the conflict-detection and resolution logic on top.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Access source registry from module4
sys.path.insert(0, str(Path(__file__).parent.parent / "module4_evidence_retrieval"))

try:
    from source_registry import credibility_for_url, get_by_slug
    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False


def weight_for_url(url: str) -> float:
    """Return credibility weight [0,1] for a source URL."""
    if _REGISTRY_AVAILABLE:
        return credibility_for_url(url)
    return 0.75   # neutral fallback


def weight_for_source_name(name: str) -> float:
    """Return approximate credibility weight by source name string."""
    name_lower = name.lower()
    # Tier 1 — official stats agencies
    if any(k in name_lower for k in ["ons", "bls", "who", "oecd", "world bank",
                                      "cbo", "nhs", "hm treasury", "nato"]):
        return 0.95
    # Tier 2 — major wire services & fact-checkers
    if any(k in name_lower for k in ["reuters", "ap", "bbc", "full fact",
                                      "politifact", "factcheck"]):
        return 0.88
    # Tier 3 — research/academic
    if any(k in name_lower for k in ["ifs", "our world in data", "brookings",
                                      "pew", "ipsos"]):
        return 0.82
    return 0.65


def detect_conflicts(sub_verdicts: list) -> bool:
    """
    Return True if sub-verdicts contain both SUPPORTED and REFUTED labels.
    Mixed NEI is not considered a conflict.
    """
    labels = set()
    for sv in sub_verdicts:
        label = sv.get("label", "") if isinstance(sv, dict) else getattr(sv, "label", "")
        if hasattr(label, "value"):
            label = label.value
        if label in ("supported", "refuted"):
            labels.add(label)
    return "supported" in labels and "refuted" in labels


def aggregate_label(sub_verdicts: list) -> tuple[str, float]:
    """
    Credibility-weighted majority vote over sub-verdicts.

    Returns
    -------
    (label_str, confidence_float)
    """
    if not sub_verdicts:
        return ("not_enough_info", 0.0)

    votes: dict[str, float] = {
        "supported": 0.0,
        "refuted":   0.0,
        "not_enough_info": 0.0,
    }

    for sv in sub_verdicts:
        if isinstance(sv, dict):
            label = sv.get("label", "not_enough_info")
            conf  = sv.get("confidence", 0.5)
            # Try to get source name from supporting passages
            w = 0.75
        else:
            label = getattr(sv, "label", "not_enough_info")
            if hasattr(label, "value"):
                label = label.value
            conf = getattr(sv, "confidence", 0.5)
            w = 0.75

        votes[label] = votes.get(label, 0.0) + conf * w

    total = sum(votes.values()) or 1.0
    best  = max(votes, key=lambda k: votes[k])
    conf  = votes[best] / total

    # If the winning margin is small and conflicts exist, call it mixed
    second = sorted(votes.values(), reverse=True)
    if len(second) > 1 and second[0] < 1.5 * second[1] and detect_conflicts(sub_verdicts):
        return ("mixed", round(conf, 4))

    return (best, round(conf, 4))

