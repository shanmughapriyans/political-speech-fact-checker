"""
schemas.py  —  Module 6 → Module 7/8 contract
==============================================
Defines Verdict: the final, conflict-resolved verdict for one claim
that Module 7 (dashboard) displays and Module 8 (database) persists.

Wire format::

    {
        "verdict_id": 5,
        "claim_id": 42,
        "claim_text": "Unemployment has fallen to 3.8%...",
        "claim_speaker": "SPEAKER_00",
        "claim_start": 4.8,
        "claim_end": 8.3,
        "claim_score": 0.991,
        "label": "supported",
        "confidence": 0.88,
        "explanation": "ONS data confirms unemployment reached 3.8%...",
        "evidence_summary": [
            {"source": "ONS", "url": "...", "snippet": "...", "supports": true}
        ],
        "sub_verdicts": [...],
        "conflicts_detected": false,
        "conflict_resolution_note": null,
        "emitted_at": 1700000000.0
    }
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import time
from enum import Enum
from typing import List, Optional

_verdict_counter: itertools.count = itertools.count(start=0)


class FinalLabel(str, Enum):
    SUPPORTED  = "supported"
    REFUTED    = "refuted"
    MIXED      = "mixed"           # sub-verdicts disagree
    NEI        = "not_enough_info"


@dataclasses.dataclass
class EvidenceSummaryItem:
    """One evidence snippet for the UI / audit trail."""
    source: str
    url: str
    snippet: str
    supports: bool    # True=supports claim, False=contradicts

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceSummaryItem":
        return cls(**d)


@dataclasses.dataclass
class Verdict:
    """Final, conflict-resolved verdict for one claim.

    Parameters
    ----------
    claim_id : int
        ClaimEvent.claim_id — unique chain ID across all modules.
    claim_text : str
        Original claim sentence.
    claim_speaker : str
        Speaker label.
    claim_start, claim_end : float
        Stream-relative timestamps.
    claim_score : float
        P(CFS) from Module 2.
    label : FinalLabel
        Aggregated verdict label.
    confidence : float
        Overall confidence in [0, 1].
    explanation : str
        One-paragraph natural-language explanation for the UI.
    evidence_summary : list[EvidenceSummaryItem]
        Traceable evidence items (shown in dashboard).
    sub_verdicts : list[dict]
        Raw SubVerdict dicts from Module 5 (kept for audit).
    conflicts_detected : bool
        Whether evidence sources disagreed.
    conflict_resolution_note : str | None
        How the conflict was resolved (or None if no conflict).
    verdict_id : int
        Auto-assigned unique ID.
    emitted_at : float
        Unix timestamp when this Verdict was created.
    """

    # Required
    claim_id: int
    claim_text: str
    claim_speaker: str
    claim_start: float
    claim_end: float
    claim_score: float
    label: FinalLabel
    confidence: float
    explanation: str
    evidence_summary: List[EvidenceSummaryItem]
    sub_verdicts: List[dict]
    conflicts_detected: bool = False
    conflict_resolution_note: Optional[str] = None

    # Auto
    emitted_at: float = dataclasses.field(default_factory=time.time)
    verdict_id: int   = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.verdict_id = next(_verdict_counter)
        if isinstance(self.label, str):
            self.label = FinalLabel(self.label)
        self.evidence_summary = [
            EvidenceSummaryItem.from_dict(e) if isinstance(e, dict) else e
            for e in self.evidence_summary
        ]

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["label"] = self.label.value
        d["evidence_summary"] = [e.to_dict() for e in self.evidence_summary]
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "Verdict":
        d = json.loads(s)
        d.pop("verdict_id", None)
        d["label"] = FinalLabel(d["label"])
        d["evidence_summary"] = [
            EvidenceSummaryItem.from_dict(e) for e in d.get("evidence_summary", [])
        ]
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"Verdict(verdict_id={self.verdict_id}, "
            f"claim_id={self.claim_id}, "
            f"label={self.label.value}, "
            f"conf={self.confidence:.2f}, "
            f"claim={self.claim_text[:50]!r})"
        )

