"""
schemas.py  —  Module 5 → Module 6 contract
============================================
Defines SubVerdict and VerdictBundle: the per-sub-question judgments
produced by the RAG verification step that Module 6 aggregates.

Wire format::

    {
        "bundle_id": 3,
        "claim_id": 42,
        "claim_text": "...",
        "sub_verdicts": [
            {
                "q_id": 0,
                "question": "What was the unemployment rate...?",
                "answer": "3.8% according to ONS (Jan 2024)",
                "label": "supported",
                "confidence": 0.91,
                "entailment_score": 0.87,
                "supporting_passage_ids": ["ons_emp_1"],
                "reasoning": "The ONS January 2024 release directly states 3.8%."
            }
        ],
        "created_at": 1700000000.0
    }
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import time
from enum import Enum
from typing import List

_bundle_counter: itertools.count = itertools.count(start=0)


class VerdictLabel(str, Enum):
    SUPPORTED  = "supported"
    REFUTED    = "refuted"
    NEI        = "not_enough_info"   # Not Enough Information


@dataclasses.dataclass
class SubVerdict:
    """Verification result for one sub-question.

    Parameters
    ----------
    q_id : int
        Matches SubQuestion.q_id.
    question : str
        The sub-question text (copied from Module 3 output).
    answer : str
        The answer the RAG verifier generated from retrieved evidence.
    label : VerdictLabel
        Supported / Refuted / NEI.
    confidence : float
        Overall confidence in this sub-verdict, [0, 1].
    entailment_score : float
        NLI entailment score from RoBERTa-MNLI, [0, 1].
    supporting_passage_ids : list[str]
        passage_id values from Module 4 that support this answer.
    reasoning : str
        One-sentence explanation of the verdict.
    """

    q_id: int
    question: str
    answer: str
    label: VerdictLabel
    confidence: float
    entailment_score: float
    supporting_passage_ids: List[str]
    reasoning: str

    def __post_init__(self) -> None:
        if isinstance(self.label, str):
            self.label = VerdictLabel(self.label)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["label"] = self.label.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SubVerdict":
        d = dict(d)
        d["label"] = VerdictLabel(d["label"])
        return cls(**d)


@dataclasses.dataclass
class VerdictBundle:
    """All sub-verdicts for one claim — Module 5 → Module 6 contract.

    Parameters
    ----------
    claim_id : int
        ClaimEvent.claim_id.
    claim_text : str
        Original claim sentence.
    claim_speaker : str
        Speaker label.
    claim_start, claim_end : float
        Stream-relative timestamps (seconds).
    claim_score : float
        P(CFS) from Module 2.
    sub_verdicts : list[SubVerdict]
        One per sub-question.
    bundle_id : int
        Auto-assigned unique ID.
    """

    claim_id: int
    claim_text: str
    claim_speaker: str
    claim_start: float
    claim_end: float
    claim_score: float
    sub_verdicts: List[SubVerdict]

    created_at: float = dataclasses.field(default_factory=time.time)
    bundle_id: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.bundle_id = next(_bundle_counter)
        self.sub_verdicts = [
            SubVerdict.from_dict(v) if isinstance(v, dict) else v
            for v in self.sub_verdicts
        ]

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["sub_verdicts"] = [v.to_dict() for v in self.sub_verdicts]
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "VerdictBundle":
        d = json.loads(s)
        d.pop("bundle_id", None)
        sub_verdicts = [SubVerdict.from_dict(v) for v in d.pop("sub_verdicts", [])]
        return cls(sub_verdicts=sub_verdicts, **d)

    def __repr__(self) -> str:
        labels = [v.label.value for v in self.sub_verdicts]
        return (
            f"VerdictBundle(bundle_id={self.bundle_id}, "
            f"claim_id={self.claim_id}, "
            f"labels={labels}, "
            f"claim={self.claim_text[:50]!r})"
        )

