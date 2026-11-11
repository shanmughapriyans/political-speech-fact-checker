"""
schemas.py  —  Module 4 → Module 5 contract
============================================
Defines EvidencePassage and EvidenceBundle: the retrieval output that
Module 5 (RAG verification) consumes.

Wire format::

    {
        "bundle_id": 3,
        "set_id": 7,
        "claim_id": 42,
        "claim_text": "...",
        "passages": [
            {
                "passage_id": "guardian_2024_p0",
                "text": "Official figures show unemployment...",
                "source_name": "The Guardian",
                "source_url": "https://...",
                "source_tier": 1,
                "credibility_score": 0.92,
                "retrieval_score": 0.87,
                "question_q_id": 0
            },
            ...
        ],
        "sub_questions": [...],
        "created_at": 1700000000.0
    }
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import time
from typing import List, Optional

_bundle_counter: itertools.count = itertools.count(start=0)


@dataclasses.dataclass
class EvidencePassage:
    """A single retrieved evidence passage.

    Parameters
    ----------
    passage_id : str
        Unique identifier (source_slug + chunk index).
    text : str
        The evidence text (a paragraph or sentence).
    source_name : str
        Human-readable source name (e.g. "ONS", "BBC News").
    source_url : str
        URL or file path of the source document.
    source_tier : int
        Credibility tier: 1=government/official, 2=major wire/newspaper,
        3=academic, 4=other.  Lower = more credible.
    credibility_score : float
        Normalised credibility score in [0, 1] (higher = more credible).
    retrieval_score : float
        Combined BM25 + dense retrieval score (higher = more relevant).
    question_q_id : int
        Which SubQuestion.q_id this passage answers.
    """

    passage_id: str
    text: str
    source_name: str
    source_url: str
    source_tier: int = 2
    credibility_score: float = 0.75
    retrieval_score: float = 0.0
    question_q_id: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvidencePassage":
        return cls(**d)


@dataclasses.dataclass
class EvidenceBundle:
    """All retrieved evidence for one SubQuestionSet.

    Parameters
    ----------
    set_id : int
        SubQuestionSet.set_id this bundle was retrieved for.
    claim_id : int
        ClaimEvent.claim_id.
    claim_text : str
        Original claim text.
    passages : list[EvidencePassage]
        All retrieved passages, sorted by retrieval_score descending.
    sub_questions : list[dict]
        The original SubQuestion dicts (passed through for Module 5).
    bundle_id : int
        Auto-assigned unique ID.
    """

    set_id: int
    claim_id: int
    claim_text: str
    passages: List[EvidencePassage]
    sub_questions: List[dict]

    created_at: float = dataclasses.field(default_factory=time.time)
    bundle_id: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.bundle_id = next(_bundle_counter)
        self.passages = [
            EvidencePassage.from_dict(p) if isinstance(p, dict) else p
            for p in self.passages
        ]

    def passages_for_question(self, q_id: int) -> List[EvidencePassage]:
        """Return passages relevant to a specific sub-question."""
        return [p for p in self.passages if p.question_q_id == q_id]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "EvidenceBundle":
        d = json.loads(s)
        d.pop("bundle_id", None)
        passages = [EvidencePassage.from_dict(p) for p in d.pop("passages", [])]
        return cls(passages=passages, **d)

    def __repr__(self) -> str:
        return (
            f"EvidenceBundle(bundle_id={self.bundle_id}, "
            f"claim_id={self.claim_id}, "
            f"n_passages={len(self.passages)}, "
            f"claim={self.claim_text[:50]!r})"
        )

