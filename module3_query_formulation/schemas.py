"""
schemas.py  —  Module 3 → Module 4 contract
============================================
Defines SubQuestion and SubQuestionSet: the structured output of the
claim decomposition step that Module 4 (evidence retrieval) consumes.

Design
------
Each ClaimEvent produces exactly one SubQuestionSet, which contains
1–N ordered SubQuestion objects.  Later sub-questions may reference
earlier ones (Chain-of-RAG pattern), so the ordering matters.

Wire format (compact JSON)::

    {
        "set_id": 7,
        "claim_id": 42,
        "claim_text": "Unemployment fell to 3.8%...",
        "claim_speaker": "SPEAKER_00",
        "claim_start": 4.8,
        "claim_end": 8.3,
        "claim_score": 0.991,
        "questions": [
            {
                "q_id": 0,
                "text": "What was the unemployment rate in <month/year>?",
                "search_query": "unemployment rate statistics <month/year>",
                "depends_on": [],
                "category": "numeric"
            },
            ...
        ],
        "created_at": 1700000000.0
    }
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import time
from typing import List, Optional

_set_counter: itertools.count = itertools.count(start=0)


@dataclasses.dataclass
class SubQuestion:
    """A single verifiable sub-question derived from a claim.

    Parameters
    ----------
    q_id : int
        Zero-based index within the parent SubQuestionSet.
    text : str
        The natural-language question to answer (e.g. "What was the
        unemployment rate in March 2024?").
    search_query : str
        A compact web/index search query derived from the question.
        May differ from `text` (shorter, keyword-focused).
    depends_on : list[int]
        Indices of other SubQuestions whose answers must be known before
        this one can be answered (Chain-of-RAG ordering).
    category : str
        One of: "numeric", "entity", "temporal", "causal", "comparative",
        "existence".  Used by the retriever to bias source selection.
    """

    q_id: int
    text: str
    search_query: str
    depends_on: List[int] = dataclasses.field(default_factory=list)
    category: str = "numeric"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SubQuestion":
        return cls(**d)


@dataclasses.dataclass
class SubQuestionSet:
    """All sub-questions for one ClaimEvent.

    Parameters
    ----------
    claim_id : int
        The ClaimEvent.claim_id this set was derived from.
    claim_text : str
        Original claim sentence (kept here so Module 4 doesn't need to
        look it up).
    claim_speaker : str
        Speaker label from the ClaimEvent.
    claim_start : float
        Stream-relative start time (seconds) of the claim.
    claim_end : float
        Stream-relative end time (seconds) of the claim.
    claim_score : float
        P(CFS) score from Module 2.
    questions : list[SubQuestion]
        Ordered list of sub-questions.
    created_at : float
        Unix timestamp of creation (auto-populated).
    set_id : int
        Auto-incrementing unique ID for this set (auto-assigned).
    """

    # Required
    claim_id: int
    claim_text: str
    claim_speaker: str
    claim_start: float
    claim_end: float
    claim_score: float
    questions: List[SubQuestion]

    # Auto
    created_at: float = dataclasses.field(default_factory=time.time)
    set_id: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.set_id = next(_set_counter)
        self.questions = [
            SubQuestion.from_dict(q) if isinstance(q, dict) else q
            for q in self.questions
        ]

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "SubQuestionSet":
        d = json.loads(s)
        d.pop("set_id", None)
        questions = [SubQuestion.from_dict(q) for q in d.pop("questions", [])]
        obj = cls(questions=questions, **d)
        return obj

    def __repr__(self) -> str:
        return (
            f"SubQuestionSet(set_id={self.set_id}, "
            f"claim_id={self.claim_id}, "
            f"n_questions={len(self.questions)}, "
            f"claim={self.claim_text[:50]!r})"
        )

