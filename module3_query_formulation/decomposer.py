"""
decomposer.py  —  Module 3: Claim → Sub-Questions
==================================================
Takes a ClaimEvent and produces a SubQuestionSet using an LLM.

The prompt is an AVeriTeC-style decomposition:
  "Break this claim into the minimum set of atomic, independently
   verifiable questions whose answers together would confirm or refute it."

The LLM is asked for structured JSON output via OpenAI's
response_format={"type": "json_object"} (or the JSON schema mode in
newer API versions).  A Pydantic model validates the output before it
becomes a SubQuestionSet, so malformed LLM responses are caught early.

Fallback
--------
If the LLM call fails (rate limit, no API key, timeout), a heuristic
fallback kicks in:
  - Extracts numbers, named entities, and temporal expressions with regex
  - Generates one template question per extracted entity type
This keeps the pipeline running in demo/offline mode.

Usage
-----
    from decomposer import ClaimDecomposer
    decomposer = ClaimDecomposer()
    sub_questions = decomposer.decompose(claim_event)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Allow running from this directory or from the project root
sys.path.insert(0, str(Path(__file__).parent))

from schemas import SubQuestion, SubQuestionSet

log = logging.getLogger("module3.decomposer")

# ---------------------------------------------------------------------------
# Prompt template (AVeriTeC-style)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a fact-checking assistant that breaks political \
claims into verifiable sub-questions.

Given a claim, produce the MINIMUM number of atomic, independently verifiable \
questions (usually 2–4) that together cover all the factual content of the claim.

Rules:
- Each question must be self-contained and answerable from a single source.
- Questions should be ordered so that later ones can depend on earlier answers.
- For each question provide:
    q_id: integer (0-based)
    text: the natural-language question
    search_query: a compact keyword search query (≤10 words)
    depends_on: list of q_id integers this question builds on (usually [])
    category: one of numeric | entity | temporal | causal | comparative | existence

Return ONLY a JSON object with key "questions" containing a list of question objects.
No other text."""

_USER_TEMPLATE = """Claim: "{claim}"
Speaker: {speaker}
Context: "{context}"

Decompose this claim into verifiable sub-questions."""

# ---------------------------------------------------------------------------
# Pydantic validation model
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, field_validator

    class _QModel(BaseModel):
        q_id: int
        text: str
        search_query: str
        depends_on: List[int] = []
        category: str = "numeric"

    class _QSetModel(BaseModel):
        questions: List[_QModel]

        @field_validator("questions")
        @classmethod
        def at_least_one(cls, v: list) -> list:
            if not v:
                raise ValueError("questions list must not be empty")
            return v

    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    log.warning("pydantic not installed — skipping strict validation")

# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM)
# ---------------------------------------------------------------------------

_NUMBER_RE  = re.compile(r"\b\d[\d,\.]*\s*(?:percent|%|billion|million|thousand|bn|mn)?\b", re.I)
_YEAR_RE    = re.compile(r"\b(19|20)\d{2}\b")
_SINCE_RE   = re.compile(r"\bsince\s+\w+\b", re.I)


def _heuristic_decompose(claim_text: str) -> List[dict]:
    """Generate template questions from claim without an LLM."""
    questions = []
    q_id = 0

    # Always: what is the current/stated value?
    questions.append({
        "q_id": q_id,
        "text": f"Is the following statement accurate: '{claim_text}'?",
        "search_query": claim_text[:60],
        "depends_on": [],
        "category": "existence",
    })
    q_id += 1

    # If numbers: ask about the specific figure
    numbers = _NUMBER_RE.findall(claim_text)
    for num in numbers[:2]:
        questions.append({
            "q_id": q_id,
            "text": f"What is the exact value of {num.strip()} mentioned in the claim?",
            "search_query": f"{num.strip()} statistics official",
            "depends_on": [],
            "category": "numeric",
        })
        q_id += 1

    # If year or "since": ask about trend
    years = _YEAR_RE.findall(claim_text)
    if years:
        y = years[0]
        questions.append({
            "q_id": q_id,
            "text": f"What was the value in {y} used as a baseline for comparison?",
            "search_query": f"{y} baseline statistics official",
            "depends_on": [0],
            "category": "temporal",
        })
        q_id += 1

    return questions


# ---------------------------------------------------------------------------
# Main decomposer class
# ---------------------------------------------------------------------------

class ClaimDecomposer:
    """
    Decomposes a ClaimEvent into a SubQuestionSet.

    Parameters
    ----------
    model : str
        OpenAI model name (default: gpt-4o-mini — fast and cheap).
    max_questions : int
        Hard cap on sub-questions per claim (default: 5).
    timeout_s : float
        LLM API timeout in seconds (default: 10).
    api_key : str | None
        OpenAI API key.  If None, reads from OPENAI_API_KEY env var.
        If neither is set, falls back to heuristic decomposition.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_questions: int = 5,
        timeout_s: float = 10.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.max_questions = max_questions
        self.timeout_s = timeout_s
        self._client = None

        key = api_key or os.getenv("OPENAI_API_KEY")
        if key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=key, timeout=timeout_s)
                log.info("OpenAI client initialised (model=%s)", model)
            except ImportError:
                log.warning("openai package not installed — using heuristic fallback")
        else:
            log.warning("OPENAI_API_KEY not set — using heuristic fallback")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, claim_event) -> SubQuestionSet:
        """
        Decompose a ClaimEvent (or any object with .claim_id, .text,
        .context, .speaker, .start, .end, .score) into a SubQuestionSet.

        Never raises — falls back to heuristic on any error.
        """
        claim_id   = getattr(claim_event, "claim_id", 0)
        claim_text = getattr(claim_event, "text", str(claim_event))
        context    = getattr(claim_event, "context", "")
        speaker    = getattr(claim_event, "speaker", "UNKNOWN")
        start      = getattr(claim_event, "start", 0.0)
        end        = getattr(claim_event, "end", 0.0)
        score      = getattr(claim_event, "score", 1.0)

        raw_questions = self._llm_decompose(claim_text, context, speaker)
        raw_questions = raw_questions[:self.max_questions]

        sub_questions = [SubQuestion(**q) for q in raw_questions]

        return SubQuestionSet(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_speaker=speaker,
            claim_start=start,
            claim_end=end,
            claim_score=score,
            questions=sub_questions,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _llm_decompose(
        self, claim_text: str, context: str, speaker: str
    ) -> list[dict]:
        """Call the LLM.  Returns raw dicts.  Falls back on any error."""
        if self._client is None:
            return _heuristic_decompose(claim_text)

        user_msg = _USER_TEMPLATE.format(
            claim=claim_text, speaker=speaker, context=context or "(none)"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = json.loads(resp.choices[0].message.content)
        except Exception as exc:
            log.warning("LLM decompose failed (%s) — using heuristic", exc)
            return _heuristic_decompose(claim_text)

        # Validate
        if _PYDANTIC_AVAILABLE:
            try:
                validated = _QSetModel(**raw)
                return [q.model_dump() for q in validated.questions]
            except Exception as exc:
                log.warning("LLM output validation failed (%s) — heuristic", exc)
                return _heuristic_decompose(claim_text)
        else:
            # Basic check
            questions = raw.get("questions", [])
            if not questions:
                return _heuristic_decompose(claim_text)
            return questions


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
        load_dotenv(Path(__file__).parent.parent / "module1_speech_asr" / ".env")
    except ImportError:
        pass

    print("=== ClaimDecomposer self-test ===\n")

    # Fake ClaimEvent
    class _FakeClaim:
        claim_id = 42
        text = "Unemployment has fallen to 3.8 percent, the lowest in a decade."
        context = "The minister addressed parliament this morning."
        speaker = "SPEAKER_00"
        start = 4.8
        end = 8.3
        score = 0.991

    decomposer = ClaimDecomposer()
    sqs = decomposer.decompose(_FakeClaim())

    print(repr(sqs))
    print(f"\nset_id={sqs.set_id}  claim_id={sqs.claim_id}")
    for q in sqs.questions:
        print(f"  Q{q.q_id} [{q.category}]: {q.text}")
        print(f"       search: {q.search_query}")
        if q.depends_on:
            print(f"       depends_on: {q.depends_on}")

    print("\nJSON round-trip:")
    restored = SubQuestionSet.from_json(sqs.to_json())
    assert restored.claim_id == sqs.claim_id
    assert len(restored.questions) == len(sqs.questions)
    print("  [PASS] round-trip OK")
    print(f"\n[PASS] {len(sqs.questions)} sub-question(s) generated.")

