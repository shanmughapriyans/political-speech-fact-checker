"""
claim_schema.py  --  Module 2 -> Module 3 output contract
==========================================================
Defines ClaimEvent, the single data structure that Module 2 (claim
detection) emits and every downstream module (3-7) consumes.

Design mirrors Module 1's output_schema.py:
  - @dataclass for IDE-friendliness and self-documenting fields
  - Module-level itertools.count() auto-generates unique claim_ids
  - to_json() via json.dumps(dataclasses.asdict(self))
  - from_json() classmethod for deserialisation on the consumer side

Field rationale
---------------
claim_id            Unique int for tracing a claim through Modules 3-7.
                    Auto-assigned; callers never set it.

text                The claim sentence itself.

context             The 1-2 sentences immediately preceding the claim,
                    as a single string.  Empty string when there is no
                    prior context (start of stream).  Included so
                    downstream fact-checkers have the surrounding
                    discourse without having to maintain their own
                    sliding window.

speaker             Copied from the Module 1 TranscriptSegment(s) that
                    this claim was extracted from.  "UNKNOWN" until
                    diarization is wired in.

start, end          Stream-relative timestamps (seconds) copied from the
                    Module 1 segment(s).  Used to anchor the claim in
                    the original audio timeline.

score               P(claim-worthy) in [0.0, 1.0] produced by the
                    classifier.  Downstream modules can threshold this
                    value independently (e.g. only verify claims with
                    score >= 0.6).

source_segment_ids  List of Module 1 TranscriptSegment.id values that
                    this claim spans.  Enables latency measurement
                    (Module 1 emitted_at -> Module 2 emitted_at) and
                    exact replay/debugging.

emitted_at          time.time() when this ClaimEvent was created,
                    mirroring Module 1's convention.

Checkpoint
----------
Run in a Python REPL to verify:

    from claim_schema import ClaimEvent
    ev = ClaimEvent(
        text="The unemployment rate fell to 3.5 percent last month.",
        context="The minister gave a press conference this morning.",
        speaker="SPEAKER_00",
        start=12.4,
        end=16.1,
        score=0.87,
        source_segment_ids=[3, 4],
    )
    print(ev.to_json())
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import time
from typing import List

# ---------------------------------------------------------------------------
# Module-level auto-incrementing ID counter (thread-safe via the GIL)
# ---------------------------------------------------------------------------
_id_counter: itertools.count = itertools.count(start=0)


@dataclasses.dataclass
class ClaimEvent:
    """A single claim-worthy sentence emitted by Module 2.

    Parameters
    ----------
    text : str
        The claim sentence.
    context : str
        The 1-2 sentences immediately before this claim in the transcript.
        Empty string if no prior context is available.
    speaker : str
        Speaker label copied from the originating TranscriptSegment(s).
        Defaults to ``"UNKNOWN"``.
    start : float
        Stream-relative start time (seconds) of the originating segment(s).
    end : float
        Stream-relative end time (seconds) of the originating segment(s).
    score : float
        P(claim-worthy) in [0.0, 1.0] from the classifier.
    source_segment_ids : list[int]
        Module 1 TranscriptSegment.id values that this claim spans.
    emitted_at : float
        Unix timestamp when this ClaimEvent was created (auto-populated).

    Notes
    -----
    ``claim_id`` is assigned automatically from a module-level counter.
    Do not pass it explicitly.
    """

    # Required fields -- must come before fields with defaults
    text: str
    context: str
    start: float
    end: float
    score: float
    source_segment_ids: List[int]

    # Optional fields with sensible defaults
    speaker: str = "UNKNOWN"
    emitted_at: float = dataclasses.field(default_factory=time.time)

    # Auto-assigned; excluded from __init__
    claim_id: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Assign the next available claim_id from the module-level counter."""
        self.claim_id = next(_id_counter)

        # Validate score range
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score must be in [0.0, 1.0], got {self.score}"
            )

        # Ensure source_segment_ids is always a plain list (not a tuple etc.)
        self.source_segment_ids = list(self.source_segment_ids)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the ClaimEvent to a JSON string.

        Returns
        -------
        str
            Compact JSON with all fields, e.g.::

                {
                    "claim_id": 0,
                    "text": "The unemployment rate fell to 3.5 percent.",
                    "context": "The minister spoke this morning.",
                    "speaker": "SPEAKER_00",
                    "start": 12.4,
                    "end": 16.1,
                    "score": 0.87,
                    "source_segment_ids": [3, 4],
                    "emitted_at": 1700000000.123456
                }
        """
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "ClaimEvent":
        """Deserialise a JSON string back into a ClaimEvent.

        The ``claim_id`` in the JSON is discarded; the new instance
        receives a fresh ID from the counter.  Use this on the consumer
        side (Modules 3-7) where IDs serve deduplication only.

        Parameters
        ----------
        json_str : str
            A JSON string previously produced by :meth:`to_json`.

        Returns
        -------
        ClaimEvent
        """
        data: dict = json.loads(json_str)
        data.pop("claim_id", None)
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"ClaimEvent(claim_id={self.claim_id}, score={self.score:.3f}, "
            f"speaker={self.speaker!r}, "
            f"text={self.text[:50]!r}{'...' if len(self.text) > 50 else ''})"
        )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== ClaimEvent checkpoint ===\n")

    # Construct and print
    ev = ClaimEvent(
        text="The unemployment rate fell to 3.5 percent last month.",
        context="The minister gave a press conference this morning.",
        speaker="SPEAKER_00",
        start=12.4,
        end=16.1,
        score=0.87,
        source_segment_ids=[3, 4],
    )

    import json as _json
    print("to_json() output:")
    print(_json.dumps(_json.loads(ev.to_json()), indent=2))
    print()

    # Verify auto-incrementing IDs
    ev2 = ClaimEvent(
        text="GDP grew by 2.1 percent in the third quarter.",
        context=ev.text,
        speaker="SPEAKER_01",
        start=20.0,
        end=24.5,
        score=0.72,
        source_segment_ids=[5],
    )
    ev3 = ClaimEvent(
        text="Inflation is at its lowest since 2019.",
        context=ev2.text,
        start=25.0,
        end=28.0,
        score=0.91,
        source_segment_ids=[6, 7],
    )
    print(f"ID auto-increment: {ev.claim_id}, {ev2.claim_id}, {ev3.claim_id}")
    assert ev.claim_id == 0 and ev2.claim_id == 1 and ev3.claim_id == 2

    # Round-trip
    restored = ClaimEvent.from_json(ev3.to_json())
    assert restored.text  == ev3.text
    assert restored.score == ev3.score
    assert restored.source_segment_ids == ev3.source_segment_ids
    print(f"from_json round-trip: text={restored.text!r}")
    print(f"                      score={restored.score}")
    print(f"                      ids={restored.source_segment_ids}")

    # Validate score guard
    try:
        ClaimEvent(
            text="bad", context="", start=0, end=1,
            score=1.5, source_segment_ids=[],
        )
        print("[FAIL] score > 1.0 should have raised ValueError")
    except ValueError as e:
        print(f"\n[PASS] score validation: {e}")

    print("\n[PASS] All ClaimEvent checkpoint tests passed.")
