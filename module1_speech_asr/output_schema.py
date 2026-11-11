"""
output_schema.py  —  Module 1 → Module 2 contract
===================================================
Defines TranscriptSegment, the single data structure that Module 1
(speech / ASR) emits and Module 2 (claim detection) consumes.

Design rules
------------
* Use a @dataclass so field names are self-documenting and IDE-friendly.
* A module-level itertools.count() auto-generates unique IDs so callers
  never have to manage or thread an id generator themselves.
* to_json() serialises via json.dumps(dataclasses.asdict(self)) so the
  wire format is plain JSON — easy to log, queue, or stream.
* is_final=False means the segment is still tentative (ASR may revise
  it).  Module 2 MUST NOT run claim detection on non-final segments.
* emitted_at is a float Unix timestamp (time.time()) used later to
  measure end-to-end latency.

Checkpoint
----------
Run this in a Python REPL to verify everything works:

    from output_schema import TranscriptSegment
    seg = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
    print(seg.to_json())
"""

import dataclasses
import itertools
import json
import time

# ---------------------------------------------------------------------------
# Module-level auto-incrementing ID counter.
# Starts at 0; each TranscriptSegment.__post_init__ advances it by 1.
# Using itertools.count() is thread-safe for the GIL-protected next() call.
# ---------------------------------------------------------------------------
_id_counter: itertools.count = itertools.count(start=0)


@dataclasses.dataclass
class TranscriptSegment:
    """A single ASR output segment emitted by Module 1.

    Parameters
    ----------
    start : float
        Segment start time in seconds, relative to the stream start
        (not wall-clock time).
    end : float
        Segment end time in seconds, relative to the stream start.
    text : str
        The recognised speech text for this segment.
    speaker : str, optional
        Speaker label.  Defaults to ``"UNKNOWN"`` until diarization is
        wired in.
    is_final : bool, optional
        ``True`` when the ASR engine has committed to this transcription.
        ``False`` (default) means the text is still tentative and may be
        revised.  **Module 2 must skip non-final segments.**
    emitted_at : float, optional
        Unix timestamp (``time.time()``) recorded when the segment is
        created.  Populated automatically if not supplied.

    Notes
    -----
    ``id`` is assigned automatically from a module-level counter and
    should not be passed by the caller.
    """

    # Required fields (no defaults) — must come before fields with defaults
    start: float
    end: float
    text: str

    # Optional fields with sensible defaults
    speaker: str = "UNKNOWN"
    is_final: bool = False
    emitted_at: float = dataclasses.field(default_factory=time.time)

    # Auto-assigned; excluded from __init__ so callers cannot set it
    id: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Assign the next available ID from the module-level counter."""
        self.id = next(_id_counter)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the segment to a JSON string.

        Returns
        -------
        str
            A compact JSON representation of all fields, e.g.::

                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "speaker": "UNKNOWN",
                    "text": "Hello world",
                    "is_final": false,
                    "emitted_at": 1700000000.123456
                }
        """
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "TranscriptSegment":
        """Deserialise a JSON string back into a TranscriptSegment.

        The ``id`` field in the JSON is ignored; the newly created
        instance receives a fresh ID from the counter.  This is
        intentional: ``from_json`` is meant for reconstructing a segment
        on the *consumer* side, where IDs are used only for deduplication
        and should not be re-used.

        Parameters
        ----------
        json_str : str
            A JSON string previously produced by :meth:`to_json`.

        Returns
        -------
        TranscriptSegment
            A new instance populated from the JSON data.
        """
        data: dict = json.loads(json_str)
        # Drop 'id' — __post_init__ will assign a fresh one
        data.pop("id", None)
        return cls(**data)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TranscriptSegment(id={self.id}, start={self.start}, "
            f"end={self.end}, speaker={self.speaker!r}, "
            f"is_final={self.is_final}, text={self.text!r})"
        )
