"""
diarization.py  —  Speaker diarization wrapper + assignment logic
=================================================================
Two pieces:

1. ``Diarizer``
   A thin wrapper around ``pyannote.audio.Pipeline`` that exposes a
   single ``diarize(audio, buffer_start_time)`` method.  Returns a list
   of ``(start, end, speaker)`` named tuples whose timestamps are already
   offset by ``buffer_start_time`` so they line up directly with the
   ``TranscriptSegment`` times produced by ``streaming_asr.py``.

   Accepts an optional ``pipeline=`` constructor parameter so you can
   inject a fake pipeline in tests without needing torch or pyannote
   installed at all.

2. ``assign_speaker(seg_start, seg_end, turns)``
   A pure function — no pyannote, no torch, no I/O.  For each speaker
   turn, compute the overlap with ``[seg_start, seg_end]`` using:

       overlap = min(seg_end, turn.end) - max(seg_start, turn.start)

   Return the speaker label of whichever turn overlaps most.
   Default to ``"UNKNOWN"`` if nothing overlaps.

   Because it is pure arithmetic on plain numbers it is fully testable
   with hand-built turn lists, independent of everything else.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Protocol, runtime_checkable

import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data type — speaker turn
# ---------------------------------------------------------------------------

class SpeakerTurn(NamedTuple):
    """A single speaker turn returned by the diarization pipeline.

    Attributes
    ----------
    start:
        Stream-relative start time in seconds.
    end:
        Stream-relative end time in seconds.
    speaker:
        Speaker label string, e.g. ``"SPEAKER_00"``.
    """

    start: float
    end: float
    speaker: str


# ---------------------------------------------------------------------------
# Protocol — what Diarizer expects from its pipeline object
# ---------------------------------------------------------------------------

@runtime_checkable
class PyAnnoteLike(Protocol):
    """Minimal interface expected from a diarization pipeline.

    Both the real ``pyannote.audio.Pipeline`` and the test stub
    ``ScriptedDiarizationPipeline`` satisfy this protocol.
    """

    def __call__(self, inputs: dict, **kwargs) -> object:
        """Run the pipeline and return a diarization annotation object."""
        ...


# ---------------------------------------------------------------------------
# Test stub — ScriptedDiarizationPipeline
# ---------------------------------------------------------------------------

class ScriptedDiarizationPipeline:
    """Returns a canned list of speaker turns, ignoring the audio entirely.

    Use this in tests to exercise ``Diarizer`` without torch or pyannote.

    Parameters
    ----------
    turns:
        List of ``(start, end, speaker)`` tuples that the pipeline will
        return on every call.

    Example
    -------
    >>> fake_pipeline = ScriptedDiarizationPipeline([
    ...     (0.0, 2.0, "SPEAKER_00"),
    ...     (2.0, 5.0, "SPEAKER_01"),
    ... ])
    >>> d = Diarizer(pipeline=fake_pipeline)
    """

    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def __call__(self, inputs: dict, **kwargs) -> "_FakeAnnotation":
        return _FakeAnnotation(self._turns)


class _FakeAnnotation:
    """Mimics the iterable interface of ``pyannote.core.Annotation``."""

    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, yield_label: bool = False):
        """Yield ``(segment, track, label)`` triples like pyannote does."""
        for start, end, speaker in self._turns:
            yield _FakeSegment(start, end), None, speaker


class _FakeSegment:
    """Mimics ``pyannote.core.Segment`` with ``.start`` and ``.end``."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


# ---------------------------------------------------------------------------
# Pure helper — assign_speaker
# ---------------------------------------------------------------------------

def assign_speaker(
    seg_start: float,
    seg_end: float,
    turns: list[SpeakerTurn],
) -> str:
    """Return the speaker label that overlaps most with ``[seg_start, seg_end]``.

    For each turn, the overlap is computed as::

        overlap = min(seg_end, turn.end) - max(seg_start, turn.start)

    The turn with the largest positive overlap wins.  If no turn overlaps,
    ``"UNKNOWN"`` is returned.

    This function is pure arithmetic — no imports beyond builtins, fully
    testable with hand-built ``SpeakerTurn`` lists.

    Parameters
    ----------
    seg_start:
        Stream-relative start time of the transcript segment (seconds).
    seg_end:
        Stream-relative end time of the transcript segment (seconds).
    turns:
        List of :class:`SpeakerTurn` objects, typically from
        :meth:`Diarizer.diarize`.

    Returns
    -------
    str
        Speaker label of the best-overlapping turn, or ``"UNKNOWN"``.

    Example
    -------
    >>> turns = [
    ...     SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
    ...     SpeakerTurn(2.0, 5.0, "SPEAKER_01"),
    ... ]
    >>> assign_speaker(1.5, 3.0, turns)
    'SPEAKER_01'
    """
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for turn in turns:
        overlap = min(seg_end, turn.end) - max(seg_start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker

    return best_speaker


# ---------------------------------------------------------------------------
# Diarizer
# ---------------------------------------------------------------------------

class Diarizer:
    """Thin wrapper around a ``pyannote.audio`` diarization pipeline.

    Parameters
    ----------
    pipeline:
        An object satisfying :class:`PyAnnoteLike`.  If ``None``, a real
        ``pyannote.audio.Pipeline`` is loaded using the model name and
        Hugging Face token defined in :mod:`config`.

    Example
    -------
    ::

        diarizer = Diarizer()
        turns = diarizer.diarize(audio_buffer, buffer_start_time=12.5)
        speaker = assign_speaker(seg.start, seg.end, turns)
    """

    def __init__(
        self,
        pipeline: Optional[PyAnnoteLike] = None,
    ) -> None:
        if pipeline is not None:
            self._pipeline: PyAnnoteLike = pipeline
            logger.debug(
                "Diarizer: using injected pipeline %s",
                type(pipeline).__name__,
            )
        else:
            self._pipeline = self._load_real_pipeline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize(
        self,
        audio: np.ndarray,
        buffer_start_time: float = 0.0,
    ) -> list[SpeakerTurn]:
        """Run diarization on ``audio`` and return offset speaker turns.

        Parameters
        ----------
        audio:
            float32 numpy array at ``config.SAMPLE_RATE`` Hz (mono).
        buffer_start_time:
            Stream-relative time that ``audio[0]`` corresponds to.
            All returned turn timestamps are offset by this value so
            they align with ``TranscriptSegment.start / .end``.

        Returns
        -------
        list[SpeakerTurn]
            Speaker turns sorted by start time.  Timestamps are
            stream-relative (already offset by ``buffer_start_time``).

        Notes
        -----
        pyannote expects a dict with key ``"waveform"`` (shape
        ``[channels, samples]``) and ``"sample_rate"``.
        """
        # pyannote wants shape (channels, samples) as a torch tensor, but
        # to avoid a hard torch dependency at import time we do the
        # conversion here only when actually calling the real pipeline.
        inputs = self._make_inputs(audio)

        logger.debug(
            "Diarizer.diarize: %.2f s of audio (buffer_start=%.2f s)",
            len(audio) / config.SAMPLE_RATE,
            buffer_start_time,
        )

        annotation = self._pipeline(inputs)

        turns: list[SpeakerTurn] = []
        for segment, _track, label in annotation.itertracks(yield_label=True):
            turns.append(SpeakerTurn(
                start=segment.start + buffer_start_time,
                end=segment.end + buffer_start_time,
                speaker=str(label),
            ))

        turns.sort(key=lambda t: t.start)

        logger.debug(
            "Diarizer.diarize: %d turns → %s",
            len(turns),
            [(t.speaker, f"{t.start:.2f}-{t.end:.2f}") for t in turns],
        )

        return turns

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_inputs(audio: np.ndarray) -> dict:
        """Build the input dict pyannote expects.

        For the real pipeline this wraps the array in a torch tensor.
        For the test stub the dict is passed as-is (the stub ignores it).
        """
        try:
            import torch  # type: ignore[import]
            waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
            return {"waveform": waveform, "sample_rate": config.SAMPLE_RATE}
        except ImportError:
            # torch not installed — likely running in test mode with a stub.
            return {"waveform": audio, "sample_rate": config.SAMPLE_RATE}

    @staticmethod
    def _load_real_pipeline() -> PyAnnoteLike:
        """Load a pyannote.audio Pipeline using settings from config."""
        try:
            from pyannote.audio import Pipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for real diarization.  "
                "Install it with:  pip install pyannote.audio\n"
                "You also need a Hugging Face token with access to the "
                "gated pyannote models.  Set the HF_TOKEN environment "
                "variable (see config.HF_TOKEN_ENV_VAR)."
            ) from exc

        token = config.get_hf_token()

        logger.info(
            "Loading pyannote pipeline: %s",
            config.DIARIZATION_MODEL,
        )
        return Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
            use_auth_token=token,
        )


# ---------------------------------------------------------------------------
# Checkpoint / unit tests  (no pyannote / torch needed)
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Verify assign_speaker and Diarizer with hand-built turn lists."""

    print("--- assign_speaker tests ---")

    turns = [
        SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
        SpeakerTurn(2.0, 5.0, "SPEAKER_01"),
    ]

    # Segment fully inside SPEAKER_00
    s = assign_speaker(0.5, 1.5, turns)
    assert s == "SPEAKER_00", f"Expected SPEAKER_00, got {s!r}"
    print(f"  [PASS] fully inside SPEAKER_00 -> {s!r}")

    # Segment fully inside SPEAKER_01
    s = assign_speaker(3.0, 4.5, turns)
    assert s == "SPEAKER_01", f"Expected SPEAKER_01, got {s!r}"
    print(f"  [PASS] fully inside SPEAKER_01 -> {s!r}")

    # Segment straddles boundary — more overlap with SPEAKER_01
    s = assign_speaker(1.5, 3.0, turns)
    assert s == "SPEAKER_01", f"Expected SPEAKER_01 (1.0 s vs 0.5 s), got {s!r}"
    print(f"  [PASS] straddling, more overlap with SPEAKER_01 -> {s!r}")

    # Segment straddles boundary — more overlap with SPEAKER_00
    s = assign_speaker(0.0, 2.5, turns)
    assert s == "SPEAKER_00", f"Expected SPEAKER_00 (2.0 s vs 0.5 s), got {s!r}"
    print(f"  [PASS] straddling, more overlap with SPEAKER_00 -> {s!r}")

    # No overlap → UNKNOWN
    s = assign_speaker(10.0, 12.0, turns)
    assert s == "UNKNOWN", f"Expected UNKNOWN, got {s!r}"
    print(f"  [PASS] no overlap -> {s!r}")

    # Exact boundary: touching but zero overlap
    s = assign_speaker(2.0, 3.0, [SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
                                    SpeakerTurn(2.0, 5.0, "SPEAKER_01")])
    # min(3.0,2.0)-max(2.0,0.0) = 2.0-2.0 = 0  → no overlap with SPEAKER_00
    # min(3.0,5.0)-max(2.0,2.0) = 3.0-2.0 = 1.0 → SPEAKER_01
    assert s == "SPEAKER_01", f"Expected SPEAKER_01 at exact boundary, got {s!r}"
    print(f"  [PASS] exact boundary -> {s!r}")

    print("\n--- Diarizer (fake pipeline) test ---")

    fake_pipeline = ScriptedDiarizationPipeline([
        (0.0, 3.0, "SPEAKER_00"),
        (3.0, 7.0, "SPEAKER_01"),
    ])
    diarizer = Diarizer(pipeline=fake_pipeline)

    dummy_audio = np.zeros(int(config.SAMPLE_RATE * 7.0), dtype=np.float32)

    # buffer_start_time=0 — no offset
    turns_out = diarizer.diarize(dummy_audio, buffer_start_time=0.0)
    assert len(turns_out) == 2
    assert turns_out[0] == SpeakerTurn(0.0, 3.0, "SPEAKER_00")
    assert turns_out[1] == SpeakerTurn(3.0, 7.0, "SPEAKER_01")
    print("  [PASS] turns without offset")

    # buffer_start_time=10.0 — timestamps must be shifted
    turns_off = diarizer.diarize(dummy_audio, buffer_start_time=10.0)
    assert turns_off[0].start == 10.0 and turns_off[0].end == 13.0
    assert turns_off[1].start == 13.0 and turns_off[1].end == 17.0
    print("  [PASS] turns with buffer_start_time=10.0 offset applied correctly")

    # End-to-end: diarize → assign_speaker
    seg_speaker = assign_speaker(11.0, 12.5, turns_off)
    assert seg_speaker == "SPEAKER_00", f"Expected SPEAKER_00, got {seg_speaker!r}"
    print(f"  [PASS] end-to-end assign_speaker after diarize -> {seg_speaker!r}")

    print("\n[PASS] All diarization checkpoint tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    _run_tests()
