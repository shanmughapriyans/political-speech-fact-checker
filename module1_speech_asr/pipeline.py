"""
pipeline.py  —  Module 1 main orchestration loop
=================================================
This file owns the main loop.  It imports from the other four files but
contains **no** ASR or diarization logic itself — just orchestration.

Responsibilities
----------------
* Drive ``AudioCapture.chunks()`` to ingest audio.
* Feed each chunk to ``StreamingASR`` and poll for new segments.
* Maintain a *separate* raw-audio accumulation buffer for diarization
  (the ASR engine trims its own buffer on every commit, so diarization
  needs its own unmodified view of the audio).
* On a coarser cadence (``config.DIARIZATION_INTERVAL_S``), re-run the
  optional ``Diarizer`` over the accumulated audio and refresh the
  speaker-turn cache.
* For every segment the ASR emits, label it with ``assign_speaker`` if a
  diarizer is configured, then call the user-supplied ``on_event``
  callback with the final :class:`~output_schema.TranscriptSegment`.

Design rules
------------
* ``diarizer=None`` is a fully-working mode — the pipeline runs without
  any speaker labelling and all segments get ``speaker="UNKNOWN"``.  This
  lets you build and demo Modules 1→2 end-to-end before you have sorted
  out the Hugging Face token for the gated pyannote model.
* The ``on_event`` callback is also optional — if not supplied, segments
  are collected and returned when ``run()`` finishes (useful for tests).
* The pipeline is intentionally single-threaded.  Diarization blocks the
  loop briefly on its cadence; for a production system you would push
  diarization onto a background thread, but that is out of scope here.

Checkpoint
----------
Run ``python pipeline.py`` — it wires up a ``ScriptedWhisperModel`` fake,
a ``ScriptedDiarizer`` fake, and a ``FileAudioCapture`` over a synthetic
sine-wave WAV, then confirms that events come out labelled correctly before
plugging in anything real.
"""

from __future__ import annotations

import logging
import time
import tempfile
import os
from typing import Callable, Optional

import numpy as np

import config
from audio_capture import FileAudioCapture
from diarization import Diarizer, SpeakerTurn, assign_speaker
from output_schema import TranscriptSegment
from streaming_asr import StreamingASR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Orchestrate audio capture, ASR, and optional diarization.

    Parameters
    ----------
    capture:
        Any object with a ``.chunks()`` generator that yields float32
        numpy arrays at ``config.SAMPLE_RATE`` Hz.
    asr:
        A :class:`~streaming_asr.StreamingASR` instance (with a real or
        injected model).
    diarizer:
        An optional :class:`~diarization.Diarizer` instance.  Pass
        ``None`` to skip speaker labelling entirely — all segments will
        have ``speaker="UNKNOWN"``.
    on_event:
        Optional callback called with each :class:`~output_schema.TranscriptSegment`
        as it is produced.  If ``None``, segments are collected and
        returned by :meth:`run`.
    poll_interval_s:
        How often (in *chunks*) to call ``asr.poll()``.  Defaults to
        ``config.POLL_INTERVAL_S`` ÷ ``chunk_s``, but since we poll after
        every chunk the effective cadence is driven by chunk size.
        Here we always poll after every chunk for simplicity; the
        ``POLL_INTERVAL_S`` constant is still honoured as a wall-clock
        guard via ``_next_poll_time``.
    """

    def __init__(
        self,
        capture,
        asr: StreamingASR,
        diarizer: Optional[Diarizer] = None,
        on_event: Optional[Callable[[TranscriptSegment], None]] = None,
    ) -> None:
        self._capture = capture
        self._asr = asr
        self._diarizer = diarizer
        self._on_event = on_event

        # Separate raw-audio accumulation buffer for diarization.
        # The ASR engine trims its own buffer on commit; we must keep our own
        # unmodified copy so diarization always sees the full recent history.
        self._diar_buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._diar_buffer_start_time: float = 0.0   # stream-relative

        # Cached speaker turns from the last diarization run.
        self._speaker_turns: list[SpeakerTurn] = []

        # Wall-clock timestamps for cadence control.
        self._next_poll_time: float = 0.0
        self._next_diar_time: float = 0.0

        # Collected segments (used when on_event is None).
        self._collected: list[TranscriptSegment] = []

        # Total audio seen (for computing stream-relative timestamps).
        self._total_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_chunks: Optional[int] = None) -> list[TranscriptSegment]:
        """Start the main loop and process audio from the capture source.

        Parameters
        ----------
        max_chunks:
            Stop after this many chunks.  ``None`` → run until the capture
            source is exhausted (or ``KeyboardInterrupt``).

        Returns
        -------
        list[TranscriptSegment]
            All segments produced during the run.  Segments are also
            delivered to ``on_event`` as they arrive if that callback was
            supplied.
        """
        now = time.monotonic()
        self._next_poll_time = now
        self._next_diar_time = now + config.DIARIZATION_INTERVAL_S

        chunks_seen = 0

        try:
            for chunk in self._capture.chunks():
                self._process_chunk(chunk)
                chunks_seen += 1
                if max_chunks is not None and chunks_seen >= max_chunks:
                    break
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")

        # Final poll to drain any remaining buffered audio.
        self._do_poll()

        logger.info(
            "Pipeline finished: %d chunks, %d segments collected",
            chunks_seen,
            len(self._collected),
        )
        return list(self._collected)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Handle one audio chunk: feed ASR, maybe poll, maybe diarize."""
        chunk_duration = len(chunk) / config.SAMPLE_RATE
        self._total_frames += len(chunk)

        # 1. Feed the chunk to the ASR engine.
        self._asr.feed(chunk)

        # 2. Accumulate into the diarization buffer.
        self._diar_buffer = np.concatenate([self._diar_buffer, chunk])

        now = time.monotonic()

        # 3. Poll ASR if enough wall-clock time has passed.
        if now >= self._next_poll_time:
            self._do_poll()
            self._next_poll_time = now + config.POLL_INTERVAL_S

        # 4. Run diarization if enough audio has accumulated since last run.
        if self._diarizer is not None and now >= self._next_diar_time:
            self._do_diarize()
            self._next_diar_time = now + config.DIARIZATION_INTERVAL_S

    def _do_poll(self) -> None:
        """Call asr.poll() and dispatch each resulting segment."""
        segments = self._asr.poll()
        for seg in segments:
            # Label the speaker using the cached diarization turns.
            if self._diarizer is not None and self._speaker_turns:
                seg.speaker = assign_speaker(seg.start, seg.end, self._speaker_turns)

            logger.debug(
                "_do_poll: %s [%.2f-%.2f] speaker=%s %r",
                "FINAL" if seg.is_final else "TENT ",
                seg.start, seg.end, seg.speaker, seg.text,
            )

            self._collected.append(seg)
            if self._on_event is not None:
                self._on_event(seg)

    def _do_diarize(self) -> None:
        """Re-run diarization over the accumulated buffer."""
        if len(self._diar_buffer) == 0:
            return

        logger.debug(
            "_do_diarize: running over %.2f s of audio (buffer_start=%.2f s)",
            len(self._diar_buffer) / config.SAMPLE_RATE,
            self._diar_buffer_start_time,
        )

        try:
            self._speaker_turns = self._diarizer.diarize(
                self._diar_buffer,
                buffer_start_time=self._diar_buffer_start_time,
            )
            logger.info(
                "_do_diarize: %d speaker turns found", len(self._speaker_turns)
            )
        except Exception as exc:  # pragma: no cover
            # Diarization errors must never crash the ASR loop.
            logger.error("_do_diarize: error during diarization: %s", exc)

        # Trim the diarization buffer to cap memory usage.
        # Keep only the last MAX_UNCOMMITTED_S seconds of audio.
        max_frames = int(config.MAX_UNCOMMITTED_S * config.SAMPLE_RATE)
        if len(self._diar_buffer) > max_frames:
            trimmed = len(self._diar_buffer) - max_frames
            self._diar_buffer_start_time += trimmed / config.SAMPLE_RATE
            self._diar_buffer = self._diar_buffer[-max_frames:]


# ---------------------------------------------------------------------------
# Checkpoint / smoke test  (no real model or pyannote needed)
# ---------------------------------------------------------------------------

class _ScriptedDiarizer:
    """Fake diarizer that returns canned turns, ignoring the audio."""

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self._turns = turns
        self.call_count = 0

    def diarize(
        self,
        audio: np.ndarray,
        buffer_start_time: float = 0.0,
    ) -> list[SpeakerTurn]:
        self.call_count += 1
        # Return turns as-is (already stream-relative in canned data).
        return list(self._turns)


def _run_checkpoint() -> None:
    """End-to-end pipeline test with fake ASR model and fake diarizer."""
    import soundfile as sf  # type: ignore[import]
    from streaming_asr import ScriptedWhisperModel

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("=== Pipeline checkpoint ===\n")

    # ----------------------------------------------------------------
    # 1. Build a synthetic 3-second 440 Hz sine-wave WAV at 16 kHz
    # ----------------------------------------------------------------
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(config.SAMPLE_RATE * duration_s), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, sine, config.SAMPLE_RATE)
    print(f"Synthetic WAV written: {tmp_path}  ({duration_s:.1f} s)\n")

    try:
        # ----------------------------------------------------------------
        # 2. Fake ASR model — scripted hypothesis sequence
        #    With AGREEMENT_N=2 this should produce two FINAL commits:
        #    "Hello" then "world"
        # ----------------------------------------------------------------
        hypotheses = [
            ["Hello"],
            ["Hello", "world"],
            ["world"],
            ["world", "today"],
        ]
        fake_model = ScriptedWhisperModel(hypotheses)

        # ----------------------------------------------------------------
        # 3. Fake diarizer — two speakers covering the full 3 s
        # ----------------------------------------------------------------
        canned_turns = [
            SpeakerTurn(0.0, 1.5, "SPEAKER_00"),
            SpeakerTurn(1.5, 3.0, "SPEAKER_01"),
        ]
        fake_diarizer = _ScriptedDiarizer(canned_turns)

        # ----------------------------------------------------------------
        # 4. Wire up the pipeline
        # ----------------------------------------------------------------
        capture = FileAudioCapture(tmp_path, chunk_s=0.5, realtime=False)
        asr     = StreamingASR(model=fake_model)

        received: list[TranscriptSegment] = []

        def on_event(seg: TranscriptSegment) -> None:
            tag = "FINAL" if seg.is_final else "TENT "
            print(f"  [{tag}] speaker={seg.speaker:11s}  [{seg.start:.2f}-{seg.end:.2f}]  {seg.text!r}")
            received.append(seg)

        pipeline = Pipeline(
            capture=capture,
            asr=asr,
            diarizer=fake_diarizer,
            on_event=on_event,
        )

        # Force DIARIZATION_INTERVAL_S to a small value so it fires in the test.
        original_diar_interval = config.DIARIZATION_INTERVAL_S
        config.DIARIZATION_INTERVAL_S = 0.0  # trigger on first opportunity

        all_segs = pipeline.run()

        config.DIARIZATION_INTERVAL_S = original_diar_interval

        # ----------------------------------------------------------------
        # 5. Assertions
        # ----------------------------------------------------------------
        finals = [s for s in all_segs if s.is_final]
        tents  = [s for s in all_segs if not s.is_final]

        print(f"\nTotal segments : {len(all_segs)}")
        print(f"  FINAL        : {len(finals)}")
        print(f"  TENTATIVE    : {len(tents)}")

        assert len(finals) >= 1, "Expected at least one FINAL segment"
        assert finals[0].text == "Hello", (
            f"Expected first commit 'Hello', got {finals[0].text!r}"
        )
        # Speaker assignment: "Hello" covers ~0.0-0.4 s -> SPEAKER_00
        assert finals[0].speaker == "SPEAKER_00", (
            f"Expected SPEAKER_00 for 'Hello', got {finals[0].speaker!r}"
        )

        print("\n[PASS] At least one FINAL commit received with correct text.")
        print(f"[PASS] Speaker assigned correctly: {finals[0].speaker!r}")

        # Test without diarizer (diarizer=None)
        print("\n--- Without diarizer (diarizer=None) ---")
        capture2 = FileAudioCapture(tmp_path, chunk_s=0.5, realtime=False)
        asr2     = StreamingASR(model=ScriptedWhisperModel(hypotheses))
        pipeline2 = Pipeline(capture=capture2, asr=asr2, diarizer=None)
        segs2 = pipeline2.run()
        finals2 = [s for s in segs2 if s.is_final]
        assert all(s.speaker == "UNKNOWN" for s in finals2), (
            "Without diarizer all speakers should be 'UNKNOWN'"
        )
        print(f"[PASS] All {len(finals2)} FINAL segments have speaker='UNKNOWN' when diarizer=None")

        print("\n[PASS] All pipeline checkpoint tests passed.")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    _run_checkpoint()
