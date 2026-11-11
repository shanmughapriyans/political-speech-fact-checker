"""
audio_capture.py  —  Module 1 audio ingestion layer
=====================================================
Defines one interface, two implementations.

Interface
---------
Both classes expose a single generator method::

    def chunks(self) -> Iterator[np.ndarray]:
        ...

Each ``yield`` produces a float32 numpy array of shape ``(N,)`` where
``N = int(SAMPLE_RATE * chunk_s)``.  All downstream code calls
``.chunks()`` and never needs to know whether audio came from a file or
a live microphone.

Classes
-------
FileAudioCapture
    Reads a WAV/FLAC/OGG file via *soundfile*.  Optionally throttles
    delivery to real-time speed so the rest of the pipeline experiences
    the same timing it would with a live mic.  Use this for all automated
    tests — it needs no hardware.

MicAudioCapture
    Captures from the default system microphone via
    *sounddevice.InputStream* with a low-latency callback that pushes
    frames into a ``queue.Queue``.  ``.chunks()`` blocks on ``q.get()``.
    Cannot be tested automatically (no fake mic) — verify manually once
    everything else works.

Checkpoint
----------
Run ``python audio_capture.py`` to generate a synthetic sine-wave WAV
and confirm FileAudioCapture yields the correct number and shape of
chunks.
"""

from __future__ import annotations

import queue
import time
import logging
from typing import Iterator

import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix a stereo (or N-channel) array to mono by averaging channels.

    Parameters
    ----------
    audio:
        Shape ``(samples,)`` for mono or ``(samples, channels)`` for multi.

    Returns
    -------
    np.ndarray
        1-D float32 array.
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Implementation 1 — File replay
# ---------------------------------------------------------------------------

class FileAudioCapture:
    """Replay a pre-recorded audio file as a stream of fixed-size chunks.

    Parameters
    ----------
    path:
        Path to the audio file (WAV, FLAC, OGG, …).  Must be 16 kHz.
    chunk_s:
        Duration of each yielded chunk in seconds.  Defaults to 0.5 s.
    realtime:
        If ``True`` (default), ``time.sleep()`` is inserted between yields
        so downstream code experiences the same cadence as a live mic.
        Set to ``False`` for fast offline processing.

    Raises
    ------
    ValueError
        If the file's sample rate is not ``config.SAMPLE_RATE`` (16 kHz).

    Example
    -------
    >>> cap = FileAudioCapture("speech.wav", chunk_s=0.5, realtime=False)
    >>> for chunk in cap.chunks():
    ...     process(chunk)
    """

    def __init__(
        self,
        path: str,
        chunk_s: float = 0.5,
        realtime: bool = True,
    ) -> None:
        self.path = path
        self.chunk_s = chunk_s
        self.realtime = realtime

    def chunks(self) -> Iterator[np.ndarray]:
        """Yield fixed-size float32 mono chunks from the audio file.

        Yields
        ------
        np.ndarray
            Shape ``(chunk_frames,)`` float32 at ``config.SAMPLE_RATE`` Hz.
            The final chunk may be shorter than ``chunk_frames`` if the
            file length is not an exact multiple.
        """
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "soundfile is required for FileAudioCapture.  "
                "Install it with:  pip install soundfile"
            ) from exc

        logger.info("FileAudioCapture: opening %s", self.path)

        # 1. Read the whole file — soundfile returns (samples, channels) or
        #    (samples,) for mono, and always respects the on-disk sample rate.
        audio, sr = sf.read(self.path, dtype="float32")

        # 2. Hard check — don't silently resample; tell the caller to fix it.
        if sr != config.SAMPLE_RATE:
            raise ValueError(
                f"File '{self.path}' has sample rate {sr} Hz, but "
                f"Whisper requires exactly {config.SAMPLE_RATE} Hz.  "
                f"Please resample the file before passing it to FileAudioCapture."
            )

        # 3. Downmix to mono if needed.
        audio = _ensure_mono(audio)

        chunk_frames = int(config.SAMPLE_RATE * self.chunk_s)
        total_frames = len(audio)
        n_chunks = (total_frames + chunk_frames - 1) // chunk_frames  # ceil

        logger.info(
            "FileAudioCapture: %.2f s of audio → %d chunks of %d frames (%.2f s each)",
            total_frames / config.SAMPLE_RATE,
            n_chunks,
            chunk_frames,
            self.chunk_s,
        )

        # 4. Slice and yield.
        for start in range(0, total_frames, chunk_frames):
            chunk = audio[start : start + chunk_frames]

            yield chunk

            # 5. Throttle to real-time if requested.
            if self.realtime:
                time.sleep(self.chunk_s)

        logger.info("FileAudioCapture: finished streaming %s", self.path)


# ---------------------------------------------------------------------------
# Implementation 2 — Live microphone
# ---------------------------------------------------------------------------

class MicAudioCapture:
    """Capture audio from the default system microphone in real time.

    Uses ``sounddevice.InputStream`` with a low-latency callback that
    pushes frames into a ``queue.Queue``.  ``.chunks()`` blocks on
    ``q.get()`` and yields each chunk as it arrives.

    Parameters
    ----------
    chunk_s:
        Duration of each yielded chunk in seconds.  Defaults to 0.5 s.
    device:
        sounddevice device index or name.  ``None`` → system default.

    Notes
    -----
    * This class cannot be tested in an automated way because there is no
      fake microphone.  Verify it manually once all other pipeline steps
      work.
    * The queue has a bounded size (``config.WS_QUEUE_MAX_SIZE``) to
      apply back-pressure if the consumer falls behind.

    Example
    -------
    >>> cap = MicAudioCapture(chunk_s=0.5)
    >>> for chunk in cap.chunks():
    ...     process(chunk)
    """

    def __init__(
        self,
        chunk_s: float = 0.5,
        device: int | str | None = None,
    ) -> None:
        self.chunk_s = chunk_s
        self.device = device
        self._q: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=config.WS_QUEUE_MAX_SIZE
        )

    def chunks(self) -> Iterator[np.ndarray]:
        """Yield float32 mono chunks captured from the microphone.

        Yields
        ------
        np.ndarray
            Shape ``(chunk_frames,)`` float32 at ``config.SAMPLE_RATE`` Hz.

        Raises
        ------
        ImportError
            If *sounddevice* is not installed.
        """
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sounddevice is required for MicAudioCapture.  "
                "Install it with:  pip install sounddevice"
            ) from exc

        chunk_frames = int(config.SAMPLE_RATE * self.chunk_s)

        def _callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            """sounddevice low-latency callback — runs in a separate thread."""
            if status:
                logger.warning("MicAudioCapture callback status: %s", status)
            # indata shape: (frames, channels) — downmix to mono float32.
            mono = _ensure_mono(indata.copy())
            try:
                self._q.put_nowait(mono)
            except queue.Full:
                logger.warning(
                    "MicAudioCapture: output queue full — dropping chunk"
                )

        logger.info(
            "MicAudioCapture: opening mic stream (device=%s, chunk_s=%.2f)",
            self.device,
            self.chunk_s,
        )

        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="float32",
            blocksize=chunk_frames,
            device=self.device,
            callback=_callback,
        ):
            logger.info("MicAudioCapture: streaming started — press Ctrl+C to stop")
            while True:
                yield self._q.get()


# ---------------------------------------------------------------------------
# Checkpoint / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os, sys
    import soundfile as sf  # type: ignore[import]

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    CHUNK_S = 0.5
    DURATION_S = 3.0
    FREQ_HZ = 440.0

    # Generate a 440 Hz sine wave at 16 kHz
    t = np.linspace(0, DURATION_S, int(config.SAMPLE_RATE * DURATION_S), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * FREQ_HZ * t)).astype(np.float32)

    wrong_sr_path = ""  # declared here so finally can always reference it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        sf.write(tmp_path, sine, config.SAMPLE_RATE)
        print(f"\nWrote synthetic WAV: {tmp_path}")
        print(f"Duration: {DURATION_S:.1f} s  |  Sample rate: {config.SAMPLE_RATE} Hz")

        cap = FileAudioCapture(tmp_path, chunk_s=CHUNK_S, realtime=False)
        expected_chunks = int(np.ceil(DURATION_S / CHUNK_S))
        chunk_frames    = int(config.SAMPLE_RATE * CHUNK_S)

        chunks = list(cap.chunks())

        print(f"\nExpected chunks : {expected_chunks}")
        print(f"Received chunks : {len(chunks)}")
        print(f"Expected frames : {chunk_frames} per chunk (last may be shorter)")

        for i, c in enumerate(chunks):
            shape_ok = c.shape[0] <= chunk_frames
            dtype_ok = c.dtype == np.float32
            status   = "OK" if (shape_ok and dtype_ok) else "FAIL"
            print(f"  chunk[{i:02d}]  shape={c.shape}  dtype={c.dtype}  [{status}]")

        assert len(chunks) == expected_chunks, (
            f"Chunk count mismatch: got {len(chunks)}, expected {expected_chunks}"
        )
        for i, c in enumerate(chunks[:-1]):
            assert c.shape == (chunk_frames,), f"chunk[{i}] wrong shape: {c.shape}"
            assert c.dtype == np.float32,       f"chunk[{i}] wrong dtype: {c.dtype}"

        print("\n[PASS] Checkpoint passed -- FileAudioCapture yields the correct chunks.")

        # Verify wrong sample-rate raises ValueError
        wrong_sr_path = tmp_path.replace(".wav", "_8k.wav")  # noqa: F841
        sf.write(wrong_sr_path, sine[:8000], 8000)
        try:
            list(FileAudioCapture(wrong_sr_path, realtime=False).chunks())
            print("[FAIL] Expected ValueError for wrong sample rate -- NOT raised!")
            sys.exit(1)
        except ValueError as e:
            print(f"[PASS] Wrong-sample-rate guard raised ValueError as expected:\n  {e}")

    finally:
        os.unlink(tmp_path)
        if os.path.exists(wrong_sr_path):
            os.unlink(wrong_sr_path)
