"""
streaming_asr.py  —  LocalAgreement-n streaming transcription engine
=====================================================================
Reference: Macháček et al., "Turning Whisper into a Real-Time
Transcription System," EMNLP 2023.  https://arxiv.org/abs/2307.14743

The problem
-----------
Whisper has no native streaming mode.  Every ``model.transcribe(audio)``
call re-transcribes the *entire* buffer from scratch.  "Streaming" must
therefore be built on top by calling Whisper repeatedly on a growing
buffer and deciding which prefix of the output is stable enough to
commit as final.

The LocalAgreement-n policy
----------------------------
If you transcribe the same buffer twice — a few tenths of a second apart
— and both transcriptions agree on the first *k* words, Whisper is
extremely unlikely to change its mind about those words no matter how
much more audio arrives.  The agreed prefix is committed as FINAL; the
tail remains tentative.

More precisely: we keep a history of ``AGREEMENT_N - 1`` prior polls.
When the current poll's word list shares a common prefix with *all*
prior polls, that prefix is committed.

State
-----
``buffer``             float32 array of audio not yet committed.
``buffer_start_time``  stream-relative time of ``buffer[0]`` (seconds).
``prior_hypotheses``   list of word lists from recent non-committing polls
                       (max length ``AGREEMENT_N - 1``).

Critical bug to avoid
---------------------
Do **not** append the current hypothesis to ``prior_hypotheses`` *before*
computing agreement — it would compare the hypothesis against itself,
trivially agree on everything, and commit the entire output on the very
first poll.  Only append when *nothing* was committed.

Word objects
------------
faster-whisper yields word objects with ``.word``, ``.start``, ``.end``
attributes.  We use lightweight ``_Word`` namedtuples internally so
tests can inject any objects that have those three attributes.
"""

from __future__ import annotations

import logging
import time
from collections import namedtuple
from typing import Iterator, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

import config
from output_schema import TranscriptSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal word representation
# ---------------------------------------------------------------------------

_Word = namedtuple("_Word", ["word", "start", "end"])


# ---------------------------------------------------------------------------
# Protocol — what StreamingASR expects from its model
# ---------------------------------------------------------------------------

@runtime_checkable
class WhisperLike(Protocol):
    """Minimal interface required from a Whisper model.

    Both the real ``faster_whisper.WhisperModel`` and the test stub
    ``ScriptedWhisperModel`` satisfy this protocol.
    """

    def transcribe(
        self,
        audio: np.ndarray,
        **kwargs,
    ):
        """Return ``(segments_iterable, info)``."""
        ...


# ---------------------------------------------------------------------------
# Test stub — ScriptedWhisperModel
# ---------------------------------------------------------------------------

class ScriptedWhisperModel:
    """Returns one canned hypothesis per call, ignoring the audio entirely.

    Use this in unit tests to verify LocalAgreement logic without
    downloading a real model.

    Parameters
    ----------
    hypotheses:
        List of word-text lists.  Each element is what one ``transcribe``
        call returns as its word sequence.  For example::

            ScriptedWhisperModel([
                ["The"],
                ["The", "economy"],
                ["The", "economy", "grew"],
            ])

    Example
    -------
    >>> model = ScriptedWhisperModel([["hello"], ["hello", "world"]])
    >>> seg = StreamingASR(model=model)
    """

    def __init__(self, hypotheses: list[list[str]]) -> None:
        self.hypotheses = hypotheses
        self.call_count = 0

    def transcribe(self, audio: np.ndarray, **kwargs):
        words_text = self.hypotheses[self.call_count % len(self.hypotheses)]
        self.call_count += 1

        # Build fake word objects with .word / .start / .end
        fake_words = [
            _Word(word=w, start=float(i * 0.5), end=float(i * 0.5 + 0.4))
            for i, w in enumerate(words_text)
        ]

        # Wrap in a fake segment with a .words list
        FakeSegment = namedtuple("FakeSegment", ["words"])
        return [FakeSegment(words=fake_words)], None


# ---------------------------------------------------------------------------
# Helper — common_prefix
# ---------------------------------------------------------------------------

def common_prefix(word_lists: list[list[_Word]]) -> list[_Word]:
    """Return the longest word prefix shared by all lists in *word_lists*.

    Comparison is case-insensitive and whitespace-stripped.  Timestamps
    are taken from the **last** list (the most recent hypothesis, which
    had the most audio context when estimating them).

    Parameters
    ----------
    word_lists:
        Two or more lists of word objects (each with a ``.word`` attr).
        Must be non-empty.

    Returns
    -------
    list[_Word]
        Agreed prefix words, using timestamps from ``word_lists[-1]``.
        Empty list if the lists disagree at position 0.

    Example
    -------
    >>> h1 = [_Word("The", 0.0, 0.3), _Word("cat", 0.3, 0.6)]
    >>> h2 = [_Word("The", 0.0, 0.3), _Word("cat", 0.3, 0.6), _Word("sat", 0.6, 0.9)]
    >>> common_prefix([h1, h2])
    [_Word(word='The', ...), _Word(word='cat', ...)]
    """
    if not word_lists:
        return []

    reference = word_lists[-1]  # timestamps come from here
    agreed: list[_Word] = []

    for pos in range(len(reference)):
        ref_text = reference[pos].word.strip().lower()
        # Check every *other* list also has this word at this position
        for other in word_lists[:-1]:
            if pos >= len(other):
                return agreed          # other list is shorter — stop here
            if other[pos].word.strip().lower() != ref_text:
                return agreed          # disagreement — stop here
        # All lists agree at pos
        agreed.append(reference[pos])

    return agreed


# ---------------------------------------------------------------------------
# Core engine — StreamingASR
# ---------------------------------------------------------------------------

class StreamingASR:
    """Streaming transcription engine using the LocalAgreement-n policy.

    Parameters
    ----------
    model:
        An object satisfying :class:`WhisperLike`.  If ``None``, a real
        ``faster_whisper.WhisperModel`` is loaded using the sizes and
        settings defined in :mod:`config`.
    buffer_start_time:
        Stream-relative timestamp of the very first audio sample (seconds).
        Leave at 0.0 unless you are joining mid-stream.

    Usage
    -----
    ::

        asr = StreamingASR()
        for chunk in audio_capture.chunks():
            asr.feed(chunk)
            for segment in asr.poll():
                downstream_queue.put(segment)

    Notes
    -----
    ``poll()`` may emit zero, one, or two :class:`~output_schema.TranscriptSegment`
    objects per call:

    * One FINAL segment when an agreed prefix is committed.
    * One TENTATIVE segment (``is_final=False``) for the remaining tail,
      if any words are still in flux.

    Module 2 **must** ignore segments where ``is_final=False``.
    """

    def __init__(
        self,
        model: Optional[WhisperLike] = None,
        buffer_start_time: float = 0.0,
    ) -> None:
        # ---- model ----
        if model is not None:
            self._model: WhisperLike = model
            logger.debug("StreamingASR: using injected model %s", type(model).__name__)
        else:
            self._model = self._load_real_model()

        # ---- state ----
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._buffer_start_time: float = buffer_start_time

        # History of word lists from polls where nothing was committed.
        # Max length: AGREEMENT_N - 1
        self._prior_hypotheses: list[list[_Word]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, chunk: np.ndarray) -> None:
        """Append a new audio chunk to the uncommitted buffer.

        Parameters
        ----------
        chunk:
            float32 numpy array at ``config.SAMPLE_RATE`` Hz.
        """
        self._buffer = np.concatenate([self._buffer, chunk])
        logger.debug(
            "feed: buffer now %.2f s",
            len(self._buffer) / config.SAMPLE_RATE,
        )

    def poll(self) -> list[TranscriptSegment]:
        """Run one transcription pass and apply the LocalAgreement policy.

        Returns
        -------
        list[TranscriptSegment]
            Between 0 and 2 segments.  A FINAL segment is emitted when an
            agreed prefix is committed; a TENTATIVE segment follows if words
            remain in flux.  An empty list is returned when the buffer is
            empty.

        Side-effects
        ------------
        * On commit: trims ``self._buffer`` and advances
          ``self._buffer_start_time``; clears ``self._prior_hypotheses``.
        * On no-commit: appends current word list to
          ``self._prior_hypotheses`` (capped at ``AGREEMENT_N - 1``).
        """
        if len(self._buffer) == 0:
            return []

        buffer_duration = len(self._buffer) / config.SAMPLE_RATE
        emitted: list[TranscriptSegment] = []

        # ----------------------------------------------------------------
        # 1. Transcribe the entire current buffer
        # ----------------------------------------------------------------
        words = self._transcribe(self._buffer)

        logger.debug(
            "poll: buffer=%.2f s  words=%s  prior_hyps=%d",
            buffer_duration,
            [w.word for w in words],
            len(self._prior_hypotheses),
        )

        # ----------------------------------------------------------------
        # 2. Determine agreed prefix (LocalAgreement-n)
        # ----------------------------------------------------------------
        if buffer_duration >= config.MAX_UNCOMMITTED_S:
            # Safety flush — buffer is getting too large; commit everything.
            agreed = words
            logger.warning(
                "poll: buffer exceeded MAX_UNCOMMITTED_S=%.1f s — force-flushing",
                config.MAX_UNCOMMITTED_S,
            )
        elif len(self._prior_hypotheses) >= config.AGREEMENT_N - 1:
            # We have enough history — find the common prefix.
            agreed = common_prefix(self._prior_hypotheses + [words])
        else:
            # Not enough history yet — cannot agree on anything.
            agreed = []

        # ----------------------------------------------------------------
        # 3. Emit and advance state
        # ----------------------------------------------------------------
        if agreed:
            # -- Emit FINAL segment for the agreed prefix --
            final_text = " ".join(w.word.strip() for w in agreed)
            final_start = self._buffer_start_time + agreed[0].start
            final_end   = self._buffer_start_time + agreed[-1].end

            emitted.append(TranscriptSegment(
                start=final_start,
                end=final_end,
                text=final_text,
                is_final=True,
                emitted_at=time.time(),
            ))
            logger.info("FINAL  [%.2f–%.2f] %r", final_start, final_end, final_text)

            # -- Trim the buffer: drop audio up through agreed[-1].end --
            trim_frames = int(agreed[-1].end * config.SAMPLE_RATE)
            trim_frames = min(trim_frames, len(self._buffer))  # safety clamp
            trim_s      = trim_frames / config.SAMPLE_RATE

            self._buffer = self._buffer[trim_frames:]
            self._buffer_start_time += trim_s

            # -- Reset hypothesis history (buffer content has shifted) --
            self._prior_hypotheses.clear()

            # -- Remainder of words = new tentative tail --
            words = words[len(agreed):]

        else:
            # Nothing committed — remember this hypothesis for next poll.
            # Cap the history at AGREEMENT_N - 1 entries.
            self._prior_hypotheses.append(words)
            if len(self._prior_hypotheses) > config.AGREEMENT_N - 1:
                self._prior_hypotheses.pop(0)

        # ----------------------------------------------------------------
        # 4. Emit TENTATIVE segment for whatever is still in flux
        # ----------------------------------------------------------------
        if words:
            tent_text  = " ".join(w.word.strip() for w in words)
            tent_start = self._buffer_start_time + words[0].start
            tent_end   = self._buffer_start_time + words[-1].end

            emitted.append(TranscriptSegment(
                start=tent_start,
                end=tent_end,
                text=tent_text,
                is_final=False,
                emitted_at=time.time(),
            ))
            logger.debug("TENTATIVE [%.2f–%.2f] %r", tent_start, tent_end, tent_text)

        return emitted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transcribe(self, audio: np.ndarray) -> list[_Word]:
        """Run the model and flatten all words across all segments.

        Returns
        -------
        list[_Word]
            Flat list of all word objects from the transcription.
        """
        segments, _info = self._model.transcribe(
            audio,
            language=config.WHISPER_LANGUAGE,
            word_timestamps=True,
            beam_size=config.WHISPER_BEAM_SIZE,
            # Re-decode independently each poll — letting Whisper condition
            # on its own (possibly wrong) previous output can cause it to
            # double-down on mistakes.
            condition_on_previous_text=False,
        )
        words: list[_Word] = []
        for seg in segments:
            for w in seg.words:
                words.append(_Word(word=w.word, start=w.start, end=w.end))
        return words

    @staticmethod
    def _load_real_model() -> WhisperLike:
        """Load a faster-whisper WhisperModel using settings from config."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required.  "
                "Install it with:  pip install faster-whisper"
            ) from exc

        logger.info(
            "Loading faster-whisper model: size=%s device=%s compute_type=%s",
            config.WHISPER_MODEL_SIZE,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
        return WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )


# ---------------------------------------------------------------------------
# Checkpoint / unit test  (no real model needed)
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Verify LocalAgreement logic against a ScriptedWhisperModel.

    Scripted sequence (AGREEMENT_N = 2, so we need 1 prior hypothesis):

    Poll 0:  ["The"]                        → prior_hyps=[["The"]]
    Poll 1:  ["The", "economy"]             → prefix=["The"] → FINAL "The"
                                              tentative="economy"
    Poll 2:  ["economy", "grew"]            → prior_hyps=[["economy","grew"]]
    Poll 3:  ["economy", "grew", "by"]      → prefix=["economy","grew"]
                                              → FINAL "economy grew"
                                              tentative="by"

    Expected: exactly 2 FINAL commits — "The" then "economy grew".
    """
    import config as cfg

    # Override AGREEMENT_N to 2 for this test
    original_n = cfg.AGREEMENT_N
    cfg.AGREEMENT_N = 2

    hypotheses = [
        ["The"],
        ["The", "economy"],
        ["economy", "grew"],
        ["economy", "grew", "by"],
    ]
    model = ScriptedWhisperModel(hypotheses)
    asr   = StreamingASR(model=model)

    # Feed a dummy 0.5-s chunk before each poll so buffer is non-empty
    dummy_chunk = np.zeros(int(cfg.SAMPLE_RATE * 0.5), dtype=np.float32)

    finals: list[TranscriptSegment] = []
    tentatives: list[TranscriptSegment] = []

    for poll_idx in range(4):
        asr.feed(dummy_chunk)
        segments = asr.poll()
        for s in segments:
            if s.is_final:
                finals.append(s)
                print(f"  [FINAL]     poll={poll_idx}  text={s.text!r}  [{s.start:.2f}–{s.end:.2f}]")
            else:
                tentatives.append(s)
                print(f"  [TENTATIVE] poll={poll_idx}  text={s.text!r}  [{s.start:.2f}–{s.end:.2f}]")

    # Restore
    cfg.AGREEMENT_N = original_n

    # Assertions
    assert len(finals) == 2, (
        f"Expected 2 FINAL segments, got {len(finals)}: {[f.text for f in finals]}"
    )
    assert finals[0].text == "The", (
        f"First commit should be 'The', got {finals[0].text!r}"
    )
    assert finals[1].text == "economy grew", (
        f"Second commit should be 'economy grew', got {finals[1].text!r}"
    )
    assert finals[0].is_final is True
    assert finals[1].is_final is True

    print("\n[PASS] LocalAgreement test: 2 commits in the right order.")
    print(f"       Commit 1: {finals[0].text!r}")
    print(f"       Commit 2: {finals[1].text!r}")

    # ---- common_prefix edge cases ----
    h1 = [_Word("hello", 0.0, 0.5), _Word("world", 0.5, 1.0)]
    h2 = [_Word("hello", 0.0, 0.5), _Word("world", 0.5, 1.0), _Word("again", 1.0, 1.5)]
    h3 = [_Word("hello", 0.0, 0.5)]

    cp = common_prefix([h1, h2])
    assert [w.word for w in cp] == ["hello", "world"], f"common_prefix failed: {cp}"

    cp2 = common_prefix([h1, h3])
    assert [w.word for w in cp2] == ["hello"], f"common_prefix (shorter list) failed: {cp2}"

    cp3 = common_prefix([[_Word("foo", 0, 1)], [_Word("bar", 0, 1)]])
    assert cp3 == [], f"common_prefix (no agreement) failed: {cp3}"

    print("[PASS] common_prefix edge cases.")

    print("\n[PASS] All checkpoint tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    _run_tests()
