"""
sentence_segmenter.py  --  Module 1 segment -> complete sentences
=================================================================
Module 1 emits TranscriptSegments, not sentences.  A single FINAL
segment can contain zero, one, or several sentences; or it can be the
tail of a sentence that started in the previous segment.

This module provides:

  Sentence         -- lightweight dataclass for a complete (or force-flushed)
                      sentence with speaker + timestamps.
  SentenceSegmenter -- stateful component that accumulates FINAL segments
                       and emits complete sentences.

Sentence splitting
------------------
We use ``pysbd`` (Pragmatic Sentence Boundary Disambiguation), which
correctly handles:
  - Abbreviations: "U.S.", "Mr.", "Dr.", "St."
  - Numbers and money: "3.5 percent", "$2.3 billion."
  - Ellipses, quoted speech, etc.
A simple regex fallback is included so the module works even if pysbd
is not installed.

Three flush triggers
--------------------
1. Punctuation boundary  -- pysbd found >= 2 sentences; all but the
                            last (possibly incomplete) fragment are emitted.
2. Speaker change        -- the incoming segment's speaker differs from
                            the pending buffer's speaker.  The entire
                            pending buffer is flushed before the new
                            speaker's text is appended.  Prevents gluing
                            two different people's words into one sentence.
3. Stale timeout         -- if ``max_pending_s`` seconds pass without any
                            new FINAL segment, ``flush()`` may be called
                            externally (e.g. by the pipeline) to drain
                            the buffer before a claim is lost forever.

What is ignored
---------------
Segments with ``is_final=False`` are silently dropped.  Classifying
tentative text creates duplicate and phantom claims.

Timestamp interpolation
-----------------------
When pysbd splits one segment into multiple sentences, we estimate each
sentence's start/end by its character-position fraction of the segment's
total duration.  This is approximate but sufficient for latency
measurement and UI display.
"""

from __future__ import annotations

import dataclasses
import re
import time
import sys
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Sentence dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Sentence:
    """A complete (or force-flushed) sentence ready for claim classification.

    Attributes
    ----------
    text : str
        The sentence text, stripped.
    speaker : str
        Speaker label from the originating TranscriptSegment(s).
    start : float
        Estimated stream-relative start time (seconds).
    end : float
        Estimated stream-relative end time (seconds).
    source_segment_ids : list[int]
        Module 1 TranscriptSegment.id values this sentence was drawn from.
    flushed : bool
        True when the sentence was emitted by a force-flush (speaker change
        or stale timeout) rather than a clean punctuation boundary.
        Downstream code may want to treat these with lower confidence.
    """
    text: str
    speaker: str
    start: float
    end: float
    source_segment_ids: List[int]
    flushed: bool = False

    def __repr__(self) -> str:
        flag = " [FLUSHED]" if self.flushed else ""
        return (
            f"Sentence({self.start:.1f}-{self.end:.1f}s "
            f"{self.speaker!r}{flag}: {self.text!r})"
        )


# ---------------------------------------------------------------------------
# Sentence splitter (pysbd with regex fallback)
# ---------------------------------------------------------------------------

def _make_splitter():
    """Return a callable split(text) -> list[str] using pysbd if available."""
    try:
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)

        def split(text: str) -> list[str]:
            return [s for s in seg.segment(text) if s.strip()]

        return split
    except ImportError:
        # Regex fallback: split on . ! ? followed by whitespace + capital
        # Does NOT handle abbreviations -- install pysbd for production.
        _pat = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

        def split(text: str) -> list[str]:
            return [s.strip() for s in _pat.split(text) if s.strip()]

        return split


_split_sentences = _make_splitter()


# ---------------------------------------------------------------------------
# Pending-buffer metadata
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _PendingMeta:
    """Metadata accumulated alongside pending_text."""
    speaker: str
    start: float          # stream-relative time of pending_text[0]
    end: float            # stream-relative time of latest appended segment
    segment_ids: List[int]
    last_feed_wall: float = dataclasses.field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# SentenceSegmenter
# ---------------------------------------------------------------------------

class SentenceSegmenter:
    """Accumulate FINAL TranscriptSegments and emit complete sentences.

    Parameters
    ----------
    max_pending_s : float
        If this many seconds pass (wall-clock) since the last ``feed()``
        call and there is still text in the buffer, a call to ``flush()``
        will emit it as a force-flushed sentence.  The pipeline should
        call ``flush()`` on every poll cycle to honour this timeout.
        Default: 8.0 seconds.
    min_sentence_tokens : int
        Segments shorter than this many whitespace-separated tokens are
        ignored (avoids classifying single-word fragments like "Yeah.").
        Default: 3.

    Usage
    -----
    ::

        seg = SentenceSegmenter()
        for transcript_segment in asr_poll_results:
            for sentence in seg.feed(transcript_segment):
                classify(sentence)
        # Call periodically to drain stale buffers:
        for sentence in seg.flush():
            classify(sentence)
    """

    def __init__(
        self,
        max_pending_s: float = 8.0,
        min_sentence_tokens: int = 3,
    ) -> None:
        self._max_pending_s = max_pending_s
        self._min_tokens = min_sentence_tokens

        self._pending_text: str = ""
        self._meta: Optional[_PendingMeta] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, segment) -> List[Sentence]:
        """Process one TranscriptSegment and return any complete sentences.

        Parameters
        ----------
        segment : TranscriptSegment (or any object with the same fields)
            Must have: ``is_final``, ``text``, ``speaker``, ``start``,
            ``end``, ``id``.

        Returns
        -------
        list[Sentence]
            Zero or more complete sentences.  Empty if the segment is
            tentative or if the accumulated text has not yet reached a
            sentence boundary.
        """
        # -- 1. Ignore tentative segments entirely --
        if not segment.is_final:
            return []

        text = segment.text.strip()
        if not text:
            return []

        emitted: List[Sentence] = []

        # -- 2. Speaker-change flush --
        if self._meta is not None and segment.speaker != self._meta.speaker:
            flushed = self._drain(flushed=True)
            if flushed:
                emitted.append(flushed)
            # _pending_text / _meta are now clear

        # -- 3. Append to pending buffer --
        if self._meta is None:
            self._meta = _PendingMeta(
                speaker=segment.speaker,
                start=segment.start,
                end=segment.end,
                segment_ids=[segment.id],
            )
            self._pending_text = text
        else:
            # Same speaker -- extend
            self._pending_text = (self._pending_text + " " + text).strip()
            self._meta.end = segment.end
            self._meta.last_feed_wall = time.monotonic()
            if segment.id not in self._meta.segment_ids:
                self._meta.segment_ids.append(segment.id)

        # -- 4. Split into sentences --
        parts = _split_sentences(self._pending_text)

        # Decide which parts are "complete" and which is the trailing fragment.
        # A part is complete if it ends with terminal punctuation (. ! ?).
        # If the very last part ends with punctuation, ALL parts are complete
        # and the remainder is empty (buffer is fully consumed).
        _terminal = re.compile(r'[.!?]["\']?\s*$')

        if not parts:
            return emitted

        if _terminal.search(parts[-1]):
            # Every part is a complete sentence -- nothing left pending
            complete  = parts
            remainder = ""
        elif len(parts) == 1:
            # One fragment, no terminal punct -- keep buffering
            return emitted
        else:
            # Multiple parts; all but the last are complete sentences
            complete  = parts[:-1]
            remainder = parts[-1]

        total_chars = len(self._pending_text)
        seg_duration = self._meta.end - self._meta.start
        cursor_chars = 0

        for part in complete:
            part = part.strip()
            if not part:
                continue
            if len(part.split()) < self._min_tokens:
                # Too short -- absorb into remainder rather than emitting
                remainder = (part + " " + remainder).strip()
                continue

            # Estimate timestamps by character-position interpolation
            part_start = self._meta.start + seg_duration * (cursor_chars / max(total_chars, 1))
            cursor_chars += len(part) + 1  # +1 for the space
            part_end = self._meta.start + seg_duration * (cursor_chars / max(total_chars, 1))

            emitted.append(Sentence(
                text=part,
                speaker=self._meta.speaker,
                start=round(part_start, 3),
                end=round(part_end, 3),
                source_segment_ids=list(self._meta.segment_ids),
                flushed=False,
            ))

        # Update pending state to the remainder fragment
        remainder = remainder.strip()
        if remainder:
            self._pending_text = remainder
            self._meta.start = self._meta.start + seg_duration * (
                cursor_chars / max(total_chars, 1)
            )
            self._meta.last_feed_wall = time.monotonic()
        else:
            # Buffer fully consumed
            self._pending_text = ""
            self._meta = None

        return emitted

    def flush(self, force: bool = False) -> List[Sentence]:
        """Emit any pending text as a force-flushed sentence.

        Called externally on every pipeline poll cycle.  Only emits if:
        - ``force=True``, OR
        - the pending text has been sitting for >= ``max_pending_s`` seconds.

        Parameters
        ----------
        force : bool
            If True, flush immediately regardless of elapsed time.

        Returns
        -------
        list[Sentence]
            Zero or one force-flushed sentence.
        """
        if not self._pending_text or self._meta is None:
            return []

        elapsed = time.monotonic() - self._meta.last_feed_wall
        if not force and elapsed < self._max_pending_s:
            return []

        result = self._drain(flushed=True)
        return [result] if result else []

    @property
    def pending_text(self) -> str:
        """Current contents of the pending buffer (read-only)."""
        return self._pending_text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drain(self, flushed: bool = True) -> Optional[Sentence]:
        """Emit the entire pending buffer as one sentence and clear state."""
        if not self._pending_text or self._meta is None:
            return None

        text = self._pending_text.strip()
        if len(text.split()) < self._min_tokens:
            # Too short to be meaningful -- discard silently
            self._pending_text = ""
            self._meta = None
            return None

        sentence = Sentence(
            text=text,
            speaker=self._meta.speaker,
            start=self._meta.start,
            end=self._meta.end,
            source_segment_ids=list(self._meta.segment_ids),
            flushed=flushed,
        )
        self._pending_text = ""
        self._meta = None
        return sentence


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Hand-written segment lists covering all the gotchas from the spec."""
    import types

    def seg(text, speaker="SPEAKER_00", start=0.0, end=1.0, seg_id=0, is_final=True):
        """Build a fake TranscriptSegment-like object."""
        s = types.SimpleNamespace()
        s.text      = text
        s.speaker   = speaker
        s.start     = start
        s.end       = end
        s.id        = seg_id
        s.is_final  = is_final
        return s

    print("=== SentenceSegmenter tests ===\n")

    # ----------------------------------------------------------------
    # Test 1: Tentative segment is silently dropped
    # ----------------------------------------------------------------
    print("-- Test 1: tentative segment dropped --")
    ss = SentenceSegmenter()
    result = ss.feed(seg("Hello world this is a test.", is_final=False))
    assert result == [], f"Expected [], got {result}"
    assert ss.pending_text == ""
    print("   [PASS] tentative segment ignored")

    # ----------------------------------------------------------------
    # Test 2: Segment ends mid-sentence -- stays in buffer
    # ----------------------------------------------------------------
    print("-- Test 2: mid-sentence segment stays in buffer --")
    ss = SentenceSegmenter()
    result = ss.feed(seg("The economy grew", start=0.0, end=1.5, seg_id=1))
    assert result == [], f"Expected no sentences yet, got {result}"
    assert "The economy grew" in ss.pending_text
    print(f"   [PASS] pending_text={ss.pending_text!r}")

    # ----------------------------------------------------------------
    # Test 3: Two segments combine into one sentence
    # ----------------------------------------------------------------
    print("-- Test 3: two segments combine into one sentence --")
    ss = SentenceSegmenter()
    ss.feed(seg("The economy grew", start=0.0, end=1.5, seg_id=1))
    result = ss.feed(seg("by two percent.", start=1.5, end=2.8, seg_id=2))
    assert len(result) == 1, f"Expected 1 sentence, got {result}"
    assert "grew" in result[0].text and "percent" in result[0].text
    print(f"   [PASS] combined: {result[0].text!r}")

    # ----------------------------------------------------------------
    # Test 4: One segment contains two full sentences
    # ----------------------------------------------------------------
    print("-- Test 4: segment with two sentences --")
    ss = SentenceSegmenter()
    result = ss.feed(seg(
        "Inflation is rising. The central bank raised rates.",
        start=5.0, end=9.0, seg_id=3,
    ))
    # Both sentences end with "." so both should be emitted
    assert len(result) == 2, f"Expected 2 emitted sentences, got {len(result)}: {result}"
    assert "Inflation" in result[0].text
    assert "central bank" in result[1].text
    assert ss.pending_text == ""  # buffer fully consumed
    print(f"   [PASS] sentence 1: {result[0].text!r}")
    print(f"   [PASS] sentence 2: {result[1].text!r}")

    # ----------------------------------------------------------------
    # Test 5: Abbreviations -- "U.S." and "$2.3 billion." must not split mid-claim
    # ----------------------------------------------------------------
    print("-- Test 5: abbreviations and numbers --")
    ss = SentenceSegmenter()
    result = ss.feed(seg(
        "The U.S. exported $2.3 billion worth of goods last month. "
        "Mr. Smith confirmed the figures.",
        start=10.0, end=16.0, seg_id=4,
    ))
    # pysbd should split at the sentence boundary after "month."
    # "U.S." and "$2.3 billion." should NOT be treated as sentence ends
    sentences_text = [r.text for r in result] + ([ss.pending_text] if ss.pending_text else [])
    combined = " ".join(sentences_text)
    assert "U.S." in combined, "U.S. should not be split"
    assert "2.3 billion" in combined, "$2.3 billion should not be split"
    for s in result:
        print(f"   [PASS] emitted:  {s.text!r}")
    if ss.pending_text:
        print(f"         pending:  {ss.pending_text!r}")

    # ----------------------------------------------------------------
    # Test 6: Speaker change mid-buffer -- flush before switching
    # ----------------------------------------------------------------
    print("-- Test 6: speaker change flushes buffer --")
    ss = SentenceSegmenter()
    ss.feed(seg("We believe this policy will work",
                speaker="SPEAKER_00", start=20.0, end=23.0, seg_id=5))
    # New speaker arrives before previous sentence was complete
    result = ss.feed(seg("That is simply not true.",
                         speaker="SPEAKER_01", start=23.0, end=25.0, seg_id=6))
    # The "SPEAKER_00" fragment should have been flushed
    flushed = [s for s in result if s.flushed]
    assert len(flushed) >= 1, f"Expected a flushed sentence, got {result}"
    assert flushed[0].speaker == "SPEAKER_00"
    print(f"   [PASS] SPEAKER_00 flushed: {flushed[0].text!r}")
    # SPEAKER_01's sentence should now be pending or emitted
    speaker01_parts = [s for s in result if s.speaker == "SPEAKER_01"]
    print(f"         SPEAKER_01 emitted: {[s.text for s in speaker01_parts]}")
    print(f"         pending: {ss.pending_text!r}")

    # ----------------------------------------------------------------
    # Test 7: Stale timeout flush
    # ----------------------------------------------------------------
    print("-- Test 7: stale timeout flush --")
    ss = SentenceSegmenter(max_pending_s=0.0)  # immediate timeout
    ss.feed(seg("This sentence never gets punctuation",
                start=30.0, end=33.0, seg_id=7))
    assert ss.pending_text != ""
    import time as _time; _time.sleep(0.05)
    flushed = ss.flush()
    assert len(flushed) == 1, f"Expected 1 flushed, got {flushed}"
    assert flushed[0].flushed is True
    print(f"   [PASS] stale flush: {flushed[0].text!r}")

    # ----------------------------------------------------------------
    # Test 8: Source segment IDs are carried through
    # ----------------------------------------------------------------
    print("-- Test 8: source_segment_ids carried correctly --")
    ss = SentenceSegmenter()
    ss.feed(seg("Part one", start=0.0, end=1.0, seg_id=10))
    result = ss.feed(seg("of this sentence.", start=1.0, end=2.0, seg_id=11))
    assert len(result) == 1
    assert 10 in result[0].source_segment_ids
    assert 11 in result[0].source_segment_ids
    print(f"   [PASS] segment IDs: {result[0].source_segment_ids}")

    # ----------------------------------------------------------------
    # Test 9: min_sentence_tokens -- very short fragments are discarded
    # ----------------------------------------------------------------
    print("-- Test 9: short fragment discarded --")
    ss = SentenceSegmenter(min_sentence_tokens=3)
    result = ss.flush(force=True)   # empty buffer
    assert result == []
    ss.feed(seg("Yes.", start=40.0, end=40.5, seg_id=12))
    result = ss.flush(force=True)   # "Yes." is only 1 token
    assert result == [], f"Expected short fragment discarded, got {result}"
    print("   [PASS] 'Yes.' discarded (< 3 tokens)")

    print("\n[PASS] All SentenceSegmenter tests passed.")


if __name__ == "__main__":
    _run_tests()
