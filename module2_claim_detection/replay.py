"""
replay.py  —  Feed a recorded .jsonl fixture into Module 2
===========================================================
Reads a .jsonl file produced by dump_segments.py and replays the segments
into the Module 2 pipeline at controllable speed.

Usage
-----
From the module2_claim_detection directory::

    # Real-time replay (honours original timestamps)
    python replay.py

    # 5× faster than real time
    python replay.py --speed 5

    # As fast as possible (no delay)
    python replay.py --speed 0

    # Different fixture file
    python replay.py --file data/my_speech.jsonl

    # Evaluate model precision/recall against hand-labels in the fixture
    python replay.py --evaluate

What it does
------------
1. Reads each line from the .jsonl file as a JSON dict.
2. Constructs a TranscriptSegment (from Module 1's schema).
3. Calls segmenter.feed(segment) to get complete sentences.
4. Scores each sentence with the classifier.
5. Passes accepted claims through the deduplicator.
6. Prints each ClaimEvent to stdout (and optionally evaluates vs. labels).

The replay script is intentionally standalone — no WebSocket, no asyncio,
just a tight synchronous loop you can run, debug, and profile instantly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow imports from both modules when run from module2 dir
# ---------------------------------------------------------------------------
import os
_MODULE2_DIR = Path(__file__).parent
_MODULE1_DIR = _MODULE2_DIR.parent / "module1_speech_asr"

# Insert module1 at position 1 (after '' for relative imports)
if str(_MODULE1_DIR) not in sys.path:
    sys.path.insert(1, str(_MODULE1_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from output_schema import TranscriptSegment          # Module 1 schema
from sentence_segmenter import SentenceSegmenter     # Module 2 Step 2
from claim_schema import ClaimEvent                  # Module 2 Step 1
from claim_classifier import BaselineClaimClassifier # Module 2 Step 6
from deduper import ClaimDeduper                     # Module 2 Step 7


# ---------------------------------------------------------------------------
# Domain-mismatch evaluator
# ---------------------------------------------------------------------------

class DomainMismatchEvaluator:
    """
    Tracks TP/FP/FN for domain-mismatch evaluation.

    The .jsonl fixture carries a 'label' field (1 = check-worthy, 0 = not)
    at the SEGMENT level.  Sentences are formed by the segmenter, so the
    evaluator inherits the label from the segment(s) that produced each
    sentence (majority vote when a sentence spans multiple segments).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0
        self._records: list[dict] = []

    def record(self, text: str, score: float, true_label: int) -> None:
        predicted = int(score >= self.threshold)
        if predicted == 1 and true_label == 1:
            self.tp += 1
        elif predicted == 1 and true_label == 0:
            self.fp += 1
        elif predicted == 0 and true_label == 1:
            self.fn += 1
        else:
            self.tn += 1
        self._records.append(
            {"text": text[:80], "score": score,
             "predicted": predicted, "true": true_label}
        )

    def report(self) -> None:
        total = self.tp + self.fp + self.fn + self.tn
        prec  = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        rec   = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc   = (self.tp + self.tn) / total if total else 0.0

        print("\n" + "=" * 60)
        print("DOMAIN-MISMATCH EVALUATION (synthetic speech fixture)")
        print("=" * 60)
        print(f"  Threshold  : {self.threshold:.2f}")
        print(f"  Sentences  : {total}")
        print(f"  TP / FP    : {self.tp} / {self.fp}")
        print(f"  FN / TN    : {self.fn} / {self.tn}")
        print(f"  Precision  : {prec:.3f}")
        print(f"  Recall     : {rec:.3f}")
        print(f"  F1         : {f1:.3f}")
        print(f"  Accuracy   : {acc:.3f}")
        print("-" * 60)
        print("Per-sentence breakdown:")
        for r in self._records:
            mark = {(1, 1): "TP", (1, 0): "FP", (0, 1): "FN", (0, 0): "TN"}[
                (r["predicted"], r["true"])
            ]
            print(f"  [{mark}] score={r['score']:.3f}  {r['text']!r}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------

def replay(
    jsonl_path: Path,
    speed: float,
    threshold: float,
    evaluate: bool,
    verbose: bool,
) -> list[ClaimEvent]:
    """
    Replay segments from a .jsonl file through the Module 2 pipeline.

    Parameters
    ----------
    jsonl_path : Path
        Path to the .jsonl fixture file.
    speed : float
        Replay speed multiplier.  0 = no delay, 1 = real-time, 5 = 5×.
    threshold : float
        Minimum classifier score for a sentence to become a ClaimEvent.
    evaluate : bool
        If True, compare against 'label' field in the .jsonl for PR stats.
    verbose : bool
        If True, print every sentence (not just claims).

    Returns
    -------
    list[ClaimEvent]
        All claims emitted during the replay.
    """
    segmenter  = SentenceSegmenter()
    classifier = BaselineClaimClassifier()
    deduper    = ClaimDeduper()
    evaluator  = DomainMismatchEvaluator(threshold=threshold) if evaluate else None

    emitted_claims: list[ClaimEvent] = []
    # Sliding context window: last 2 sentences for the ClaimEvent.context field
    _context_buf: list[str] = []

    def _get_context() -> str:
        return " ".join(_context_buf[-2:]) if _context_buf else ""

    # Read all segments so we can honour original timing
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    segments_raw: list[dict] = [json.loads(l) for l in lines if l.strip()]

    if not segments_raw:
        print(f"[replay] No segments found in {jsonl_path}")
        return []

    print(f"\n{'='*60}")
    print(f"REPLAY: {jsonl_path.name}  ({len(segments_raw)} segments)")
    print(f"  speed={speed}×  threshold={threshold}  evaluate={evaluate}")
    print(f"{'='*60}\n")

    prev_end_time = segments_raw[0]["start"]
    replay_start  = time.monotonic()

    for raw in segments_raw:
        # --- Timing ---
        if speed > 0:
            gap_s = (raw["start"] - prev_end_time) / speed
            if gap_s > 0:
                time.sleep(gap_s)
        prev_end_time = raw.get("end", raw["start"])

        # --- Build TranscriptSegment ---
        seg = TranscriptSegment(
            start=raw["start"],
            end=raw.get("end", raw["start"] + 1.0),
            text=raw["text"],
            speaker=raw.get("speaker", "UNKNOWN"),
            is_final=raw.get("is_final", True),
            emitted_at=raw.get("emitted_at", time.time()),
        )
        true_label: int = raw.get("label", -1)  # -1 = unlabelled

        # --- Feed to segmenter ---
        sentences = segmenter.feed(seg)

        for sent in sentences:
            if verbose:
                print(f"  [sent] {sent.speaker} t={sent.start:.1f}s  {sent.text!r}")

            # --- Score ---
            score = classifier.score(sent.text, context=_get_context())

            # --- Evaluate if requested ---
            if evaluator and true_label >= 0:
                evaluator.record(sent.text, score, true_label)

            # --- Threshold + dedup ---
            if score >= threshold:
                claim = ClaimEvent(
                    text=sent.text,
                    context=_get_context(),
                    speaker=sent.speaker,
                    start=sent.start,
                    end=sent.end,
                    score=score,
                    source_segment_ids=[raw["id"]],
                )
                if deduper.check_and_add(claim):
                    # Duplicate — skip
                    if verbose:
                        print(f"    [dedup] skipped duplicate")
                    continue

                emitted_claims.append(claim)
                _context_buf.append(sent.text)

                elapsed = time.monotonic() - replay_start
                print(
                    f"  CLAIM [{claim.claim_id:3d}]  "
                    f"score={score:.3f}  "
                    f"t={sent.start:.1f}–{sent.end:.1f}s  "
                    f"speaker={sent.speaker}\n"
                    f"    \"{sent.text}\""
                )
            else:
                _context_buf.append(sent.text)

    # --- Flush any remaining text in the segmenter buffer ---
    for sent in segmenter.flush():
        score = classifier.score(sent.text, context=_get_context())
        if evaluator and true_label >= 0:
            evaluator.record(sent.text, score, true_label)
        if score >= threshold:
            claim = ClaimEvent(
                text=sent.text,
                context=_get_context(),
                speaker=sent.speaker,
                start=sent.start,
                end=sent.end,
                score=score,
                source_segment_ids=[],
            )
            if not deduper.check_and_add(claim):
                emitted_claims.append(claim)
                print(f"  CLAIM (flush) score={score:.3f}  \"{sent.text}\"")

    # --- Summary ---
    elapsed_total = time.monotonic() - replay_start
    print(f"\n{'='*60}")
    print(f"Replay finished in {elapsed_total:.1f}s")
    print(f"  Segments : {len(segments_raw)}")
    print(f"  Claims   : {len(emitted_claims)}")
    dedup_stats = deduper.stats()
    print(f"  Dedup rate: {dedup_stats['dedup_rate']:.1%}  "
          f"(pass1={dedup_stats['pass1_duplicates']}, "
          f"pass2={dedup_stats['pass2_duplicates']})")
    print(f"{'='*60}")

    if evaluator:
        evaluator.report()

    return emitted_claims


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay a .jsonl fixture through the Module 2 pipeline"
    )
    ap.add_argument(
        "--file", default="data/sample_recording.jsonl",
        help="Path to .jsonl fixture (default: data/sample_recording.jsonl)",
    )
    ap.add_argument(
        "--speed", type=float, default=0.0,
        help="Replay speed multiplier (0=instant, 1=real-time, default: 0)",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.5,
        help="Minimum classifier score to emit a ClaimEvent (default: 0.5)",
    )
    ap.add_argument(
        "--evaluate", action="store_true",
        help="Compare against hand-labels in the .jsonl for domain-mismatch PR stats",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print every sentence, not just emitted claims",
    )
    args = ap.parse_args()

    jsonl_path = Path(args.file)
    if not jsonl_path.exists():
        print(f"[replay] File not found: {jsonl_path}")
        print("Run first:  python dump_segments.py --source synthetic")
        sys.exit(1)

    replay(
        jsonl_path=jsonl_path,
        speed=args.speed,
        threshold=args.threshold,
        evaluate=args.evaluate,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

