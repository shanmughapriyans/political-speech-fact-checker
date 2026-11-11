"""
dump_segments.py  —  Record Module 1 output to a .jsonl file
=============================================================
Run Module 1 against a WAV file (or mic) and save every FINAL
TranscriptSegment to a newline-delimited JSON file.  The resulting
.jsonl becomes your reproducible development fixture for Module 2.

Usage
-----
From the module2_claim_detection directory::

    # Dump a WAV from Module 1 (Module 1 must be running):
    python dump_segments.py --source ws://localhost:8765 --out data/my_speech.jsonl

    # Or generate a synthetic fixture for unit-testing (no Module 1 needed):
    python dump_segments.py --source synthetic --out data/sample_recording.jsonl

The file format is one JSON object per line::

    {"id": 0, "start": 0.0, "end": 2.3, "text": "Hello.", ...}
    {"id": 1, "start": 2.5, "end": 5.1, "text": "The rate fell...", ...}

Pass --source synthetic to produce a ~40-segment political speech fixture
complete with hand-labels (a "label" field: 1 = check-worthy, 0 = not)
for domain-mismatch evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic fixture — a realistic political speech with 40 segments,
# covering a range of claim types and non-claim filler.
# "label" field: 1 = check-worthy claim, 0 = not check-worthy.
# This is your hand-labelled domain-mismatch test set.
# ---------------------------------------------------------------------------

SYNTHETIC_SEGMENTS: list[dict] = [
    # Segment 0  — opening filler
    {"id": 0, "start": 0.0,  "end": 2.1,  "speaker": "SPEAKER_00", "is_final": True, "text": "Thank you all for being here today.", "label": 0},
    # 1 — opinion, not a checkable fact
    {"id": 1, "start": 2.2,  "end": 4.6,  "speaker": "SPEAKER_00", "is_final": True, "text": "It is a great honour to address this chamber.", "label": 0},
    # 2 — specific statistical claim ✓
    {"id": 2, "start": 4.8,  "end": 8.3,  "speaker": "SPEAKER_00", "is_final": True, "text": "Unemployment has fallen to 3.8 percent, the lowest in a decade.", "label": 1},
    # 3 — mid-sentence stub (no punctuation yet)
    {"id": 3, "start": 8.4,  "end": 10.1, "speaker": "SPEAKER_00", "is_final": True, "text": "Our GDP grew by", "label": 0},
    # 4 — completes the sentence from 3
    {"id": 4, "start": 10.2, "end": 13.0, "speaker": "SPEAKER_00", "is_final": True, "text": "2.1 percent last quarter.", "label": 1},
    # 5 — causal claim ✓
    {"id": 5, "start": 13.1, "end": 17.4, "speaker": "SPEAKER_00", "is_final": True, "text": "We have added 450,000 jobs in the manufacturing sector since 2021.", "label": 1},
    # 6 — filler
    {"id": 6, "start": 17.5, "end": 20.0, "speaker": "SPEAKER_00", "is_final": True, "text": "I want to be very clear about what we stand for.", "label": 0},
    # 7 — normative, not factual
    {"id": 7, "start": 20.1, "end": 23.5, "speaker": "SPEAKER_00", "is_final": True, "text": "Every child in this country deserves a quality education.", "label": 0},
    # 8 — specific budget claim ✓
    {"id": 8, "start": 23.6, "end": 27.9, "speaker": "SPEAKER_00", "is_final": True, "text": "We have invested $4.2 billion in public schools over the past three years.", "label": 1},
    # 9 — comparative claim ✓
    {"id": 9, "start": 28.0, "end": 32.1, "speaker": "SPEAKER_00", "is_final": True, "text": "Test scores have risen by 12 points on average since our reforms began.", "label": 1},
    # 10 — rhetorical question
    {"id": 10, "start": 32.2, "end": 35.0, "speaker": "SPEAKER_00", "is_final": True, "text": "What kind of future do we want for our children?", "label": 0},
    # 11 — health stat ✓
    {"id": 11, "start": 35.1, "end": 39.4, "speaker": "SPEAKER_00", "is_final": True, "text": "Infant mortality has dropped by 18 percent since 2019.", "label": 1},
    # 12 — filler transition
    {"id": 12, "start": 39.5, "end": 42.0, "speaker": "SPEAKER_00", "is_final": True, "text": "Let me turn now to the economy.", "label": 0},
    # 13 — inflation claim ✓
    {"id": 13, "start": 42.1, "end": 46.8, "speaker": "SPEAKER_00", "is_final": True, "text": "Inflation is down to 2.4 percent, below the OECD average of 3.1 percent.", "label": 1},
    # 14 — anecdotal
    {"id": 14, "start": 46.9, "end": 50.5, "speaker": "SPEAKER_00", "is_final": True, "text": "I met a nurse last week who told me she finally feels secure.", "label": 0},
    # 15 — health spending ✓
    {"id": 15, "start": 50.6, "end": 55.2, "speaker": "SPEAKER_00", "is_final": True, "text": "Healthcare spending per capita is now $3,800, up from $3,100 in 2020.", "label": 1},
    # 16 — opinion
    {"id": 16, "start": 55.3, "end": 58.0, "speaker": "SPEAKER_00", "is_final": True, "text": "We believe everyone deserves access to affordable housing.", "label": 0},
    # 17 — housing stat ✓  — also tests abbreviation ("U.S.")
    {"id": 17, "start": 58.1, "end": 62.5, "speaker": "SPEAKER_00", "is_final": True, "text": "We built 120,000 new affordable units last year, more than any U.S. state.", "label": 1},
    # 18 — filler
    {"id": 18, "start": 62.6, "end": 64.9, "speaker": "SPEAKER_00", "is_final": True, "text": "These are real achievements.", "label": 0},
    # 19 — speaker change, now SPEAKER_01 (opposition response)
    {"id": 19, "start": 65.0, "end": 69.3, "speaker": "SPEAKER_01", "is_final": True, "text": "With respect, the government's record is far less impressive.", "label": 0},
    # 20 — opposition claim ✓
    {"id": 20, "start": 69.4, "end": 74.0, "speaker": "SPEAKER_01", "is_final": True, "text": "In reality, real wages have fallen by 4.7 percent over the same period.", "label": 1},
    # 21 — opposition claim ✓
    {"id": 21, "start": 74.1, "end": 78.8, "speaker": "SPEAKER_01", "is_final": True, "text": "Food bank usage has tripled since 2021, reaching a record 2.3 million visits.", "label": 1},
    # 22 — normative
    {"id": 22, "start": 78.9, "end": 81.5, "speaker": "SPEAKER_01", "is_final": True, "text": "That is not the record of a government that cares about working people.", "label": 0},
    # 23 — specific record claim ✓
    {"id": 23, "start": 81.6, "end": 86.0, "speaker": "SPEAKER_01", "is_final": True, "text": "NHS waiting lists stand at a record high of 7.6 million patients.", "label": 1},
    # 24 — filler / transition
    {"id": 24, "start": 86.1, "end": 88.5, "speaker": "SPEAKER_01", "is_final": True, "text": "I could go on, but time does not permit.", "label": 0},
    # 25 — climate claim ✓
    {"id": 25, "start": 88.6, "end": 93.0, "speaker": "SPEAKER_01", "is_final": True, "text": "Carbon emissions rose by 2.9 percent last year, reversing a decade of progress.", "label": 1},
    # 26 — opinion
    {"id": 26, "start": 93.1, "end": 95.5, "speaker": "SPEAKER_01", "is_final": True, "text": "We owe it to future generations to do better.", "label": 0},
    # 27 — back to SPEAKER_00 — multi-sentence segment
    {"id": 27, "start": 95.6, "end": 102.0, "speaker": "SPEAKER_00", "is_final": True, "text": "Our renewable energy capacity has doubled. Solar installations grew by 34 percent.", "label": 1},
    # 28 — specific trade claim ✓
    {"id": 28, "start": 102.1, "end": 107.5, "speaker": "SPEAKER_00", "is_final": True, "text": "Exports reached a record $580 billion, a 9.2 percent increase year-on-year.", "label": 1},
    # 29 — filler
    {"id": 29, "start": 107.6, "end": 110.0, "speaker": "SPEAKER_00", "is_final": True, "text": "We are proud of this record.", "label": 0},
    # 30 — crime stat ✓
    {"id": 30, "start": 110.1, "end": 115.3, "speaker": "SPEAKER_00", "is_final": True, "text": "Violent crime is down 15 percent compared with five years ago.", "label": 1},
    # 31 — normative
    {"id": 31, "start": 115.4, "end": 118.0, "speaker": "SPEAKER_00", "is_final": True, "text": "Safe streets are the foundation of a good society.", "label": 0},
    # 32 — defense spending ✓
    {"id": 32, "start": 118.1, "end": 123.4, "speaker": "SPEAKER_00", "is_final": True, "text": "Defense spending has reached 2.1 percent of GDP, meeting our NATO commitment.", "label": 1},
    # 33 — causal opinion
    {"id": 33, "start": 123.5, "end": 126.5, "speaker": "SPEAKER_01", "is_final": True, "text": "That increase came at the expense of social services.", "label": 0},
    # 34 — poverty stat ✓
    {"id": 34, "start": 126.6, "end": 131.0, "speaker": "SPEAKER_01", "is_final": True, "text": "Child poverty has increased to 24 percent, up from 19 percent in 2020.", "label": 1},
    # 35 — normative
    {"id": 35, "start": 131.1, "end": 133.5, "speaker": "SPEAKER_01", "is_final": True, "text": "That is a moral failure.", "label": 0},
    # 36 — debt claim ✓
    {"id": 36, "start": 133.6, "end": 138.0, "speaker": "SPEAKER_01", "is_final": True, "text": "National debt has grown to 102 percent of GDP under this government.", "label": 1},
    # 37 — filler
    {"id": 37, "start": 138.1, "end": 140.5, "speaker": "SPEAKER_00", "is_final": True, "text": "The honourable member is entitled to her view.", "label": 0},
    # 38 — productivity claim ✓
    {"id": 38, "start": 140.6, "end": 145.0, "speaker": "SPEAKER_00", "is_final": True, "text": "Productivity growth has averaged 1.8 percent annually over the past four years.", "label": 1},
    # 39 — closing filler
    {"id": 39, "start": 145.1, "end": 148.0, "speaker": "SPEAKER_00", "is_final": True, "text": "Thank you, Mr Speaker.", "label": 0},
]

# Quick summary: 22 check-worthy (label=1), 18 non-check-worthy (label=0)


# ---------------------------------------------------------------------------
# WebSocket receiver — connects to Module 1's server and dumps to file
# ---------------------------------------------------------------------------

async def _receive_from_ws(uri: str, out_path: Path, max_segments: int) -> None:
    """Connect to Module 1 ws server and write FINAL segments to file."""
    try:
        import websockets  # type: ignore[import]
    except ImportError:
        print("websockets not installed — pip install websockets", file=sys.stderr)
        raise

    count = 0
    print(f"Connecting to {uri} ...")
    async with websockets.connect(uri) as ws:
        print(f"Connected. Writing to {out_path} (Ctrl-C to stop)")
        with out_path.open("w", encoding="utf-8") as fh:
            async for raw_msg in ws:
                data = json.loads(raw_msg)
                # Only keep final segments
                if not data.get("is_final", False):
                    continue
                fh.write(json.dumps(data) + "\n")
                fh.flush()
                count += 1
                text_preview = data.get("text", "")[:60]
                print(f"  [{count:4d}] t={data.get('start',0):.1f}s  {text_preview!r}")
                if max_segments and count >= max_segments:
                    print(f"Reached --max {max_segments}; stopping.")
                    break
    print(f"\nDone. {count} segments saved to {out_path}")


def _write_synthetic(out_path: Path) -> None:
    """Write the built-in synthetic fixture to a .jsonl file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for seg in SYNTHETIC_SEGMENTS:
            # Add emitted_at so the schema matches TranscriptSegment exactly
            import time
            seg_copy = {**seg, "emitted_at": time.time()}
            fh.write(json.dumps(seg_copy) + "\n")
    n_claims = sum(1 for s in SYNTHETIC_SEGMENTS if s.get("label") == 1)
    n_total  = len(SYNTHETIC_SEGMENTS)
    print(f"Written {n_total} segments to {out_path}")
    print(f"  check-worthy (label=1): {n_claims}")
    print(f"  not check-worthy (label=0): {n_total - n_claims}")
    print()
    print("Hand-label summary (ground truth for domain-mismatch evaluation):")
    for seg in SYNTHETIC_SEGMENTS:
        marker = "Y" if seg["label"] == 1 else " "
        print(f"  [{marker}] id={seg['id']:2d} t={seg['start']:5.1f}s  {seg['text'][:70]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump Module 1 segments to .jsonl")
    ap.add_argument(
        "--source", default="synthetic",
        help="'synthetic' (default) or ws://host:port of Module 1 server",
    )
    ap.add_argument(
        "--out", default="data/sample_recording.jsonl",
        help="Output .jsonl path (default: data/sample_recording.jsonl)",
    )
    ap.add_argument(
        "--max", type=int, default=0,
        help="Max segments to record from WebSocket (0 = unlimited)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.source == "synthetic":
        _write_synthetic(out_path)
    else:
        asyncio.run(_receive_from_ws(args.source, out_path, args.max))


if __name__ == "__main__":
    main()
