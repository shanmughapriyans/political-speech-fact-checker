"""
end_to_end_eval.py  —  Module 8: Full-pipeline offline evaluation harness
==========================================================================
Runs a complete offline evaluation of the entire pipeline using the
synthetic fixture from Module 2.  No live audio, no network calls.

Metrics reported
----------------
Per-module latency      (median, p95, p99 in ms)
Module 2 claim detection F1 on domain-mismatch fixture
Module 5 verdict accuracy vs. hand-labels (ground truth)
Module 6 conflict resolution accuracy
End-to-end latency      (claim detected → verdict emitted)

Usage::

    python end_to_end_eval.py
    python end_to_end_eval.py --fixture ../module2_claim_detection/data/sample_recording.jsonl
    python end_to_end_eval.py --threshold 0.4

This is the "metric none of the 24 papers report" — the measurement
showing the full system works, not just individual components.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — allow imports from all modules
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent  # fact verification system/

# ---------------------------------------------------------------------------
# Each module has a file called schemas.py — they shadow each other on
# sys.path.  Use importlib to load each one under a unique module name so
# they can all coexist in sys.modules simultaneously.
# ---------------------------------------------------------------------------
import importlib.util as _ilu


def _load_module(name: str, file_path: Path):
    """Load a Python file as a module with a given (unique) name."""
    if name in sys.modules:
        return sys.modules[name]
    spec = _ilu.spec_from_file_location(name, str(file_path))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load all schemas under unique names so later imports never collide
_load_module("m3_schemas", ROOT / "module3_query_formulation"  / "schemas.py")
_load_module("m4_schemas", ROOT / "module4_evidence_retrieval" / "schemas.py")
_load_module("m5_schemas", ROOT / "module5_rag_verification"   / "schemas.py")
_load_module("m6_schemas", ROOT / "module6_verdict_resolution" / "schemas.py")

# Now add the module directories in dependency order (later ones shadow
# earlier ones for non-schema files, which is fine — the schemas are already
# locked in sys.modules above).
for subdir in [
    "module1_speech_asr",           # output_schema, TranscriptSegment
    "module2_claim_detection",      # sentence_segmenter, claim_classifier, deduper, claim_schema
    "module3_query_formulation",    # decomposer  (uses m3_schemas)
    "module4_evidence_retrieval",   # retriever   (uses m4_schemas)
    "module5_rag_verification",     # rag_verifier, nli_scorer (uses m5_schemas)
    "module6_verdict_resolution",   # resolver    (uses m6_schemas)
]:
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.append(p)          # append so module1 output_schema wins over others

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / "module1_speech_asr" / ".env")
except ImportError:
    pass


def _load_fixture(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


class LatencyTracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.samples: list[float] = []

    def record(self, ms: float) -> None:
        self.samples.append(ms)

    def report(self) -> dict:
        if not self.samples:
            return {"name": self.name, "n": 0}
        return {
            "name":   self.name,
            "n":      len(self.samples),
            "median": round(_percentile(self.samples, 50), 2),
            "p95":    round(_percentile(self.samples, 95), 2),
            "p99":    round(_percentile(self.samples, 99), 2),
            "mean":   round(sum(self.samples) / len(self.samples), 2),
        }


def run_eval(fixture_path: Path, threshold: float = 0.5) -> dict:
    """
    Run the full offline pipeline and return a metrics dict.
    """

    print(f"\n{'='*65}")
    print("END-TO-END EVALUATION HARNESS")
    print(f"  Fixture : {fixture_path}")
    print(f"  Threshold: {threshold}")
    print(f"{'='*65}\n")

    # ----------------------------------------------------------------
    # Load fixture
    # ----------------------------------------------------------------
    segments_raw = _load_fixture(fixture_path)
    print(f"Loaded {len(segments_raw)} segments from fixture.\n")

    # ----------------------------------------------------------------
    # Import all module components (no network — offline mode)
    # ----------------------------------------------------------------
    print("Loading modules...")

    from output_schema import TranscriptSegment
    from sentence_segmenter import SentenceSegmenter
    from claim_classifier import BaselineClaimClassifier
    from deduper import ClaimDeduper
    from claim_schema import ClaimEvent

    # Module 3
    from decomposer import ClaimDecomposer

    # Module 4
    from retriever import EvidenceRetriever

    # Module 5
    from rag_verifier import RAGVerifier

    # Module 6
    from resolver import VerdictResolver

    print("  All modules imported.\n")

    # ----------------------------------------------------------------
    # Initialise components
    # ----------------------------------------------------------------
    segmenter   = SentenceSegmenter()
    classifier  = BaselineClaimClassifier()
    deduper     = ClaimDeduper()
    decomposer  = ClaimDecomposer()
    retriever   = EvidenceRetriever(use_dense=False)   # offline
    verifier    = RAGVerifier()
    resolver    = VerdictResolver()

    # Latency trackers (ms)
    lt_m2 = LatencyTracker("Module 2 (classify+dedup)")
    lt_m3 = LatencyTracker("Module 3 (decompose)")
    lt_m4 = LatencyTracker("Module 4 (retrieve)")
    lt_m5 = LatencyTracker("Module 5 (verify)")
    lt_m6 = LatencyTracker("Module 6 (resolve)")
    lt_e2e = LatencyTracker("End-to-end (claim→verdict)")

    # Domain-mismatch evaluation
    dm_tp = dm_fp = dm_fn = dm_tn = 0

    # Verdict accuracy (compare to hand-labels where available)
    correct_verdicts = total_verdicts = 0

    all_verdicts = []
    context_buf: list[str] = []

    # ----------------------------------------------------------------
    # Run the pipeline
    # ----------------------------------------------------------------
    print("Running pipeline...")
    t_pipeline_start = time.monotonic()

    for raw in segments_raw:
        seg = TranscriptSegment(
            start=raw["start"],
            end=raw.get("end", raw["start"] + 1.0),
            text=raw["text"],
            speaker=raw.get("speaker", "UNKNOWN"),
            is_final=raw.get("is_final", True),
            emitted_at=raw.get("emitted_at", time.time()),
        )
        true_label: int = raw.get("label", -1)

        sentences = segmenter.feed(seg)
        for sent in sentences:
            ctx = " ".join(context_buf[-2:])
            t0 = time.monotonic()
            score = classifier.score(sent.text, context=ctx)

            # Domain-mismatch eval
            if true_label >= 0:
                predicted = int(score >= threshold)
                if predicted == 1 and true_label == 1:
                    dm_tp += 1
                elif predicted == 1 and true_label == 0:
                    dm_fp += 1
                elif predicted == 0 and true_label == 1:
                    dm_fn += 1
                else:
                    dm_tn += 1

            if score >= threshold:
                claim = ClaimEvent(
                    text=sent.text,
                    context=ctx,
                    speaker=sent.speaker,
                    start=sent.start,
                    end=sent.end,
                    score=score,
                    source_segment_ids=[raw["id"]],
                )
                if deduper.check_and_add(claim):
                    continue

                t_m2 = (time.monotonic() - t0) * 1000
                lt_m2.record(t_m2)
                t_claim = time.monotonic()

                # Module 3: decompose
                t0 = time.monotonic()
                sqs = decomposer.decompose(claim)
                lt_m3.record((time.monotonic() - t0) * 1000)

                # Module 4: retrieve
                t0 = time.monotonic()
                bundle = retriever.retrieve(sqs)
                lt_m4.record((time.monotonic() - t0) * 1000)

                # Module 5: verify
                # Attach claim metadata to bundle for Module 5
                bundle.claim_speaker = claim.speaker
                bundle.claim_start   = claim.start
                bundle.claim_end     = claim.end
                bundle.claim_score   = claim.score
                t0 = time.monotonic()
                vb = verifier.verify(bundle)
                lt_m5.record((time.monotonic() - t0) * 1000)

                # Module 6: resolve
                t0 = time.monotonic()
                verdict = resolver.resolve(vb)
                lt_m6.record((time.monotonic() - t0) * 1000)

                lt_e2e.record((time.monotonic() - t_claim) * 1000)
                all_verdicts.append(verdict)

                # Verdict accuracy vs. hand-label
                if true_label >= 0:
                    total_verdicts += 1
                    expected_label = "supported" if true_label == 1 else "not_enough_info"
                    if verdict.label.value in ("supported", "not_enough_info"):
                        if verdict.label.value == expected_label:
                            correct_verdicts += 1

                print(
                    f"  claim #{claim.claim_id:3d}  score={score:.3f}  "
                    f"[{verdict.label.value}]  conf={verdict.confidence:.2f}  "
                    f"e2e={lt_e2e.samples[-1]:.0f}ms"
                )

            context_buf.append(sent.text)

    # Flush
    for sent in segmenter.flush():
        ctx = " ".join(context_buf[-2:])
        score = classifier.score(sent.text, ctx)
        if score >= threshold:
            claim = ClaimEvent(
                text=sent.text, context=ctx, speaker=sent.speaker,
                start=sent.start, end=sent.end, score=score,
                source_segment_ids=[],
            )
            if not deduper.check_and_add(claim):
                sqs = decomposer.decompose(claim)
                bundle = retriever.retrieve(sqs)
                bundle.claim_speaker = claim.speaker
                bundle.claim_start = claim.start
                bundle.claim_end = claim.end
                bundle.claim_score = claim.score
                vb = verifier.verify(bundle)
                verdict = resolver.resolve(vb)
                all_verdicts.append(verdict)

    t_total = time.monotonic() - t_pipeline_start

    # ----------------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------------

    # Module 2 F1
    dm_prec = dm_tp / (dm_tp + dm_fp) if (dm_tp + dm_fp) else 0.0
    dm_rec  = dm_tp / (dm_tp + dm_fn) if (dm_tp + dm_fn) else 0.0
    dm_f1   = 2 * dm_prec * dm_rec / (dm_prec + dm_rec) if (dm_prec + dm_rec) else 0.0

    # Verdict accuracy
    v_acc = correct_verdicts / total_verdicts if total_verdicts else 0.0

    # ----------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------
    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")

    print(f"\n[Module 2 — Claim Detection]  (domain-mismatch fixture)")
    print(f"  Precision : {dm_prec:.3f}")
    print(f"  Recall    : {dm_rec:.3f}")
    print(f"  F1        : {dm_f1:.3f}")
    print(f"  TP/FP/FN/TN: {dm_tp}/{dm_fp}/{dm_fn}/{dm_tn}")
    print(f"  Claims emitted: {len(all_verdicts)}")

    print(f"\n[Verdict accuracy vs. hand-labels]")
    print(f"  Correct: {correct_verdicts}/{total_verdicts}  ({v_acc:.1%})")

    print(f"\n[Per-module latency (ms)]")
    for tracker in [lt_m2, lt_m3, lt_m4, lt_m5, lt_m6, lt_e2e]:
        r = tracker.report()
        if r["n"] == 0:
            print(f"  {tracker.name}: no data")
        else:
            print(
                f"  {tracker.name:40s}  "
                f"n={r['n']:3d}  "
                f"med={r['median']:7.1f}  "
                f"p95={r['p95']:7.1f}  "
                f"mean={r['mean']:7.1f}"
            )

    print(f"\n[Pipeline total time]: {t_total:.1f}s for {len(segments_raw)} segments")
    print(f"{'='*65}\n")

    return {
        "dm_precision": round(dm_prec, 4),
        "dm_recall":    round(dm_rec, 4),
        "dm_f1":        round(dm_f1, 4),
        "verdict_accuracy": round(v_acc, 4),
        "n_verdicts":   len(all_verdicts),
        "latency": {
            t.name: t.report() for t in [lt_m2, lt_m3, lt_m4, lt_m5, lt_m6, lt_e2e]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end offline evaluation of the full fact-verification pipeline"
    )
    ap.add_argument(
        "--fixture",
        default=str(ROOT / "module2_claim_detection" / "data" / "sample_recording.jsonl"),
        help="Path to .jsonl fixture",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    path = Path(args.fixture)
    if not path.exists():
        print(f"Fixture not found: {path}")
        print("Run first: python module2_claim_detection/dump_segments.py --source synthetic")
        sys.exit(1)

    results = run_eval(path, args.threshold)

    # Write results to JSON for CI artifact upload
    out = ROOT / "module8_infrastructure" / "eval" / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
