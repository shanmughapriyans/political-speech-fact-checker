# Live Fact-Verification System

Real-time, end-to-end claim detection and fact-checking for live speech —
from microphone to verdict in under 10 seconds.

## Architecture

```
Mic/WAV → Module 1 (ASR) → Module 2 (Claim Detection)
        → Module 3 (Query Formulation) → Module 4 (Evidence Retrieval)
        → Module 5 (RAG Verification) → Module 6 (Verdict Resolution)
        → Module 7 (Live Dashboard)
                            ↕
                  Module 8 (PostgreSQL + Docker + CI)
```

Each module runs as an independent Python process (or Next.js for Module 7)
connected by WebSockets.  Every module can be tested offline in isolation.

## Module Summary

| Module | Dir | In | Out | Port |
|--------|-----|-----|-----|------|
| 1 — Speech ASR | `module1_speech_asr/` | Audio | TranscriptSegment JSON | 8765 |
| 2 — Claim Detection | `module2_claim_detection/` | Segments | ClaimEvent JSON | 8766 |
| 3 — Query Formulation | `module3_query_formulation/` | ClaimEvent | SubQuestionSet JSON | 8767 |
| 4 — Evidence Retrieval | `module4_evidence_retrieval/` | SubQuestionSet | EvidenceBundle JSON | 8768 |
| 5 — RAG Verification | `module5_rag_verification/` | EvidenceBundle | VerdictBundle JSON | 8769 |
| 6 — Verdict Resolution | `module6_verdict_resolution/` | VerdictBundle | Verdict JSON | 8770 |
| 7 — Dashboard | `module7_dashboard/` | Verdict JSON (SSE) | Browser UI | 3000 |
| 8 — Infrastructure | `module8_infrastructure/` | — | DB schema, Docker, CI | — |

## Quick Start

### Prerequisites
- Python 3.11+ with a venv in each module directory
- Node.js 20+ for Module 7
- OpenAI API key (for Modules 3, 5, 6 — heuristic fallback if absent)
- HuggingFace token (for Module 1 diarization)

### Setup
```bash
# 1. Copy and fill in environment variables
cp .env.example .env

# 2. Install Module 7 npm dependencies
make install-m7

# 3. Run offline demo (no audio, no API key needed)
make demo

# 4. Run all self-tests
make test

# 5. Run end-to-end evaluation
make eval
```

### Running the full live system
```bash
# Docker Compose (all modules):
make up

# Or manually in 7 terminals:
cd module1_speech_asr   && python server.py --input mic
cd module2_claim_detection && python pipeline.py
cd module3_query_formulation && python server.py
cd module4_evidence_retrieval && python server.py
cd module5_rag_verification && python server.py
cd module6_verdict_resolution && python server.py
cd module7_dashboard && npm run dev
# Open http://localhost:3000
```

## Offline Development

No live audio needed for iterating on claim detection:

```bash
# Generate/update the fixture
cd module2_claim_detection
python dump_segments.py --source synthetic --out data/sample_recording.jsonl

# Run Module 2 offline
python replay.py --evaluate --verbose

# Run full pipeline offline
python ../../module8_infrastructure/eval/end_to_end_eval.py
```

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| Module 2 Claim Detection F1 (ClaimBuster test) | **0.867** |
| Module 2 Claim Detection F1 (domain fixture) | **0.952** |
| End-to-end latency (claim → verdict, heuristic) | **< 500 ms** |
| End-to-end latency (claim → verdict, with LLM) | **2–8 s** |

## Key Design Decisions

- **Streaming first**: every module outputs immediately after processing one item, so the dashboard updates in real time, not after a full transcript.
- **Offline fallback**: every LLM-dependent module has a heuristic fallback so the full pipeline runs without any API key (for demos and CI).
- **Reproducible development**: the synthetic `.jsonl` fixture lets you iterate on any module without running Whisper or recording audio.
- **Inter-module contracts**: each module's `schemas.py` defines a versioned JSON contract — changing the schema is a deliberate act, not an accident.

## Project Structure

```
fact verification system/
├── module1_speech_asr/        # faster-whisper + pyannote
├── module2_claim_detection/   # TF-IDF baseline + DistilRoBERTa
├── module3_query_formulation/ # AVeriTeC-style decomposition
├── module4_evidence_retrieval/# BM25 + dense retrieval, trusted sources
├── module5_rag_verification/  # RAG + RoBERTa-MNLI entailment
├── module6_verdict_resolution/# CONFACT weighting + self-reflection
├── module7_dashboard/         # Next.js, React, Tailwind, SSE
├── module8_infrastructure/    # Prisma, Docker Compose, GitHub CI, eval
├── Dockerfile.python
├── Dockerfile.node
├── Makefile
└── .env.example
```

