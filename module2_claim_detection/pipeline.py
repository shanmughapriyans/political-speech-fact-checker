"""
pipeline.py  —  Module 2 orchestration: WebSocket client + server
=================================================================
Wires together:

  Module 1 WebSocket (ws://localhost:8765)
      ↓  TranscriptSegment JSON
  SentenceSegmenter
      ↓  Sentence objects
  ClaimClassifier  (score ≥ threshold)
      ↓  candidate claims
  ClaimDeduper
      ↓  novel claims only
  ClaimEvent queue
      ↓  broadcast to all Module 3 subscribers
  Module 3 WebSocket server (ws://localhost:8766)

Threading / asyncio model
-------------------------
Module 1 is consumed by an asyncio task (``_consume_module1``).
The Module 3 broadcast server runs as an asyncio server in the same loop.
All shared state (segmenter, classifier, deduper, context window) lives
in a single coroutine, which is single-threaded from asyncio's POV.
A ``ClaimEvent`` is put on an ``asyncio.Queue`` and the ``_broadcaster``
task pops it and sends to all connected Module 3 clients.

Usage
-----
From the module2_claim_detection directory::

    # Connect to a live Module 1 stream, serve claims on port 8766
    python pipeline.py

    # Custom ports / threshold
    python pipeline.py --m1-uri ws://localhost:8765 --port 8766 --threshold 0.5

    # Offline mode: pipe from replay.py instead of live Module 1
    # (run replay.py as the event source; pipeline.py still serves Module 3)
    python pipeline.py --m1-uri ws://localhost:8900  # replay server port

Reconnect logic
---------------
If the Module 1 WebSocket drops, the client backs off and retries
with exponential back-off (cap: 30 s).  Module 3 clients are unaffected.

Stale-flush watchdog
--------------------
An asyncio periodic task calls ``segmenter.flush()`` every
``STALE_FLUSH_INTERVAL_S`` seconds so a claim can never be stuck in the
buffer indefinitely if speech pauses without terminal punctuation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Set

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_MODULE2_DIR = Path(__file__).parent
_MODULE1_DIR = _MODULE2_DIR.parent / "module1_speech_asr"
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

try:
    import websockets                                 # type: ignore[import]
    from websockets.exceptions import ConnectionClosed
    try:
        from websockets.server import ServerConnection as _WSConn
    except ImportError:
        from websockets.legacy.server import WebSocketServerProtocol as _WSConn  # type: ignore[assignment]
except ImportError:
    websockets = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_M1_URI      = "ws://localhost:8765"
DEFAULT_M3_HOST     = "localhost"
DEFAULT_M3_PORT     = 8766
DEFAULT_THRESHOLD   = 0.5
STALE_FLUSH_S       = 8.0      # flush buffer if no new segment for 8 s
M1_RECONNECT_BASE_S = 1.0
M1_RECONNECT_MAX_S  = 30.0
CONTEXT_WINDOW      = 2        # sentences to keep for ClaimEvent.context

log = logging.getLogger("module2.pipeline")


# ---------------------------------------------------------------------------
# Pipeline state (plain object, not a class, to keep the coroutine simple)
# ---------------------------------------------------------------------------

class _PipelineState:
    """All mutable state shared by coroutines."""

    def __init__(self, threshold: float) -> None:
        self.segmenter   = SentenceSegmenter()
        self.classifier  = BaselineClaimClassifier()
        self.deduper     = ClaimDeduper()
        self.threshold   = threshold
        self._ctx: list[str] = []          # last CONTEXT_WINDOW sentences

        # asyncio queue: pipeline → broadcaster
        self.claim_queue: asyncio.Queue[ClaimEvent] = asyncio.Queue(maxsize=256)
        # Set of currently-connected Module 3 WebSocket connections
        self.m3_clients: Set[_WSConn] = set()

        # Stats
        self.n_segments   = 0
        self.n_sentences  = 0
        self.n_above_thr  = 0
        self.n_deduped    = 0
        self.n_emitted    = 0

    def context_str(self) -> str:
        return " ".join(self._ctx[-CONTEXT_WINDOW:])

    def push_context(self, text: str) -> None:
        self._ctx.append(text)
        if len(self._ctx) > CONTEXT_WINDOW * 4:   # keep buffer bounded
            self._ctx = self._ctx[-CONTEXT_WINDOW * 4:]

    def process_sentences(self, sentences, source_seg_id: int) -> list[ClaimEvent]:
        """Score, threshold, dedup, and return new ClaimEvents."""
        new_claims: list[ClaimEvent] = []
        for sent in sentences:
            self.n_sentences += 1
            ctx = self.context_str()
            score = self.classifier.score(sent.text, context=ctx)

            if score >= self.threshold:
                self.n_above_thr += 1
                claim = ClaimEvent(
                    text=sent.text,
                    context=ctx,
                    speaker=sent.speaker,
                    start=sent.start,
                    end=sent.end,
                    score=score,
                    source_segment_ids=[source_seg_id],
                )
                if self.deduper.check_and_add(claim):
                    self.n_deduped += 1
                    log.debug("dedup: %s", sent.text[:60])
                else:
                    self.n_emitted += 1
                    new_claims.append(claim)
                    log.info(
                        "CLAIM #%d  score=%.3f  %r",
                        claim.claim_id, score, sent.text[:60],
                    )

            self.push_context(sent.text)
        return new_claims


# ---------------------------------------------------------------------------
# Module 3 WebSocket server — broadcast ClaimEvents
# ---------------------------------------------------------------------------

async def _m3_handler(state: _PipelineState, ws: _WSConn) -> None:
    """Handle a new Module 3 client connection."""
    remote = getattr(ws, "remote_address", "?")
    log.info("Module 3 client connected: %s", remote)
    state.m3_clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        state.m3_clients.discard(ws)
        log.info("Module 3 client disconnected: %s", remote)


async def _broadcaster(state: _PipelineState) -> None:
    """Pop ClaimEvents from the queue and broadcast to all Module 3 clients."""
    while True:
        claim: ClaimEvent = await state.claim_queue.get()
        payload = claim.to_json()

        dead: list[_WSConn] = []
        for ws in list(state.m3_clients):
            try:
                await ws.send(payload)
            except ConnectionClosed:
                dead.append(ws)
            except Exception as exc:
                log.warning("Send to Module 3 client failed: %s", exc)
                dead.append(ws)
        for ws in dead:
            state.m3_clients.discard(ws)


# ---------------------------------------------------------------------------
# Module 1 WebSocket client — receive TranscriptSegments
# ---------------------------------------------------------------------------

async def _process_segment(state: _PipelineState, raw_msg: str) -> None:
    """Parse a Module 1 JSON message and push through the pipeline."""
    try:
        data = json.loads(raw_msg)
    except json.JSONDecodeError:
        log.warning("Received non-JSON from Module 1: %r", raw_msg[:100])
        return

    # Build TranscriptSegment
    try:
        seg = TranscriptSegment(
            start       = float(data["start"]),
            end         = float(data["end"]),
            text        = data["text"],
            speaker     = data.get("speaker", "UNKNOWN"),
            is_final    = bool(data.get("is_final", False)),
            emitted_at  = float(data.get("emitted_at", time.time())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Malformed segment from Module 1: %s — %s", exc, data)
        return

    state.n_segments += 1

    # Feed to segmenter (non-final segments are dropped inside the segmenter)
    sentences = state.segmenter.feed(seg)
    seg_id    = data.get("id", -1)

    for claim in state.process_sentences(sentences, seg_id):
        await state.claim_queue.put(claim)


async def _consume_module1(state: _PipelineState, uri: str) -> None:
    """
    Connect to Module 1 WebSocket with exponential back-off reconnect.
    Runs forever until the process is stopped.
    """
    delay = M1_RECONNECT_BASE_S
    while True:
        try:
            log.info("Connecting to Module 1 at %s", uri)
            async with websockets.connect(uri) as ws:
                log.info("Connected to Module 1.")
                delay = M1_RECONNECT_BASE_S   # reset back-off on success
                async for raw_msg in ws:
                    await _process_segment(state, raw_msg)
        except ConnectionClosed:
            log.warning("Module 1 connection closed. Reconnecting in %.1fs", delay)
        except OSError as exc:
            log.warning("Module 1 connection error: %s. Retrying in %.1fs", exc, delay)
        except Exception as exc:
            log.error("Unexpected error from Module 1: %s. Retrying in %.1fs", exc, delay)

        await asyncio.sleep(delay)
        delay = min(delay * 2, M1_RECONNECT_MAX_S)


# ---------------------------------------------------------------------------
# Stale-flush watchdog
# ---------------------------------------------------------------------------

async def _stale_flush_watchdog(state: _PipelineState) -> None:
    """Periodically flush the segmenter buffer to avoid stuck claims."""
    while True:
        await asyncio.sleep(STALE_FLUSH_S)
        flushed = state.segmenter.flush()
        if flushed:
            log.info("Stale flush: %d sentence(s)", len(flushed))
            for claim in state.process_sentences(flushed, source_seg_id=-1):
                await state.claim_queue.put(claim)


# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------

async def _stats_printer(state: _PipelineState, interval_s: float = 30.0) -> None:
    """Periodically log pipeline stats."""
    while True:
        await asyncio.sleep(interval_s)
        ds = state.deduper.stats()
        log.info(
            "STATS | segments=%d  sentences=%d  above_thr=%d  "
            "deduped=%d  emitted=%d  dedup_rate=%.1f%%",
            state.n_segments, state.n_sentences, state.n_above_thr,
            state.n_deduped, state.n_emitted,
            ds["dedup_rate"] * 100,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _run(m1_uri: str, m3_host: str, m3_port: int, threshold: float) -> None:
    if websockets is None:
        raise RuntimeError(
            "websockets package not installed. "
            "Run: pip install websockets"
        )

    state = _PipelineState(threshold=threshold)

    # Module 3 server
    m3_server = await websockets.serve(
        lambda ws: _m3_handler(state, ws),
        m3_host,
        m3_port,
    )
    log.info("Module 3 WebSocket server listening on ws://%s:%d", m3_host, m3_port)

    # Launch all tasks
    async with asyncio.TaskGroup() as tg:
        tg.create_task(_consume_module1(state, m1_uri))
        tg.create_task(_broadcaster(state))
        tg.create_task(_stale_flush_watchdog(state))
        tg.create_task(_stats_printer(state))

    m3_server.close()
    await m3_server.wait_closed()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Module 2 Pipeline — consume Module 1 ASR stream, "
            "detect claims, serve to Module 3"
        )
    )
    ap.add_argument(
        "--m1-uri", default=DEFAULT_M1_URI,
        help=f"Module 1 WebSocket URI (default: {DEFAULT_M1_URI})",
    )
    ap.add_argument(
        "--host", default=DEFAULT_M3_HOST,
        help=f"Module 3 server host (default: {DEFAULT_M3_HOST})",
    )
    ap.add_argument(
        "--port", type=int, default=DEFAULT_M3_PORT,
        help=f"Module 3 server port (default: {DEFAULT_M3_PORT})",
    )
    ap.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Classifier score threshold (default: {DEFAULT_THRESHOLD})",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(_run(args.m1_uri, args.host, args.port, args.threshold))
    except KeyboardInterrupt:
        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()

