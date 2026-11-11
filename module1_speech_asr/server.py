"""
server.py  —  Module 1 WebSocket output server
===============================================
Bridges the blocking ``Pipeline.run()`` main loop with an asyncio
WebSocket server so Module 2 (or any other consumer) can subscribe to
the transcript stream over a plain WebSocket connection.

Threading model
---------------
The pipeline's main loop is a blocking generator — it runs forever,
consuming audio chunks one at a time.  An asyncio event loop, on the
other hand, requires the main thread.  These two don't mix directly.
The solution is the standard Python pattern:

1. Start the asyncio event loop on the **main thread**.
2. Spawn ``Pipeline.run()`` on a **background daemon thread**.
3. The pipeline's ``on_event`` callback (called from the worker thread)
   uses ``asyncio.run_coroutine_threadsafe(broadcast(payload), loop)``
   to hand the segment off to the event loop safely.
4. The ``broadcast`` coroutine (running on the event loop) serialises the
   segment to JSON and sends it to every currently-connected WebSocket
   client, silently dropping any that have disconnected.

CLI
---
::

    python server.py --input file --file speech.wav
    python server.py --input file --file speech.wav --diarize
    python server.py --input mic
    python server.py --input mic --diarize

    Optional overrides:
    --host  HOST   (default: config.WS_HOST)
    --port  PORT   (default: config.WS_PORT)
    --model SIZE   (default: config.WHISPER_MODEL_SIZE)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from typing import Optional

import websockets  # type: ignore[import]
try:
    # websockets >= 12 uses ServerConnection
    from websockets.server import ServerConnection as _WSConn
except ImportError:
    # websockets < 12 fallback
    from websockets.legacy.server import WebSocketServerProtocol as _WSConn  # type: ignore[assignment]

import config
from audio_capture import FileAudioCapture, MicAudioCapture
from diarization import Diarizer
from output_schema import TranscriptSegment
from pipeline import Pipeline
from streaming_asr import StreamingASR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connected clients registry
# ---------------------------------------------------------------------------

# All currently-connected WebSocket clients.  Accessed from the asyncio
# event loop only — no locking required.
_connected: set["_WSConn"] = set()


# ---------------------------------------------------------------------------
# WebSocket handler & broadcast
# ---------------------------------------------------------------------------

async def _handler(websocket: "_WSConn") -> None:
    """Register a new client and keep the connection alive until it closes."""
    _connected.add(websocket)
    remote = websocket.remote_address
    logger.info("Client connected: %s  (total: %d)", remote, len(_connected))
    try:
        # Block until the client disconnects.
        await websocket.wait_closed()
    finally:
        _connected.discard(websocket)
        logger.info("Client disconnected: %s  (total: %d)", remote, len(_connected))


async def _broadcast(payload: str) -> None:
    """Send *payload* to every currently-connected client.

    Clients that have disconnected between the time the message was
    enqueued and now are silently dropped — their sockets are removed
    from ``_connected``.
    """
    if not _connected:
        return

    dead: set[WebSocketServerProtocol] = set()
    for ws in list(_connected):
        try:
            await ws.send(payload)
        except websockets.exceptions.ConnectionClosed:
            dead.add(ws)
        except Exception as exc:  # pragma: no cover
            logger.warning("Unexpected send error for %s: %s", ws.remote_address, exc)
            dead.add(ws)

    _connected.difference_update(dead)
    if dead:
        logger.debug("Removed %d dead connection(s)", len(dead))


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def _build_on_event(
    loop: asyncio.AbstractEventLoop,
) -> "Callable[[TranscriptSegment], None]":
    """Return an ``on_event`` callback that is safe to call from any thread.

    The callback serialises the segment to JSON, then schedules
    ``_broadcast`` on the asyncio event loop using
    ``asyncio.run_coroutine_threadsafe`` — the correct pattern for
    pushing work from a worker thread into a running event loop.
    """
    from typing import Callable  # local import to avoid circular at module level

    def on_event(seg: TranscriptSegment) -> None:
        payload = seg.to_json()
        logger.debug(
            "on_event: scheduling broadcast  is_final=%s  text=%r",
            seg.is_final,
            seg.text,
        )
        asyncio.run_coroutine_threadsafe(_broadcast(payload), loop)

    return on_event


def _run_pipeline_thread(
    pipeline: Pipeline,
    ready_event: threading.Event,
) -> None:
    """Target function for the pipeline background thread."""
    logger.info("Pipeline thread starting.")
    ready_event.set()       # signal that the thread is alive
    try:
        pipeline.run()
    except Exception as exc:  # pragma: no cover
        logger.error("Pipeline thread crashed: %s", exc, exc_info=True)
    logger.info("Pipeline thread finished.")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def serve(
    capture,
    asr: StreamingASR,
    diarizer: Optional[Diarizer],
    host: str,
    port: int,
) -> None:
    """Start the WebSocket server and the pipeline background thread.

    Parameters
    ----------
    capture:
        Audio capture source (``FileAudioCapture`` or ``MicAudioCapture``).
    asr:
        Configured ``StreamingASR`` instance.
    diarizer:
        Optional ``Diarizer`` instance.  ``None`` disables speaker labelling.
    host, port:
        WebSocket bind address.
    """
    loop = asyncio.get_running_loop()
    on_event = _build_on_event(loop)

    pipeline = Pipeline(
        capture=capture,
        asr=asr,
        diarizer=diarizer,
        on_event=on_event,
    )

    # Start the pipeline in a background daemon thread.
    ready = threading.Event()
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(pipeline, ready),
        daemon=True,
        name="pipeline-worker",
    )
    thread.start()
    ready.wait()  # wait until the thread has actually started

    logger.info("WebSocket server listening on ws://%s:%d", host, port)

    async with websockets.serve(_handler, host, port):
        # Run until the pipeline thread finishes (file mode) or forever (mic).
        await loop.run_in_executor(None, thread.join)

    logger.info("Server shut down.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 1 — Speech ASR WebSocket server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py --input file --file speech.wav
  python server.py --input file --file speech.wav --diarize
  python server.py --input mic
  python server.py --input mic --diarize
        """,
    )

    parser.add_argument(
        "--input",
        choices=["mic", "file"],
        default="mic",
        help="Audio source: 'mic' for live microphone, 'file' for a WAV file.",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        default=None,
        help="Path to the audio file (required when --input file).",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Enable speaker diarization (requires HF_TOKEN env var).",
    )
    parser.add_argument(
        "--host",
        default=config.WS_HOST,
        help=f"WebSocket host (default: {config.WS_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.WS_PORT,
        help=f"WebSocket port (default: {config.WS_PORT}).",
    )
    parser.add_argument(
        "--model",
        default=config.WHISPER_MODEL_SIZE,
        help=f"faster-whisper model size (default: {config.WHISPER_MODEL_SIZE}).",
    )
    parser.add_argument(
        "--chunk-s",
        type=float,
        default=0.5,
        dest="chunk_s",
        help="Audio chunk duration in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help=f"Logging verbosity (default: {config.LOG_LEVEL}).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build components, and start the server."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- validate args ----
    if args.input == "file" and not args.file:
        print("ERROR: --file PATH is required when --input file", file=sys.stderr)
        sys.exit(1)

    # ---- build capture ----
    if args.input == "file":
        capture = FileAudioCapture(args.file, chunk_s=args.chunk_s, realtime=True)
        logger.info("Input: file  (%s)", args.file)
    else:
        capture = MicAudioCapture(chunk_s=args.chunk_s)
        logger.info("Input: microphone")

    # ---- build ASR (override model size from CLI) ----
    if args.model != config.WHISPER_MODEL_SIZE:
        config.WHISPER_MODEL_SIZE = args.model
    asr = StreamingASR()
    logger.info("ASR: faster-whisper  model=%s", config.WHISPER_MODEL_SIZE)

    # ---- build optional diarizer ----
    diarizer: Optional[Diarizer] = None
    if args.diarize:
        diarizer = Diarizer()
        logger.info("Diarization: enabled  model=%s", config.DIARIZATION_MODEL)
    else:
        logger.info("Diarization: disabled")

    # ---- run ----
    try:
        asyncio.run(
            serve(
                capture=capture,
                asr=asr,
                diarizer=diarizer,
                host=args.host,
                port=args.port,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
