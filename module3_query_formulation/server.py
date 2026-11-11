"""
server.py  —  Module 3 WebSocket bridge
========================================
Consumes ClaimEvent JSON from Module 2 (ws://localhost:8766),
decomposes each claim, and broadcasts SubQuestionSet JSON to
Module 4 subscribers (ws://localhost:8767).

Run::

    python server.py
    python server.py --m2-uri ws://localhost:8766 --port 8767
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent / "module2_claim_detection"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(Path(__file__).parent.parent / "module1_speech_asr" / ".env")
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import websockets
from websockets.exceptions import ConnectionClosed
try:
    from websockets.server import ServerConnection as _WSConn
except ImportError:
    from websockets.legacy.server import WebSocketServerProtocol as _WSConn  # type: ignore

from decomposer import ClaimDecomposer

log = logging.getLogger("module3.server")

DEFAULT_M2_URI = "ws://localhost:8766"
DEFAULT_PORT   = 8767
RECONNECT_MAX  = 30.0


class _State:
    def __init__(self, decomposer: ClaimDecomposer) -> None:
        self.decomposer = decomposer
        self.clients: set[_WSConn] = set()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        self.n_received = 0
        self.n_emitted  = 0


async def _handler(state: _State, ws: _WSConn) -> None:
    log.info("Module 4 client connected: %s", getattr(ws, "remote_address", "?"))
    state.clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        state.clients.discard(ws)


async def _broadcaster(state: _State) -> None:
    while True:
        payload = await state.queue.get()
        dead = []
        for ws in list(state.clients):
            try:
                await ws.send(payload)
            except ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            state.clients.discard(ws)


async def _consume_m2(state: _State, uri: str) -> None:
    delay = 1.0
    while True:
        try:
            log.info("Connecting to Module 2 at %s", uri)
            async with websockets.connect(uri) as ws:
                log.info("Connected to Module 2.")
                delay = 1.0
                async for raw in ws:
                    state.n_received += 1
                    try:
                        # Import here to avoid circular at module level
                        import sys as _sys
                        _sys.path.insert(0, str(Path(__file__).parent.parent / "module2_claim_detection"))
                        from claim_schema import ClaimEvent
                        claim = ClaimEvent.from_json(raw)
                    except Exception as exc:
                        log.warning("Could not parse ClaimEvent: %s", exc)
                        continue

                    # Decompose (CPU-bound but fast for heuristic)
                    try:
                        loop = asyncio.get_event_loop()
                        sqs = await loop.run_in_executor(
                            None, state.decomposer.decompose, claim
                        )
                    except Exception as exc:
                        log.error("Decompose failed: %s", exc)
                        continue

                    log.info(
                        "claim #%d -> %d sub-questions",
                        claim.claim_id, len(sqs.questions),
                    )
                    state.n_emitted += 1
                    await state.queue.put(sqs.to_json())

        except ConnectionClosed:
            log.warning("Module 2 closed. Retry in %.1fs", delay)
        except OSError as exc:
            log.warning("Module 2 error: %s. Retry in %.1fs", exc, delay)

        await asyncio.sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX)


async def _run(m2_uri: str, host: str, port: int) -> None:
    decomposer = ClaimDecomposer()
    state = _State(decomposer)

    server = await websockets.serve(
        lambda ws: _handler(state, ws), host, port
    )
    log.info("Module 3 → Module 4 server on ws://%s:%d", host, port)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_consume_m2(state, m2_uri))
        tg.create_task(_broadcaster(state))


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 3 — Query Formulation server")
    ap.add_argument("--m2-uri", default=DEFAULT_M2_URI)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        asyncio.run(_run(args.m2_uri, args.host, args.port))
    except KeyboardInterrupt:
        print("\nModule 3 stopped.")


if __name__ == "__main__":
    main()

