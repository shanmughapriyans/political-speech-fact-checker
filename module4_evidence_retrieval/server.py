"""
server.py  —  Module 4 WebSocket bridge
========================================
Consumes SubQuestionSet JSON from Module 3 (ws://localhost:8767),
retrieves evidence, and broadcasts EvidenceBundle JSON to Module 5
subscribers (ws://localhost:8768).

Run::

    python server.py
    python server.py --m3-uri ws://localhost:8767 --port 8768
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent / "module3_query_formulation"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import websockets
from websockets.exceptions import ConnectionClosed
try:
    from websockets.server import ServerConnection as _WSConn
except ImportError:
    from websockets.legacy.server import WebSocketServerProtocol as _WSConn  # type: ignore

from retriever import EvidenceRetriever
from schemas import EvidenceBundle

log = logging.getLogger("module4.server")

DEFAULT_M3_URI = "ws://localhost:8767"
DEFAULT_PORT   = 8768
RECONNECT_MAX  = 30.0


class _State:
    def __init__(self, retriever: EvidenceRetriever) -> None:
        self.retriever = retriever
        self.clients: set[_WSConn] = set()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=64)


async def _handler(state: _State, ws: _WSConn) -> None:
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


async def _consume_m3(state: _State, uri: str) -> None:
    delay = 1.0
    while True:
        try:
            log.info("Connecting to Module 3 at %s", uri)
            async with websockets.connect(uri) as ws:
                delay = 1.0
                async for raw in ws:
                    try:
                        # Import Module 3 schema
                        sys.path.insert(0, str(Path(__file__).parent.parent / "module3_query_formulation"))
                        from schemas import SubQuestionSet
                        sqs = SubQuestionSet.from_json(raw)
                    except Exception as exc:
                        log.warning("Could not parse SubQuestionSet: %s", exc)
                        continue

                    try:
                        loop = asyncio.get_event_loop()
                        bundle: EvidenceBundle = await loop.run_in_executor(
                            None, state.retriever.retrieve, sqs
                        )
                    except Exception as exc:
                        log.error("Retrieve failed: %s", exc)
                        continue

                    log.info(
                        "claim #%d -> %d passages",
                        bundle.claim_id, len(bundle.passages),
                    )
                    await state.queue.put(bundle.to_json())

        except (ConnectionClosed, OSError) as exc:
            log.warning("Module 3 error: %s. Retry in %.1fs", exc, delay)

        await asyncio.sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX)


async def _run(m3_uri: str, host: str, port: int) -> None:
    retriever = EvidenceRetriever()
    state = _State(retriever)

    await websockets.serve(lambda ws: _handler(state, ws), host, port)
    log.info("Module 4 -> Module 5 server on ws://%s:%d", host, port)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_consume_m3(state, m3_uri))
        tg.create_task(_broadcaster(state))


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 4 — Evidence Retrieval server")
    ap.add_argument("--m3-uri", default=DEFAULT_M3_URI)
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
        asyncio.run(_run(args.m3_uri, args.host, args.port))
    except KeyboardInterrupt:
        print("\nModule 4 stopped.")


if __name__ == "__main__":
    main()

