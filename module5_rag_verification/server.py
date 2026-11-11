"""
server.py  —  Module 5 WebSocket bridge
========================================
Consumes EvidenceBundle JSON from Module 4 (ws://localhost:8768),
verifies each claim, and broadcasts VerdictBundle JSON to Module 6
subscribers (ws://localhost:8769).

Run::

    python server.py
    python server.py --m4-uri ws://localhost:8768 --port 8769
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent / "module4_evidence_retrieval"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / "module1_speech_asr" / ".env")
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

# Import from local module
from rag_verifier import RAGVerifier
from schemas import VerdictBundle

# Import Module 4 schema via importlib to avoid name collision
import importlib.util as _ilu
_m4_spec = _ilu.spec_from_file_location(
    "m4_schemas_srv",
    Path(__file__).parent.parent / "module4_evidence_retrieval" / "schemas.py"
)
_m4_mod = _ilu.module_from_spec(_m4_spec)
sys.modules["m4_schemas_srv"] = _m4_mod   # must register before exec on Py3.14
_m4_spec.loader.exec_module(_m4_mod)
M4EvidenceBundle = _m4_mod.EvidenceBundle

log = logging.getLogger("module5.server")

DEFAULT_M4_URI = "ws://localhost:8768"
DEFAULT_PORT   = 8769
RECONNECT_MAX  = 30.0


class _State:
    def __init__(self, verifier: RAGVerifier) -> None:
        self.verifier = verifier
        self.clients: set[_WSConn] = set()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=32)


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


async def _consume_m4(state: _State, uri: str) -> None:
    delay = 1.0
    while True:
        try:
            log.info("Connecting to Module 4 at %s", uri)
            async with websockets.connect(uri) as ws:
                delay = 1.0
                async for raw in ws:
                    try:
                        bundle = M4EvidenceBundle.from_json(raw)
                    except Exception as exc:
                        log.warning("Could not parse EvidenceBundle: %s", exc)
                        continue

                    try:
                        loop = asyncio.get_event_loop()
                        vb: VerdictBundle = await loop.run_in_executor(
                            None, state.verifier.verify, bundle
                        )
                    except Exception as exc:
                        log.error("Verification failed: %s", exc)
                        continue

                    labels = [v.label.value for v in vb.sub_verdicts]
                    log.info("claim #%d -> %s", vb.claim_id, labels)
                    await state.queue.put(vb.to_json())

        except (ConnectionClosed, OSError) as exc:
            log.warning("Module 4 error: %s. Retry in %.1fs", exc, delay)

        await asyncio.sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX)


async def _run(m4_uri: str, host: str, port: int) -> None:
    verifier = RAGVerifier()
    state = _State(verifier)

    await websockets.serve(lambda ws: _handler(state, ws), host, port)
    log.info("Module 5 -> Module 6 server on ws://%s:%d", host, port)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_consume_m4(state, m4_uri))
        tg.create_task(_broadcaster(state))


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 5 — RAG Verification server")
    ap.add_argument("--m4-uri", default=DEFAULT_M4_URI)
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
        asyncio.run(_run(args.m4_uri, args.host, args.port))
    except KeyboardInterrupt:
        print("\nModule 5 stopped.")


if __name__ == "__main__":
    main()

