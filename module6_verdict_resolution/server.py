"""
server.py  —  Module 6 WebSocket bridge
========================================
Consumes VerdictBundle JSON from Module 5 (ws://localhost:8769),
resolves to a final Verdict, and broadcasts to Module 7 & 8 (ws://localhost:8770).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# NOTE: do NOT insert module5's path here — we load it via importlib below
#       to avoid the schemas.py name collision chain that reaches module4.

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

# Load module6 schemas via importlib BEFORE importing resolver/credibility,
# because credibility.py inserts module4's path at sys.path[0] which would
# cause a bare `from schemas import Verdict` to find module4's schemas.py.
import importlib.util as _ilu

def _load_schema(name: str, fpath: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = _ilu.spec_from_file_location(name, fpath)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_m6s = _load_schema("m6_schemas_srv", Path(__file__).parent / "schemas.py")
Verdict = _m6s.Verdict

_m5s = _load_schema("m5_schemas_srv2",
                    Path(__file__).parent.parent / "module5_rag_verification" / "schemas.py")
M5VerdictBundle = _m5s.VerdictBundle

from resolver import VerdictResolver

log = logging.getLogger("module6.server")

DEFAULT_M5_URI = "ws://localhost:8769"
DEFAULT_PORT   = 8770
RECONNECT_MAX  = 30.0


class _State:
    def __init__(self, resolver: VerdictResolver) -> None:
        self.resolver = resolver
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


async def _consume_m5(state: _State, uri: str) -> None:
    delay = 1.0
    while True:
        try:
            log.info("Connecting to Module 5 at %s", uri)
            async with websockets.connect(uri) as ws:
                delay = 1.0
                async for raw in ws:
                    try:
                        bundle = M5VerdictBundle.from_json(raw)
                    except Exception as exc:
                        log.warning("Could not parse VerdictBundle: %s", exc)
                        continue

                    try:
                        loop = asyncio.get_event_loop()
                        verdict: Verdict = await loop.run_in_executor(
                            None, state.resolver.resolve, bundle
                        )
                    except Exception as exc:
                        log.error("Resolution failed: %s", exc)
                        continue

                    log.info(
                        "claim #%d -> [%s] conf=%.2f conflicts=%s",
                        verdict.claim_id, verdict.label.value,
                        verdict.confidence, verdict.conflicts_detected,
                    )
                    await state.queue.put(verdict.to_json())

        except (ConnectionClosed, OSError) as exc:
            log.warning("Module 5 error: %s. Retry in %.1fs", exc, delay)

        await asyncio.sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX)


async def _run(m5_uri: str, host: str, port: int) -> None:
    resolver = VerdictResolver()
    state = _State(resolver)

    await websockets.serve(lambda ws: _handler(state, ws), host, port)
    log.info("Module 6 -> Module 7/8 server on ws://%s:%d", host, port)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_consume_m5(state, m5_uri))
        tg.create_task(_broadcaster(state))


def main() -> None:
    ap = argparse.ArgumentParser(description="Module 6 — Verdict Resolution server")
    ap.add_argument("--m5-uri", default=DEFAULT_M5_URI)
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
        asyncio.run(_run(args.m5_uri, args.host, args.port))
    except KeyboardInterrupt:
        print("\nModule 6 stopped.")


if __name__ == "__main__":
    main()

