// app/api/events/route.ts  —  SSE relay: all pipeline WebSockets → browser
//
// Connects server-side to every module's output WebSocket and relays each
// message as a named Server-Sent Event so the browser can build a live
// pipeline-flow view without touching WebSockets directly.
//
// Events emitted:
//   connected          — handshake on SSE open
//   segment            — Module 1 final TranscriptSegment
//   claim              — Module 2 ClaimEvent   (claim_id, text, score)
//   subquestions       — Module 3 SubQuestionSet
//   evidence           — Module 4 EvidenceBundle
//   subverdicts        — Module 5 VerdictBundle
//   verdict            — Module 6 Verdict (final)
//   error              — any upstream WS error
//   <module>_disconnected — upstream closed

import { NextRequest } from "next/server";
import WebSocket from "ws";

const M1_WS_URI = process.env.M1_WS_URI ?? "ws://localhost:8765";
const M2_WS_URI = process.env.M2_WS_URI ?? "ws://localhost:8766";
const M3_WS_URI = process.env.M3_WS_URI ?? "ws://localhost:8767";
const M4_WS_URI = process.env.M4_WS_URI ?? "ws://localhost:8768";
const M5_WS_URI = process.env.M5_WS_URI ?? "ws://localhost:8769";
const M6_WS_URI = process.env.M6_WS_URI ?? "ws://localhost:8770";

export const dynamic = "force-dynamic";   // never cache SSE responses
export const runtime = "nodejs";          // need Node.js for 'ws' package

export async function GET(req: NextRequest) {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      let closed = false;

      const send = (event: string, data: unknown) => {
        if (closed) return;
        const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
        try { controller.enqueue(encoder.encode(payload)); } catch { /* client gone */ }
      };

      // Helper: attach a WebSocket, relay messages under `eventName`,
      // filter with optional predicate, and reconnect on close.
      const attach = (
        uri: string,
        moduleName: string,
        eventName: string,
        filter?: (parsed: Record<string, unknown>) => boolean,
      ) => {
        let ws: WebSocket;
        const connect = () => {
          if (closed) return;
          ws = new WebSocket(uri);
          ws.on("message", (raw: Buffer) => {
            try {
              const parsed = JSON.parse(raw.toString()) as Record<string, unknown>;
              if (!filter || filter(parsed)) send(eventName, parsed);
            } catch { /* skip malformed */ }
          });
          ws.on("error", () =>
            send("error", { source: moduleName, ts: Date.now() })
          );
          ws.on("close", () => {
            send(`${moduleName}_disconnected`, { ts: Date.now() });
            // Retry after 3 s (module may not be running yet)
            if (!closed) setTimeout(connect, 3000);
          });
        };
        connect();
        return () => { closed = true; try { ws?.close(); } catch { /* ok */ } };
      };

      send("connected", { ts: Date.now() });

      // M1 — only final segments (is_final flag)
      const closeM1 = attach(M1_WS_URI, "module1", "segment",
        (p) => Boolean(p["is_final"]));

      // M2 — claim events
      const closeM2 = attach(M2_WS_URI, "module2", "claim");

      // M3 — sub-question sets
      const closeM3 = attach(M3_WS_URI, "module3", "subquestions");

      // M4 — evidence bundles
      const closeM4 = attach(M4_WS_URI, "module4", "evidence");

      // M5 — verdict bundles (sub-verdicts per sub-question)
      const closeM5 = attach(M5_WS_URI, "module5", "subverdicts");

      // M6 — final resolved verdicts
      const closeM6 = attach(M6_WS_URI, "module6", "verdict");

      req.signal.addEventListener("abort", () => {
        closed = true;
        [closeM1, closeM2, closeM3, closeM4, closeM5, closeM6].forEach(fn => fn());
        try { controller.close(); } catch { /* ok */ }
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type":    "text/event-stream",
      "Cache-Control":   "no-cache",
      "Connection":      "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

