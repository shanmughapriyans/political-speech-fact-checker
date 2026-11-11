"use client";

// app/page.tsx  —  Live Fact-Verification Dashboard
// Three-panel layout:
//   Left   — Live Transcript
//   Middle — Pipeline Flow (per-claim, per-module progress)
//   Right  — Verified Claims (final verdicts)

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Verdict, TranscriptSegment,
  SubQuestionSet, EvidenceBundle, VerdictBundle,
  PipelineEntry, PipelineStage,
} from "@/lib/types";
import { ClaimCard }        from "@/components/ClaimCard";
import { TranscriptPanel }  from "@/components/TranscriptPanel";
import { PipelineTracker }  from "@/components/PipelineTracker";

const MAX_SEGMENTS = 300;
const MAX_VERDICTS = 100;
const MAX_PIPELINE = 20;   // keep last N pipeline entries

function StatusDot({ connected }: { connected: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-xs">
      <span
        className={`h-2 w-2 rounded-full ${
          connected ? "bg-green-500 animate-pulse" : "bg-slate-400"
        }`}
      />
      {connected ? "Live" : "Disconnected"}
    </span>
  );
}

// ── Helper: upsert a pipeline entry ──────────────────────────────────────────
function upsertPipeline(
  prev: PipelineEntry[],
  claim_id: number,
  claim_text: string,
  stage: PipelineStage,
  patch: Partial<PipelineEntry>,
): PipelineEntry[] {
  const now = Date.now();
  const existing = prev.find((e) => e.claim_id === claim_id);
  if (existing) {
    return prev.map((e) =>
      e.claim_id === claim_id
        ? { ...e, stage, updated_at: now, ...patch }
        : e
    );
  }
  const entry: PipelineEntry = {
    claim_id,
    claim_text,
    stage,
    started_at: now,
    updated_at: now,
    ...patch,
  };
  return [entry, ...prev].slice(0, MAX_PIPELINE);
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [segments,  setSegments]  = useState<TranscriptSegment[]>([]);
  const [verdicts,  setVerdicts]  = useState<Verdict[]>([]);
  const [pipeline,  setPipeline]  = useState<PipelineEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const [newVerdictId, setNewVerdictId] = useState<number | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const connect = useCallback(() => {
    esRef.current?.close();
    const es = new EventSource("/api/events");
    esRef.current = es;

    es.addEventListener("connected",          () => setConnected(true));
    es.addEventListener("module6_disconnected", () => setConnected(false));
    es.addEventListener("error",              () => setConnected(false));

    // ── M1 — transcript segment ──
    es.addEventListener("segment", (e: MessageEvent) => {
      try {
        const seg: TranscriptSegment = JSON.parse(e.data);
        setSegments((prev) => [...prev, seg].slice(-MAX_SEGMENTS));
      } catch {}
    });

    // ── M2 — claim detected ──
    es.addEventListener("claim", (e: MessageEvent) => {
      try {
        const c = JSON.parse(e.data) as {
          claim_id: number; claim_text: string; score?: number;
        };
        setPipeline((prev) =>
          upsertPipeline(prev, c.claim_id, c.claim_text, "detected", {
            claim_score: c.score,
          })
        );
      } catch {}
    });

    // ── M3 — sub-questions ──
    es.addEventListener("subquestions", (e: MessageEvent) => {
      try {
        const sqs: SubQuestionSet = JSON.parse(e.data);
        setPipeline((prev) =>
          upsertPipeline(prev, sqs.claim_id, sqs.claim_text, "decomposed", {
            sub_questions: sqs.questions,
          })
        );
      } catch {}
    });

    // ── M4 — evidence ──
    es.addEventListener("evidence", (e: MessageEvent) => {
      try {
        const eb: EvidenceBundle = JSON.parse(e.data);
        setPipeline((prev) =>
          upsertPipeline(prev, eb.claim_id, eb.claim_text, "retrieved", {
            passages: eb.passages,
          })
        );
      } catch {}
    });

    // ── M5 — sub-verdicts ──
    es.addEventListener("subverdicts", (e: MessageEvent) => {
      try {
        const vb: VerdictBundle = JSON.parse(e.data);
        setPipeline((prev) =>
          upsertPipeline(prev, vb.claim_id, vb.claim_text, "verified", {
            sub_verdicts: vb.sub_verdicts,
          })
        );
      } catch {}
    });

    // ── M6 — final verdict ──
    es.addEventListener("verdict", (e: MessageEvent) => {
      try {
        const verdict: Verdict = JSON.parse(e.data);

        // Update pipeline to resolved
        setPipeline((prev) =>
          upsertPipeline(prev, verdict.claim_id, verdict.claim_text, "resolved", {
            verdict,
          })
        );

        // Add to verdict list
        setVerdicts((prev) => {
          const filtered = prev.filter((v) => v.verdict_id !== verdict.verdict_id);
          return [...filtered, verdict].slice(-MAX_VERDICTS);
        });
        setNewVerdictId(verdict.verdict_id);
        setTimeout(() => setNewVerdictId(null), 3000);
      } catch {}
    });

    es.onerror = () => {
      setConnected(false);
      setTimeout(connect, 3000);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => esRef.current?.close();
  }, [connect]);

  const sortedVerdicts = [...verdicts].sort((a, b) => b.emitted_at - a.emitted_at);

  return (
    <div className="flex flex-col h-screen bg-slate-100 font-sans">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-slate-200 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold text-slate-800">Live Fact Checker</span>
          <span className="text-xs text-slate-400 font-mono bg-slate-100 px-2 py-0.5 rounded">v0.2</span>
        </div>
        <div className="flex items-center gap-4 text-slate-500 text-sm">
          <StatusDot connected={connected} />
          <span>{verdicts.length} claim{verdicts.length !== 1 ? "s" : ""} verified</span>
          <span>{segments.length} segment{segments.length !== 1 ? "s" : ""}</span>
          <button
            onClick={() => { setSegments([]); setVerdicts([]); setPipeline([]); }}
            className="text-xs text-slate-400 hover:text-slate-600 underline"
          >
            Clear
          </button>
        </div>
      </header>

      {/* Three-panel layout */}
      <main className="flex flex-1 overflow-hidden">

        {/* Left — Live Transcript (25%) */}
        <section className="flex flex-col w-1/4 border-r border-slate-200 bg-white">
          <div className="px-4 py-2 border-b border-slate-100 bg-slate-50">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Live Transcript
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto px-2">
            <TranscriptPanel segments={segments} verdicts={verdicts} autoScroll />
          </div>
        </section>

        {/* Middle — Pipeline Flow (40%) */}
        <section className="flex flex-col w-2/5 border-r border-slate-200 bg-slate-50">
          <div className="px-4 py-2 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Pipeline Flow
            </h2>
            <div className="flex gap-2 text-[10px] text-slate-400 font-mono">
              <span className="text-blue-500">M2</span>→
              <span className="text-violet-500">M3</span>→
              <span className="text-amber-500">M4</span>→
              <span className="text-orange-500">M5</span>→
              <span className="text-green-500">M6</span>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-3">
            <PipelineTracker entries={pipeline} maxVisible={6} />
          </div>
        </section>

        {/* Right — Final Verdicts (35%) */}
        <section className="flex flex-col flex-1 bg-slate-50">
          <div className="px-4 py-2 border-b border-slate-100 bg-slate-50">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Verified Claims
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {sortedVerdicts.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-2 py-20">
                <div className="text-4xl">🔍</div>
                <p className="text-sm">No claims verified yet.</p>
                <p className="text-xs text-center max-w-xs">
                  Claims detected in the transcript will appear here with supporting evidence.
                </p>
              </div>
            ) : (
              sortedVerdicts.map((v) => (
                <ClaimCard key={v.verdict_id} verdict={v} isNew={v.verdict_id === newVerdictId} />
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
