// components/PipelineTracker.tsx
// Shows a live pipeline flow card for each in-flight claim.
// Stages light up one by one as SSE events arrive from each module.

"use client";

import clsx from "clsx";
import { PipelineEntry, PipelineStage } from "@/lib/types";

// ── Stage metadata ────────────────────────────────────────────────────────────
const STAGES: { key: PipelineStage; label: string; module: string; color: string }[] = [
  { key: "detected",   label: "Detected",   module: "M2", color: "blue"   },
  { key: "decomposed", label: "Decomposed", module: "M3", color: "violet" },
  { key: "retrieved",  label: "Evidence",   module: "M4", color: "amber"  },
  { key: "verified",   label: "NLI Scored", module: "M5", color: "orange" },
  { key: "resolved",   label: "Verdict",    module: "M6", color: "green"  },
];

const STAGE_ORDER: PipelineStage[] = STAGES.map((s) => s.key);

function stageIndex(stage: PipelineStage): number {
  return STAGE_ORDER.indexOf(stage);
}

// ── Dot component ─────────────────────────────────────────────────────────────
function StageDot({
  meta,
  reached,
  active,
}: {
  meta: (typeof STAGES)[number];
  reached: boolean;
  active: boolean;
}) {
  const colorMap: Record<string, string> = {
    blue:   "bg-blue-500   border-blue-500   text-blue-700   bg-blue-50",
    violet: "bg-violet-500 border-violet-500 text-violet-700 bg-violet-50",
    amber:  "bg-amber-400  border-amber-400  text-amber-700  bg-amber-50",
    orange: "bg-orange-500 border-orange-500 text-orange-700 bg-orange-50",
    green:  "bg-green-500  border-green-500  text-green-700  bg-green-50",
  };
  const [dotCls, , labelCls, bgCls] = colorMap[meta.color].split("  ");

  return (
    <div className="flex flex-col items-center gap-1 min-w-[52px]">
      <div
        className={clsx(
          "h-7 w-7 rounded-full border-2 flex items-center justify-center text-xs font-bold text-white transition-all duration-500",
          reached ? dotCls : "bg-slate-200 border-slate-300 text-slate-400",
          active && "ring-2 ring-offset-1 ring-blue-400 animate-pulse"
        )}
      >
        {meta.module}
      </div>
      <span
        className={clsx(
          "text-[10px] font-medium transition-colors duration-300",
          reached ? labelCls : "text-slate-400"
        )}
      >
        {meta.label}
      </span>
    </div>
  );
}

// ── Connector line ────────────────────────────────────────────────────────────
function Connector({ filled }: { filled: boolean }) {
  return (
    <div className="flex-1 h-0.5 mt-3.5 mx-0.5 transition-colors duration-500"
      style={{ background: filled ? "#6366f1" : "#e2e8f0" }}
    />
  );
}

// ── Detail rows shown below the stage dots ────────────────────────────────────
function DetailSection({ entry }: { entry: PipelineEntry }) {
  const idx = stageIndex(entry.stage);

  return (
    <div className="mt-3 space-y-1.5 text-xs text-slate-600">
      {/* M2 — claim score */}
      {idx >= 0 && entry.claim_score !== undefined && (
        <div className="flex gap-2">
          <span className="font-semibold text-blue-600 w-16 shrink-0">M2 score</span>
          <span>{(entry.claim_score * 100).toFixed(1)}% check-worthy</span>
        </div>
      )}

      {/* M3 — sub-questions */}
      {idx >= 1 && entry.sub_questions && entry.sub_questions.length > 0 && (
        <div className="flex gap-2">
          <span className="font-semibold text-violet-600 w-16 shrink-0">
            M3 ({entry.sub_questions.length}Q)
          </span>
          <ul className="list-none space-y-0.5">
            {entry.sub_questions.map((q) => (
              <li key={q.q_id} className="text-slate-500">
                <span className="font-mono text-violet-500">Q{q.q_id}</span>{" "}
                {q.question}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* M4 — evidence sources */}
      {idx >= 2 && entry.passages && entry.passages.length > 0 && (
        <div className="flex gap-2">
          <span className="font-semibold text-amber-600 w-16 shrink-0">
            M4 ({entry.passages.length}P)
          </span>
          <ul className="list-none space-y-0.5">
            {entry.passages.slice(0, 3).map((p) => (
              <li key={p.passage_id} className="text-slate-500 truncate max-w-xs">
                <span className="font-medium text-amber-700">[{p.source_name}]</span>{" "}
                {p.text.slice(0, 70)}…
              </li>
            ))}
            {entry.passages.length > 3 && (
              <li className="text-slate-400">+{entry.passages.length - 3} more</li>
            )}
          </ul>
        </div>
      )}

      {/* M5 — NLI sub-verdicts */}
      {idx >= 3 && entry.sub_verdicts && entry.sub_verdicts.length > 0 && (
        <div className="flex gap-2">
          <span className="font-semibold text-orange-600 w-16 shrink-0">M5 NLI</span>
          <ul className="list-none space-y-0.5">
            {entry.sub_verdicts.map((sv) => {
              const badgeColor =
                sv.label === "supported" ? "bg-green-100 text-green-700" :
                sv.label === "refuted"   ? "bg-red-100   text-red-700"   :
                                           "bg-slate-100 text-slate-500";
              return (
                <li key={sv.q_id} className="flex items-center gap-1.5">
                  <span className={clsx("px-1 py-0.5 rounded text-[10px] font-bold", badgeColor)}>
                    {sv.label === "not_enough_info" ? "NEI" : sv.label.toUpperCase()}
                  </span>
                  <span className="text-slate-500 truncate max-w-[200px]">Q{sv.q_id}: {sv.question}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* M6 — final verdict (compact, full card shown separately) */}
      {idx >= 4 && entry.verdict && (
        <div className="flex gap-2 items-center">
          <span className="font-semibold text-green-700 w-16 shrink-0">M6 final</span>
          <span
            className={clsx("px-2 py-0.5 rounded-full text-[11px] font-bold uppercase", {
              "bg-green-100 text-green-700": entry.verdict.label === "supported",
              "bg-red-100   text-red-700":   entry.verdict.label === "refuted",
              "bg-amber-100 text-amber-700": entry.verdict.label === "mixed",
              "bg-slate-100 text-slate-500": entry.verdict.label === "not_enough_info",
            })}
          >
            {entry.verdict.label.replace(/_/g, " ")}
          </span>
          <span className="text-slate-400">conf {(entry.verdict.confidence * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}

// ── Main exported component ───────────────────────────────────────────────────
interface PipelineTrackerProps {
  entries: PipelineEntry[];   // most recent first
  maxVisible?: number;
}

export function PipelineTracker({ entries, maxVisible = 5 }: PipelineTrackerProps) {
  const visible = entries.slice(0, maxVisible);

  if (visible.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-slate-400 gap-2">
        <div className="text-3xl">⚙️</div>
        <p className="text-sm">Pipeline idle — waiting for claims…</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {visible.map((entry) => {
        const currentIdx = stageIndex(entry.stage);
        const elapsed = ((Date.now() - entry.started_at) / 1000).toFixed(1);

        return (
          <div
            key={entry.claim_id}
            className={clsx(
              "rounded-xl border bg-white p-4 shadow-sm transition-all duration-300",
              entry.stage === "resolved" ? "border-slate-200 opacity-80" : "border-blue-200"
            )}
          >
            {/* Claim text + elapsed */}
            <div className="flex items-start justify-between mb-3">
              <p className="text-xs font-medium text-slate-700 leading-snug flex-1 mr-2 line-clamp-2">
                &ldquo;{entry.claim_text}&rdquo;
              </p>
              <span className="text-[10px] text-slate-400 shrink-0 font-mono">
                {entry.stage === "resolved" ? "done" : `${elapsed}s`}
              </span>
            </div>

            {/* Stage dots + connectors */}
            <div className="flex items-start">
              {STAGES.map((s, i) => (
                <div key={s.key} className="flex items-start flex-1">
                  <StageDot
                    meta={s}
                    reached={currentIdx >= i}
                    active={currentIdx === i && entry.stage !== "resolved"}
                  />
                  {i < STAGES.length - 1 && (
                    <Connector filled={currentIdx > i} />
                  )}
                </div>
              ))}
            </div>

            {/* Detail section */}
            <DetailSection entry={entry} />
          </div>
        );
      })}
    </div>
  );
}

