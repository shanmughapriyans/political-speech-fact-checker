// components/ClaimCard.tsx  —  Per-claim verdict card with expandable evidence

"use client";

import { useState } from "react";
import { Verdict, EvidenceSummaryItem } from "@/lib/types";
import { VerdictBadge } from "./VerdictBadge";
import clsx from "clsx";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function EvidenceRow({ item }: { item: EvidenceSummaryItem }) {
  return (
    <div
      className={clsx(
        "flex items-start gap-2 rounded p-2 text-sm",
        item.supports ? "bg-green-50" : "bg-red-50"
      )}
    >
      <span
        className={clsx(
          "mt-0.5 h-4 w-4 flex-shrink-0 rounded-full text-xs font-bold text-white flex items-center justify-center",
          item.supports ? "bg-green-500" : "bg-red-500"
        )}
      >
        {item.supports ? "+" : "−"}
      </span>
      <div>
        <span className="font-medium text-slate-700">{item.source}</span>
        {item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noreferrer"
            className="ml-1 text-blue-600 hover:underline text-xs"
          >
            [link]
          </a>
        )}
        <p className="text-slate-600 mt-0.5">{item.snippet}</p>
      </div>
    </div>
  );
}

interface ClaimCardProps {
  verdict: Verdict;
  isNew?: boolean;
}

export function ClaimCard({ verdict, isNew = false }: ClaimCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={clsx(
        "rounded-xl border bg-white shadow-sm transition-all duration-300",
        isNew && "ring-2 ring-blue-400 ring-offset-1",
        {
          "border-green-200": verdict.label === "supported",
          "border-red-200":   verdict.label === "refuted",
          "border-amber-200": verdict.label === "mixed",
          "border-slate-200": verdict.label === "not_enough_info",
        }
      )}
    >
      {/* Header */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
              <span className="font-mono">
                {formatTime(verdict.claim_start)}–{formatTime(verdict.claim_end)}
              </span>
              <span>·</span>
              <span>{verdict.claim_speaker}</span>
              <span>·</span>
              <span>claim #{verdict.claim_id}</span>
              {verdict.conflicts_detected && (
                <>
                  <span>·</span>
                  <span className="text-amber-600 font-medium">conflict resolved</span>
                </>
              )}
            </div>
            <p className="text-slate-900 font-medium leading-snug">
              &ldquo;{verdict.claim_text}&rdquo;
            </p>
          </div>
          <VerdictBadge
            label={verdict.label}
            confidence={verdict.confidence}
            size="md"
          />
        </div>

        {/* Explanation */}
        <p className="mt-2 text-sm text-slate-600">{verdict.explanation}</p>
      </div>

      {/* Expandable evidence */}
      {verdict.evidence_summary.length > 0 && (
        <div className="border-t border-slate-100">
          <button
            onClick={() => setExpanded((e) => !e)}
            className="w-full px-4 py-2 text-left text-xs font-medium text-slate-500 hover:bg-slate-50 flex items-center justify-between"
          >
            <span>
              {verdict.evidence_summary.length} evidence item
              {verdict.evidence_summary.length !== 1 ? "s" : ""}
              {" · "}
              {verdict.sub_verdicts.length} sub-question
              {verdict.sub_verdicts.length !== 1 ? "s" : ""}
            </span>
            <span>{expanded ? "▲" : "▼"}</span>
          </button>

          {expanded && (
            <div className="px-4 pb-4 space-y-2">
              {verdict.evidence_summary.map((item, i) => (
                <EvidenceRow key={i} item={item} />
              ))}

              {verdict.conflict_resolution_note && (
                <div className="rounded bg-amber-50 border border-amber-200 p-2 text-xs text-amber-700">
                  <strong>Conflict note:</strong>{" "}
                  {verdict.conflict_resolution_note}
                </div>
              )}

              {/* Sub-verdict table */}
              <details className="mt-2">
                <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-600">
                  Sub-question details
                </summary>
                <div className="mt-2 space-y-1">
                  {verdict.sub_verdicts.map((sv) => (
                    <div
                      key={sv.q_id}
                      className="text-xs bg-slate-50 rounded p-2 border border-slate-100"
                    >
                      <div className="flex items-center gap-2 font-medium text-slate-700">
                        <VerdictBadge label={sv.label as any} size="sm" />
                        <span>Q{sv.q_id}: {sv.question}</span>
                      </div>
                      <p className="text-slate-500 mt-1 ml-1">
                        A: {sv.answer}
                      </p>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

