// components/TranscriptPanel.tsx  —  Live scrolling transcript with claim highlights

"use client";

import { useEffect, useRef } from "react";
import { TranscriptSegment, Verdict } from "@/lib/types";
import { VerdictBadge } from "./VerdictBadge";
import clsx from "clsx";

interface TranscriptPanelProps {
  segments: TranscriptSegment[];
  verdicts: Verdict[];
  autoScroll?: boolean;
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function TranscriptPanel({
  segments,
  verdicts,
  autoScroll = true,
}: TranscriptPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Build a map: segment text → verdict (approximate match by time)
  const verdictByTime = new Map<string, Verdict>();
  for (const v of verdicts) {
    // Key by rounded start time
    verdictByTime.set(`${Math.round(v.claim_start)}`, v);
  }

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [segments.length, autoScroll]);

  if (segments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-2">
        <div className="text-4xl">🎙️</div>
        <p className="text-sm">Waiting for audio stream…</p>
        <p className="text-xs">Start Module 1 to begin transcription.</p>
      </div>
    );
  }

  return (
    <div className="space-y-1 py-2">
      {segments.map((seg) => {
        // Find a matching verdict (within ±2s of segment start)
        const verdictMatch = verdicts.find(
          (v) => Math.abs(v.claim_start - seg.start) < 2.5
        );

        return (
          <div
            key={seg.id}
            className={clsx(
              "flex gap-3 px-2 py-1.5 rounded-lg transition-colors",
              verdictMatch
                ? {
                    "bg-green-50": verdictMatch.label === "supported",
                    "bg-red-50":   verdictMatch.label === "refuted",
                    "bg-amber-50": verdictMatch.label === "mixed",
                    "bg-slate-50": verdictMatch.label === "not_enough_info",
                  }
                : "hover:bg-slate-50"
            )}
          >
            {/* Timestamp */}
            <span className="flex-shrink-0 text-xs text-slate-400 font-mono pt-0.5 w-10">
              {formatTime(seg.start)}
            </span>

            {/* Speaker label */}
            <span className="flex-shrink-0 text-xs font-medium text-slate-500 pt-0.5 w-20 truncate">
              {seg.speaker}
            </span>

            {/* Text */}
            <p
              className={clsx(
                "flex-1 text-sm leading-relaxed",
                seg.is_final ? "text-slate-900" : "text-slate-400 italic"
              )}
            >
              {seg.text}
            </p>

            {/* Inline verdict badge */}
            {verdictMatch && (
              <div className="flex-shrink-0 self-start pt-0.5">
                <VerdictBadge
                  label={verdictMatch.label}
                  confidence={verdictMatch.confidence}
                  size="sm"
                />
              </div>
            )}
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}

