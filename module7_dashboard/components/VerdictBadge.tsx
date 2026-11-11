// components/VerdictBadge.tsx  —  Colour-coded verdict label pill

import { VerdictLabel } from "@/lib/types";
import clsx from "clsx";

const LABEL_CONFIG: Record<
  VerdictLabel,
  { text: string; bg: string; border: string; dot: string }
> = {
  supported: {
    text: "Supported",
    bg: "bg-green-50",
    border: "border-green-400",
    dot: "bg-green-500",
  },
  refuted: {
    text: "Refuted",
    bg: "bg-red-50",
    border: "border-red-400",
    dot: "bg-red-500",
  },
  mixed: {
    text: "Mixed",
    bg: "bg-amber-50",
    border: "border-amber-400",
    dot: "bg-amber-500",
  },
  not_enough_info: {
    text: "Unverified",
    bg: "bg-slate-50",
    border: "border-slate-300",
    dot: "bg-slate-400",
  },
};

interface VerdictBadgeProps {
  label: VerdictLabel;
  confidence?: number;
  size?: "sm" | "md" | "lg";
}

export function VerdictBadge({
  label,
  confidence,
  size = "md",
}: VerdictBadgeProps) {
  const cfg = LABEL_CONFIG[label] ?? LABEL_CONFIG.not_enough_info;

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border font-medium",
        cfg.bg,
        cfg.border,
        {
          "px-2 py-0.5 text-xs": size === "sm",
          "px-3 py-1 text-sm": size === "md",
          "px-4 py-1.5 text-base": size === "lg",
        }
      )}
    >
      <span className={clsx("h-2 w-2 rounded-full", cfg.dot)} />
      {cfg.text}
      {confidence !== undefined && (
        <span className="text-slate-500 font-normal">
          {Math.round(confidence * 100)}%
        </span>
      )}
    </span>
  );
}

