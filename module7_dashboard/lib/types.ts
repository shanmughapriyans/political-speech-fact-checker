// lib/types.ts  —  Shared TypeScript types (mirrors Python schemas)

export type VerdictLabel = "supported" | "refuted" | "mixed" | "not_enough_info";

// ── Module 3 output ──────────────────────────────────────────────────────────
export interface SubQuestion {
  q_id: number;
  question: string;
  search_query: string;
  question_type: string;
}

export interface SubQuestionSet {
  set_id: number;
  claim_id: number;
  claim_text: string;
  questions: SubQuestion[];
  n_questions: number;
  generated_at: number;
}

// ── Module 4 output ──────────────────────────────────────────────────────────
export interface EvidencePassage {
  passage_id: string;
  source_name: string;
  source_url: string;
  text: string;
  retrieval_score: number;
  credibility_score: number;
  q_id: number;
}

export interface EvidenceBundle {
  bundle_id: number;
  claim_id: number;
  claim_text: string;
  passages: EvidencePassage[];
  n_passages: number;
  retrieved_at: number;
}

// ── Module 5 output ──────────────────────────────────────────────────────────
export interface SubVerdictIntermediate {
  q_id: number;
  question: string;
  answer: string;
  label: "supported" | "refuted" | "not_enough_info";
  confidence: number;
  reasoning: string;
}

export interface VerdictBundle {
  bundle_id: number;
  claim_id: number;
  claim_text: string;
  sub_verdicts: SubVerdictIntermediate[];
  verified_at: number;
}

// ── Module 6 output (final verdict) ─────────────────────────────────────────
export interface EvidenceSummaryItem {
  source: string;
  url: string;
  snippet: string;
  supports: boolean;
}

export interface SubVerdict {
  q_id: number;
  question: string;
  answer: string;
  label: "supported" | "refuted" | "not_enough_info";
  confidence: number;
  entailment_score: number;
  supporting_passage_ids: string[];
  reasoning: string;
}

export interface Verdict {
  verdict_id: number;
  claim_id: number;
  claim_text: string;
  claim_speaker: string;
  claim_start: number;   // stream-relative seconds
  claim_end: number;
  claim_score: number;
  label: VerdictLabel;
  confidence: number;
  explanation: string;
  evidence_summary: EvidenceSummaryItem[];
  sub_verdicts: SubVerdict[];
  conflicts_detected: boolean;
  conflict_resolution_note: string | null;
  emitted_at: number;    // Unix timestamp
}

export interface TranscriptSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  speaker: string;
  is_final: boolean;
  emitted_at: number;
}

export interface DashboardState {
  segments: TranscriptSegment[];
  verdicts: Verdict[];
  connected: boolean;
  lastUpdate: number;
}

// ── Pipeline tracker ─────────────────────────────────────────────────────────
// One entry per claim_id, built up stage by stage as SSE events arrive.
export type PipelineStage =
  | "detected"       // M2 emitted ClaimEvent
  | "decomposed"     // M3 emitted SubQuestionSet
  | "retrieved"      // M4 emitted EvidenceBundle
  | "verified"       // M5 emitted VerdictBundle
  | "resolved";      // M6 emitted Verdict

export interface PipelineEntry {
  claim_id: number;
  claim_text: string;
  stage: PipelineStage;
  claim_score?: number;
  sub_questions?: SubQuestion[];
  passages?: EvidencePassage[];
  sub_verdicts?: SubVerdictIntermediate[];
  verdict?: Verdict;
  started_at: number;   // ms timestamp when first seen
  updated_at: number;
}

