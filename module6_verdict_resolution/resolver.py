"""
resolver.py  —  Module 6: Verdict Aggregation & Self-Reflection
===============================================================
Takes a VerdictBundle (per-sub-question judgments from Module 5) and
produces a single Verdict (the final, conflict-resolved verdict).

Steps
-----
1. Aggregate sub-verdicts with credibility-weighted majority vote.
2. Detect conflicts (SUPPORTED + REFUTED sub-verdicts for the same claim).
3. If conflicted, run a self-reflection pass via the LLM to pick the
   better-supported side (MultiReflect pattern).
4. Generate a natural-language explanation.
5. Build the evidence_summary for the UI.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import importlib.util as _m6_ilu

_m6_dir = str(Path(__file__).parent)
if _m6_dir not in sys.path:
    sys.path.insert(0, _m6_dir)

def _load_m6_schemas():
    name = "m6_schemas"
    if name in sys.modules:
        return sys.modules[name]
    spec = _m6_ilu.spec_from_file_location(name, Path(__file__).parent / "schemas.py")
    mod = _m6_ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_m6_schemas = _load_m6_schemas()
EvidenceSummaryItem = _m6_schemas.EvidenceSummaryItem
FinalLabel          = _m6_schemas.FinalLabel
Verdict             = _m6_schemas.Verdict

from credibility import aggregate_label, detect_conflicts, weight_for_source_name

log = logging.getLogger("module6.resolver")

_REFLECT_SYSTEM = """You are a senior fact-checking editor resolving a disputed claim.
You will be given a claim, and evidence that partially supports and partially refutes it.
Decide the most defensible verdict (supported/refuted/mixed/not_enough_info) based on
source credibility and strength of evidence.  Return JSON: {"label": "...", "explanation": "..."}"""

_REFLECT_USER = """Claim: "{claim}"

Sub-verdicts:
{sub_verdicts}

Resolve the conflict and provide a 2-sentence explanation."""


class VerdictResolver:
    """
    Aggregates sub-verdicts into a final Verdict with self-reflection.

    Parameters
    ----------
    model : str
        OpenAI model for reflection pass.
    api_key : str | None
        OpenAI API key (falls back to env OPENAI_API_KEY).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self._client = None

        key = api_key or os.getenv("OPENAI_API_KEY")
        if key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=key, timeout=15.0)
                log.info("OpenAI resolver client ready (model=%s)", model)
            except ImportError:
                log.warning("openai not installed — no self-reflection pass")
        else:
            log.warning("No OPENAI_API_KEY — no self-reflection pass")

    def resolve(self, verdict_bundle) -> Verdict:
        """
        Resolve a VerdictBundle into a final Verdict.

        Parameters
        ----------
        verdict_bundle
            Any object with .claim_id, .claim_text, .claim_speaker,
            .claim_start, .claim_end, .claim_score, .sub_verdicts.
        """
        claim_id   = getattr(verdict_bundle, "claim_id", 0)
        claim_text = getattr(verdict_bundle, "claim_text", "")
        speaker    = getattr(verdict_bundle, "claim_speaker",
                             getattr(verdict_bundle, "speaker", "UNKNOWN"))
        start      = getattr(verdict_bundle, "claim_start", 0.0)
        end        = getattr(verdict_bundle, "claim_end", 0.0)
        score      = getattr(verdict_bundle, "claim_score", 1.0)
        sub_vs     = getattr(verdict_bundle, "sub_verdicts", [])

        # Convert to dicts for downstream processing
        sub_v_dicts = []
        for sv in sub_vs:
            if isinstance(sv, dict):
                sub_v_dicts.append(sv)
            elif hasattr(sv, "to_dict"):
                sub_v_dicts.append(sv.to_dict())
            else:
                sub_v_dicts.append({
                    "q_id": getattr(sv, "q_id", 0),
                    "question": getattr(sv, "question", ""),
                    "answer": getattr(sv, "answer", ""),
                    "label": getattr(sv, "label", "not_enough_info"),
                    "confidence": getattr(sv, "confidence", 0.5),
                    "entailment_score": getattr(sv, "entailment_score", 0.0),
                    "supporting_passage_ids": getattr(sv, "supporting_passage_ids", []),
                    "reasoning": getattr(sv, "reasoning", ""),
                })
            # Normalize label enum → string
            sv_dict = sub_v_dicts[-1]
            if hasattr(sv_dict.get("label"), "value"):
                sv_dict["label"] = sv_dict["label"].value

        # --- Step 1: aggregate ---
        label_str, confidence = aggregate_label(sub_v_dicts)
        conflicts = detect_conflicts(sub_v_dicts)

        conflict_note: Optional[str] = None

        # --- Step 2: self-reflection on conflict ---
        if conflicts and self._client:
            label_str, confidence, conflict_note, explanation = self._reflect(
                claim_text, sub_v_dicts
            )
        else:
            explanation = self._explain(claim_text, label_str, sub_v_dicts)
            if conflicts:
                conflict_note = (
                    "Evidence sources disagreed. The majority-weighted verdict is shown. "
                    "No LLM reflection available to resolve further."
                )

        # --- Step 3: evidence summary ---
        evidence_summary = self._build_evidence_summary(sub_v_dicts)

        return Verdict(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_speaker=speaker,
            claim_start=start,
            claim_end=end,
            claim_score=score,
            label=FinalLabel(label_str),
            confidence=confidence,
            explanation=explanation,
            evidence_summary=evidence_summary,
            sub_verdicts=sub_v_dicts,
            conflicts_detected=conflicts,
            conflict_resolution_note=conflict_note,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _reflect(
        self, claim_text: str, sub_v_dicts: list
    ) -> tuple[str, float, str, str]:
        """LLM self-reflection to resolve conflicting sub-verdicts."""
        sv_summary = "\n".join(
            f"Q{sv.get('q_id',i)}: [{sv.get('label','?')}] "
            f"conf={sv.get('confidence',0):.2f} — {sv.get('reasoning','')}"
            for i, sv in enumerate(sub_v_dicts)
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _REFLECT_SYSTEM},
                    {"role": "user", "content":
                     _REFLECT_USER.format(
                         claim=claim_text, sub_verdicts=sv_summary)},
                ],
                temperature=0.0,
            )
            data = json.loads(resp.choices[0].message.content)
            lbl  = data.get("label", "mixed")
            expl = data.get("explanation", "")
            note = "Conflict resolved via LLM self-reflection pass."
            # confidence stays from aggregate but slightly boosted for reflection
            _, conf = aggregate_label(sub_v_dicts)
            conf = min(0.95, conf + 0.05)
            return (lbl, round(conf, 4), note, expl)
        except Exception as exc:
            log.warning("Self-reflection failed: %s", exc)
            # Fall back
            label_str, conf = aggregate_label(sub_v_dicts)
            expl = self._explain(claim_text, label_str, sub_v_dicts)
            note = f"Conflict detected but reflection unavailable ({exc}). Majority vote used."
            return (label_str, conf, note, expl)

    def _explain(
        self, claim_text: str, label_str: str, sub_v_dicts: list
    ) -> str:
        """Generate a plain-English explanation without LLM."""
        supported = [sv for sv in sub_v_dicts if sv.get("label") == "supported"]
        refuted   = [sv for sv in sub_v_dicts if sv.get("label") == "refuted"]
        nei       = [sv for sv in sub_v_dicts if sv.get("label") == "not_enough_info"]

        if label_str == "supported":
            reasons = "; ".join(sv.get("reasoning", "")[:80] for sv in supported[:2])
            return f"The claim appears to be supported by available evidence. {reasons}"
        elif label_str == "refuted":
            reasons = "; ".join(sv.get("reasoning", "")[:80] for sv in refuted[:2])
            return f"The claim appears to be refuted by available evidence. {reasons}"
        elif label_str == "mixed":
            return (
                f"Evidence is mixed: {len(supported)} sub-question(s) supported, "
                f"{len(refuted)} refuted. The claim may be partially accurate."
            )
        else:
            return (
                f"Insufficient evidence was found to verify or refute this claim. "
                f"{len(nei)} of {len(sub_v_dicts)} sub-questions returned no conclusive evidence."
            )

    def _build_evidence_summary(
        self, sub_v_dicts: list
    ) -> list[EvidenceSummaryItem]:
        """Build the evidence_summary list for the UI from sub-verdicts."""
        seen: set[str] = set()
        items: list[EvidenceSummaryItem] = []

        for sv in sub_v_dicts:
            answer   = sv.get("answer", "")
            label    = sv.get("label", "not_enough_info")
            supports = label == "supported"
            # We don't have passage URLs here (they're in Module 4),
            # so we construct a summary item from the reasoning
            reasoning = sv.get("reasoning", "")
            source    = reasoning.split(" ")[0] if reasoning else "Evidence"
            snippet   = answer[:120] if answer else sv.get("question", "")

            key = snippet[:40]
            if key in seen:
                continue
            seen.add(key)

            items.append(EvidenceSummaryItem(
                source=source,
                url="",        # URL not propagated through Module 5 (extend later)
                snippet=snippet,
                supports=supports,
            ))
        return items


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / "module1_speech_asr" / ".env")
    except ImportError:
        pass

    print("=== VerdictResolver self-test ===\n")

    class _FakeBundle:
        claim_id    = 42
        claim_text  = "Unemployment has fallen to 3.8 percent, the lowest in a decade."
        claim_speaker = "SPEAKER_00"
        claim_start = 4.8
        claim_end   = 8.3
        claim_score = 0.991
        sub_verdicts = [
            {
                "q_id": 0, "question": "What was the UK unemployment rate in early 2024?",
                "answer": "3.8% per ONS January 2024 release.",
                "label": "supported", "confidence": 0.91,
                "entailment_score": 0.87, "supporting_passage_ids": ["ons_emp_1"],
                "reasoning": "ONS data directly states 3.8% unemployment in January 2024.",
            },
            {
                "q_id": 1, "question": "Is 3.8% the lowest unemployment in a decade?",
                "answer": "Records show 3.8% is not the decade low; it reached 3.6% in 2022.",
                "label": "refuted", "confidence": 0.72,
                "entailment_score": 0.21, "supporting_passage_ids": [],
                "reasoning": "ONS data shows unemployment was lower (3.6%) in 2022.",
            },
        ]

    resolver = VerdictResolver()
    verdict = resolver.resolve(_FakeBundle())

    print(repr(verdict))
    print(f"\nLabel: {verdict.label.value}")
    print(f"Confidence: {verdict.confidence:.2f}")
    print(f"Conflicts: {verdict.conflicts_detected}")
    print(f"Explanation: {verdict.explanation[:150]}")
    print(f"Evidence items: {len(verdict.evidence_summary)}")

    print("\nJSON round-trip:")
    restored = Verdict.from_json(verdict.to_json())
    assert restored.claim_id == verdict.claim_id
    assert restored.label == verdict.label
    print("  [PASS] round-trip OK")
    print("\n[PASS] VerdictResolver self-test passed.")

