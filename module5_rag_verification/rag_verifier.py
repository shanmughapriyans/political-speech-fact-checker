"""
rag_verifier.py  —  Module 5: Multi-step RAG Verification
==========================================================
Takes an EvidenceBundle (retrieved passages + sub-questions from Module 4)
and produces a VerdictBundle (per-sub-question verdicts).

Pipeline per sub-question
--------------------------
1. Gather passages for this q_id from the EvidenceBundle.
2. Build a context string from the top-N passages (ranked by retrieval_score).
3. Call the LLM to answer the sub-question given that context (RAG step).
4. Run NLI scoring: score the LLM answer against each passage to get
   entailment/contradiction probabilities.
5. Aggregate NLI scores (weighted by passage credibility) → VerdictLabel.
6. If confidence < NEI_THRESHOLD, label as NEI.

LLM fallback
------------
If no API key is set, the verifier uses the top passage's text as the
"answer" directly (passage-only mode), which is weaker but functional
for demos.

Chain-of-RAG
------------
Sub-questions with depends_on=[q_id] receive the previous sub-question's
answer appended to their context, enabling multi-hop reasoning.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import importlib.util as _m5_ilu, sys as _m5_sys

_m5_dir = str(Path(__file__).parent)
if _m5_dir not in _m5_sys.path:
    _m5_sys.path.insert(0, _m5_dir)

def _load_own_schemas():
    name = "m5_schemas"
    if name in _m5_sys.modules:
        return _m5_sys.modules[name]
    spec = _m5_ilu.spec_from_file_location(name, Path(__file__).parent / "schemas.py")
    mod = _m5_ilu.module_from_spec(spec)
    _m5_sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_m5_schemas = _load_own_schemas()
SubVerdict   = _m5_schemas.SubVerdict
VerdictBundle = _m5_schemas.VerdictBundle
VerdictLabel  = _m5_schemas.VerdictLabel

from nli_scorer import NLIScorer

log = logging.getLogger("module5.rag_verifier")

NEI_THRESHOLD  = 0.45    # below this, label as NEI regardless
TOP_PASSAGES   = 3       # passages fed to LLM context per sub-question
MAX_CTX_CHARS  = 1500    # truncate context to avoid token overflow

_SYSTEM_PROMPT = """You are a fact-checking assistant. Answer the question using
ONLY the provided evidence passages. Be concise (1-2 sentences).
If the evidence does not answer the question, say "Insufficient evidence."
Do NOT use external knowledge."""

_USER_TEMPLATE = """Evidence:
{context}

Question: {question}

Answer:"""


class RAGVerifier:
    """
    Multi-step RAG verifier.

    Parameters
    ----------
    nli_scorer : NLIScorer | None
        NLI scorer instance.  Created automatically if None.
    model : str
        OpenAI model for RAG answers.
    api_key : str | None
        OpenAI API key.  Falls back to env var OPENAI_API_KEY.
    """

    def __init__(
        self,
        nli_scorer: Optional[NLIScorer] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.nli = nli_scorer or NLIScorer()
        self.model = model
        self._client = None

        key = api_key or os.getenv("OPENAI_API_KEY")
        if key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=key, timeout=15.0)
                log.info("OpenAI RAG client ready (model=%s)", model)
            except ImportError:
                log.warning("openai not installed — passage-only mode")
        else:
            log.warning("No OPENAI_API_KEY — passage-only mode")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def verify(self, evidence_bundle) -> VerdictBundle:
        """
        Verify all sub-questions in an EvidenceBundle.

        Parameters
        ----------
        evidence_bundle
            Any object with .claim_id, .claim_text, .claim_speaker,
            .claim_start, .claim_end, .claim_score, .sub_questions,
            .passages, and a .passages_for_question(q_id) method.

        Returns
        -------
        VerdictBundle
        """
        claim_id  = getattr(evidence_bundle, "claim_id", 0)
        claim_text = getattr(evidence_bundle, "claim_text", "")

        sub_questions = getattr(evidence_bundle, "sub_questions", [])
        sub_verdicts: list[SubVerdict] = []
        # Map q_id -> answer for Chain-of-RAG
        previous_answers: Dict[int, str] = {}

        for q_raw in sub_questions:
            q_id    = q_raw.get("q_id", 0) if isinstance(q_raw, dict) else q_raw.q_id
            q_text  = q_raw.get("text", "") if isinstance(q_raw, dict) else q_raw.text
            depends = q_raw.get("depends_on", []) if isinstance(q_raw, dict) else q_raw.depends_on

            # Gather passages
            try:
                passages = evidence_bundle.passages_for_question(q_id)
            except AttributeError:
                passages = [
                    p for p in evidence_bundle.passages
                    if getattr(p, "question_q_id", p.get("question_q_id", -1) if isinstance(p, dict) else -1) == q_id
                ]

            if not passages:
                # No evidence — NEI
                sub_verdicts.append(SubVerdict(
                    q_id=q_id,
                    question=q_text,
                    answer="Insufficient evidence.",
                    label=VerdictLabel.NEI,
                    confidence=0.0,
                    entailment_score=0.0,
                    supporting_passage_ids=[],
                    reasoning="No relevant evidence passages were retrieved for this question.",
                ))
                continue

            # Build context (top passages by retrieval_score)
            sorted_passages = sorted(
                passages,
                key=lambda p: getattr(p, "retrieval_score", 0.0) if not isinstance(p, dict) else p.get("retrieval_score", 0.0),
                reverse=True,
            )
            top = sorted_passages[:TOP_PASSAGES]

            context_parts = []
            for i, p in enumerate(top):
                text = getattr(p, "text", "") if not isinstance(p, dict) else p.get("text", "")
                src  = getattr(p, "source_name", "") if not isinstance(p, dict) else p.get("source_name", "")
                context_parts.append(f"[{i+1}] ({src}) {text}")
            context = "\n".join(context_parts)[:MAX_CTX_CHARS]

            # Append Chain-of-RAG context from dependencies
            if depends:
                prior = " | ".join(
                    f"Q{d}: {previous_answers[d]}"
                    for d in depends if d in previous_answers
                )
                if prior:
                    context = f"Prior answers: {prior}\n\n{context}"

            # RAG answer
            answer = self._answer(q_text, context)
            previous_answers[q_id] = answer

            # NLI scoring
            passage_texts = [
                getattr(p, "text", "") if not isinstance(p, dict) else p.get("text", "")
                for p in top
            ]
            passage_creds = [
                getattr(p, "credibility_score", 0.75) if not isinstance(p, dict) else p.get("credibility_score", 0.75)
                for p in top
            ]
            nli_results = self.nli.score_batch(passage_texts, answer)

            # Weighted aggregation by credibility
            total_cred = sum(passage_creds) or 1.0
            agg = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
            for scores, cred in zip(nli_results, passage_creds):
                w = cred / total_cred
                for k in agg:
                    agg[k] += w * scores[k]

            # Determine label
            entail_s = agg["entailment"]
            contra_s = agg["contradiction"]

            if max(entail_s, contra_s) < NEI_THRESHOLD:
                label = VerdictLabel.NEI
                confidence = 1.0 - max(entail_s, contra_s)
            elif entail_s >= contra_s:
                label = VerdictLabel.SUPPORTED
                confidence = entail_s
            else:
                label = VerdictLabel.REFUTED
                confidence = contra_s

            supporting_ids = [
                getattr(p, "passage_id", "") if not isinstance(p, dict) else p.get("passage_id", "")
                for p in top if agg["entailment"] > 0.4
            ]

            reasoning = self._reasoning(q_text, answer, label, top)

            sub_verdicts.append(SubVerdict(
                q_id=q_id,
                question=q_text,
                answer=answer,
                label=label,
                confidence=round(confidence, 4),
                entailment_score=round(entail_s, 4),
                supporting_passage_ids=supporting_ids,
                reasoning=reasoning,
            ))
            log.info(
                "Q%d [%s] conf=%.2f  %r",
                q_id, label.value, confidence, q_text[:60],
            )

        return VerdictBundle(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_speaker=getattr(evidence_bundle, "claim_speaker",
                                  getattr(evidence_bundle, "speaker", "UNKNOWN")),
            claim_start=getattr(evidence_bundle, "claim_start", 0.0),
            claim_end=getattr(evidence_bundle, "claim_end", 0.0),
            claim_score=getattr(evidence_bundle, "claim_score", 1.0),
            sub_verdicts=sub_verdicts,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _answer(self, question: str, context: str) -> str:
        """Call LLM or return best passage text."""
        if self._client is None:
            # Passage-only mode: first sentence of context
            first_line = context.split("\n")[0]
            return first_line[first_line.find(")") + 1:].strip()[:200]

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content":
                     _USER_TEMPLATE.format(context=context, question=question)},
                ],
                max_tokens=120,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("LLM answer failed: %s", exc)
            first_line = context.split("\n")[0]
            return first_line[first_line.find(")") + 1:].strip()[:200]

    def _reasoning(
        self, question: str, answer: str,
        label: VerdictLabel, passages: list
    ) -> str:
        """Generate a one-sentence reasoning string."""
        if not passages:
            return "No evidence was found to evaluate this question."
        src = getattr(passages[0], "source_name", "the evidence") if not isinstance(passages[0], dict) else passages[0].get("source_name", "the evidence")
        if label == VerdictLabel.SUPPORTED:
            return f"{src} directly supports the claim: '{answer[:100]}'"
        elif label == VerdictLabel.REFUTED:
            return f"{src} contradicts the claim: '{answer[:100]}'"
        else:
            return f"Available evidence ({src}) does not conclusively confirm or refute the claim."


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

    print("=== RAGVerifier self-test ===\n")

    # Build a fake EvidenceBundle
    class _FakePassage:
        def __init__(self, pid, text, source, cred, score, qid):
            self.passage_id = pid
            self.text = text
            self.source_name = source
            self.credibility_score = cred
            self.retrieval_score = score
            self.question_q_id = qid

    class _FakeBundle:
        claim_id     = 42
        claim_text   = "Unemployment has fallen to 3.8 percent, the lowest in a decade."
        claim_speaker = "SPEAKER_00"
        claim_start  = 4.8
        claim_end    = 8.3
        claim_score  = 0.991
        sub_questions = [
            {"q_id": 0, "text": "What was the UK unemployment rate in early 2024?",
             "search_query": "UK unemployment rate 2024", "depends_on": [], "category": "numeric"},
            {"q_id": 1, "text": "What was the previous record low unemployment rate?",
             "search_query": "UK unemployment decade lowest", "depends_on": [0], "category": "temporal"},
        ]
        passages = [
            _FakePassage("ons_emp_1",
                "UK unemployment rate fell to 3.8% in January 2024, the lowest since 2020.",
                "ONS", 0.98, 0.91, 0),
            _FakePassage("ons_emp_2",
                "The employment rate was 74.5% in December 2023, the highest since records began.",
                "ONS", 0.98, 0.72, 0),
            _FakePassage("bbc_emp_1",
                "Unemployment held at 3.8% but economists warn of a gradual rise in 2024.",
                "BBC", 0.86, 0.65, 1),
        ]
        def passages_for_question(self, q_id):
            return [p for p in self.passages if p.question_q_id == q_id]

    verifier = RAGVerifier()
    bundle = verifier.verify(_FakeBundle())

    print(repr(bundle))
    for sv in bundle.sub_verdicts:
        print(f"\n  Q{sv.q_id}: [{sv.label.value}] conf={sv.confidence:.2f}")
        print(f"    Answer: {sv.answer[:100]}")
        print(f"    Reason: {sv.reasoning}")

    print("\nJSON round-trip:")
    restored = VerdictBundle.from_json(bundle.to_json())
    assert restored.claim_id == bundle.claim_id
    assert len(restored.sub_verdicts) == len(bundle.sub_verdicts)
    print("  [PASS] round-trip OK")
    print(f"\n[PASS] {len(bundle.sub_verdicts)} sub-verdicts produced.")

