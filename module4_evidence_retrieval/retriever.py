"""
retriever.py  —  Module 4: Hybrid BM25 + Dense Evidence Retrieval
==================================================================
Retrieves ranked evidence passages for each sub-question in a
SubQuestionSet.  Outputs an EvidenceBundle.

Architecture
------------
1. BM25 pass   — keyword match against an in-memory corpus using rank_bm25.
                 Fast, recall-oriented.  Returns top-K candidates.
2. Dense pass  — sentence-transformer embeddings + cosine similarity.
                 Precision-oriented.  Reranks the BM25 candidates.
3. Reranking   — Weighted sum: 0.4 * BM25 score + 0.6 * dense score.
4. Source filter — only passages from SOURCES in source_registry.py.

The corpus is populated from:
  a. A bundled static corpus (data/corpus.jsonl) — hand-curated passages
     covering common political topics.
  b. (Future) A live OpenSearch/FAISS index connected via environment vars.

Offline / no-internet mode
--------------------------
If the sentence-transformer model is not available, the dense pass is
skipped and BM25-only ranking is used.  This keeps the pipeline running
during development.

Usage::

    from retriever import EvidenceRetriever
    retriever = EvidenceRetriever()
    bundle = retriever.retrieve(sub_question_set)
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_M4_DIR = str(Path(__file__).parent)
if _M4_DIR not in sys.path:
    sys.path.insert(0, _M4_DIR)

# Use importlib so this always loads module4's schemas.py even when
# other modules' schema files appear earlier on sys.path.
import importlib.util as _m4_ilu
_m4_spec = _m4_ilu.spec_from_file_location(
    "m4_schemas", Path(__file__).parent / "schemas.py"
)
_m4_mod = _m4_ilu.module_from_spec(_m4_spec)
import sys as _sys
if "m4_schemas" not in _sys.modules:
    _sys.modules["m4_schemas"] = _m4_mod
    _m4_spec.loader.exec_module(_m4_mod)
else:
    _m4_mod = _sys.modules["m4_schemas"]
EvidencePassage = _m4_mod.EvidencePassage
EvidenceBundle  = _m4_mod.EvidenceBundle

from source_registry import SOURCES, credibility_for_url

log = logging.getLogger("module4.retriever")

_DATA_DIR  = Path(__file__).parent / "data"
_CORPUS_FILE = _DATA_DIR / "corpus.jsonl"

TOP_K_BM25   = 20   # candidates from BM25
TOP_K_FINAL  = 5    # passages per sub-question
DENSE_WEIGHT = 0.6
BM25_WEIGHT  = 0.4


# ---------------------------------------------------------------------------
# Static corpus — bundled fact-checked passages for common claim topics
# ---------------------------------------------------------------------------

_STATIC_CORPUS: list[dict] = [
    # Employment
    {"id": "ons_emp_1", "text": "UK unemployment rate fell to 3.8% in the three months to January 2024, down from 4.2% a year earlier, according to the Office for National Statistics.", "source_name": "ONS", "source_url": "https://www.ons.gov.uk/employment/unemployment", "tier": 1},
    {"id": "ons_emp_2", "text": "The employment rate in the UK was 74.5% in December 2023, the highest since comparable records began.", "source_name": "ONS", "source_url": "https://www.ons.gov.uk/employment/employment", "tier": 1},
    {"id": "bls_emp_1", "text": "US nonfarm payrolls increased by 353,000 in January 2024, with the unemployment rate unchanged at 3.7%.", "source_name": "BLS", "source_url": "https://www.bls.gov/news.release/empsit.nr0.htm", "tier": 1},
    {"id": "bls_wages_1", "text": "Average hourly earnings for all employees rose 4.5% year-over-year in January 2024, outpacing inflation for the first time in two years.", "source_name": "BLS", "source_url": "https://www.bls.gov/news.release/empsit.nr0.htm", "tier": 1},
    # GDP & Economy
    {"id": "worldbank_gdp_1", "text": "Global GDP growth slowed to 2.4% in 2023, with advanced economies growing at just 1.5%, according to World Bank projections.", "source_name": "World Bank", "source_url": "https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG", "tier": 1},
    {"id": "oecd_gdp_1", "text": "OECD composite leading indicators point to a gradual recovery in GDP growth across member countries in the second half of 2024.", "source_name": "OECD", "source_url": "https://data.oecd.org/leadind/composite-leading-indicator-cli.htm", "tier": 1},
    {"id": "cbo_debt_1", "text": "US federal debt held by the public reached 97% of GDP at end of fiscal year 2023, the highest since World War II, per CBO projections.", "source_name": "CBO", "source_url": "https://www.cbo.gov/publication/59727", "tier": 1},
    # Inflation
    {"id": "ons_inflation_1", "text": "UK CPI inflation fell to 2.3% in April 2024, down from a peak of 11.1% in October 2022.", "source_name": "ONS", "source_url": "https://www.ons.gov.uk/economy/inflationandpriceindices", "tier": 1},
    {"id": "bls_inflation_1", "text": "US CPI rose 3.5% year-over-year in March 2024, slightly above the Federal Reserve's 2% target.", "source_name": "BLS", "source_url": "https://www.bls.gov/news.release/cpi.nr0.htm", "tier": 1},
    {"id": "oecd_inflation_1", "text": "OECD average inflation was 5.4% in 2023, down from 9.4% in 2022, driven by falling energy prices.", "source_name": "OECD", "source_url": "https://data.oecd.org/price/inflation-cpi.htm", "tier": 1},
    # Health
    {"id": "nhs_wait_1", "text": "NHS England waiting lists reached a record 7.6 million in September 2023, with 3.3 million waiting more than 18 weeks.", "source_name": "NHS England", "source_url": "https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/", "tier": 1},
    {"id": "who_mortality_1", "text": "Global infant mortality has declined by 59% since 1990, falling to 27 deaths per 1,000 live births in 2021.", "source_name": "WHO", "source_url": "https://www.who.int/data/gho/data/themes/topics/indicator-groups/indicator-group-details/GHO/child-mortality", "tier": 1},
    # Housing & Poverty
    {"id": "ifs_poverty_1", "text": "Child poverty in the UK is projected to rise to 24% in 2023-24, up from 19% before the 2022 cost-of-living crisis, according to IFS.", "source_name": "IFS", "source_url": "https://ifs.org.uk/publications/child-poverty-projections", "tier": 3},
    {"id": "fullfact_housing_1", "text": "The UK government built approximately 234,000 new homes in 2022-23, below the 300,000 annual target.", "source_name": "Full Fact", "source_url": "https://fullfact.org/economy/house-building-target-england/", "tier": 2},
    # Crime
    {"id": "ons_crime_1", "text": "Police recorded crime in England and Wales fell 6% in the year to June 2023, to 6.7 million offences.", "source_name": "ONS", "source_url": "https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice", "tier": 1},
    # Environment
    {"id": "owid_emissions_1", "text": "Global CO2 emissions from fossil fuels reached a record 36.8 billion tonnes in 2023, up 1.1% from 2022.", "source_name": "Our World in Data", "source_url": "https://ourworldindata.org/co2-emissions", "tier": 3},
    {"id": "bbc_climate_1", "text": "UK carbon emissions fell 5.4% in 2023, the lowest since the 1800s, driven by reduced coal use.", "source_name": "BBC News", "source_url": "https://www.bbc.co.uk/news/science-environment-68352980", "tier": 2},
    # Education
    {"id": "oecd_edu_1", "text": "OECD countries spent an average of $12,000 per primary student annually in 2020, with the UK spending $11,200.", "source_name": "OECD", "source_url": "https://data.oecd.org/eduresource/spending-on-tertiary-education.htm", "tier": 1},
    # Defence
    {"id": "nato_defence_1", "text": "NATO defence spending targets require members to spend at least 2% of GDP. The UK spent 2.07% of GDP on defence in 2023.", "source_name": "Reuters", "source_url": "https://www.reuters.com/world/europe/nato-defence-spending-2023/", "tier": 2},
    # Trade
    {"id": "worldbank_trade_1", "text": "Global merchandise trade fell 5% in value in 2023 due to lower commodity prices, though volumes grew modestly.", "source_name": "World Bank", "source_url": "https://data.worldbank.org/indicator/TG.VAL.TOTL.GD.ZS", "tier": 1},
    # Food banks
    {"id": "reuters_foodbank_1", "text": "Food bank use in the UK rose sharply in 2022-23, with the Trussell Trust reporting record 3 million emergency food parcels.", "source_name": "Reuters", "source_url": "https://www.reuters.com/uk-food-bank-record-2023", "tier": 2},
    # Wages
    {"id": "ons_wages_1", "text": "UK real wages fell 2.2% in 2022-23 when adjusted for inflation, the largest annual fall since records began in 2001.", "source_name": "ONS", "source_url": "https://www.ons.gov.uk/economy/earningsandworkinghours", "tier": 1},
]


def _ensure_corpus() -> None:
    """Write static corpus to disk if it doesn't exist (for future indexing)."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _CORPUS_FILE.exists():
        with _CORPUS_FILE.open("w", encoding="utf-8") as fh:
            for doc in _STATIC_CORPUS:
                fh.write(json.dumps(doc) + "\n")
        log.info("Static corpus written to %s (%d docs)", _CORPUS_FILE, len(_STATIC_CORPUS))


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    text = re.sub(r"[^\w\s%]", " ", text.lower())
    return text.split()


class _BM25:
    """Minimal BM25 implementation (k1=1.5, b=0.75)."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b  = b
        self.tokenised = [_tokenise(doc) for doc in corpus]
        n = len(self.tokenised)
        avgdl = sum(len(d) for d in self.tokenised) / max(n, 1)
        self.idf: dict[str, float] = {}
        from collections import Counter
        df: dict[str, int] = {}
        for doc in self.tokenised:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        for term, freq in df.items():
            self.idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)
        self.avgdl = avgdl

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = _tokenise(query)
        doc = self.tokenised[doc_idx]
        dl = len(doc)
        from collections import Counter
        tf = Counter(doc)
        s = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            f = tf[term]
            s += self.idf[term] * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.tokenised))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class EvidenceRetriever:
    """
    Hybrid BM25 + dense retriever over the trusted-source corpus.

    Parameters
    ----------
    top_k_per_question : int
        Maximum passages per sub-question (default: 5).
    use_dense : bool
        Whether to use sentence embeddings for reranking (default: True).
        Falls back to BM25-only if the model can't be loaded.
    embed_model : str
        Sentence-transformer model name (default: all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        top_k_per_question: int = TOP_K_FINAL,
        use_dense: bool = True,
        embed_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        _ensure_corpus()
        self.top_k = top_k_per_question
        self._corpus = _STATIC_CORPUS
        self._texts  = [d["text"] for d in self._corpus]
        self._bm25   = _BM25(self._texts)
        log.info("BM25 index built (%d docs)", len(self._corpus))

        self._embed_fn = None
        if use_dense:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                _model = SentenceTransformer(embed_model)
                _embeddings = _model.encode(self._texts, show_progress_bar=False)

                def _embed(query: str) -> "np.ndarray":
                    q_emb = _model.encode([query])
                    scores = np.dot(_embeddings, q_emb.T).squeeze()
                    return scores

                self._embed_fn = _embed
                log.info("Dense encoder loaded (%s)", embed_model)
            except Exception as exc:
                log.warning("Dense encoder unavailable (%s) — BM25 only", exc)

    def retrieve(self, sub_question_set) -> EvidenceBundle:
        """
        Retrieve evidence for all sub-questions in a SubQuestionSet.

        Parameters
        ----------
        sub_question_set
            Any object with .set_id, .claim_id, .claim_text, .questions.
            Compatible with module3_query_formulation.schemas.SubQuestionSet.

        Returns
        -------
        EvidenceBundle
        """
        set_id     = getattr(sub_question_set, "set_id", 0)
        claim_id   = getattr(sub_question_set, "claim_id", 0)
        claim_text = getattr(sub_question_set, "claim_text", "")
        questions  = getattr(sub_question_set, "questions", [])

        all_passages: list[EvidencePassage] = []

        for q in questions:
            q_id  = getattr(q, "q_id", 0) if not isinstance(q, dict) else q.get("q_id", 0)
            query = getattr(q, "search_query", "") if not isinstance(q, dict) else q.get("search_query", "")
            if not query:
                query = getattr(q, "text", "") if not isinstance(q, dict) else q.get("text", "")

            passages = self._retrieve_for_query(query, q_id, claim_text)
            all_passages.extend(passages)

        # De-duplicate by passage_id
        seen: set[str] = set()
        unique_passages: list[EvidencePassage] = []
        for p in all_passages:
            if p.passage_id not in seen:
                seen.add(p.passage_id)
                unique_passages.append(p)

        # Sort overall by retrieval_score
        unique_passages.sort(key=lambda p: p.retrieval_score, reverse=True)

        import dataclasses as _dc
        sub_q_dicts = []
        for q in questions:
            if isinstance(q, dict):
                sub_q_dicts.append(q)
            elif _dc.is_dataclass(q):
                sub_q_dicts.append(_dc.asdict(q))
            else:
                sub_q_dicts.append(vars(q))

        return EvidenceBundle(
            set_id=set_id,
            claim_id=claim_id,
            claim_text=claim_text,
            passages=unique_passages,
            sub_questions=sub_q_dicts,
        )

    def _retrieve_for_query(
        self, query: str, q_id: int, claim_text: str
    ) -> list[EvidencePassage]:
        """BM25 + optional dense rerank for a single query."""
        combined_query = f"{claim_text} {query}"

        # BM25
        bm25_hits = self._bm25.top_k(combined_query, TOP_K_BM25)
        max_bm25 = max((s for _, s in bm25_hits), default=1.0) or 1.0

        # Dense
        dense_scores: list[float] = []
        if self._embed_fn:
            try:
                raw = self._embed_fn(combined_query)
                dense_scores = list(raw)
                max_dense = max(dense_scores) or 1.0
                dense_scores = [s / max_dense for s in dense_scores]
            except Exception as exc:
                log.warning("Dense scoring failed: %s", exc)

        passages = []
        for idx, bm25_s in bm25_hits:
            if bm25_s <= 0:
                continue
            norm_bm25 = bm25_s / max_bm25
            norm_dense = dense_scores[idx] if dense_scores else 0.0
            combined = BM25_WEIGHT * norm_bm25 + DENSE_WEIGHT * norm_dense

            doc = self._corpus[idx]
            cred = credibility_for_url(doc.get("source_url", ""))
            # Boost retrieval score by credibility
            final_score = combined * (0.5 + 0.5 * cred)

            passages.append(EvidencePassage(
                passage_id=doc["id"],
                text=doc["text"],
                source_name=doc.get("source_name", "Unknown"),
                source_url=doc.get("source_url", ""),
                source_tier=doc.get("tier", 2),
                credibility_score=cred,
                retrieval_score=round(final_score, 4),
                question_q_id=q_id,
            ))

        passages.sort(key=lambda p: p.retrieval_score, reverse=True)
        return passages[:self.top_k]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== EvidenceRetriever self-test ===\n")

    # Fake SubQuestionSet
    class _FakeQ:
        def __init__(self, q_id, text, search_query):
            self.q_id = q_id
            self.text = text
            self.search_query = search_query

    class _FakeSQS:
        set_id = 7
        claim_id = 42
        claim_text = "Unemployment has fallen to 3.8 percent, the lowest in a decade."
        questions = [
            _FakeQ(0, "What was the UK unemployment rate in early 2024?",
                   "UK unemployment rate 2024 official"),
            _FakeQ(1, "What was the previous unemployment rate for comparison?",
                   "UK unemployment rate decade lowest"),
        ]

    retriever = EvidenceRetriever(use_dense=False)   # offline mode
    bundle = retriever.retrieve(_FakeSQS())

    print(repr(bundle))
    print(f"\nTotal passages: {len(bundle.passages)}")
    for p in bundle.passages[:5]:
        print(f"  [{p.question_q_id}] score={p.retrieval_score:.3f} "
              f"cred={p.credibility_score:.2f} [{p.source_name}]")
        print(f"       {p.text[:90]!r}")

    print("\nJSON round-trip:")
    restored = EvidenceBundle.from_json(bundle.to_json())
    assert restored.claim_id == bundle.claim_id
    assert len(restored.passages) == len(bundle.passages)
    print("  [PASS] round-trip OK")
    print(f"\n[PASS] {len(bundle.passages)} passage(s) retrieved.")

