"""
deduper.py  --  Two-pass claim deduplication for Module 2
==========================================================
Speakers repeat themselves ("jobs, jobs, jobs"), and Whisper polls
can occasionally produce near-identical committed segments.  Without
deduplication, the same claim could be sent to Module 3 five times,
wasting fact-checking budget and polluting the output.

Two-pass strategy
-----------------
Pass 1 — cheap normalisation + exact match (microseconds)
    Normalise: lowercase, strip punctuation, collapse number formatting,
               collapse whitespace.
    Compare against a sliding window of recent normalised fingerprints.

Pass 2 — sentence-embedding cosine similarity (milliseconds)
    Model: all-MiniLM-L6-v2 (~80 MB, loads lazily on first use)
    Cosine similarity >= ``embed_threshold`` (default 0.92) within the
    same time window → duplicate.

Critical number guard
---------------------
"Unemployment fell to 4 percent" and "unemployment fell to 5 percent"
are very similar embeddings but are DIFFERENT claims.  After pass 2
triggers, we extract all numeric tokens from both texts.  If the sets
differ, we override the duplicate verdict and let the claim through.

Sliding time window
-------------------
Claims older than ``window_s`` seconds are evicted automatically on
every ``check_and_add()`` call.  No unbounded memory growth.

Thread safety
-------------
The internal window is protected by a threading.Lock so the pipeline
can call ``check_and_add()`` from any thread.

Usage
-----
::

    from deduper import ClaimDeduper
    from claim_schema import ClaimEvent

    deduper = ClaimDeduper()        # lazy — no model loaded yet

    for claim in claim_stream:
        if not deduper.check_and_add(claim):
            send_to_module3(claim)  # genuine new claim

Checkpoint
----------
Run directly for tests::

    python deduper.py               # all unit tests
    python deduper.py --benchmark   # latency of both passes
"""

from __future__ import annotations

import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_S      = 120.0   # evict claims older than 2 minutes
DEFAULT_EMBED_THRESH  = 0.92    # cosine similarity >= this -> duplicate
DEFAULT_EMBED_MODEL   = "all-MiniLM-L6-v2"
MAX_WINDOW_SIZE       = 200     # hard cap on sliding window (memory safety)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Canonical number formats: "3,500" -> "3500", "2.5%" -> "2.5%"
_COMMA_IN_NUM = re.compile(r'(\d),(\d)')
_PUNCT        = re.compile(r"[^\w\s%]")   # strip punctuation, keep % and words
_WHITESPACE   = re.compile(r"\s+")

# Numeric token extraction for the number guard
_NUM_TOKEN = re.compile(r"\b\d+(?:\.\d+)?\s*%?")


def _normalise(text: str) -> str:
    """Cheap normalisation for pass-1 fingerprinting.

    Steps
    -----
    1. Lowercase
    2. Remove commas inside numbers ("3,500" -> "3500")
    3. Strip non-word punctuation (keep digits, letters, %, spaces)
    4. Collapse whitespace and strip
    """
    t = text.lower()
    t = _COMMA_IN_NUM.sub(r"\1\2", t)
    t = _PUNCT.sub(" ", t)
    t = _WHITESPACE.sub(" ", t).strip()
    return t


def _normalise_nums(text: str) -> str:
    """Partial normalisation for number extraction only.

    Lowercase + remove commas-inside-numbers, but keep dots so that
    '2.3' remains '2.3' and is not split into '2' and '3'.
    """
    t = text.lower()
    return _COMMA_IN_NUM.sub(r"\1\2", t)


def _numeric_tokens(text: str) -> frozenset[str]:
    """Extract numeric tokens, preserving decimals like '2.3'.

    Uses _normalise_nums() (not _normalise()) so dots are preserved.

    Examples::
        'fell to 4 percent'    -> frozenset({'4'})
        'GDP grew 2.3 percent' -> frozenset({'2.3'})
        'GDP grew 2.3%'        -> frozenset({'2.3%'})
        '3,500 jobs added'     -> frozenset({'3500'})  # comma stripped
    """
    return frozenset(
        m.strip() for m in _NUM_TOKEN.findall(_normalise_nums(text))
        if m.strip()
    )


# ---------------------------------------------------------------------------
# Window entry
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    claim_id:    int
    fingerprint: str           # normalised text (pass 1 key)
    text:        str           # original text (pass 2 / number guard)
    embedding:   Optional[np.ndarray]  # lazy; None until pass 2 needed
    added_at:    float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# ClaimDeduper
# ---------------------------------------------------------------------------

class ClaimDeduper:
    """Two-pass sliding-window claim deduplicator.

    Parameters
    ----------
    window_s : float
        How many seconds to keep claims in the window before evicting.
        Default: 120.0 (2 minutes).
    embed_threshold : float
        Cosine similarity >= this triggers a duplicate verdict in pass 2.
        Default: 0.92.
    embed_model : str
        sentence-transformers model name.  Loaded lazily on the first
        pass-2 comparison.
    use_embeddings : bool
        Set to False to skip pass 2 entirely (useful for unit tests or
        CPU-constrained environments where sentence-transformers is not
        installed).  Default: True.
    """

    def __init__(
        self,
        window_s:        float = DEFAULT_WINDOW_S,
        embed_threshold: float = DEFAULT_EMBED_THRESH,
        embed_model:     str   = DEFAULT_EMBED_MODEL,
        use_embeddings:  bool  = True,
    ) -> None:
        self._window_s        = window_s
        self._embed_threshold = embed_threshold
        self._embed_model_name = embed_model
        self._use_embeddings  = use_embeddings

        self._window: List[_Entry] = []
        self._lock    = threading.Lock()
        self._model   = None          # lazy-loaded sentence-transformer
        self._model_lock = threading.Lock()

        # Stats
        self._total_seen       = 0
        self._pass1_duplicates = 0
        self._pass2_duplicates = 0
        self._number_overrides = 0   # pass-2 would have deduped but numbers differ

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_add(self, claim) -> bool:
        """Check if ``claim`` is a duplicate; if not, add it to the window.

        Parameters
        ----------
        claim : ClaimEvent (or any object with ``.claim_id``, ``.text``,
                ``.emitted_at`` attributes)

        Returns
        -------
        bool
            True  → duplicate (do NOT forward to Module 3)
            False → genuine new claim (forward it)
        """
        with self._lock:
            self._total_seen += 1
            self._evict()

            fingerprint = _normalise(claim.text)

            # ---- Pass 1: exact normalised match ----
            for entry in self._window:
                if entry.fingerprint == fingerprint:
                    self._pass1_duplicates += 1
                    return True   # duplicate

            # ---- Pass 2: embedding similarity ----
            if self._use_embeddings and len(self._window) > 0:
                embedding = self._embed(claim.text)
                is_dup, overridden = self._pass2(claim.text, embedding)
                if overridden:
                    self._number_overrides += 1
                elif is_dup:
                    self._pass2_duplicates += 1
                    return True   # duplicate

            # ---- Not a duplicate — add to window ----
            embedding = self._embed(claim.text) if self._use_embeddings else None
            self._window.append(_Entry(
                claim_id    = claim.claim_id,
                fingerprint = fingerprint,
                text        = claim.text,
                embedding   = embedding,
                added_at    = time.monotonic(),
            ))

            # Hard cap
            if len(self._window) > MAX_WINDOW_SIZE:
                self._window.pop(0)

            return False   # genuine new claim

    def stats(self) -> dict:
        """Return deduplication statistics."""
        with self._lock:
            total_dup = self._pass1_duplicates + self._pass2_duplicates
            return {
                "total_seen":         self._total_seen,
                "pass1_duplicates":   self._pass1_duplicates,
                "pass2_duplicates":   self._pass2_duplicates,
                "number_overrides":   self._number_overrides,
                "total_duplicates":   total_dup,
                "dedup_rate":         round(
                    total_dup / max(self._total_seen, 1), 3
                ),
                "window_size":        len(self._window),
            }

    def reset(self) -> None:
        """Clear the sliding window and reset stats."""
        with self._lock:
            self._window.clear()
            self._total_seen       = 0
            self._pass1_duplicates = 0
            self._pass2_duplicates = 0
            self._number_overrides = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Remove entries older than ``window_s`` (called under lock)."""
        cutoff = time.monotonic() - self._window_s
        self._window = [e for e in self._window if e.added_at >= cutoff]

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Return the unit-normalised embedding for ``text``."""
        model = self._get_model()
        if model is None:
            return None
        vec = model.encode(text, normalize_embeddings=True)
        return vec

    def _get_model(self):
        """Lazily load the sentence-transformer model (thread-safe)."""
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:   # double-checked
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._embed_model_name)
            except ImportError:
                print(
                    "[deduper] sentence-transformers not installed; "
                    "pass-2 disabled.  pip install sentence-transformers"
                )
                self._use_embeddings = False
                self._model = None
            except Exception as exc:
                print(f"[deduper] Could not load embedding model: {exc}")
                self._use_embeddings = False
                self._model = None
        return self._model

    def _pass2(
        self,
        new_text: str,
        new_emb:  Optional[np.ndarray],
    ) -> tuple[bool, bool]:
        """Compare new_emb against all window embeddings.

        Returns
        -------
        (is_duplicate, number_overridden)
            is_duplicate      : True if cosine sim >= threshold
            number_overridden : True if pass-2 would have flagged it
                                but numeric tokens differ
        """
        if new_emb is None:
            return False, False

        new_nums = _numeric_tokens(new_text)

        for entry in self._window:
            if entry.embedding is None:
                continue
            sim = float(np.dot(new_emb, entry.embedding))   # unit vecs -> cosine
            if sim >= self._embed_threshold:
                # Check number guard
                entry_nums = _numeric_tokens(entry.text)
                if new_nums != entry_nums:
                    # Numbers differ -- different claim, do NOT deduplicate
                    return False, True
                return True, False

        return False, False


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _make_claim(text: str, claim_id: int = 0):
    """Minimal fake ClaimEvent for tests."""
    import types
    c = types.SimpleNamespace()
    c.text      = text
    c.claim_id  = claim_id
    c.emitted_at = time.time()
    return c


def _run_tests() -> None:
    print("=== ClaimDeduper tests ===\n")

    # ----------------------------------------------------------------
    # Test 1: identical text -> pass-1 duplicate
    # ----------------------------------------------------------------
    print("-- Test 1: identical text is a duplicate --")
    d = ClaimDeduper(use_embeddings=False)
    c1 = _make_claim("The unemployment rate fell to 3.5 percent.", 1)
    c2 = _make_claim("The unemployment rate fell to 3.5 percent.", 2)
    assert d.check_and_add(c1) is False,  "first seen should not be duplicate"
    assert d.check_and_add(c2) is True,   "exact repeat should be duplicate"
    print("   [PASS] exact repeat caught by pass 1")

    # ----------------------------------------------------------------
    # Test 2: normalisation — different punctuation / capitalisation
    # ----------------------------------------------------------------
    print("-- Test 2: normalisation catches near-identical surface forms --")
    d = ClaimDeduper(use_embeddings=False)
    c1 = _make_claim("The unemployment rate fell to 3.5 percent.", 1)
    c2 = _make_claim("the unemployment rate fell to 3.5 percent", 2)  # no period, lower
    c3 = _make_claim("The unemployment rate fell to 3,500 percent!", 3) # comma, !
    assert d.check_and_add(c1) is False
    assert d.check_and_add(c2) is True,  "lower + no period should match"
    d2 = ClaimDeduper(use_embeddings=False)
    d2.check_and_add(c1)
    # "3,500" normalises differently from "3.5" -- should NOT match
    assert d2.check_and_add(c3) is False, "3,500 != 3.5 after normalisation"
    print("   [PASS] normalisation deduplication works")

    # ----------------------------------------------------------------
    # Test 3: different claim text -> not a duplicate
    # ----------------------------------------------------------------
    print("-- Test 3: genuinely different claims pass through --")
    d = ClaimDeduper(use_embeddings=False)
    claims = [
        _make_claim("The unemployment rate fell to 3.5 percent.", 1),
        _make_claim("GDP grew by 2.1 percent in Q3.", 2),
        _make_claim("Inflation is at its lowest since 2019.", 3),
    ]
    results = [d.check_and_add(c) for c in claims]
    assert results == [False, False, False], f"All should be new: {results}"
    print("   [PASS] three distinct claims all accepted")

    # ----------------------------------------------------------------
    # Test 4: sliding window eviction
    # ----------------------------------------------------------------
    print("-- Test 4: sliding window evicts old claims --")
    d = ClaimDeduper(use_embeddings=False, window_s=0.05)  # 50ms window
    c1 = _make_claim("The inflation rate rose to 7 percent.", 1)
    c2 = _make_claim("The inflation rate rose to 7 percent.", 2)
    assert d.check_and_add(c1) is False
    time.sleep(0.1)    # let the window expire
    assert d.check_and_add(c2) is False, "expired entry should not block new claim"
    print("   [PASS] expired window entry does not block re-admission")

    # ----------------------------------------------------------------
    # Test 5: number guard -- similar text, DIFFERENT number -> not dup
    # ----------------------------------------------------------------
    print("-- Test 5: number guard -- different numbers = different claim --")
    # We test the number guard in isolation (no embedding model needed)
    text_a = "Unemployment fell to 4 percent last month."
    text_b = "Unemployment fell to 5 percent last month."
    nums_a = _numeric_tokens(text_a)
    nums_b = _numeric_tokens(text_b)
    assert nums_a != nums_b, f"Number sets should differ: {nums_a} vs {nums_b}"
    print(f"   nums_a={nums_a}  nums_b={nums_b}")
    print("   [PASS] number guard correctly distinguishes 4% vs 5%")

    # ----------------------------------------------------------------
    # Test 6: number guard -- same number -> would be duplicate
    # ----------------------------------------------------------------
    print("-- Test 6: number guard -- same numbers -> same claim --")
    text_c = "Unemployment fell to 4 percent."
    text_d = "The unemployment fell to 4 percent."   # slightly different wording
    nums_c = _numeric_tokens(text_c)
    nums_d = _numeric_tokens(text_d)
    assert nums_c == nums_d, f"Same numbers should match: {nums_c} vs {nums_d}"
    print(f"   nums_c={nums_c}  nums_d={nums_d}")
    print("   [PASS] number guard does not block same-number near-duplicates")

    # ----------------------------------------------------------------
    # Test 7: stats tracking
    # ----------------------------------------------------------------
    print("-- Test 7: stats tracking --")
    d = ClaimDeduper(use_embeddings=False)
    for i, text in enumerate([
        "Claim A.", "Claim A.", "Claim B.", "Claim B.", "Claim C."
    ]):
        d.check_and_add(_make_claim(text, i))
    s = d.stats()
    assert s["total_seen"]       == 5, s
    assert s["pass1_duplicates"] == 2, s
    assert s["total_duplicates"] == 2, s
    assert s["window_size"]      == 3, s   # A, B, C
    print(f"   [PASS] stats: {s}")

    # ----------------------------------------------------------------
    # Test 8: reset()
    # ----------------------------------------------------------------
    print("-- Test 8: reset clears window and stats --")
    d = ClaimDeduper(use_embeddings=False)
    d.check_and_add(_make_claim("Some claim.", 1))
    d.reset()
    assert d.stats()["total_seen"] == 0
    assert d.check_and_add(_make_claim("Some claim.", 2)) is False, \
        "After reset, same claim should be accepted again"
    print("   [PASS] reset works correctly")

    print("\n[PASS] All ClaimDeduper tests passed.")


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def _run_benchmark() -> None:
    print("\n=== ClaimDeduper latency benchmark ===\n")

    sentences = [
        "The unemployment rate fell to 3.5 percent last month.",
        "GDP grew by 2.1 percent in the third quarter.",
        "Inflation is at its lowest since 2019.",
        "We exported 2.3 billion dollars of goods.",
        "The central bank raised rates by 25 basis points.",
        "Tax revenue increased by 4 percent year on year.",
        "Healthcare spending rose to 18 percent of GDP.",
        "Renewable energy now accounts for 30 percent of power.",
    ]

    # -- Pass 1 only --
    print("Pass-1 only (no embeddings):")
    d = ClaimDeduper(use_embeddings=False)
    for i, s in enumerate(sentences):
        d.check_and_add(_make_claim(s, i))
    # Re-run same sentences -> all duplicates
    t0 = time.monotonic()
    N = 1000
    for _ in range(N):
        for i, s in enumerate(sentences):
            d.check_and_add(_make_claim(s, i + 100))
    elapsed = time.monotonic() - t0
    print(f"  {N * len(sentences)} lookups in {elapsed*1000:.1f}ms")
    print(f"  mean per lookup: {elapsed*1000 / (N*len(sentences)):.4f}ms")

    # -- Pass 2 (embedding) --
    print("\nPass-2 embedding (lazy model load on first call):")
    try:
        d2 = ClaimDeduper(use_embeddings=True)
        c0 = _make_claim(sentences[0], 0)
        t0 = time.monotonic()
        d2.check_and_add(c0)    # triggers model download + load
        print(f"  First call (model load): {(time.monotonic()-t0)*1000:.0f}ms")

        t0 = time.monotonic()
        for i, s in enumerate(sentences[1:], 1):
            d2.check_and_add(_make_claim(s, i))
        elapsed = time.monotonic() - t0
        print(f"  Subsequent {len(sentences)-1} calls: "
              f"{elapsed*1000:.1f}ms total, "
              f"{elapsed*1000/(len(sentences)-1):.1f}ms each")
        print(f"  Stats: {d2.stats()}")
    except Exception as e:
        print(f"  [SKIP] sentence-transformers not available: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ClaimDeduper tests and benchmark")
    p.add_argument("--benchmark", action="store_true",
                   help="Run latency benchmark after tests")
    p.add_argument("--no-embed",  action="store_true",
                   help="Skip embedding pass (pass-1 only)")
    args = p.parse_args()

    _run_tests()
    if args.benchmark:
        _run_benchmark()
