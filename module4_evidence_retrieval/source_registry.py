"""
source_registry.py  —  Curated trusted-source list
====================================================
Defines the set of sources Module 4 is allowed to retrieve from.

Each source has:
  - name        : human-readable label
  - slug        : short identifier used in passage_id
  - url         : canonical URL
  - tier        : 1=official/government, 2=major outlet, 3=academic, 4=other
  - credibility : float in [0,1]; used by Module 6 for conflict resolution
  - topics      : list of topic tags for query routing (bias toward relevant sources)

This file is the single source of truth for which sources are "trusted".
To add a source, append an entry here — no other file needs to change.

Extending to live retrieval
---------------------------
When you connect a real search API (Bing, Brave, News API), pass each
result through `registry.is_trusted(url)` to filter untrusted domains.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional
import re


@dataclasses.dataclass(frozen=True)
class TrustedSource:
    name: str
    slug: str
    url: str
    tier: int               # 1=best, 4=lowest
    credibility: float      # 0–1
    topics: tuple           # topic tags


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

SOURCES: List[TrustedSource] = [
    # --- Tier 1: Official / government ---
    TrustedSource("UK Office for National Statistics", "ons",
                  "https://www.ons.gov.uk", 1, 0.98,
                  ("statistics", "economy", "employment", "population")),
    TrustedSource("UK HM Treasury", "hmt",
                  "https://www.gov.uk/government/organisations/hm-treasury", 1, 0.96,
                  ("budget", "economy", "public-finance")),
    TrustedSource("US Bureau of Labor Statistics", "bls",
                  "https://www.bls.gov", 1, 0.97,
                  ("employment", "wages", "inflation")),
    TrustedSource("US Congressional Budget Office", "cbo",
                  "https://www.cbo.gov", 1, 0.96,
                  ("budget", "economy", "public-finance")),
    TrustedSource("World Bank Open Data", "worldbank",
                  "https://data.worldbank.org", 1, 0.95,
                  ("gdp", "development", "trade", "economy")),
    TrustedSource("OECD Data", "oecd",
                  "https://data.oecd.org", 1, 0.95,
                  ("economy", "employment", "education", "health")),
    TrustedSource("WHO Global Health Observatory", "who",
                  "https://www.who.int/data/gho", 1, 0.95,
                  ("health", "mortality", "disease")),
    TrustedSource("NHS England Statistics", "nhs",
                  "https://www.england.nhs.uk/statistics", 1, 0.94,
                  ("health", "nhs", "waiting-times")),

    # --- Tier 2: Major fact-checkers & wire services ---
    TrustedSource("Full Fact (UK)", "fullfact",
                  "https://fullfact.org", 2, 0.92,
                  ("politics", "economy", "health", "crime")),
    TrustedSource("PolitiFact", "politifact",
                  "https://www.politifact.com", 2, 0.90,
                  ("politics", "economy", "health")),
    TrustedSource("FactCheck.org", "factcheckorg",
                  "https://www.factcheck.org", 2, 0.90,
                  ("politics", "economy", "science")),
    TrustedSource("Reuters", "reuters",
                  "https://www.reuters.com", 2, 0.88,
                  ("economy", "politics", "trade", "crime")),
    TrustedSource("AP News", "apnews",
                  "https://apnews.com", 2, 0.88,
                  ("politics", "economy", "health", "crime")),
    TrustedSource("BBC News", "bbc",
                  "https://www.bbc.co.uk/news", 2, 0.86,
                  ("politics", "economy", "health", "crime")),

    # --- Tier 3: Academic / research ---
    TrustedSource("Our World in Data", "owid",
                  "https://ourworldindata.org", 3, 0.85,
                  ("health", "environment", "economy", "education")),
    TrustedSource("Institute for Fiscal Studies", "ifs",
                  "https://ifs.org.uk", 3, 0.85,
                  ("economy", "public-finance", "welfare")),
]

_BY_SLUG: Dict[str, TrustedSource] = {s.slug: s for s in SOURCES}
_TRUSTED_DOMAINS: set[str] = set()
for _s in SOURCES:
    # Extract domain from URL for quick filtering
    _m = re.match(r"https?://(?:www\.)?([^/]+)", _s.url)
    if _m:
        _TRUSTED_DOMAINS.add(_m.group(1))


def get_by_slug(slug: str) -> Optional[TrustedSource]:
    return _BY_SLUG.get(slug)


def is_trusted(url: str) -> bool:
    """Return True if the URL's domain is in the trusted-source list."""
    m = re.match(r"https?://(?:www\.)?([^/]+)", url)
    if not m:
        return False
    domain = m.group(1)
    return any(domain == d or domain.endswith("." + d) for d in _TRUSTED_DOMAINS)


def sources_for_topic(topic: str) -> List[TrustedSource]:
    """Return sources whose topic list contains the given tag."""
    return [s for s in SOURCES if topic in s.topics]


def credibility_for_url(url: str) -> float:
    """Return credibility score for a URL, or 0.5 if domain not found."""
    m = re.match(r"https?://(?:www\.)?([^/]+)", url)
    if not m:
        return 0.5
    domain = m.group(1)
    for s in SOURCES:
        src_domain = re.sub(r"https?://(?:www\.)?", "", s.url).split("/")[0]
        if domain == src_domain or domain.endswith("." + src_domain):
            return s.credibility
    return 0.5

