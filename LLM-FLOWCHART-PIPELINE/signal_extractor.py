"""Signal extraction: H1/H2/H3 headings, repeated nouns, named entities, stats.

This is Step 1 of the LLM Flowchart Pipeline.  It operates on a list of
CleanDocument objects (produced by nlp_keyword_pipeline.normalizer) and
returns a SignalBundle that summarises the most salient signals in the corpus
before any LLM call is made.

Rich extraction (RepeatedConcept, ExtractedEntity, NumericStat) is stored on
``SignalBundle.rich`` when available. Legacy flat fields mirror the rich data
for backward compatibility.

Rules (from pipeline brief):
  - Do NOT rely on predefined categories.
  - Infer topic dimensions from the title and source signals.
  - Prefer terms explicitly present in the corpus.
  - Any LLM-generated term must later be checked for corpus support.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import spacy

from signal_models import (
    ExtractedEntity,
    NumericStat,
    RepeatedConcept,
    RichPageSignals,
)

if TYPE_CHECKING:
    from nlp_keyword_pipeline.models import CleanDocument

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Number + unit / percent / bare percent
_STAT_PATTERN = re.compile(
    r"""
    (?:
        \b\d+(?:\.\d+)?\s*[\-–]\s*\d+(?:\.\d+)?\s*
        (?:
            years?|months?|weeks?|days?|%|percent|lbs?|kg|inches?|cm|feet|ft
        )?\b
    )
    |
    (?:
        \b\d+(?:\.\d+)?          # leading number
        [\s\-–]*                 # optional separator
        (?:
            years?|months?|weeks?|days?|   # time
            pounds?|lbs?|kg|kilograms?|    # weight
            inches?|cm|feet|ft|            # size
            hours?|minutes?|               # duration
            %|percent|                     # percentage
            calories?|kcal|                # energy
            cups?|ml|liters?|              # volume
            mph|km\/h|                     # speed
            x|times?                       # multiples
        )
        \b
    )
    |
    (?:
        \b\d+(?:\.\d+)?\s*%
    )
    |
    (?:
        \$\s*\d+(?:,\d{3})*(?:\.\d{2})?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")

_ENTITY_TYPES = frozenset(
    {
        "PERSON",
        "ORG",
        "PRODUCT",
        "GPE",
        "NORP",
        "FAC",
        "WORK_OF_ART",
        "EVENT",
        "LANGUAGE",
        "LAW",
    }
)

_ENTITY_BLOCKLIST = frozenset(
    {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "today", "yesterday", "tomorrow",
        "google", "facebook", "twitter", "instagram", "youtube",
        "amazon", "walmart",
    }
)

# Generic / weak noun phrases to drop
_FILLER_LEMMAS = frozenset(
    {
        "thing", "way", "time", "year", "day", "people", "person", "lot",
        "part", "number", "kind", "sort", "fact", "information", "article",
        "page", "site", "website", "post", "blog", "video", "photo", "image",
        "example", "case", "point", "idea", "reason", "place", "home", "world",
    }
)

_SPACY_LABEL_TO_TYPE: dict[str, str] = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "PRODUCT": "product",
    "NORP": "group",
    "FAC": "facility",
    "WORK_OF_ART": "work_of_art",
    "EVENT": "event",
    "LANGUAGE": "language",
    "LAW": "law",
}

_LABEL_CONFIDENCE_BASE: dict[str, float] = {
    "PERSON": 0.72,
    "ORG": 0.70,
    "PRODUCT": 0.74,
    "GPE": 0.68,
    "NORP": 0.65,
    "FAC": 0.62,
    "WORK_OF_ART": 0.60,
    "EVENT": 0.63,
    "LANGUAGE": 0.68,
    "LAW": 0.66,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SignalBundle:
    """All corpus signals extracted for a given (title, docs) pair."""

    title: str
    headings: list[str] = field(default_factory=list)
    """All unique heading texts (H1/H2/H3) across the corpus."""

    repeated_nouns: list[tuple[str, int]] = field(default_factory=list)
    """(noun_phrase, count) sorted by count desc; top-N cross-document nouns."""

    named_entities: list[str] = field(default_factory=list)
    """Unique named entity strings (filtered by type)."""

    stats: list[str] = field(default_factory=list)
    """Legacy: context snippets for numeric matches (backward compatible)."""

    rich: RichPageSignals | None = None
    """Structured extraction; when set, drives consolidated anchors."""

    def as_dict(self) -> dict:
        d = {
            "title": self.title,
            "headings": self.headings,
            "repeated_nouns": [{"noun": n, "count": c} for n, c in self.repeated_nouns],
            "named_entities": self.named_entities,
            "stats": self.stats,
        }
        if self.rich is not None:
            d["rich"] = self.rich.as_dict()
        return d

    def as_compact_text(self) -> str:
        """Human-readable summary for LLM prompt injection."""
        parts: list[str] = [f"Title: {self.title}"]
        if self.headings:
            parts.append("Headings: " + " | ".join(self.headings[:30]))
        if self.repeated_nouns:
            noun_str = ", ".join(n for n, _ in self.repeated_nouns[:20])
            parts.append(f"Repeated nouns: {noun_str}")
        if self.named_entities:
            parts.append("Named entities: " + ", ".join(self.named_entities[:20]))
        if self.stats:
            parts.append("Stats/numbers: " + " | ".join(self.stats[:15]))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import warnings

            warnings.warn(
                "en_core_web_sm not found; NER and noun-chunk extraction will be skipped. "
                "Run: python -m spacy download en_core_web_sm",
                stacklevel=2,
            )
            _nlp = spacy.blank("en")
    return _nlp


def _corpus_haystack(docs: list[CleanDocument]) -> str:
    parts: list[str] = []
    for doc in docs:
        parts.append(doc.full_clean_text)
    return "\n".join(parts)


def _snippet_at(haystack: str, start: int, end: int, pad: int = 90) -> str:
    a = max(0, start - pad)
    b = min(len(haystack), end + pad)
    return " ".join(haystack[a:b].split())


def _find_evidence(haystack: str, query: str) -> str | None:
    if not query or not haystack:
        return None
    low = haystack.lower()
    q = query.strip().lower()
    idx = low.find(q)
    if idx == -1:
        return None
    return _snippet_at(haystack, idx, idx + len(query))


def _noun_chunk_key(chunk) -> str | None:
    """Lemma-based key for grouping singular/plural variants."""
    lemmas: list[str] = []
    for t in chunk:
        if t.pos_ in {"DET", "PUNCT"}:
            continue
        if t.is_stop and len(chunk) > 1:
            continue
        lemmas.append(t.lemma_.lower())
    key = " ".join(lemmas).strip()
    if len(key) < 3:
        return None
    first = key.split()[0] if key.split() else ""
    if first in _FILLER_LEMMAS and len(key.split()) == 1:
        return None
    return key


def _normalized_phrase_from_chunk(chunk) -> str:
    parts: list[str] = []
    for t in chunk:
        if t.pos_ == "PUNCT":
            continue
        if t.is_stop and len(chunk) > 1:
            continue
        parts.append(t.lemma_.lower())
    return " ".join(parts).strip()


def _is_filler_token(token) -> bool:
    if token.pos_ not in {"NOUN", "PROPN"}:
        return True
    if token.lemma_.lower() in _FILLER_LEMMAS and len(token.text) < 5:
        return True
    return False


def _entity_confidence(label: str, text: str) -> float:
    base = _LABEL_CONFIDENCE_BASE.get(label, 0.62)
    extra = min(len(text) / 55.0, 0.22)
    return max(0.35, min(0.95, base + extra))


def _classify_stat_value(raw: str, context_lower: str) -> tuple[str, str]:
    """Return (stat_type, normalized_interpretation)."""
    r = raw.strip()
    rl = r.lower()

    if r.startswith("$") or "$" in r:
        return "price", re.sub(r"\s+", "", r)

    if "%" in r or "percent" in rl:
        num = re.search(r"[\d.]+", r)
        return "percent", f"{num.group(0)}%" if num else r

    if re.search(r"\d+\s*[\-–]\s*\d+", r) and any(
        u in rl for u in ("year", "month", "week", "day", "lb", "kg", "%", "inch", "cm")
    ):
        return "range", r

    if any(u in rl for u in ("year", "years", "month", "months", "week", "weeks", "day", "days")):
        return "duration", r

    if any(u in rl for u in ("hour", "hours", "minute", "minutes", "mph", "km/h")):
        return "duration", r

    if any(u in rl for u in ("inch", "inches", "cm", "ft", "feet", "lb", "kg", "pound")):
        return "measurement", r

    if "calorie" in rl or "kcal" in rl:
        return "measurement", r

    if re.fullmatch(r"(?:19|20)\d{2}", r.strip()):
        return "date", r

    if "time" in context_lower and any(x in context_lower for x in ("×", "x", " per ")):
        return "frequency", r

    if re.search(r"^\d+$", r.strip()) and any(
        x in context_lower for x in (" dogs", " breeds", " puppies", " cases", " types")
    ):
        return "count", r

    return "other", r


def _extract_headings(docs: list[CleanDocument], max_level: int = 3) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for doc in docs:
        for level, text in doc.heading_hierarchy:
            if level <= max_level:
                key = text.strip().lower()
                if key and key not in seen and len(text.strip()) > 2:
                    seen.add(key)
                    result.append(text.strip())
    return result


def _extract_repeated_concepts(
    docs: list[CleanDocument],
    nlp: spacy.language.Language,
    haystack: str,
    top_n: int = 40,
    min_count: int = 2,
) -> tuple[list[tuple[str, int]], list[RepeatedConcept]]:
    """Count lemma-grouped noun phrases; build RepeatedConcept rows."""
    key_counts: Counter[str] = Counter()
    key_surface: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for doc in docs:
        combined = " ".join(
            [doc.title_text, " ".join(doc.heading_texts), doc.body_text]
        )
        spacy_doc = nlp(combined[:12000])

        if hasattr(spacy_doc, "noun_chunks"):
            try:
                for chunk in spacy_doc.noun_chunks:
                    if all(t.is_stop for t in chunk):
                        continue
                    key = _noun_chunk_key(chunk)
                    if not key:
                        continue
                    surface = chunk.text.strip().lower()
                    if len(surface) < 3:
                        continue
                    key_counts[key] += 1
                    key_surface[key][surface] += 1
            except Exception:
                pass

        for token in spacy_doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                if _is_filler_token(token):
                    continue
                lem = token.lemma_.lower()
                if len(lem) < 3 or lem in _FILLER_LEMMAS:
                    continue
                key_counts[lem] += 1
                surf = token.text.strip().lower()
                key_surface[lem][surf] += 1

    top_keys = [k for k, c in key_counts.most_common(top_n * 4) if c >= min_count][
        :top_n
    ]

    concepts: list[RepeatedConcept] = []
    legacy: list[tuple[str, int]] = []

    for key in top_keys:
        count = key_counts[key]
        surfaces = key_surface.get(key)
        if surfaces:
            best_surface = surfaces.most_common(1)[0][0]
        else:
            best_surface = key
        term = best_surface
        normalized = key if " " in key else key
        ev = _find_evidence(haystack, term) or _find_evidence(haystack, normalized)
        concepts.append(
            RepeatedConcept(
                term=term,
                normalized=normalized,
                approximate_frequency=count,
                evidence_snippet=ev,
            )
        )
        legacy.append((term, count))

    legacy.sort(key=lambda x: -x[1])
    return legacy, concepts


def _extract_entities_rich(
    docs: list[CleanDocument],
    nlp: spacy.language.Language,
    haystack: str,
    max_entities: int = 35,
) -> tuple[list[str], list[ExtractedEntity]]:
    if "ner" not in nlp.pipe_names:
        return [], []

    seen: set[str] = set()
    strings: list[str] = []
    rich: list[ExtractedEntity] = []

    for doc in docs:
        text = " ".join([doc.title_text, " ".join(doc.heading_texts), doc.intro_text])
        spacy_doc = nlp(text[:8000])
        for ent in spacy_doc.ents:
            if ent.label_ not in _ENTITY_TYPES:
                continue
            el = ent.text.strip().lower()
            if len(el) < 2 or el in _ENTITY_BLOCKLIST:
                continue
            if el in seen:
                continue
            seen.add(el)
            etype = _SPACY_LABEL_TO_TYPE.get(ent.label_, ent.label_.lower())
            ev = _find_evidence(haystack, ent.text.strip()) or _find_evidence(
                haystack, ent.text
            )
            rich.append(
                ExtractedEntity(
                    entity_text=ent.text.strip(),
                    entity_type=etype,
                    confidence=_entity_confidence(ent.label_, ent.text),
                    evidence_snippet=ev,
                    spacy_label=ent.label_,
                )
            )
            strings.append(ent.text.strip())
            if len(strings) >= max_entities:
                return strings, rich

    return strings, rich


def _extract_numeric_stats(
    docs: list[CleanDocument], max_stats: int = 30
) -> tuple[list[str], list[NumericStat]]:
    seen_match: set[str] = set()
    legacy_snippets: list[str] = []
    stats: list[NumericStat] = []

    for doc in docs:
        text = doc.full_clean_text
        for match in _STAT_PATTERN.finditer(text):
            raw = match.group(0).strip()
            key = raw.lower()
            if key in seen_match:
                continue
            seen_match.add(key)
            start, end = match.start(), match.end()
            phrase = _snippet_at(text, start, end, pad=100)
            ctx_low = phrase.lower()
            stype, norm = _classify_stat_value(raw, ctx_low)
            stats.append(
                NumericStat(
                    value=raw,
                    normalized_interpretation=norm,
                    stat_type=stype,
                    related_phrase=phrase,
                )
            )
            legacy_snippets.append(phrase)
            if len(stats) >= max_stats:
                return legacy_snippets, stats

        for ymatch in _YEAR_PATTERN.finditer(text):
            raw = ymatch.group(1)
            key = f"year:{raw}"
            if key in seen_match:
                continue
            seen_match.add(key)
            start, end = ymatch.start(), ymatch.end()
            phrase = _snippet_at(text, start, end, pad=80)
            stats.append(
                NumericStat(
                    value=raw,
                    normalized_interpretation=raw,
                    stat_type="date",
                    related_phrase=phrase,
                )
            )
            legacy_snippets.append(phrase)
            if len(stats) >= max_stats:
                return legacy_snippets, stats

    return legacy_snippets, stats


def _build_rich_page_signals(
    title: str,
    headings: list[str],
    concepts: list[RepeatedConcept],
    entities: list[ExtractedEntity],
    numeric: list[NumericStat],
) -> RichPageSignals:
    return RichPageSignals(
        title=title,
        headings=headings,
        repeated_concepts=concepts,
        extracted_entities=entities,
        numeric_stats=numeric,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_signals(
    title: str,
    docs: list[CleanDocument],
    *,
    max_headings: int = 40,
    top_n_nouns: int = 30,
    min_noun_count: int = 2,
    max_entities: int = 30,
    max_stats: int = 25,
) -> SignalBundle:
    """Extract all corpus signals for a given title and document set."""
    if not docs:
        return SignalBundle(title=title)

    nlp = _get_nlp()
    haystack = _corpus_haystack(docs)

    headings = _extract_headings(docs)[:max_headings]
    repeated_nouns, concepts = _extract_repeated_concepts(
        docs,
        nlp,
        haystack,
        top_n=top_n_nouns,
        min_count=min_noun_count,
    )

    named_entities, entities_rich = _extract_entities_rich(
        docs, nlp, haystack, max_entities=max_entities
    )
    stats_legacy, numeric_stats = _extract_numeric_stats(docs, max_stats=max_stats)

    rich = _build_rich_page_signals(title, headings, concepts, entities_rich, numeric_stats)

    return SignalBundle(
        title=title,
        headings=headings,
        repeated_nouns=repeated_nouns,
        named_entities=named_entities,
        stats=stats_legacy,
        rich=rich,
    )
