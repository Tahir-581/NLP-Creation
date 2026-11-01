"""Deterministic consolidation of rich signals into topic anchor lists."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from signal_models import RichPageSignals, TopicSignalSummary


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def consolidate(
    rich: RichPageSignals,
    *,
    max_title_anchors: int = 8,
    max_heading_anchors: int = 40,
    max_concept_anchors: int = 35,
    max_entity_anchors: int = 30,
    max_numeric_anchors: int = 25,
) -> TopicSignalSummary:
    """Deduplicate and cap signals into a TopicSignalSummary for downstream use."""
    title_anchors: list[str] = []
    seen_t: set[str] = set()
    if rich.title and rich.title.strip():
        parts = [p.strip() for p in re.split(r"[|:–—]", rich.title) if p.strip()]
        if not parts:
            parts = [rich.title.strip()]
        for p in parts:
            k = _norm_key(p)
            if k and k not in seen_t and len(p) > 1:
                seen_t.add(k)
                title_anchors.append(p)
                if len(title_anchors) >= max_title_anchors:
                    break
        if not title_anchors and rich.title.strip():
            title_anchors = [rich.title.strip()]

    heading_anchors: list[str] = []
    seen_h: set[str] = set()
    for h in rich.headings:
        k = _norm_key(h)
        if k and k not in seen_h:
            seen_h.add(k)
            heading_anchors.append(h.strip())
            if len(heading_anchors) >= max_heading_anchors:
                break

    concept_anchors: list[str] = []
    seen_c: set[str] = set()
    sorted_concepts = sorted(
        rich.repeated_concepts,
        key=lambda c: (-c.approximate_frequency, len(c.normalized)),
    )
    for c in sorted_concepts:
        anchor = c.normalized.strip() or c.term.strip()
        k = _norm_key(anchor)
        if k and k not in seen_c:
            seen_c.add(k)
            concept_anchors.append(anchor)
            if len(concept_anchors) >= max_concept_anchors:
                break

    entity_anchors: list[str] = []
    seen_e: set[str] = set()
    for e in rich.extracted_entities:
        t = e.entity_text.strip()
        k = _norm_key(t)
        if k and k not in seen_e:
            seen_e.add(k)
            entity_anchors.append(t)
            if len(entity_anchors) >= max_entity_anchors:
                break

    numeric_anchors: list[str] = []
    seen_n: set[str] = set()
    for s in rich.numeric_stats:
        brief = s.normalized_interpretation.strip() or s.value.strip()
        k = _norm_key(brief)
        if k and k not in seen_n:
            seen_n.add(k)
            numeric_anchors.append(f"{brief} [{s.stat_type}]")
            if len(numeric_anchors) >= max_numeric_anchors:
                break

    return TopicSignalSummary(
        title_anchors=title_anchors,
        heading_anchors=heading_anchors,
        repeated_concept_anchors=concept_anchors,
        entity_anchors=entity_anchors,
        numeric_stat_anchors=numeric_anchors,
    )
