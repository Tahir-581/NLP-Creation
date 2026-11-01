"""Structured types for rich page-signal extraction and category inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepeatedConcept:
    """A repeated noun phrase with normalization and evidence."""

    term: str
    normalized: str
    approximate_frequency: int
    evidence_snippet: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "normalized": self.normalized,
            "approximate_frequency": self.approximate_frequency,
            "evidence_snippet": self.evidence_snippet,
        }


@dataclass
class ExtractedEntity:
    """A named entity with mapped type and heuristic confidence."""

    entity_text: str
    entity_type: str
    confidence: float
    evidence_snippet: str | None = None
    spacy_label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type,
            "confidence": round(self.confidence, 4),
            "evidence_snippet": self.evidence_snippet,
            "spacy_label": self.spacy_label,
        }


@dataclass
class NumericStat:
    """A numeric expression with classification and context."""

    value: str
    normalized_interpretation: str
    stat_type: str
    related_phrase: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "normalized_interpretation": self.normalized_interpretation,
            "type": self.stat_type,
            "related_phrase": self.related_phrase,
        }


@dataclass
class TopicSignalSummary:
    """Consolidated anchors for downstream keyword generation."""

    title_anchors: list[str] = field(default_factory=list)
    heading_anchors: list[str] = field(default_factory=list)
    repeated_concept_anchors: list[str] = field(default_factory=list)
    entity_anchors: list[str] = field(default_factory=list)
    numeric_stat_anchors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title_anchors": self.title_anchors,
            "heading_anchors": self.heading_anchors,
            "repeated_concept_anchors": self.repeated_concept_anchors,
            "entity_anchors": self.entity_anchors,
            "numeric_stat_anchors": self.numeric_stat_anchors,
        }

    def as_compact_text(self) -> str:
        parts: list[str] = []
        if self.title_anchors:
            parts.append("Title anchors: " + " | ".join(self.title_anchors))
        if self.heading_anchors:
            parts.append("Heading anchors: " + " | ".join(self.heading_anchors[:40]))
        if self.repeated_concept_anchors:
            parts.append("Concept anchors: " + " | ".join(self.repeated_concept_anchors[:30]))
        if self.entity_anchors:
            parts.append("Entity anchors: " + " | ".join(self.entity_anchors[:25]))
        if self.numeric_stat_anchors:
            parts.append("Numeric anchors: " + " | ".join(self.numeric_stat_anchors[:20]))
        return "\n".join(parts)


@dataclass
class InferredCategory:
    """One inferred topic dimension with confidence and rationale."""

    name: str
    confidence: float
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
        }


@dataclass
class RichPageSignals:
    """Full structured extraction before consolidation."""

    title: str
    headings: list[str]
    repeated_concepts: list[RepeatedConcept]
    extracted_entities: list[ExtractedEntity]
    numeric_stats: list[NumericStat]

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "headings": self.headings,
            "repeated_concepts": [c.as_dict() for c in self.repeated_concepts],
            "extracted_entities": [e.as_dict() for e in self.extracted_entities],
            "numeric_stats": [s.as_dict() for s in self.numeric_stats],
        }
