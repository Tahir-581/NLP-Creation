"""Data models for the keyword generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz


@dataclass
class PageDocument:
    """Validated page-level document from JSON corpus."""

    doc_id: str
    file_path: str
    url: str
    domain: str
    title: str = ""
    description: str = ""
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    content: str = ""
    word_count: int | None = None
    heading_count: int | None = None
    para_count: int | None = None
    authority: int | None = None
    # Source modeling / diagnostics (upgrade)
    source_relevance_score: float = 0.0
    source_relevance_reasons: list[str] = field(default_factory=list)
    boilerplate_ratio: float = 0.0
    has_error_signals: bool = False
    usable_text_fields: list[str] = field(default_factory=list)
    paragraph_count: int = 0


@dataclass
class CleanDocument:
    """Normalized document with structural text zones preserved."""

    doc_id: str
    url: str
    domain: str
    title_text: str
    description_text: str
    heading_texts: list[str]
    paragraph_texts: list[str]
    intro_text: str
    body_text: str
    full_clean_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Structured zones (upgrade)
    zone_weights: dict[str, float] = field(default_factory=dict)
    sentences: list[str] = field(default_factory=list)
    heading_hierarchy: list[tuple[int, str]] = field(default_factory=list)
    faq_text: str = ""
    meta_text: str = ""


@dataclass
class Candidate:
    """A consolidated keyword candidate with provenance and scores."""

    surface_form: str
    normalized_form: str
    source_methods: set[str] = field(default_factory=set)
    source_docs: set[str] = field(default_factory=set)
    docs_in_title: set[str] = field(default_factory=set)
    docs_in_heading: set[str] = field(default_factory=set)
    docs_in_body: set[str] = field(default_factory=set)
    docs_in_intro: set[str] = field(default_factory=set)
    title_hits: int = 0
    heading_hits: int = 0
    body_hits: int = 0
    intro_hits: int = 0
    doc_prevalence: float = 0.0
    semantic_score: float | None = None
    score: float = 0.0
    candidate_type: str | None = None
    importance: str = "low"
    example_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    subscores: dict[str, float] = field(default_factory=dict)
    is_entity: bool = False
    from_lexicon: bool = False
    is_question: bool = False
    # Phase 2/3: explainability and semantic grouping
    audit_trail: list[str] = field(default_factory=list)
    cluster_id: int | None = None
    cluster_representative: str | None = None
    cluster_label: str | None = None
    # Canonicalization / explainability (upgrade)
    canonical_form: str = ""
    display_form: str = ""
    source_relevance_weighted_prevalence: float = 0.0
    extractor_confidence: dict[str, float] = field(default_factory=dict)
    hygiene_result: str = "pass"
    hygiene_reasons: list[str] = field(default_factory=list)
    inclusion_reason: str = ""
    penalties: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def merge_from(self, other: Candidate) -> None:
        """Merge another candidate (same normalized key) into this one."""
        def _fit(sf: str) -> float:
            if not sf:
                return 0.0
            return float(fuzz.ratio(sf.lower(), self.normalized_form))

        if other.surface_form:
            if not self.surface_form:
                self.surface_form = other.surface_form
            elif _fit(other.surface_form) > _fit(self.surface_form):
                self.surface_form = other.surface_form
            elif _fit(other.surface_form) == _fit(self.surface_form) and len(
                other.surface_form
            ) > len(self.surface_form):
                self.surface_form = other.surface_form
        self.source_methods |= other.source_methods
        self.source_docs |= other.source_docs
        self.docs_in_title |= other.docs_in_title
        self.docs_in_heading |= other.docs_in_heading
        self.docs_in_body |= other.docs_in_body
        self.docs_in_intro |= other.docs_in_intro
        self.title_hits += other.title_hits
        self.heading_hits += other.heading_hits
        self.body_hits += other.body_hits
        self.intro_hits += other.intro_hits
        self.is_entity = self.is_entity or other.is_entity
        self.from_lexicon = self.from_lexicon or other.from_lexicon
        self.is_question = self.is_question or other.is_question
        self.notes.extend(n for n in other.notes if n not in self.notes)
        if other.canonical_form and not self.canonical_form:
            self.canonical_form = other.canonical_form
        if other.display_form and (
            not self.display_form or len(other.display_form) > len(self.display_form)
        ):
            self.display_form = other.display_form
        for k, v in other.extractor_confidence.items():
            self.extractor_confidence[k] = max(self.extractor_confidence.get(k, 0.0), v)
        if other.hygiene_result == "hard_reject" or (
            self.hygiene_result != "hard_reject" and other.hygiene_result == "soft_penalty"
        ):
            self.hygiene_result = other.hygiene_result
        for r in other.hygiene_reasons:
            if r not in self.hygiene_reasons:
                self.hygiene_reasons.append(r)
        for k, v in other.penalties.items():
            self.penalties[k] = max(self.penalties.get(k, 0.0), v)
