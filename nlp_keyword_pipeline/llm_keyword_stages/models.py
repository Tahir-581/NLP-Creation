"""Dataclasses for llm_corpus / llm_generated workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SupportLabel = Literal["none", "weak", "strong"]
Decision = Literal["keep", "demote", "reject"]
Source = Literal["llm_corpus", "llm_generated", "corpus"]


@dataclass
class LLMCorpusTerm:
    """Stage 1: corpus-grounded extraction (LLM)."""

    term: str
    normalized: str
    likely_category: str = ""
    evidence_snippet: str = ""
    corpus_support_estimate: float = 0.0
    reason_for_inclusion: str = ""
    source: Source = "llm_corpus"

    def as_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "normalized": self.normalized,
            "source": self.source,
            "likely_category": self.likely_category,
            "evidence_snippet": self.evidence_snippet,
            "corpus_support_estimate": round(self.corpus_support_estimate, 4),
            "reason_for_inclusion": self.reason_for_inclusion,
        }


@dataclass
class LLMGeneratedTerm:
    """Stage 2: expansion candidates (LLM)."""

    term: str
    normalized: str
    likely_category: str = ""
    why_supported: str = ""
    supporting_signals: list[str] = field(default_factory=list)
    source: Source = "llm_generated"
    salience: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "normalized": self.normalized,
            "source": self.source,
            "likely_category": self.likely_category,
            "why_supported": self.why_supported,
            "supporting_signals": list(self.supporting_signals),
            "salience": round(self.salience, 4),
        }


@dataclass
class MergedCandidate:
    """Normalized pool entry after merge (before support eval)."""

    term: str
    normalized: str
    source: Source
    likely_category: str = ""
    evidence_snippet: str = ""
    reason_for_inclusion: str = ""
    why_supported: str = ""
    supporting_signals: list[str] = field(default_factory=list)
    corpus_support_estimate: float = 0.0
    salience: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "term": self.term,
            "normalized": self.normalized,
            "source": self.source,
            "likely_category": self.likely_category,
        }
        if self.evidence_snippet:
            d["evidence_snippet"] = self.evidence_snippet
        if self.reason_for_inclusion:
            d["reason_for_inclusion"] = self.reason_for_inclusion
        if self.why_supported:
            d["why_supported"] = self.why_supported
        if self.supporting_signals:
            d["supporting_signals"] = self.supporting_signals
        if self.corpus_support_estimate:
            d["corpus_support_estimate"] = round(self.corpus_support_estimate, 4)
        if self.salience != 0.5:
            d["salience"] = round(self.salience, 4)
        return d


@dataclass
class SupportEvaluation:
    """Per-term corpus support (heuristic or LLM-refined)."""

    corpus_support_score: float
    support_label: SupportLabel
    exact_match_count: int
    heading_or_title_presence: bool
    supporting_snippets: list[str]
    rationale: str
    matched_doc_ids: list[str] = field(default_factory=list)
    best_fuzzy_score: int = 0
    in_title: bool = False
    in_heading: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus_support_score": round(self.corpus_support_score, 4),
            "support_label": self.support_label,
            "exact_match_count": self.exact_match_count,
            "heading_or_title_presence": self.heading_or_title_presence,
            "in_title": self.in_title,
            "in_heading": self.in_heading,
            "supporting_snippets": self.supporting_snippets[:5],
            "rationale": self.rationale,
            "matched_doc_ids": self.matched_doc_ids,
            "best_fuzzy_score": self.best_fuzzy_score,
        }


@dataclass
class TermDecision:
    decision: Decision
    rationale: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "rationale": self.rationale,
            "confidence": round(self.confidence, 4),
        }
