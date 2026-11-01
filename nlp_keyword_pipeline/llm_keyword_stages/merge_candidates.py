"""Merge llm_corpus and llm_generated lists; corpus wins on duplicate keys."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from nlp_keyword_pipeline.llm_keyword_stages.models import (
    LLMCorpusTerm,
    LLMGeneratedTerm,
    MergedCandidate,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Near-duplicate fold: if generated term fuzzes to an existing corpus key
_NEAR_DUP_RATIO = 92


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _corpus_to_merged(ct: LLMCorpusTerm) -> MergedCandidate:
    return MergedCandidate(
        term=ct.term,
        normalized=ct.normalized,
        source="llm_corpus",
        likely_category=ct.likely_category,
        evidence_snippet=ct.evidence_snippet,
        reason_for_inclusion=ct.reason_for_inclusion,
        corpus_support_estimate=ct.corpus_support_estimate,
    )


def _generated_to_merged(gt: LLMGeneratedTerm) -> MergedCandidate:
    return MergedCandidate(
        term=gt.term,
        normalized=gt.normalized,
        source="llm_generated",
        likely_category=gt.likely_category,
        why_supported=gt.why_supported,
        supporting_signals=list(gt.supporting_signals),
        salience=gt.salience,
    )


def _is_near_duplicate_of_any(norm: str, corpus_norms: set[str]) -> bool:
    if norm in corpus_norms:
        return True
    for c in corpus_norms:
        if len(norm) < 3 or len(c) < 3:
            continue
        if fuzz.ratio(norm, c) >= _NEAR_DUP_RATIO:
            return True
    return False


def merge_candidate_lists(
    corpus_terms: list[LLMCorpusTerm],
    generated_terms: list[LLMGeneratedTerm],
) -> list[MergedCandidate]:
    """Normalize duplicates; llm_corpus wins on exact or near-duplicate overlap."""
    merged: dict[str, MergedCandidate] = {}
    corpus_norms: set[str] = set()

    for ct in corpus_terms:
        key = ct.normalized or _normalize(ct.term)
        if not key:
            continue
        merged[key] = _corpus_to_merged(ct)
        corpus_norms.add(key)

    for gt in generated_terms:
        key = gt.normalized or _normalize(gt.term)
        if not key:
            continue
        if key in merged:
            logger.debug("[merge] skip generated (exact key): %s", gt.term)
            continue
        if _is_near_duplicate_of_any(key, corpus_norms):
            logger.debug("[merge] skip generated (near-dup of corpus): %s", gt.term)
            continue
        merged[key] = _generated_to_merged(gt)

    return list(merged.values())


def merged_from_mixed_dicts(
    rows: list[dict],
    *,
    source_key: str = "source",
) -> list[MergedCandidate]:
    """Merge list of dicts with source llm_corpus | llm_generated (e.g. GPT-OSS export)."""
    corpus: list[LLMCorpusTerm] = []
    generated: list[LLMGeneratedTerm] = []
    for r in rows:
        src = (r.get(source_key) or "").strip().lower()
        term = (r.get("term") or "").strip()
        if not term:
            continue
        norm = _normalize(term)
        if src == "llm_corpus":
            corpus.append(
                LLMCorpusTerm(
                    term=term,
                    normalized=norm,
                    likely_category=str(r.get("likely_category") or r.get("category") or ""),
                    evidence_snippet=str(r.get("evidence_snippet") or ""),
                    corpus_support_estimate=float(r.get("corpus_support_estimate") or 0.0),
                    reason_for_inclusion=str(r.get("reason_for_inclusion") or ""),
                )
            )
        elif src == "llm_generated":
            generated.append(
                LLMGeneratedTerm(
                    term=term,
                    normalized=norm,
                    likely_category=str(r.get("likely_category") or ""),
                    why_supported=str(r.get("why_supported") or ""),
                    supporting_signals=list(r.get("supporting_signals") or []),
                    salience=float(r.get("salience") or r.get("score") or 0.5),
                )
            )
    return merge_candidate_lists(corpus, generated)
