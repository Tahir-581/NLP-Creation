"""Final ranking of corpus and LLM-validated terms.

Scoring formulas (from the plan):

  Corpus terms:
      score = 0.65 * prevalence_rate + 0.35 * zone_bonus

  LLM terms (strong / kept):
      score = 0.50 * salience + 0.40 * support_score + 0.10 * zone_bonus

  LLM terms (weak / demoted):
      score = (0.50 * salience + 0.40 * support_score + 0.10 * zone_bonus)
              * DEMOTE_FACTOR

After scoring, both sets are merged, de-duplicated by normalized form, and
sorted descending.  A final cap (MAX_TERMS) is applied before return.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from corpus_extractor import CorpusTerm
    from llm_generator import LLMTerm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (override at call-site if needed)
# ---------------------------------------------------------------------------

MAX_TERMS: int = 150
"""Maximum number of terms in the final ranked list."""

DEMOTE_FACTOR: float = 0.55
"""Multiplicative penalty applied to demoted (weak-support) LLM terms."""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class RankedTerm:
    """A scored, ranked keyword candidate ready for export."""

    term: str
    normalized: str
    score: float
    source: str  # "corpus" | "llm_corpus" | "llm_generated"
    status: str  # "keep" | "demote" | "reject"  (reject terms are never ranked)

    # Corpus-side metadata
    doc_count: int = 0
    total_docs: int = 0
    prevalence_rate: float = 0.0
    zone_bonus: float = 0.0
    title_hits: int = 0
    heading_hits: int = 0
    body_hits: int = 0
    intro_hits: int = 0
    is_entity: bool = False
    from_lexicon: bool = False
    is_question: bool = False
    extractors: list[str] = field(default_factory=list)

    # LLM-side metadata
    dimension: str = ""
    salience: float = 0.0
    support_score: float = 0.0
    matched_docs: list[str] = field(default_factory=list)

    # Shared llm_keyword_stages pipeline (llm_corpus / llm_generated merged path)
    support_label: str = ""
    corpus_support_eval_score: float = 0.0
    supporting_snippets: list[str] = field(default_factory=list)
    decision_rationale: str = ""
    decision_confidence: float = 0.0
    evidence_snippet: str = ""
    likely_category: str = ""
    why_supported: str = ""
    supporting_signals: list[str] = field(default_factory=list)
    exact_match_count: int = 0
    heading_or_title_presence: bool = False
    rank: int = 0
    final_score: float = 0.0
    ranking_rationale: str = ""
    title_alignment: bool = False
    heading_alignment: bool = False
    specificity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        fs = round(self.score, 4)
        d: dict[str, Any] = {
            "term": self.term,
            "normalized": self.normalized,
            "rank": self.rank,
            "final_score": fs,
            "score": fs,
            "ranking_rationale": self.ranking_rationale,
            "source": self.source,
            "status": self.status,
            "title_alignment": self.title_alignment,
            "heading_alignment": self.heading_alignment,
            "specificity": round(self.specificity, 4),
        }
        if self.source == "corpus":
            d.update(
                {
                    "doc_count": self.doc_count,
                    "total_docs": self.total_docs,
                    "prevalence_rate": round(self.prevalence_rate, 4),
                    "zone_bonus": round(self.zone_bonus, 4),
                    "title_hits": self.title_hits,
                    "heading_hits": self.heading_hits,
                    "body_hits": self.body_hits,
                    "intro_hits": self.intro_hits,
                    "is_entity": self.is_entity,
                    "from_lexicon": self.from_lexicon,
                    "is_question": self.is_question,
                    "extractors": self.extractors,
                }
            )
        elif self.source == "llm_corpus":
            d.update(
                {
                    "likely_category": self.likely_category,
                    "salience": round(self.salience, 4),
                    "support_score": round(self.support_score, 4),
                    "corpus_support_eval_score": round(self.corpus_support_eval_score, 4),
                    "support_label": self.support_label,
                    "exact_match_count": self.exact_match_count,
                    "heading_or_title_presence": self.heading_or_title_presence,
                    "evidence_snippet": self.evidence_snippet,
                    "supporting_snippets": self.supporting_snippets[:5],
                    "decision_rationale": self.decision_rationale,
                    "decision_confidence": round(self.decision_confidence, 4),
                    "matched_docs": self.matched_docs,
                }
            )
        else:
            d.update(
                {
                    "dimension": self.dimension,
                    "likely_category": self.likely_category or self.dimension,
                    "salience": round(self.salience, 4),
                    "support_score": round(self.support_score, 4),
                    "corpus_support_eval_score": round(self.corpus_support_eval_score, 4),
                    "support_label": self.support_label,
                    "why_supported": self.why_supported,
                    "supporting_signals": self.supporting_signals,
                    "supporting_snippets": self.supporting_snippets[:5],
                    "decision_rationale": self.decision_rationale,
                    "decision_confidence": round(self.decision_confidence, 4),
                    "matched_docs": self.matched_docs,
                }
            )
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _score_corpus_term(ct: CorpusTerm) -> float:
    return 0.65 * ct.prevalence_rate + 0.35 * ct.zone_bonus


def _score_llm_term(lt: LLMTerm) -> float:
    # zone_bonus approximation for LLM terms: heading in matched_docs gives bonus
    zone_bonus = 0.2 if lt.matched_docs else 0.0
    base = 0.50 * lt.salience + 0.40 * lt.support_score + 0.10 * zone_bonus
    if lt.support_status == "demote":
        return base * DEMOTE_FACTOR
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank_terms(
    corpus_terms: list[CorpusTerm],
    llm_terms: list[LLMTerm],
    *,
    max_terms: int = MAX_TERMS,
) -> list[RankedTerm]:
    """Merge, score, de-duplicate, and sort all accepted terms.

    Parameters
    ----------
    corpus_terms:
        Output of corpus_extractor.extract_corpus_terms().
    llm_terms:
        Output of support_checker.check_support() (all terms, including rejects).
    max_terms:
        Cap on the final list length.

    Returns
    -------
    list[RankedTerm]
        Sorted descending by score; length <= max_terms.
        Rejected LLM terms are excluded.
    """
    # normalized_form → RankedTerm  (corpus terms win on collision)
    merged: dict[str, RankedTerm] = {}

    # ---- Corpus terms -------------------------------------------------------
    for ct in corpus_terms:
        score = _score_corpus_term(ct)
        rt = RankedTerm(
            term=ct.term,
            normalized=ct.normalized,
            score=score,
            source="corpus",
            status="keep",
            doc_count=ct.doc_count,
            total_docs=ct.total_docs,
            prevalence_rate=ct.prevalence_rate,
            zone_bonus=ct.zone_bonus,
            title_hits=ct.title_hits,
            heading_hits=ct.heading_hits,
            body_hits=ct.body_hits,
            intro_hits=ct.intro_hits,
            is_entity=ct.is_entity,
            from_lexicon=ct.from_lexicon,
            is_question=ct.is_question,
            extractors=ct.extractors,
        )
        merged[ct.normalized] = rt

    # ---- LLM terms (skip rejects) ------------------------------------------
    for lt in llm_terms:
        if lt.support_status == "reject":
            continue
        norm = lt.normalized
        if norm in merged:
            # Corpus term already occupies this key; boost score slightly if
            # the LLM also found it as important.
            existing = merged[norm]
            existing.score = min(1.0, existing.score + 0.02 * lt.salience)
            # Annotate with LLM dimension for richer export
            if not existing.dimension:
                existing.dimension = lt.dimension
            continue

        score = _score_llm_term(lt)
        rt = RankedTerm(
            term=lt.term,
            normalized=norm,
            score=score,
            source="llm_generated",
            status=lt.support_status,
            dimension=lt.dimension,
            salience=lt.salience,
            support_score=lt.support_score,
            matched_docs=lt.matched_docs,
        )
        merged[norm] = rt

    # ---- Sort and cap -------------------------------------------------------
    from nlp_keyword_pipeline.llm_keyword_stages.ranking import specificity_score

    ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    ranked = ranked[:max_terms]
    for i, r in enumerate(ranked, start=1):
        r.rank = i
        r.final_score = r.score
        r.specificity = specificity_score(r.normalized)
        r.title_alignment = r.title_hits > 0
        r.heading_alignment = r.heading_hits > 0
        r.ranking_rationale = (
            f"classical merge; prevalence {r.prevalence_rate:.2f}; "
            f"zone_bonus {r.zone_bonus:.2f}"
        )

    logger.info(
        "[ranker] Final list: %d terms (corpus=%d, llm_keep=%d, llm_demote=%d)",
        len(ranked),
        sum(1 for r in ranked if r.source == "corpus"),
        sum(1 for r in ranked if r.source == "llm_generated" and r.status == "keep"),
        sum(1 for r in ranked if r.source == "llm_generated" and r.status == "demote"),
    )
    return ranked


def rank_from_merged_pipeline(
    evaluated: list[
        tuple[
            "MergedCandidate",
            "SupportEvaluation",
            "TermDecision",
        ]
    ],
    *,
    max_terms: int = MAX_TERMS,
) -> list[RankedTerm]:
    """Build ranked list from merge + support evaluation + decisions."""
    from nlp_keyword_pipeline.llm_keyword_stages.ranking import (
        compute_merged_rank,
        ranking_sort_key,
    )

    rows: list[RankedTerm] = []
    for cand, ev, dec in evaluated:
        if dec.decision == "reject":
            continue
        final, rationale, spec = compute_merged_rank(cand, ev, dec)
        src = cand.source
        status: str = "keep" if dec.decision == "keep" else "demote"
        rt = RankedTerm(
            term=cand.term,
            normalized=cand.normalized,
            score=final,
            final_score=final,
            source=src,
            status=status,
            support_score=ev.corpus_support_score,
            corpus_support_eval_score=ev.corpus_support_score,
            support_label=ev.support_label,
            supporting_snippets=list(ev.supporting_snippets),
            decision_rationale=dec.rationale,
            decision_confidence=dec.confidence,
            matched_docs=ev.matched_doc_ids,
            exact_match_count=ev.exact_match_count,
            heading_or_title_presence=ev.heading_or_title_presence,
            salience=cand.salience,
            likely_category=cand.likely_category,
            ranking_rationale=rationale,
            title_alignment=ev.in_title,
            heading_alignment=ev.in_heading,
            specificity=spec,
        )
        if src == "llm_corpus":
            rt.evidence_snippet = cand.evidence_snippet
            rt.salience = cand.corpus_support_estimate or rt.salience
        if src == "llm_generated":
            rt.why_supported = cand.why_supported
            rt.supporting_signals = list(cand.supporting_signals)
            rt.dimension = cand.likely_category
        rows.append(rt)

    rows.sort(
        key=lambda r: ranking_sort_key(
            final_score=r.score,
            source=r.source,
            corpus_support_score=r.corpus_support_eval_score,
            in_title=r.title_alignment,
            in_heading=r.heading_alignment,
            specificity=r.specificity,
            normalized=r.normalized,
        ),
        reverse=True,
    )
    rows = rows[:max_terms]
    for i, r in enumerate(rows, start=1):
        r.rank = i
    logger.info(
        "[ranker] merged pipeline: %d terms (llm_corpus=%d, llm_generated=%d)",
        len(rows),
        sum(1 for r in rows if r.source == "llm_corpus"),
        sum(1 for r in rows if r.source == "llm_generated"),
    )
    return rows
