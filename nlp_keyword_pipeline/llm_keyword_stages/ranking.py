"""Score merged candidates + support evaluations + decisions for final ranking."""

from __future__ import annotations

import re

from nlp_keyword_pipeline.llm_keyword_stages.models import (
    MergedCandidate,
    SupportEvaluation,
    TermDecision,
)

DEMOTE_FACTOR = 0.55

# Priority: strong corpus support > title > heading > specificity > LLM aux signals
_W_CORPUS = 0.48
_W_TITLE = 0.18
_W_HEADING = 0.12
_W_SPEC = 0.12
_W_LLM_CORPUS_EST = 0.10
_W_LLM_GEN_SALIENCE = 0.08
_GENERATED_SCORE_DAMP = 0.98  # keeps weakly supported generated terms from edging corpus ties


def specificity_score(normalized: str) -> float:
    """Heuristic in [0, 1]: prefer 2–5 word phrases; downweight bare single tokens."""
    s = re.sub(r"\s+", " ", (normalized or "").strip().lower())
    parts = [p for p in s.split() if p]
    n = len(parts)
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.35
    if 2 <= n <= 5:
        return min(0.95, 0.78 + 0.04 * (n - 2))
    return max(0.55, 0.92 - 0.04 * (n - 5))


def source_tie_priority(source: str) -> int:
    """Higher sorts first when final_score ties (corpus > llm_corpus > llm_generated)."""
    if source == "corpus":
        return 3
    if source == "llm_corpus":
        return 2
    return 1


def build_ranking_rationale(
    candidate: MergedCandidate,
    evaluation: SupportEvaluation,
    *,
    specificity: float,
) -> str:
    parts: list[str] = []
    label = evaluation.support_label
    if label == "strong":
        parts.append(
            f"strong corpus support ({evaluation.exact_match_count} doc(s), score {evaluation.corpus_support_score:.2f})"
        )
    elif label == "weak":
        parts.append(f"weak corpus support (score {evaluation.corpus_support_score:.2f})")
    else:
        parts.append("limited corpus support")
    if evaluation.in_title:
        parts.append("title alignment")
    if evaluation.in_heading:
        parts.append("heading alignment")
    parts.append(f"specificity {specificity:.2f}")
    if candidate.source == "llm_corpus":
        parts.append("corpus-derived (llm_corpus)")
    elif candidate.source == "llm_generated":
        parts.append("generated expansion")
    else:
        parts.append("classical corpus")
    return "; ".join(parts)


def compute_merged_rank(
    candidate: MergedCandidate,
    evaluation: SupportEvaluation,
    decision: TermDecision,
) -> tuple[float, str, float]:
    """Returns (final_score, ranking_rationale, specificity)."""
    if decision.decision == "reject":
        return 0.0, "rejected", specificity_score(candidate.normalized)

    spec = specificity_score(candidate.normalized)
    t = 1.0 if evaluation.in_title else 0.0
    h = 1.0 if evaluation.in_heading else 0.0
    cs = evaluation.corpus_support_score

    if candidate.source == "llm_corpus":
        est = candidate.corpus_support_estimate or 0.5
        raw = (
            _W_CORPUS * cs
            + _W_TITLE * t
            + _W_HEADING * h
            + _W_SPEC * spec
            + _W_LLM_CORPUS_EST * est
        )
    elif candidate.source == "corpus":
        zone = 1.0 if (evaluation.in_title or evaluation.in_heading) else 0.25
        raw = 0.65 * cs + 0.35 * zone
        raw = raw * 0.98 + 0.02 * spec  # slight specificity nudge
    else:
        raw = (
            _W_CORPUS * cs
            + _W_TITLE * t
            + _W_HEADING * h
            + _W_SPEC * spec
            + _W_LLM_GEN_SALIENCE * candidate.salience
        )
        raw *= _GENERATED_SCORE_DAMP

    if decision.decision == "demote":
        raw *= DEMOTE_FACTOR

    final = min(1.0, max(0.0, raw))
    rationale = build_ranking_rationale(candidate, evaluation, specificity=spec)
    return final, rationale, spec


def score_merged_term(
    candidate: MergedCandidate,
    evaluation: SupportEvaluation,
    decision: TermDecision,
) -> float:
    """Backward-compatible scalar score for callers that only need a float."""
    final, _rationale, _spec = compute_merged_rank(candidate, evaluation, decision)
    return final


def ranking_sort_key(
    *,
    final_score: float,
    source: str,
    corpus_support_score: float,
    in_title: bool,
    in_heading: bool,
    specificity: float,
    normalized: str,
) -> tuple[float, int, float, int, int, float, str]:
    """Descending sort: higher tuple compares greater (use reverse=True on sort)."""
    return (
        final_score,
        source_tie_priority(source),
        corpus_support_score,
        1 if in_title else 0,
        1 if in_heading else 0,
        specificity,
        normalized,
    )
