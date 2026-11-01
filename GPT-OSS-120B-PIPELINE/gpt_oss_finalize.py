"""Shared llm_keyword_stages merge, corpus support, and decisions for GPT-OSS output."""

from __future__ import annotations

import re
from typing import Any

from nlp_keyword_pipeline.llm_keyword_stages import (
    apply_decision,
    evaluate_support_batch,
    merge_candidate_lists,
)
from nlp_keyword_pipeline.llm_keyword_stages.models import LLMCorpusTerm, LLMGeneratedTerm
from nlp_keyword_pipeline.llm_keyword_stages.ranking import score_merged_term
from nlp_keyword_pipeline.models import CleanDocument


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def apply_support_pipeline(
    aggregated: list[dict[str, Any]],
    gap_terms: list[dict[str, Any]],
    documents: list[CleanDocument],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """
    Merge llm_corpus (aggregated) + llm_generated (gap), evaluate support, apply decisions.

    Returns
    -------
    filtered_aggregated, filtered_gap, stats
    """
    corpus_llm: list[LLMCorpusTerm] = []
    for a in aggregated:
        term = str(a.get("term", "")).strip()
        if not term:
            continue
        corpus_llm.append(
            LLMCorpusTerm(
                term=term,
                normalized=_norm(term),
                likely_category=str(a.get("category") or a.get("likely_category") or "other"),
                evidence_snippet="",
                corpus_support_estimate=float(a.get("avg_salience") or a.get("salience") or 0.5),
                reason_for_inclusion="Aggregated per-page GPT-OSS extraction",
            )
        )

    gen: list[LLMGeneratedTerm] = []
    for g in gap_terms:
        term = str(g.get("term", "")).strip()
        if not term:
            continue
        gen.append(
            LLMGeneratedTerm(
                term=term,
                normalized=_norm(term),
                likely_category=str(g.get("likely_category") or g.get("category") or "other"),
                why_supported=str(g.get("why_supported") or ""),
                supporting_signals=list(g.get("supporting_signals") or []),
                salience=float(g.get("salience") or 0.5),
            )
        )

    merged = merge_candidate_lists(corpus_llm, gen)
    pairs = evaluate_support_batch(merged, documents)
    triple_map: dict[str, tuple[Any, Any, Any]] = {}
    stats = {"reject": 0, "demote": 0, "keep": 0}
    for cand, ev in pairs:
        dec = apply_decision(cand, ev)
        triple_map[cand.normalized] = (cand, ev, dec)
        if dec.decision == "reject":
            stats["reject"] += 1
        elif dec.decision == "demote":
            stats["demote"] += 1
        else:
            stats["keep"] += 1

    def enrich_row(
        base: dict[str, Any],
        norm: str,
    ) -> dict[str, Any] | None:
        trip = triple_map.get(norm)
        if not trip:
            return None
        cand, ev, dec = trip
        if dec.decision == "reject":
            return None
        row = dict(base)
        row["score"] = round(score_merged_term(cand, ev, dec), 4)
        row["support_label"] = ev.support_label
        row["corpus_support_eval_score"] = round(ev.corpus_support_score, 4)
        row["decision"] = dec.decision
        row["decision_rationale"] = dec.rationale
        row["supporting_snippets"] = ev.supporting_snippets[:3]
        return row

    out_agg: list[dict[str, Any]] = []
    for a in aggregated:
        term = str(a.get("term", "")).strip()
        if not term:
            continue
        enriched = enrich_row(a, _norm(term))
        if enriched is not None:
            out_agg.append(enriched)

    out_gap: list[dict[str, Any]] = []
    for g in gap_terms:
        term = str(g.get("term", "")).strip()
        if not term:
            continue
        enriched = enrich_row(g, _norm(term))
        if enriched is not None:
            out_gap.append(enriched)

    return out_agg, out_gap, stats
