"""Unit tests for llm_keyword_stages merge, decisions, corpus_support."""

from __future__ import annotations

from nlp_keyword_pipeline.llm_keyword_stages.corpus_support import evaluate_support
from nlp_keyword_pipeline.llm_keyword_stages.decisions import apply_decision
from nlp_keyword_pipeline.llm_keyword_stages.merge_candidates import merge_candidate_lists
from nlp_keyword_pipeline.llm_keyword_stages.models import (
    LLMCorpusTerm,
    LLMGeneratedTerm,
    MergedCandidate,
    SupportEvaluation,
)
from nlp_keyword_pipeline.models import CleanDocument


def _doc(text: str, title: str = "t") -> CleanDocument:
    return CleanDocument(
        doc_id="1",
        url="http://x",
        domain="x",
        title_text=title,
        description_text="",
        heading_texts=[],
        paragraph_texts=[],
        intro_text="",
        body_text=text,
        full_clean_text=text,
        sentences=text.replace(". ", ".").split(".") if text else [],
        heading_hierarchy=[],
    )


def test_merge_corpus_wins_on_exact_duplicate() -> None:
    c = [
        LLMCorpusTerm(
            term="guard dog",
            normalized="guard dog",
            likely_category="topic",
            evidence_snippet="x",
            corpus_support_estimate=0.9,
            reason_for_inclusion="in heading",
        )
    ]
    g = [
        LLMGeneratedTerm(
            term="guard dog",
            normalized="guard dog",
            likely_category="x",
            why_supported="y",
        )
    ]
    m = merge_candidate_lists(c, g)
    assert len(m) == 1
    assert m[0].source == "llm_corpus"


def test_decision_generated_reject_on_none() -> None:
    cand = MergedCandidate(
        term="foo",
        normalized="foo",
        source="llm_generated",
        salience=0.8,
    )
    ev = SupportEvaluation(
        corpus_support_score=0.0,
        support_label="none",
        exact_match_count=0,
        heading_or_title_presence=False,
        supporting_snippets=[],
        rationale="none",
    )
    d = apply_decision(cand, ev)
    assert d.decision == "reject"


def test_decision_generated_demote_on_weak() -> None:
    cand = MergedCandidate(term="x", normalized="x", source="llm_generated")
    ev = SupportEvaluation(
        corpus_support_score=0.5,
        support_label="weak",
        exact_match_count=1,
        heading_or_title_presence=False,
        supporting_snippets=["a"],
        rationale="weak",
    )
    d = apply_decision(cand, ev)
    assert d.decision == "demote"


def test_corpus_support_strong_when_two_docs() -> None:
    d1 = _doc("The guard dog breed is popular.")
    d2 = _doc("Another guard dog mention here.")
    cand = MergedCandidate(
        term="guard dog",
        normalized="guard dog",
        source="llm_generated",
    )
    ev = evaluate_support(cand, [d1, d2])
    assert ev.support_label == "strong"
    assert ev.exact_match_count == 2


def test_corpus_support_splits_title_and_heading() -> None:
    d = CleanDocument(
        doc_id="1",
        url="http://x",
        domain="x",
        title_text="best guard dogs for families",
        description_text="",
        heading_texts=["training your guard dog"],
        paragraph_texts=[],
        intro_text="",
        body_text="unrelated body text without the phrase",
        full_clean_text="unrelated body text without the phrase",
        sentences=["unrelated body text without the phrase"],
        heading_hierarchy=[(2, "training your guard dog")],
    )
    cand = MergedCandidate(
        term="best guard dogs for families",
        normalized="best guard dogs for families",
        source="llm_corpus",
    )
    ev = evaluate_support(cand, [d])
    assert ev.in_title is True
    assert ev.in_heading is False
    assert ev.heading_or_title_presence is True

    cand2 = MergedCandidate(
        term="training your guard dog",
        normalized="training your guard dog",
        source="llm_corpus",
    )
    ev2 = evaluate_support(cand2, [d])
    assert ev2.in_heading is True


def test_specificity_prefers_phrases() -> None:
    from nlp_keyword_pipeline.llm_keyword_stages.ranking import specificity_score

    assert specificity_score("dog") < specificity_score("guard dog breed")


def test_ranking_sort_key_prefers_llm_corpus_on_tie() -> None:
    from nlp_keyword_pipeline.llm_keyword_stages.ranking import ranking_sort_key

    k_gen = ranking_sort_key(
        final_score=0.75,
        source="llm_generated",
        corpus_support_score=0.9,
        in_title=False,
        in_heading=False,
        specificity=0.5,
        normalized="z",
    )
    k_corpus = ranking_sort_key(
        final_score=0.75,
        source="llm_corpus",
        corpus_support_score=0.9,
        in_title=False,
        in_heading=False,
        specificity=0.5,
        normalized="a",
    )
    assert k_corpus > k_gen
