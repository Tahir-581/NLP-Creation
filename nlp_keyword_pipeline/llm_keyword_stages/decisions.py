"""Keep / demote / reject from source + SupportEvaluation."""

from __future__ import annotations

from nlp_keyword_pipeline.llm_keyword_stages.models import (
    MergedCandidate,
    SupportEvaluation,
    TermDecision,
)


def apply_decision(
    candidate: MergedCandidate,
    evaluation: SupportEvaluation,
    *,
    reject_llm_corpus_on_none: bool = True,
) -> TermDecision:
    """
    Rules:
    - llm_corpus: keep unless strong reason (support_label none → reject if flag).
    - llm_generated: none or score 0 → reject; weak → demote; strong → keep.
    - corpus (classical): keep.
    """
    src = candidate.source
    label = evaluation.support_label
    escore = evaluation.corpus_support_score

    if src == "corpus":
        return TermDecision(
            decision="keep",
            rationale="Classical corpus extraction; treated as grounded.",
            confidence=0.95,
        )

    if src == "llm_corpus":
        if reject_llm_corpus_on_none and label == "none":
            return TermDecision(
                decision="reject",
                rationale="Corpus-derived term has no support in evaluated text.",
                confidence=0.85,
            )
        return TermDecision(
            decision="keep",
            rationale="Corpus LLM term; retained with evaluated support.",
            confidence=min(1.0, 0.7 + 0.1 * escore),
        )

    # llm_generated
    if escore <= 0.0 or label == "none":
        return TermDecision(
            decision="reject",
            rationale="Generated term lacks corpus support.",
            confidence=0.9,
        )
    if label == "weak":
        return TermDecision(
            decision="demote",
            rationale="Weak corpus support for generated term.",
            confidence=0.75,
        )
    return TermDecision(
        decision="keep",
        rationale="Strong corpus support for generated term.",
        confidence=0.8,
    )
