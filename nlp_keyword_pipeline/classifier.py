"""Bucket-aware classification using explicit rules + topic anchor overlap."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nlp_keyword_pipeline.models import Candidate
from nlp_keyword_pipeline.topic_anchor import anchor_token_overlap

if TYPE_CHECKING:
    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.topic_anchor import TopicAnchor

logger = logging.getLogger(__name__)

IGNORE_BUCKET = "ignore_or_noise_terms"


@dataclass
class BucketRule:
    name: str
    min_score: float
    min_prevalence: float
    min_word_count: int
    max_word_count: int
    require_heading_docs: int
    require_entity_or_lexicon: bool
    require_corpus_question: bool
    max_output_count: int


def _word_count(c: Candidate) -> int:
    return len(c.normalized_form.split())


def _noisy_heading_split(norm: str) -> bool:
    low = norm.lower()
    if " breeds to " in low and "breeds to avoid" not in low:
        return True
    if low.startswith("dog breeds to ") and "avoid" not in low:
        return True
    return False


def _rules_for_bucket(bucket: str, config: PipelineConfig) -> BucketRule:
    caps = config.bucket_caps
    cap_map = {
        "primary_topic_term": caps.primary_topic_term,
        "core_entity": caps.core_entity,
        "heading_candidate": caps.heading_candidate,
        "subtopic": caps.subtopic,
        "supporting_phrase": caps.supporting_phrase,
        "intent_phrase": caps.intent_phrase,
        "overused_terms": caps.overused_terms,
    }
    return BucketRule(
        name=bucket,
        min_score=config.min_absolute_score,
        min_prevalence=0.0,
        min_word_count=1,
        max_word_count=20,
        require_heading_docs=0,
        require_entity_or_lexicon=False,
        require_corpus_question=False,
        max_output_count=cap_map.get(bucket, 50),
    )


def classify_candidate(
    c: Candidate,
    config: PipelineConfig,
    topic_anchor: TopicAnchor | None = None,
) -> str:
    wc = _word_count(c)
    hd = len(c.docs_in_heading)
    hh = c.heading_hits
    prev = c.doc_prevalence
    entity_like = c.is_entity or c.from_lexicon
    conf = c.confidence
    pq = float(c.subscores.get("phrase_quality", 0.0))
    sem = float(c.subscores.get("semantic_relevance", 0.0))

    primary_min = config.effective_primary_min_prevalence()
    core_min = config.effective_core_entity_min_prevalence()
    sup_min = config.effective_supporting_min_prevalence()
    sub_hd = config.effective_subtopic_min_heading_docs()
    over_prev = getattr(config, "overused_prevalence_threshold", 0.7)

    anchor_overlap = (
        anchor_token_overlap(topic_anchor, c.normalized_form) if topic_anchor else 0
    )

    if c.score < config.min_absolute_score:
        return IGNORE_BUCKET
    if c.subscores.get("junk_penalty_raw", 0) >= 0.65:
        return IGNORE_BUCKET
    if _noisy_heading_split(c.normalized_form):
        return IGNORE_BUCKET

    if c.is_question:
        has_corpus_intent = bool(
            c.source_methods & {"intent_heading", "intent_phrase_body", "intent_corpus_backed"}
        )
        if not has_corpus_intent and "intent_template" in c.source_methods:
            return IGNORE_BUCKET
        if not has_corpus_intent:
            return IGNORE_BUCKET
        return "intent_phrase"

    if (
        not entity_like
        and prev >= over_prev
        and wc == 1
        and c.normalized_form in config.generic_tokens
    ):
        return "overused_terms"
    if (
        not entity_like
        and prev >= over_prev
        and wc == 2
        and all(t in config.generic_tokens for t in c.normalized_form.split())
    ):
        return "overused_terms"

    if entity_like and prev >= core_min:
        return "core_entity"

    p_lo, p_hi = config.primary_phrase_len
    if prev >= primary_min and p_lo <= wc <= p_hi:
        return "primary_topic_term"

    if hd >= sub_hd and wc >= 3:
        return "subtopic"

    h_lo, h_hi = config.heading_candidate_len
    if hh >= 2 and h_lo <= wc <= h_hi and anchor_overlap >= 1 and conf >= 0.45:
        return "heading_candidate"
    if hd >= sub_hd and h_lo <= wc <= h_hi and conf >= 0.4:
        return "heading_candidate"

    if prev >= sup_min and wc >= 2 and pq >= 0.55 and sem >= 0.22:
        if len(c.docs_in_body) == 0 and c.score < 0.34 and len(c.source_methods) <= 2:
            return IGNORE_BUCKET
        return "supporting_phrase"

    if entity_like:
        return "core_entity"

    if wc >= 2 and prev >= 0.08 and pq >= 0.5 and sem >= 0.18:
        if len(c.docs_in_body) == 0 and c.score < 0.32:
            return IGNORE_BUCKET
        return "supporting_phrase"

    return IGNORE_BUCKET


def classify_all(
    registry: dict[str, Candidate],
    config: PipelineConfig,
    topic_anchor: TopicAnchor | None = None,
) -> None:
    for c in registry.values():
        c.candidate_type = classify_candidate(c, config, topic_anchor)
    logger.info("Classified %d candidates", len(registry))
