"""Explicit topic anchor: lexical + semantic seed for scoring and classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nlp_keyword_pipeline.domain_lexicon import DOMAIN_LEXICON
from nlp_keyword_pipeline.normalizer import normalize_text_for_dup, strip_leading_determiners

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class TopicAnchor:
    """Control signal derived from the configured topic phrase (and optional corpus hints)."""

    raw_title: str
    normalized_title: str
    tokens: list[str]
    noun_phrases: list[str]
    content_type: str
    topic_lexicon: frozenset[str]
    embedding: Any | None = None
    semantic_neighborhoods: list[str] = field(default_factory=list)


def _infer_content_type(anchor: str, config: PipelineConfig) -> str:
    explicit = getattr(config, "topic_anchor_content_type", "auto")
    if explicit and explicit != "auto":
        return explicit
    low = anchor.lower()
    if "breed" in low or "dog" in low:
        return "breed_listicle"
    if "compare" in low or "vs" in low or "versus" in low:
        return "comparison"
    if "train" in low or "care" in low or "health" in low:
        return "breed_care"
    return "generic"


def build_topic_anchor(
    topic_anchor_str: str,
    config: PipelineConfig,
    nlp: Language,
    st_model: Any | None = None,
) -> TopicAnchor:
    raw = (topic_anchor_str or "").strip()
    normalized = strip_leading_determiners(normalize_text_for_dup(raw))
    doc = nlp(raw[:500] if raw else "topic")
    tokens = [t.text.lower() for t in doc if t.is_alpha and len(t.text) > 1]
    noun_phrases: list[str] = []
    seen: set[str] = set()
    for chunk in doc.noun_chunks:
        t = chunk.text.strip()
        low = normalize_text_for_dup(t)
        if len(low.split()) >= 1 and low not in seen and len(low) > 2:
            seen.add(low)
            noun_phrases.append(low)

    content_type = _infer_content_type(raw, config)
    # Topic lexicon: anchor tokens + noun phrases + overlapping domain terms
    extra: set[str] = set(tokens) | set(noun_phrases)
    for term in DOMAIN_LEXICON:
        if len(term) < 4:
            continue
        if term in normalized or any(tok in term.split() for tok in tokens):
            extra.add(term)
    topic_lexicon = frozenset(extra) | frozenset({normalized}) if normalized else frozenset(extra)

    embedding = None
    semantic_neighborhoods: list[str] = []
    if st_model is not None and raw:
        try:
            import numpy as np

            embedding = st_model.encode([raw], convert_to_numpy=True)[0]
            # Optional: nearest neighbors from small seed set (lexicon sample)
            seeds = [s for s in list(DOMAIN_LEXICON)[:80] if len(s) > 5]
            if seeds:
                emb_seeds = st_model.encode(seeds[:40], convert_to_numpy=True, show_progress_bar=False)
                e = np.asarray(embedding, dtype=np.float64)
                e = e / (np.linalg.norm(e) + 1e-9)
                sims = emb_seeds @ e
                top_idx = np.argsort(-sims)[:8]
                semantic_neighborhoods = [seeds[i] for i in top_idx if sims[i] > 0.35]
        except Exception as exc:
            logger.debug("Topic anchor embedding/neighbors skipped: %s", exc)

    return TopicAnchor(
        raw_title=raw,
        normalized_title=normalized,
        tokens=tokens,
        noun_phrases=noun_phrases,
        content_type=content_type,
        topic_lexicon=topic_lexicon,
        embedding=embedding,
        semantic_neighborhoods=semantic_neighborhoods,
    )


def anchor_token_overlap(anchor: TopicAnchor, phrase_norm: str) -> int:
    ptoks = set(phrase_norm.split())
    return len(ptoks & set(anchor.tokens))
