"""KeyBERT keyphrase extraction (Phase 3 — semantic phrase recall)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)


def extract(
    doc: CleanDocument,
    keybert_model: Any,
    registry: dict,
    config: PipelineConfig,
) -> None:
    if not config.enable_keybert or keybert_model is None:
        return
    text = doc.body_text or doc.full_clean_text
    if not text or len(text.strip()) < 100:
        return
    text = text[: getattr(config, "keybert_body_max_chars", 6000)]
    lo, hi = config.keybert_ngram_range
    try:
        kws = keybert_model.extract_keywords(
            text,
            keyphrase_ngram_range=(lo, hi),
            stop_words="english",
            top_n=config.keybert_top_n,
            use_mmr=True,
            diversity=config.keybert_mmr_diversity,
        )
    except Exception as exc:
        logger.debug("KeyBERT skip for doc %s: %s", doc.doc_id, exc)
        return
    for item in kws:
        phrase = item[0] if isinstance(item, tuple) else str(item)
        phrase = phrase.strip()
        if len(phrase) < 3:
            continue
        register_hit(registry, phrase, doc.doc_id, "keybert", {"body"}, config=config)
