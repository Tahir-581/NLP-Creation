"""YAKE keyphrase extraction per document."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument


def extract(
    doc: CleanDocument,
    yake_extractor,
    registry: dict,
    config: PipelineConfig,
) -> None:
    body = doc.body_text or doc.full_clean_text
    if not body or len(body.strip()) < 50:
        return
    lim = getattr(config, "yake_body_max_chars", 0) or len(body)
    body = body[:lim]
    try:
        kws = yake_extractor.extract_keywords(body)
    except Exception:
        return
    for item in kws[: config.yake_top]:
        if isinstance(item, tuple):
            phrase = item[0]
        else:
            phrase = str(item)
        phrase = phrase.strip()
        if len(phrase) < 3:
            continue
        register_hit(registry, phrase, doc.doc_id, "yake", {"body"}, config=config)
