"""spaCy noun chunk extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spacy.lang.en.stop_words import STOP_WORDS

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument


def _zone_for_char_idx(clean_doc, idx: int) -> str:
    from nlp_keyword_pipeline.zones import char_zone

    return char_zone(clean_doc, idx)


def extract(
    clean_doc: CleanDocument,
    nlp: Language,
    registry: dict,
    config: PipelineConfig,
) -> None:
    text = clean_doc.full_clean_text
    if not text.strip():
        return
    cap = min(
        len(text),
        getattr(config, "noun_chunk_max_chars", 10_000),
        nlp.max_length,
    )
    spacy_doc = nlp(text[:cap])
    for chunk in spacy_doc.noun_chunks:
        phrase = chunk.text.strip()
        toks = [t.text.lower() for t in chunk if not t.is_space]
        if len(toks) < 2:
            continue
        if all(t in STOP_WORDS or len(t) < 2 for t in toks):
            continue
        start = chunk.start_char
        zone = _zone_for_char_idx(clean_doc, start)
        register_hit(
            registry,
            phrase,
            clean_doc.doc_id,
            "noun_chunk",
            {zone},
            config=config,
        )
