"""N-gram phrase extraction (bigrams and trigrams)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from spacy.lang.en.stop_words import STOP_WORDS

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']*")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _phrase_ok(tokens_tuple: tuple[str, ...]) -> bool:
    if not tokens_tuple:
        return False
    if tokens_tuple[0] in STOP_WORDS or tokens_tuple[-1] in STOP_WORDS:
        return False
    if any(len(t) < 2 for t in tokens_tuple):
        return False
    return True


def _emit(
    registry: dict,
    doc_id: str,
    tokens_tuple: tuple[str, ...],
    zone: str,
    config: PipelineConfig,
) -> None:
    surface = " ".join(tokens_tuple)
    if len(surface) < 4:
        return
    register_hit(
        registry,
        surface,
        doc_id,
        "ngram",
        {zone},
        config=config,
    )


def extract(doc: CleanDocument, registry: dict, config: PipelineConfig) -> None:
    n_min = config.ngram_min_len
    n_max = config.ngram_max_len
    body = doc.body_text or ""
    lim = getattr(config, "ngram_body_max_chars", 0) or len(body)
    body_slice = body[:lim] if body else ""
    zones_text: list[tuple[str, str]] = [
        ("title", doc.title_text),
        ("body", body_slice),
    ]
    for h in doc.heading_texts:
        zones_text.append(("heading", h))

    for zone, text in zones_text:
        if not text.strip():
            continue
        toks = _tokens(text)
        for n in range(n_min, n_max + 1):
            for ng in _ngrams(toks, n):
                if _phrase_ok(ng):
                    _emit(registry, doc.doc_id, ng, zone, config)
