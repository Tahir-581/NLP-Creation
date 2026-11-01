"""Candidate extractors: n-grams, chunks, entities, headings, YAKE, KeyBERT, questions, POS patterns."""

from nlp_keyword_pipeline.extractors.registry_ops import (
    extract_all_for_document,
    normalize_phrase_key,
)

__all__ = ["extract_all_for_document", "normalize_phrase_key"]
