"""Register extraction hits into the candidate registry."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from nlp_keyword_pipeline.models import Candidate
from nlp_keyword_pipeline.normalizer import strip_leading_determiners

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig

_WS = re.compile(r"\s+")


def normalize_phrase_key(text: str) -> str:
    t = (text or "").strip()
    t = _WS.sub(" ", t)
    return t.lower()


def _trust_for_method(method: str, config: PipelineConfig) -> float:
    w = getattr(config, "extractor_trust_weights", None) or {}
    return float(w.get(method, 0.65))


def register_hit(
    registry: dict[str, Candidate],
    surface_form: str,
    doc_id: str,
    method: str,
    zones: set[str],
    *,
    is_entity: bool = False,
    from_lexicon: bool = False,
    is_question: bool = False,
    config: PipelineConfig | None = None,
    skip_phrase_hygiene: bool = False,
) -> None:
    if not surface_form or len(surface_form.strip()) < 2:
        return
    surface_form = _WS.sub(" ", surface_form.strip())
    norm = strip_leading_determiners(normalize_phrase_key(surface_form))
    if not norm:
        return
    hy = None
    if config is not None and not skip_phrase_hygiene:
        from nlp_keyword_pipeline.phrase_hygiene import evaluate_phrase_hygiene

        hy = evaluate_phrase_hygiene(surface_form, norm, config)
        if hy.verdict == "hard_reject":
            return
    c = registry.get(norm)
    if c is None:
        c = Candidate(
            surface_form=surface_form,
            normalized_form=norm,
            is_entity=is_entity,
            from_lexicon=from_lexicon,
            is_question=is_question,
        )
        registry[norm] = c
    else:

        def _fit(sf: str) -> float:
            if not sf:
                return 0.0
            return float(fuzz.ratio(sf.lower(), c.normalized_form))

        if surface_form:
            if not c.surface_form:
                c.surface_form = surface_form
            elif _fit(surface_form) > _fit(c.surface_form):
                c.surface_form = surface_form
            elif _fit(surface_form) == _fit(c.surface_form) and len(surface_form) > len(
                c.surface_form
            ):
                c.surface_form = surface_form
        c.is_entity = c.is_entity or is_entity
        c.from_lexicon = c.from_lexicon or from_lexicon
        c.is_question = c.is_question or is_question

    if hy is not None and hy.verdict == "soft_penalty":
        c.hygiene_result = "soft_penalty"
        if hy.reason and hy.reason not in c.hygiene_reasons:
            c.hygiene_reasons.append(hy.reason)
        c.penalties["hygiene_soft"] = max(c.penalties.get("hygiene_soft", 0.0), hy.penalty_weight)

    trust = _trust_for_method(method, config) if config is not None else 0.65
    c.extractor_confidence[method] = max(c.extractor_confidence.get(method, 0.0), trust)

    c.source_methods.add(method)
    c.source_docs.add(doc_id)
    if "title" in zones:
        if doc_id not in c.docs_in_title:
            c.title_hits += 1
        c.docs_in_title.add(doc_id)
    if "heading" in zones:
        if doc_id not in c.docs_in_heading:
            c.heading_hits += 1
        c.docs_in_heading.add(doc_id)
    if "body" in zones:
        if doc_id not in c.docs_in_body:
            c.body_hits += 1
        c.docs_in_body.add(doc_id)
    if "intro" in zones:
        if doc_id not in c.docs_in_intro:
            c.intro_hits += 1
        c.docs_in_intro.add(doc_id)


def extract_all_for_document(
    doc,
    nlp: Language,
    yake_extractor,
    keybert_model,
    registry: dict[str, Candidate],
    config,
) -> None:
    from nlp_keyword_pipeline.extractors import (
        attribute_relations,
        context_phrases,
        entities,
        headings,
        keybert_ext,
        ngrams,
        noun_chunks,
        questions,
        yake_ext,
    )  # noqa: PLC0415

    ngrams.extract(doc, registry, config)
    noun_chunks.extract(doc, nlp, registry, config)
    entities.extract(doc, nlp, registry, config)
    headings.extract(doc, registry, config)
    yake_ext.extract(doc, yake_extractor, registry, config)
    keybert_ext.extract(doc, keybert_model, registry, config)
    questions.extract(doc, nlp, registry, config)
    context_phrases.extract(doc, nlp, registry, config)
    from nlp_keyword_pipeline.extractors import phrase_patterns  # noqa: PLC0415

    phrase_patterns.extract(doc, nlp, registry, config)
    attribute_relations.extract(doc, nlp, registry, config)
