"""Named entities and domain lexicon (dog breeds / guard-dog topic)."""

from __future__ import annotations

import re
import string
from typing import TYPE_CHECKING

from nlp_keyword_pipeline.domain_lexicon import DOMAIN_LEXICON, canonical_breed_phrase
from nlp_keyword_pipeline.extractors.registry_ops import register_hit

# Single-token NER surface forms that are breed/proper names (allow without full lexicon hit)
_SINGLE_TOKEN_ALLOW = frozenset(
    {
        "akita",
        "boxer",
        "komondor",
        "kuvasz",
        "malinois",
        "ridgeback",
        "rottweiler",
        "husky",
        "wolfhound",
        "boerboel",
        "ovcharka",
        "sarplaninac",
        "leonberger",
        "newfoundland",
    }
)

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

NER_LABELS = frozenset({"ORG", "PERSON", "GPE", "NORP", "FAC", "PRODUCT", "EVENT"})
_ORPHAN_GEO = frozenset(
    {"german", "belgian", "american", "british", "french", "italian", "russian", "chinese", "australian"}
)


def _find_lexicon_spans(text_lower: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for phrase in sorted(DOMAIN_LEXICON, key=len, reverse=True):
        for m in re.finditer(re.escape(phrase), text_lower):
            spans.append((m.start(), m.end(), phrase))
    spans.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    used: list[tuple[int, int]] = []
    out: list[tuple[int, int, str]] = []
    for s, e, p in spans:
        if any(not (e <= u[0] or s >= u[1]) for u in used):
            continue
        used.append((s, e))
        out.append((s, e, p))
    out.sort(key=lambda x: x[0])
    return out


def _zone_at_offset(clean_doc: CleanDocument, idx: int) -> str:
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
        getattr(config, "entity_match_max_chars", 10_000),
        nlp.max_length,
    )
    spacy_doc = nlp(text[:cap])
    for ent in spacy_doc.ents:
        if ent.label_ not in NER_LABELS:
            continue
        surface = ent.text.strip()
        if len(surface) < 3:
            continue
        words = surface.split()
        low = surface.lower()
        if len(words) == 1 and low in _ORPHAN_GEO:
            continue
        if len(words) == 1 and low not in DOMAIN_LEXICON and low not in _SINGLE_TOKEN_ALLOW:
            continue
        zone = _zone_at_offset(clean_doc, ent.start_char)
        canon = canonical_breed_phrase(low)
        surface_out = string.capwords(canon) if canon != low else surface
        register_hit(
            registry,
            surface_out,
            clean_doc.doc_id,
            "ner",
            {zone},
            is_entity=True,
            config=config,
        )

    lower = text[:cap].lower()
    slice_text = text[:cap]
    for _s, _e, phrase in _find_lexicon_spans(lower):
        surface = slice_text[_s:_e].strip() or phrase.title()
        zone = _zone_at_offset(clean_doc, _s)
        register_hit(
            registry,
            surface,
            clean_doc.doc_id,
            "lexicon",
            {zone},
            from_lexicon=True,
            config=config,
        )
