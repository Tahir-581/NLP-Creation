"""Context / lifestyle phrase extractor — Opt-3.

Extracts prepositional context phrases (for apartments, with children),
owner-type phrases (experienced owners, first-time owners), and role/usage
bigrams (military work, police work, search and rescue) using spaCy's
dependency parser.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)

# Preposition heads that signal lifestyle/use-case context
_CONTEXT_PREPS = frozenset(
    {"for", "with", "around", "near", "in", "within", "among", "alongside"}
)

# Owner / role type modifiers that form useful phrases
_OWNER_MOD_HEADS = frozenset(
    {
        "owners",
        "owner",
        "handlers",
        "handler",
        "families",
        "family",
        "households",
        "household",
        "environments",
        "environment",
    }
)

# Known role/use-case bigram seeds to scan body for
_ROLE_SEEDS: list[tuple[str, str]] = [
    ("military", "work"),
    ("military", "service"),
    ("police", "work"),
    ("police", "dog"),
    ("police", "force"),
    ("search", "rescue"),
    ("personal", "protection"),
    ("estate", "guardian"),
    ("property", "guard"),
    ("farm", "dog"),
    ("livestock", "guardian"),
    ("working", "dog"),
    ("herding", "dog"),
    ("therapy", "dog"),
    ("service", "dog"),
    ("rescue", "dog"),
    ("apartment", "dog"),
    ("family", "companion"),
    ("family", "dog"),
    ("companion", "dog"),
    ("suitable", "families"),
    ("good", "children"),
    ("good", "kids"),
    ("family", "life"),
    ("daily", "exercise"),
    ("high", "exercise"),
    ("exercise", "needs"),
    ("mental", "stimulation"),
    ("natural", "protector"),
    ("territorial", "nature"),
    ("early", "socialization"),
    ("responds", "training"),
    ("owner", "fit"),
    ("experienced", "owners"),
]


def _extract_prep_phrases(
    spacy_doc,
    clean_doc: "CleanDocument",
    registry: dict,
    config: "PipelineConfig",
    seen: set,
    emitted: int,
    max_emit: int,
) -> int:
    """Extract PREP → POBJ phrases signaling lifestyle context."""
    for token in spacy_doc:
        if emitted >= max_emit:
            break
        if token.dep_ == "prep" and token.text.lower() in _CONTEXT_PREPS:
            for child in token.children:
                if child.dep_ == "pobj":
                    # Include compound children of pobj (e.g. "first-time owners")
                    left_parts = [
                        c.text for c in child.lefts if c.dep_ in {"compound", "amod", "nummod"}
                    ]
                    parts = left_parts + [child.text]
                    phrase = " ".join(parts)
                    full_phrase = f"{token.text} {phrase}"
                    low = full_phrase.lower()
                    if len(full_phrase.split()) >= 2 and low not in seen:
                        seen.add(low)
                        register_hit(
                            registry,
                            full_phrase,
                            clean_doc.doc_id,
                            "context_phrase",
                            {"body"},
                            config=config,
                        )
                        emitted += 1
    return emitted


def _extract_owner_phrases(
    spacy_doc,
    clean_doc: "CleanDocument",
    registry: dict,
    config: "PipelineConfig",
    seen: set,
    emitted: int,
    max_emit: int,
) -> int:
    """Extract owner/handler type phrases: [ADJ|NOUN compound] + owner/family head."""
    for token in spacy_doc:
        if emitted >= max_emit:
            break
        if token.text.lower() in _OWNER_MOD_HEADS:
            mods = [
                c.text
                for c in token.lefts
                if c.dep_ in {"amod", "compound", "nummod"} and c.tag_ not in {"CD"}
            ]
            if mods:
                phrase = " ".join(mods) + " " + token.text
                low = phrase.lower()
                if low not in seen:
                    seen.add(low)
                    register_hit(
                        registry,
                        phrase,
                        clean_doc.doc_id,
                        "context_phrase",
                        {"body"},
                        config=config,
                    )
                    emitted += 1
    return emitted


def _extract_role_seeds(
    text_lower: str,
    clean_doc: "CleanDocument",
    registry: dict,
    config: "PipelineConfig",
    seen: set,
    emitted: int,
    max_emit: int,
) -> int:
    """Scan lowercased body text for known role/use-case bigrams."""
    for w1, w2 in _ROLE_SEEDS:
        if emitted >= max_emit:
            break
        phrase = f"{w1} {w2}"
        if phrase in text_lower and phrase not in seen:
            seen.add(phrase)
            register_hit(
                registry,
                phrase,
                clean_doc.doc_id,
                "context_phrase",
                {"body"},
                config=config,
            )
            emitted += 1
    return emitted


def extract(
    clean_doc: "CleanDocument",
    nlp: "Language",
    registry: dict,
    config: "PipelineConfig",
) -> None:
    """Main entry point for context/lifestyle phrase extraction."""
    if not getattr(config, "enable_context_phrase_extractor", True):
        return
    max_emit = getattr(config, "context_phrase_max_per_doc", 60)

    text = clean_doc.body_text or clean_doc.intro_text
    if not text or len(text.strip()) < 40:
        return

    text_slice = text[: min(len(text), nlp.max_length)]
    spacy_doc = nlp(text_slice)
    seen: set[str] = set()
    emitted = 0

    emitted = _extract_role_seeds(
        text_slice.lower(), clean_doc, registry, config, seen, emitted, max_emit
    )
    emitted = _extract_prep_phrases(
        spacy_doc, clean_doc, registry, config, seen, emitted, max_emit
    )
    emitted = _extract_owner_phrases(
        spacy_doc, clean_doc, registry, config, seen, emitted, max_emit
    )

    logger.debug(
        "context_phrases: doc=%s emitted=%d", clean_doc.doc_id, emitted
    )
