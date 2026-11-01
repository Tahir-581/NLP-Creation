"""POS + dependency patterns: ADJ+NOUN, NOUN+NOUN, ADJ+NOUN+NOUN, dep amod pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spacy.lang.en.stop_words import STOP_WORDS

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

_ALLOWED_ADJ = {"JJ", "JJR", "JJS"}
_ALLOWED_NOUN = {"NN", "NNS", "NNP", "NNPS"}


def _span_token_idx(doc, start_i: int, end_i_inclusive: int) -> str:
    return doc[start_i : end_i_inclusive + 1].text.strip()


def _ok_tokens(text: str) -> bool:
    parts = text.lower().split()
    if len(parts) < 2:
        return False
    if parts[0] in STOP_WORDS and parts[-1] in STOP_WORDS:
        return False
    return True


def _extract_dep_parse(
    spacy_doc,
    clean_doc: CleanDocument,
    registry: dict,
    config: PipelineConfig,
    seen: set[str],
    emitted: int,
) -> int:
    max_emit = getattr(config, "pos_pattern_max_per_doc", 80)
    for token in spacy_doc:
        if emitted >= max_emit:
            break
        if token.dep_ == "advmod" and token.head.dep_ == "amod":
            adj = token.head
            noun = adj.head
            if adj.tag_ in _ALLOWED_ADJ and noun.tag_ in _ALLOWED_NOUN:
                phrase = f"{token.text} {adj.text} {noun.text}"
                if _ok_tokens(phrase) and phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    register_hit(
                        registry,
                        phrase,
                        clean_doc.doc_id,
                        "dep_pattern",
                        {"body"},
                        config=config,
                    )
                    emitted += 1
                    continue
        if token.dep_ == "amod" and token.head.tag_ in _ALLOWED_NOUN:
            adj = token
            noun = token.head
            phrase = f"{adj.text} {noun.text}"
            if _ok_tokens(phrase) and phrase.lower() not in seen:
                seen.add(phrase.lower())
                register_hit(
                    registry,
                    phrase,
                    clean_doc.doc_id,
                    "dep_pattern",
                    {"body"},
                    config=config,
                )
                emitted += 1
    return emitted


def extract(
    clean_doc: CleanDocument,
    nlp: Language,
    registry: dict,
    config: PipelineConfig,
) -> None:
    text = clean_doc.body_text or clean_doc.intro_text
    if not text or len(text.strip()) < 40:
        return
    spacy_doc = nlp(text[: min(len(text), nlp.max_length)])
    toks = [t for t in spacy_doc if not t.is_space]
    n = len(toks)
    i = 0
    seen: set[str] = set()
    max_emit = getattr(config, "pos_pattern_max_per_doc", 80)
    emitted = 0
    while i < n and emitted < max_emit:
        t0 = toks[i]
        if (
            i + 2 < n
            and t0.tag_ in _ALLOWED_ADJ
            and toks[i + 1].tag_ in _ALLOWED_ADJ
            and toks[i + 2].tag_ in _ALLOWED_NOUN
        ):
            phrase = _span_token_idx(spacy_doc, t0.i, toks[i + 2].i)
            if _ok_tokens(phrase) and phrase.lower() not in seen:
                seen.add(phrase.lower())
                register_hit(
                    registry,
                    phrase,
                    clean_doc.doc_id,
                    "pos_pattern",
                    {"body"},
                    config=config,
                )
                emitted += 1
            i += 1
            continue
        if (
            i + 2 < n
            and t0.tag_ in _ALLOWED_ADJ
            and toks[i + 1].tag_ in _ALLOWED_NOUN
            and toks[i + 2].tag_ in _ALLOWED_NOUN
        ):
            phrase = _span_token_idx(spacy_doc, t0.i, toks[i + 2].i)
            if _ok_tokens(phrase) and phrase.lower() not in seen:
                seen.add(phrase.lower())
                register_hit(
                    registry,
                    phrase,
                    clean_doc.doc_id,
                    "pos_pattern",
                    {"body"},
                    config=config,
                )
                emitted += 1
            i += 1
            continue
        if emitted >= max_emit:
            break
        if i + 1 < n and t0.tag_ in _ALLOWED_ADJ and toks[i + 1].tag_ in _ALLOWED_NOUN:
            phrase = _span_token_idx(spacy_doc, t0.i, toks[i + 1].i)
            if _ok_tokens(phrase) and phrase.lower() not in seen:
                seen.add(phrase.lower())
                register_hit(
                    registry,
                    phrase,
                    clean_doc.doc_id,
                    "pos_pattern",
                    {"body"},
                    config=config,
                )
                emitted += 1
        if emitted >= max_emit:
            break
        if i + 1 < n and t0.tag_ in _ALLOWED_NOUN and toks[i + 1].tag_ in _ALLOWED_NOUN:
            phrase = _span_token_idx(spacy_doc, t0.i, toks[i + 1].i)
            if _ok_tokens(phrase) and phrase.lower() not in seen:
                seen.add(phrase.lower())
                register_hit(
                    registry,
                    phrase,
                    clean_doc.doc_id,
                    "pos_pattern",
                    {"body"},
                    config=config,
                )
                emitted += 1
        i += 1
    _extract_dep_parse(spacy_doc, clean_doc, registry, config, seen, emitted)
