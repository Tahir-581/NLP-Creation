"""Entity–attribute and trait phrases via dependency parse (recall upgrade)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

_ATTRIBUTE_HEADS = frozenset(
    {
        "temperament",
        "instinct",
        "instincts",
        "demeanor",
        "presence",
        "needs",
        "need",
        "training",
        "socialization",
        "behavior",
        "loyalty",
        "courage",
        "stimulation",
        "exercise",
        "role",
        "traits",
        "trait",
        "ability",
        "abilities",
    }
)
_ALLOWED_ADJ = {"JJ", "JJR", "JJS"}
_ALLOWED_NOUN = {"NN", "NNS", "NNP", "NNPS"}


def extract(
    clean_doc: CleanDocument,
    nlp: Language,
    registry: dict,
    config: PipelineConfig,
) -> None:
    text = clean_doc.body_text or clean_doc.intro_text
    if not text or len(text.strip()) < 50:
        return
    spacy_doc = nlp(text[: min(len(text), nlp.max_length)])
    seen: set[str] = set()
    cap = getattr(config, "attribute_relation_max_per_doc", 40)
    emitted = 0
    for token in spacy_doc:
        if emitted >= cap:
            break
        h = token.head
        if token.dep_ == "compound" and h.text.lower() in _ATTRIBUTE_HEADS:
            phrase = f"{token.text} {h.text}".strip()
            low = phrase.lower()
            if len(low) >= 6 and low not in seen:
                seen.add(low)
                register_hit(
                    registry,
                    phrase,
                    clean_doc.doc_id,
                    "attribute_relation",
                    {"body"},
                    config=config,
                )
                emitted += 1
            continue
        if token.dep_ == "amod" and h.text.lower() in _ATTRIBUTE_HEADS:
            phrase = f"{token.text} {h.text}".strip()
            low = phrase.lower()
            if token.tag_ in _ALLOWED_ADJ and len(low) >= 6 and low not in seen:
                seen.add(low)
                register_hit(
                    registry,
                    phrase,
                    clean_doc.doc_id,
                    "attribute_relation",
                    {"body"},
                    config=config,
                )
                emitted += 1
            continue
        if token.dep_ == "acomp" and token.head.pos_ in {"AUX", "VERB"}:
            pred = token.head
            for c in pred.children:
                if c.dep_ not in {"nsubj", "nsubjpass"}:
                    continue
                if len(c.text) < 3:
                    continue
                phrase = f"{c.text} {token.text}".strip()
                low = phrase.lower()
                if low not in seen and 2 <= len(phrase.split()) <= 4:
                    seen.add(low)
                    register_hit(
                        registry,
                        phrase,
                        clean_doc.doc_id,
                        "attribute_relation",
                        {"body"},
                        config=config,
                    )
                    emitted += 1
        if emitted >= cap:
            break
    for chunk in spacy_doc.noun_chunks:
        if emitted >= cap:
            break
        root = chunk.root
        if root.text.lower() not in _ATTRIBUTE_HEADS:
            continue
        left = [t for t in chunk if t.i < root.i and t.tag_ in _ALLOWED_ADJ | _ALLOWED_NOUN]
        if not left:
            continue
        mod = left[-1]
        phrase = f"{mod.text} {root.text}".strip()
        low = phrase.lower()
        if len(low) >= 8 and low not in seen:
            seen.add(low)
            register_hit(
                registry,
                phrase,
                clean_doc.doc_id,
                "attribute_relation",
                {"body"},
                config=config,
            )
            emitted += 1
