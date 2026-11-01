"""Corpus-based term extraction.

Wraps the existing nlp_keyword_pipeline extractor stack
(ngrams, noun_chunks, entities, YAKE, headings, context_phrases, etc.)
and runs them against every CleanDocument in the corpus.

All extracted terms carry ``source = "corpus"`` and go directly to the
**Keep** branch — they are not subject to LLM corpus-support checking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CorpusTerm:
    """A term extracted directly from the corpus."""

    term: str
    normalized: str
    source: str = "corpus"

    doc_count: int = 0
    """Number of documents this term appears in."""

    total_docs: int = 0
    """Total documents considered (for prevalence computation)."""

    title_hits: int = 0
    heading_hits: int = 0
    body_hits: int = 0
    intro_hits: int = 0

    is_entity: bool = False
    from_lexicon: bool = False
    is_question: bool = False

    extractors: list[str] = field(default_factory=list)
    """Extraction methods that found this term."""

    @property
    def prevalence_rate(self) -> float:
        if self.total_docs == 0:
            return 0.0
        return min(1.0, self.doc_count / self.total_docs)

    @property
    def zone_bonus(self) -> float:
        """0.0–1.0 bonus based on high-value zone presence."""
        bonus = 0.0
        if self.title_hits:
            bonus += 0.4
        if self.heading_hits:
            bonus += 0.35
        if self.intro_hits:
            bonus += 0.15
        return min(1.0, bonus)

    def as_dict(self) -> dict:
        return {
            "term": self.term,
            "normalized": self.normalized,
            "source": self.source,
            "doc_count": self.doc_count,
            "total_docs": self.total_docs,
            "prevalence_rate": round(self.prevalence_rate, 4),
            "zone_bonus": round(self.zone_bonus, 4),
            "title_hits": self.title_hits,
            "heading_hits": self.heading_hits,
            "body_hits": self.body_hits,
            "intro_hits": self.intro_hits,
            "is_entity": self.is_entity,
            "from_lexicon": self.from_lexicon,
            "is_question": self.is_question,
            "extractors": self.extractors,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_corpus_terms(
    docs: list[CleanDocument],
    *,
    min_doc_count: int = 1,
    spacy_model: str = "en_core_web_sm",
) -> list[CorpusTerm]:
    """Run the full nlp_keyword_pipeline extractor stack over all documents.

    Parameters
    ----------
    docs:
        CleanDocument list from build_clean_documents().
    min_doc_count:
        Minimum number of documents a term must appear in to be kept.
    spacy_model:
        spaCy model name (must be installed).

    Returns
    -------
    list[CorpusTerm]
        Corpus-sourced terms, not yet scored — scoring is done in ranker.py.
    """
    if not docs:
        logger.warning("[corpus_extractor] No documents provided; skipping extraction.")
        return []

    # ------------------------------------------------------------------
    # Lazy imports: only pulled in when this function actually runs,
    # so the module can be imported cheaply.
    # ------------------------------------------------------------------
    import spacy
    import yake as _yake_lib

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.extractors.registry_ops import (
        extract_all_for_document,
        normalize_phrase_key,
    )
    from nlp_keyword_pipeline.models import Candidate
    from nlp_keyword_pipeline.normalizer import strip_leading_determiners

    cfg = PipelineConfig()

    # Load spaCy
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        logger.warning(
            "[corpus_extractor] spaCy model %r not found; falling back to blank model. "
            "Many extractors will produce sparse results. "
            "Run: python -m spacy download %s",
            spacy_model,
            spacy_model,
        )
        nlp = spacy.blank("en")

    # Build YAKE extractor (shared across docs)
    yake_extractor = _yake_lib.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.9,
        top=30,
        features=None,
    )

    # KeyBERT is optional — skip gracefully if not installed
    keybert_model = None
    try:
        from keybert import KeyBERT  # type: ignore

        keybert_model = KeyBERT()
    except ImportError:
        logger.debug("[corpus_extractor] KeyBERT not installed; skipping keybert extractor.")

    # Shared candidate registry (normalized_form → Candidate)
    registry: dict[str, Candidate] = {}

    for doc in docs:
        try:
            extract_all_for_document(doc, nlp, yake_extractor, keybert_model, registry, cfg)
        except Exception as exc:
            logger.warning(
                "[corpus_extractor] Extraction failed for doc %s: %s",
                doc.doc_id,
                exc,
            )

    logger.info("[corpus_extractor] Registry: %d unique candidates", len(registry))

    total_docs = len(docs)
    result: list[CorpusTerm] = []

    for norm_key, cand in registry.items():
        if cand.hygiene_result == "hard_reject":
            continue
        dc = len(cand.source_docs)
        if dc < min_doc_count:
            continue

        ct = CorpusTerm(
            term=cand.surface_form or cand.normalized_form,
            normalized=norm_key,
            source="corpus",
            doc_count=dc,
            total_docs=total_docs,
            title_hits=cand.title_hits,
            heading_hits=cand.heading_hits,
            body_hits=cand.body_hits,
            intro_hits=cand.intro_hits,
            is_entity=cand.is_entity,
            from_lexicon=cand.from_lexicon,
            is_question=cand.is_question,
            extractors=sorted(cand.source_methods),
        )
        result.append(ct)

    logger.info(
        "[corpus_extractor] %d terms kept (min_doc_count=%d)",
        len(result),
        min_doc_count,
    )
    return result
