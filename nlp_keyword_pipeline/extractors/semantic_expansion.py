"""Semantic synonym expansion post-pass — Opt-5.

After all per-document extractors have run, iterates over high-confidence
candidate phrases in the registry and uses spaCy's en_core_web_md word
vectors to find semantically similar single words (synonyms, related terms)
that were not extracted from the corpus directly.

This is a standalone post-pass, NOT a per-document extractor.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import Candidate

logger = logging.getLogger(__name__)

# Candidate word categories worth expanding into (POS tags to include)
_EXPAND_POS = frozenset({"NOUN", "ADJ", "VERB"})

# Hard block-list: words that are too generic to ever be useful single-word keywords
_BLOCK_SINGLES = frozenset(
    {
        "dog", "dogs", "breed", "breeds", "pet", "pets", "animal", "animals",
        "puppy", "puppies", "one", "two", "three", "many", "some", "good",
        "great", "big", "small", "old", "new", "well", "best", "top", "most",
        "more", "also", "just", "very", "need", "make", "take", "come",
        "know", "think", "find", "use", "help", "give", "keep", "way", "time",
        "day", "thing", "get", "go", "see", "man", "look", "first", "people",
        "right", "back", "life", "place", "number", "work", "large", "high",
    }
)


def _get_seed_words(
    registry: dict[str, "Candidate"],
    n_docs: int,
    min_heading: int,
    min_docs_frac: float,
) -> list[str]:
    """Pick single-token seeds with heading support and multi-doc evidence."""
    seeds: list[str] = []
    nd = max(n_docs, 1)
    for norm, cand in registry.items():
        if cand.heading_hits < min_heading:
            continue
        if len(cand.source_methods) < 2:
            continue
        if len(cand.source_docs) / nd < min_docs_frac:
            continue
        if (
            len(norm.split()) == 1
            and len(norm) >= 4
            and norm not in _BLOCK_SINGLES
            and (cand.is_entity or cand.from_lexicon or len(cand.source_docs) >= 2)
        ):
            seeds.append(norm)
    return seeds


def _get_multi_token_seeds(
    registry: dict[str, "Candidate"],
    n_docs: int,
    min_heading: int,
    min_docs_frac: float,
) -> list[str]:
    """Multi-token entity phrases → head noun as vector seed."""
    seeds: list[str] = []
    nd = max(n_docs, 1)
    for norm, cand in registry.items():
        words = norm.split()
        if cand.heading_hits < min_heading:
            continue
        if len(cand.source_docs) / nd < min_docs_frac:
            continue
        if (
            2 <= len(words) <= 3
            and (cand.is_entity or cand.from_lexicon)
        ):
            seeds.append(words[-1])
    return list(set(seeds))


def run(
    registry: dict[str, "Candidate"],
    nlp: "Language",
    config: "PipelineConfig",
    *,
    corpus_lower: str = "",
    n_docs: int = 1,
) -> None:
    """Entry point: expand registry with semantically similar words (guardrailed)."""
    if not getattr(config, "enable_semantic_expansion", True):
        return

    sim_thresh = getattr(config, "semantic_expansion_similarity_threshold", 0.72)
    max_per_seed = getattr(config, "semantic_expansion_max_per_seed", 4)
    min_heading = getattr(config, "semantic_expansion_min_seed_heading_hits", 1)
    min_docs_frac = getattr(config, "semantic_expansion_min_seed_doc_fraction", 0.12)
    min_corpus_hits = getattr(config, "semantic_expansion_min_corpus_presence", 1)

    # Check that the model has vectors
    if not nlp.vocab.vectors.shape[0]:
        logger.info("semantic_expansion: no word vectors in spaCy model, skipping.")
        return

    single_seeds = _get_seed_words(registry, n_docs, min_heading, min_docs_frac)
    multi_seeds = _get_multi_token_seeds(registry, n_docs, min_heading, min_docs_frac)
    all_seeds = list(set(single_seeds + multi_seeds))

    if not all_seeds:
        logger.info("semantic_expansion: no suitable seeds found in registry.")
        return

    logger.info(
        "semantic_expansion: expanding from %d seed words (sim_thresh=%.2f, max_per_seed=%d)",
        len(all_seeds),
        sim_thresh,
        max_per_seed,
    )

    expanded_count = 0
    # Use a fixed synthetic doc_id for expansion hits
    synthetic_doc_id = "__semantic_expansion__"

    for seed_word in all_seeds:
        seed_token = nlp(seed_word)
        if not seed_token or not seed_token[0].has_vector:
            continue

        seed_vec = seed_token[0]
        candidates_found = 0

        # Walk vocabulary for similar words
        seen_expansions: set[str] = set()
        queries = nlp.vocab.vectors.most_similar(
            seed_vec.vector.reshape(1, -1), n=20
        )
        # most_similar returns (keys, best_rows, scores)
        similar_keys = queries[0][0] if queries and len(queries[0]) else []
        scores = queries[2][0] if queries and len(queries[2]) else []

        for key, score in zip(similar_keys, scores):
            if candidates_found >= max_per_seed:
                break
            if float(score) < sim_thresh:
                continue
            word = nlp.vocab.strings[key].lower()
            if (
                not word.isalpha()
                or len(word) < 4
                or word in _BLOCK_SINGLES
                or word in seen_expansions
                or word == seed_word
                or word in registry
            ):
                continue
            if corpus_lower and min_corpus_hits > 0:
                if len(re.findall(rf"\b{re.escape(word)}\b", corpus_lower)) < min_corpus_hits:
                    continue
            tok = nlp(word)
            if tok and tok[0].pos_ in _EXPAND_POS:
                seen_expansions.add(word)
                register_hit(
                    registry,
                    word,
                    synthetic_doc_id,
                    "semantic_expansion",
                    {"body"},
                    config=config,
                )
                expanded_count += 1
                candidates_found += 1

    logger.info("semantic_expansion: added %d expansion candidates to registry.", expanded_count)
