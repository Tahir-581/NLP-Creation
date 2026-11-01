"""Cached sentence-transformers model, corpus centroid, and fused phrase similarity."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)


def lexical_topic_overlap(anchor: str, phrase: str) -> float:
    """Fallback when embeddings are unavailable (Jaccard on tokens)."""
    a = set(anchor.lower().split())
    b = set(phrase.lower().split())
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@lru_cache(maxsize=4)
def get_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(model_name)
        logger.info("Loaded SentenceTransformer: %s", model_name)
        return m
    except Exception as exc:
        logger.warning("Could not load SentenceTransformer %s: %s", model_name, exc)
        return None


def build_corpus_summaries(clean_docs: list[CleanDocument], max_chars: int = 480) -> list[str]:
    out: list[str] = []
    for d in clean_docs:
        parts = [d.title_text, d.description_text, d.intro_text or d.body_text[:max_chars]]
        s = " ".join(p for p in parts if p).strip()
        if len(s) > max_chars:
            s = s[:max_chars]
        out.append(s if s else d.url)
    return out


def corpus_centroid_embedding(
    model,
    clean_docs: list[CleanDocument],
    batch_size: int,
) -> "np.ndarray | None":
    import numpy as np

    texts = build_corpus_summaries(clean_docs)
    if not texts:
        return None
    try:
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=min(batch_size, 32),
        )
        c = np.mean(emb, axis=0)
        n = np.linalg.norm(c)
        if n == 0:
            return None
        return c / n
    except Exception as exc:
        logger.warning("Corpus centroid embedding failed: %s", exc)
        return None


def anchor_embedding(model, topic_anchor: str) -> "np.ndarray | None":
    import numpy as np

    try:
        v = model.encode(
            [topic_anchor],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        n = np.linalg.norm(v)
        if n == 0:
            return None
        return v / n
    except Exception as exc:
        logger.warning("Anchor embedding failed: %s", exc)
        return None


def fused_semantic_scores(
    phrases: list[str],
    clean_docs: list[CleanDocument],
    config: PipelineConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Returns (anchor_sim_by_phrase_key, corpus_sim_by_phrase_key, fused_by_phrase_key).
    Phrase keys are lowercase normalized phrases matching Candidate.normalized_form.
    """
    model = get_sentence_transformer(config.sentence_transformer_model)
    empty_a: dict[str, float] = {}
    empty_c: dict[str, float] = {}
    empty_f: dict[str, float] = {}
    if not phrases:
        return empty_a, empty_c, empty_f

    def _fallback() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        anchor = config.topic_anchor
        fa = {p: lexical_topic_overlap(anchor, p) for p in phrases}
        return fa, dict(fa), dict(fa)

    if model is None:
        return _fallback()

    import numpy as np

    a_vec = anchor_embedding(model, config.topic_anchor)
    c_vec = corpus_centroid_embedding(model, clean_docs, config.semantic_batch_size)
    if a_vec is None:
        return _fallback()

    wa = config.semantic_anchor_weight
    wc = config.semantic_corpus_weight
    if c_vec is None:
        wa, wc = 1.0, 0.0

    batch = config.semantic_batch_size
    anchor_sims: dict[str, float] = {}
    corpus_sims: dict[str, float] = {}
    fused: dict[str, float] = {}

    for i in range(0, len(phrases), batch):
        chunk = phrases[i : i + batch]
        try:
            emb = model.encode(
                chunk,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=min(batch, 64),
            )
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            emb = emb / norms
            sim_a = (emb @ a_vec).clip(0.0, 1.0)
            if c_vec is not None:
                sim_c = (emb @ c_vec).clip(0.0, 1.0)
            else:
                sim_c = np.zeros(len(chunk))
            for j, p in enumerate(chunk):
                sa = float(sim_a[j])
                sc = float(sim_c[j])
                anchor_sims[p] = sa
                corpus_sims[p] = sc
                fused[p] = float(np.clip(wa * sa + wc * sc, 0.0, 1.0))
        except Exception as exc:
            logger.warning("Semantic batch encode failed: %s", exc)
            for p in chunk:
                v = lexical_topic_overlap(config.topic_anchor, p)
                anchor_sims[p] = v
                corpus_sims[p] = v
                fused[p] = v

    return anchor_sims, corpus_sims, fused
