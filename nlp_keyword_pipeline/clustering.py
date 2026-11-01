"""Semantic clustering for heading/subtopic candidates + diversity demotion."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from nlp_keyword_pipeline.classifier import IGNORE_BUCKET
from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.models import Candidate
from nlp_keyword_pipeline.semantic_service import get_sentence_transformer

logger = logging.getLogger(__name__)


def assign_semantic_clusters(
    registry: dict[str, Candidate],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """
    Cluster heading-backed and subtopic/heading_candidate phrases.
    Sets cluster_id, cluster_representative, cluster_label; applies mild diversity demotion.
    """
    for c in registry.values():
        c.cluster_id = None
        c.cluster_representative = None
        c.cluster_label = None

    if not config.enable_semantic_clustering:
        return []

    model = get_sentence_transformer(config.sentence_transformer_model)
    if model is None:
        logger.info("Semantic clustering skipped: no embedding model")
        return []

    pool = [
        c
        for c in registry.values()
        if c.candidate_type not in (IGNORE_BUCKET, "intent_phrase", None, "overused_terms")
        and c.score >= config.min_absolute_score
        and (
            len(c.docs_in_heading) >= config.cluster_min_heading_docs
            or c.candidate_type in ("subtopic", "heading_candidate")
        )
    ]
    pool.sort(key=lambda x: -x.score)
    pool = pool[: config.cluster_max_candidates]
    if len(pool) < config.cluster_min_candidates:
        return []

    texts = [(c.surface_form or c.normalized_form).strip() for c in pool]
    try:
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=min(64, config.semantic_batch_size),
        )
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        emb = emb / norms
    except Exception as exc:
        logger.warning("Semantic clustering encode failed: %s", exc)
        return []

    try:
        clus = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=config.cluster_distance_threshold,
            linkage="average",
            metric="cosine",
        )
        labels = clus.fit_predict(emb)
    except Exception as exc:
        logger.warning("Semantic clustering fit failed: %s", exc)
        return []

    by_label: dict[int, list[Candidate]] = {}
    for c, lab in zip(pool, labels):
        by_label.setdefault(int(lab), []).append(c)

    summaries: list[dict[str, Any]] = []
    for lab in sorted(by_label.keys()):
        members = by_label[lab]
        members.sort(key=lambda x: -x.score)
        rep = members[0]
        rep_phrase = rep.surface_form or rep.normalized_form
        cluster_conf = min(1.0, 0.55 + 0.08 * min(len(members), 8))
        for i, m in enumerate(members):
            m.cluster_id = lab
            m.cluster_representative = rep_phrase
            m.cluster_label = rep_phrase
            if i >= 1:
                m.score *= 0.94
                m.notes.append("cluster_diversity_demotion")
            note = (
                f"semantic_subtopic_cluster: id={lab}, representative={rep_phrase!r}, "
                f"cluster_size={len(members)}"
            )
            if note not in m.audit_trail:
                m.audit_trail.append(note)
        summaries.append(
            {
                "cluster_id": lab,
                "cluster_label": rep_phrase,
                "representative_keyword": rep_phrase,
                "representative_normalized": rep.normalized_form,
                "size": len(members),
                "cluster_confidence": round(cluster_conf, 4),
                "members": [
                    {
                        "keyword": m.surface_form or m.normalized_form,
                        "normalized_form": m.normalized_form,
                        "score": round(m.score, 4),
                        "keyword_type": m.candidate_type,
                    }
                    for m in members[:25]
                ],
            }
        )

    logger.info("Semantic clusters: %d from %d candidates", len(summaries), len(pool))
    return summaries
