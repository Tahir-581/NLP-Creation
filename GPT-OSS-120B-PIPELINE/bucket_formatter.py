"""Bucket LLM terms by category for display."""

from __future__ import annotations

from typing import Any

BUCKET_ORDER = [
    "anchor_terms",
    "core_entities",
    "supporting_terms",
    "context_phrases",
    "questions",
    "suggested_additions",
]

CATEGORY_TO_BUCKET: dict[str, str] = {
    "anchor_term": "anchor_terms",
    "health_condition": "core_entities",
    "care_routine": "core_entities",
    "longevity_factor": "core_entities",
    "breed_name": "supporting_terms",
    "temporal": "supporting_terms",
    "lifestyle": "supporting_terms",
    "breed_trait": "context_phrases",
    "emotional": "context_phrases",
    "quality_of_life": "context_phrases",
    "question_phrase": "questions",
    "other": "supporting_terms",
}


def format_llm_output(
    aggregated_terms: list[dict[str, Any]],
    gap_terms: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Merge corpus-backed and generated terms into display-ready buckets.
    Each bucket is sorted by score descending.
    """
    buckets: dict[str, list[dict[str, Any]]] = {b: [] for b in BUCKET_ORDER}

    for term in aggregated_terms:
        bucket = CATEGORY_TO_BUCKET.get(term.get("category", "other"), "supporting_terms")
        buckets[bucket].append(term)

    for term in gap_terms:
        buckets["suggested_additions"].append(term)

    for bucket in buckets:
        buckets[bucket].sort(key=lambda x: float(x.get("score", 0)), reverse=True)

    return buckets
