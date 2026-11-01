"""Aggregate per-page LLM extraction results into a ranked list."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def aggregate_llm_terms(
    per_page_results: list[list[dict[str, Any]]],
    total_docs: int,
    prevalence_weight: float = 0.6,
    salience_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Aggregate per-page LLM extraction results into a single ranked list.

    score = (prevalence * prevalence_weight) + (avg_salience * salience_weight)
    """
    if total_docs <= 0:
        return []

    term_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "salience_sum": 0.0,
            "doc_count": 0,
            "categories": [],
        }
    )

    for page_terms in per_page_results:
        seen_this_page: set[str] = set()
        for entry in page_terms:
            key = (entry.get("normalized") or entry.get("term") or "").strip().lower()
            if not key:
                continue
            if key in seen_this_page:
                continue
            seen_this_page.add(key)
            td = term_data[key]
            try:
                sal = float(entry.get("salience", 0.5))
            except (TypeError, ValueError):
                sal = 0.5
            td["salience_sum"] += sal
            td["doc_count"] += 1
            td["categories"].append(str(entry.get("category", "other")))

    aggregated: list[dict[str, Any]] = []
    for term, data in term_data.items():
        doc_count = int(data["doc_count"])
        prevalence = doc_count / total_docs
        avg_salience = data["salience_sum"] / doc_count
        final_score = (prevalence * prevalence_weight) + (
            avg_salience * salience_weight
        )
        category = Counter(data["categories"]).most_common(1)[0][0]

        aggregated.append(
            {
                "term": term,
                "category": category,
                "prevalence": round(prevalence, 3),
                "avg_salience": round(avg_salience, 3),
                "score": round(final_score, 4),
                "doc_count": doc_count,
                "source": "llm_corpus",
            }
        )

    return sorted(aggregated, key=lambda x: x["score"], reverse=True)
