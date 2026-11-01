"""Export keywords.json, improved_* artifacts, comparison report, exclusion_report."""

from __future__ import annotations

import csv
import json
import logging
import string
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.evaluator import generate_comparison_report
from nlp_keyword_pipeline.models import Candidate

logger = logging.getLogger(__name__)

BUCKET_ORDER = [
    "primary_topic_term",
    "core_entity",
    "supporting_phrase",
    "subtopic",
    "heading_candidate",
    "intent_phrase",
    "overused_terms",
    "ignore_or_noise_terms",
]


def _display_keyword(c: Candidate) -> str:
    if not c.surface_form:
        return string.capwords(c.normalized_form)
    if fuzz.ratio(c.surface_form.lower(), c.normalized_form) < 88:
        return string.capwords(c.normalized_form)
    return c.surface_form


def _confidence_score(c: Candidate) -> float:
    if getattr(c, "confidence", 0) > 0:
        mult = {"high": 1.0, "medium": 0.98, "low": 0.95}.get(c.importance, 0.95)
        return round(min(1.0, 0.5 * c.confidence + 0.5 * c.score * mult), 4)
    mult = {"high": 1.0, "medium": 0.92, "low": 0.84}.get(c.importance, 0.84)
    method_bonus = min(0.12, 0.025 * max(0, len(c.source_methods) - 1))
    return round(min(1.0, c.score * mult + method_bonus), 4)


def _inclusion_reason(c: Candidate) -> str:
    bits: list[str] = []
    hdc = len(c.docs_in_heading)
    if hdc:
        bits.append(f"Heading evidence in {hdc} document(s)")
    tdc = len(c.docs_in_title)
    if tdc:
        bits.append(f"Title mention in {tdc} document(s)")
    sem = float(c.subscores.get("semantic_relevance", 0))
    if sem >= 0.38:
        bits.append("strong semantic alignment with topic anchor")
    elif sem >= 0.22:
        bits.append("moderate semantic relevance")
    if c.is_entity or c.from_lexicon:
        bits.append("entity or lexicon-backed term")
    if c.cluster_label:
        bits.append(f"Subtopic cluster: {c.cluster_label}")
    if "attribute_relation" in c.source_methods:
        bits.append("attribute/trait phrase extraction")
    methods = len(c.source_methods)
    if methods >= 3:
        bits.append(f"high extractor consensus ({methods} methods)")
    elif methods >= 2:
        bits.append("multi-extractor support")
    return "; ".join(bits) if bits else "Prevalence and quality-weighted corpus support"


def _candidate_to_dict(c: Candidate) -> dict[str, Any]:
    urls = list(c.example_sources)[:3]
    canon = c.canonical_form or c.normalized_form
    disp = c.display_form or _display_keyword(c)
    return {
        "keyword": disp,
        "canonical_form": canon,
        "display_form": disp,
        "normalized_form": c.normalized_form,
        "bucket": c.candidate_type,
        "keyword_type": c.candidate_type,
        "score": round(c.score, 4),
        "confidence": _confidence_score(c),
        "importance": c.importance,
        "document_prevalence": round(c.doc_prevalence, 4),
        "prevalence": round(c.doc_prevalence, 4),
        "source_relevance_weighted_prevalence": round(
            float(getattr(c, "source_relevance_weighted_prevalence", 0)), 4
        ),
        "heading_presence": len(c.docs_in_heading) > 0,
        "title_presence": len(c.docs_in_title) > 0,
        "intro_presence": len(c.docs_in_intro) > 0,
        "heading_indicator": len(c.docs_in_heading) > 0,
        "title_indicator": len(c.docs_in_title) > 0,
        "heading_doc_count": len(c.docs_in_heading),
        "title_doc_count": len(c.docs_in_title),
        "intro_doc_count": len(c.docs_in_intro),
        "extractor_hits": sorted(c.source_methods),
        "source_methods": sorted(c.source_methods),
        "supporting_documents": urls,
        "supporting_urls": urls,
        "example_sources": c.example_sources,
        "penalties": {k: round(float(v), 4) for k, v in c.penalties.items()},
        "notes": list(c.notes),
        "subscores": {k: round(float(v), 4) for k, v in c.subscores.items()},
        "semantic_anchor_similarity": round(
            float(c.subscores.get("semantic_anchor_similarity", 0)), 4
        ),
        "semantic_corpus_similarity": round(
            float(c.subscores.get("semantic_corpus_similarity", 0)), 4
        ),
        "semantic_fused_relevance": round(
            float(c.subscores.get("semantic_relevance", 0)), 4
        ),
        "audit_trail": list(c.audit_trail),
        "cluster_id": c.cluster_id,
        "cluster_representative": c.cluster_representative,
        "cluster_label": c.cluster_label,
        "inclusion_reason": _inclusion_reason(c),
        "is_entity": c.is_entity,
        "from_lexicon": c.from_lexicon,
        "is_question": c.is_question,
        "hygiene_result": c.hygiene_result,
    }


def _remove_near_dupes_ranked(items: list[Candidate], threshold: int = 92) -> list[Candidate]:
    out: list[Candidate] = []
    for c in sorted(items, key=lambda x: -x.score):
        drop = False
        for kept in out:
            if fuzz.token_sort_ratio(c.normalized_form, kept.normalized_form) >= threshold:
                drop = True
                break
        if not drop:
            out.append(c)
    return out


def export_results(
    registry: dict[str, Candidate],
    output_dir: str | Path,
    config: PipelineConfig,
    parse_issues: list[dict],
    filter_exclusions: list[dict],
    *,
    semantic_cluster_summaries: list[dict[str, Any]] | None = None,
    pre_consolidation_candidate_count: int | None = None,
    junk_rejection_count: int = 0,
    clean_docs: list[Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    caps = config.bucket_caps_dict()
    min_score = config.min_absolute_score
    ignore_bucket = "ignore_or_noise_terms"

    by_bucket: dict[str, list[Candidate]] = {b: [] for b in BUCKET_ORDER}
    for c in registry.values():
        bt = c.candidate_type or ignore_bucket
        if bt not in by_bucket:
            by_bucket[bt] = []
        by_bucket[bt].append(c)

    grouped: dict[str, list[dict[str, Any]]] = {}
    flat_selected: list[dict[str, Any]] = []

    for bucket in BUCKET_ORDER:
        items = [c for c in by_bucket.get(bucket, []) if c.score >= min_score]
        if bucket != ignore_bucket and config.remove_near_duplicates_after_ranking:
            items = _remove_near_dupes_ranked(items)
        items.sort(
            key=lambda x: (
                -x.score,
                -float(x.subscores.get("authority_support", 0)),
                -float(x.subscores.get("semantic_relevance", 0)),
            )
        )
        cap = caps.get(bucket, 50) if bucket != ignore_bucket else min(100, len(items))
        if bucket == ignore_bucket:
            cap = min(45, len(items))
        selected = items[:cap]
        grouped[bucket] = [_candidate_to_dict(c) for c in selected]
        flat_selected.extend(grouped[bucket])

    flat_all = [
        _candidate_to_dict(c)
        for c in sorted(
            registry.values(),
            key=lambda x: (
                -x.score,
                -float(x.subscores.get("authority_support", 0)),
                -float(x.subscores.get("semantic_relevance", 0)),
            ),
        )
    ]

    for path_name, data in (
        ("keywords.json", flat_selected),
        ("keywords_full.json", flat_all),
        ("keywords_grouped.json", grouped),
    ):
        (out / path_name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Improved outputs (richer schema)
    (out / "improved_keywords.json").write_text(
        json.dumps(flat_selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "improved_keywords_grouped.json").write_text(
        json.dumps(grouped, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out / "keywords.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "keyword",
                "normalized_form",
                "keyword_type",
                "score",
                "prevalence",
                "heading_doc_count",
                "title_doc_count",
                "source_methods",
            ]
        )
        for row in flat_selected:
            w.writerow(
                [
                    row["keyword"],
                    row["normalized_form"],
                    row["keyword_type"],
                    row["score"],
                    row["prevalence"],
                    row["heading_doc_count"],
                    row["title_doc_count"],
                    ";".join(row["source_methods"]),
                ]
            )

    imp_csv = out / "improved_keywords.csv"
    with imp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "keyword",
                "normalized_form",
                "bucket",
                "score",
                "confidence",
                "document_prevalence",
                "heading_doc_count",
                "title_doc_count",
                "source_methods",
                "supporting_urls",
                "notes",
            ]
        )
        for row in flat_selected:
            w.writerow(
                [
                    row["keyword"],
                    row["normalized_form"],
                    row["bucket"],
                    row["score"],
                    row["confidence"],
                    row["document_prevalence"],
                    row["heading_doc_count"],
                    row["title_doc_count"],
                    ";".join(row["source_methods"]),
                    "|".join(row["supporting_urls"]),
                    " | ".join(row["notes"]) if row["notes"] else "",
                ]
            )

    report = {
        "parse_issues": parse_issues,
        "filter_exclusions": filter_exclusions,
    }
    (out / "exclusion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    clusters = semantic_cluster_summaries or []
    (out / "semantic_subtopic_clusters.json").write_text(
        json.dumps(clusters, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "cluster_summary.json").write_text(
        json.dumps(clusters, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ignore_bucket = "ignore_or_noise_terms"
    rejected = [
        _candidate_to_dict(c)
        for c in registry.values()
        if c.candidate_type == ignore_bucket or c.hygiene_result == "hard_reject"
    ]
    (out / "rejected_candidates.json").write_text(
        json.dumps(rejected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    debug_pool = [
        _candidate_to_dict(c)
        for c in registry.values()
        if c.candidate_type != ignore_bucket
    ]
    (out / "candidate_debug_report.json").write_text(
        json.dumps(debug_pool, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    source_rows: list[dict[str, Any]] = []
    if clean_docs:
        for d in clean_docs:
            source_rows.append(
                {
                    "doc_id": d.doc_id,
                    "url": d.url,
                    "domain": d.domain,
                    "kept": True,
                    "source_relevance_score": d.metadata.get("source_relevance_score"),
                    "source_relevance_reasons": d.metadata.get("source_relevance_reasons", []),
                    "boilerplate_ratio": d.metadata.get("boilerplate_ratio"),
                }
            )
    for row in filter_exclusions:
        source_rows.append(
            {
                "doc_id": row.get("doc_id"),
                "url": row.get("url"),
                "domain": "",
                "kept": False,
                "exclusion_reason": row.get("reason"),
                "source_relevance_score": row.get("source_relevance_score"),
                "source_relevance_reasons": row.get("source_relevance_reasons", []),
            }
        )
    (out / "source_filter_report.json").write_text(
        json.dumps(source_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pre_n = pre_consolidation_candidate_count or len(registry)
    noise_bucket = sum(1 for c in registry.values() if c.candidate_type == ignore_bucket)
    attr_n = sum(1 for c in registry.values() if "attribute_relation" in c.source_methods)
    avg_conf = (
        sum(_confidence_score(c) for c in registry.values() if c.candidate_type != ignore_bucket)
        / max(1, sum(1 for c in registry.values() if c.candidate_type != ignore_bucket))
    )
    non_q = [c for c in registry.values() if c.candidate_type != ignore_bucket and not c.is_question]
    cluster_cov = (
        sum(1 for c in non_q if c.cluster_id is not None) / max(len(non_q), 1) if non_q else 0.0
    )
    generate_comparison_report(
        out,
        flat_selected,
        pre_consolidation_candidates=pre_n,
        post_consolidation_candidates=len(registry),
        junk_rejections=junk_rejection_count,
        baseline_csv=config.baseline_csv_path,
        extended_metrics={
            "noise_bucket_count": noise_bucket,
            "attribute_phrase_candidates": attr_n,
            "avg_confidence_non_ignored": round(avg_conf, 4),
            "cluster_coverage_non_intent": round(cluster_cov, 4),
        },
    )

    logger.info("Exported to %s (Layer2 surfaced=%d)", out, len(flat_selected))
    return {
        "flat_selected_count": len(flat_selected),
        "registry_count": len(registry),
    }
