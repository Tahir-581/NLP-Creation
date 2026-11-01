"""Before/after metrics and comparison_report.json."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_baseline_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def _metrics_from_flat(
    flat: list[dict[str, Any]],
    *,
    raw_pool_size: int,
    pre_consolidation_size: int,
    junk_rejections: int,
    extended: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_bucket: Counter[str] = Counter()
    lengths: list[int] = []
    heading_sup = 0
    multi_src = 0
    for row in flat:
        bt = row.get("keyword_type") or row.get("bucket") or "unknown"
        by_bucket[bt] += 1
        kw = row.get("keyword") or row.get("normalized_form") or ""
        lengths.append(len(kw.split()))
        hdc = int(row.get("heading_doc_count") or 0)
        if hdc > 0:
            heading_sup += 1
        sm = row.get("source_methods") or []
        if isinstance(sm, str):
            parts = [p for p in sm.split(";") if p.strip()]
        else:
            parts = list(sm)
        if len(parts) >= 2:
            multi_src += 1
    n = max(len(flat), 1)
    collapse = 0.0
    if pre_consolidation_size > 0:
        collapse = 1.0 - (raw_pool_size / pre_consolidation_size)
    confs: list[float] = []
    attr_export = 0
    cluster_hits = 0
    for row in flat:
        cf = row.get("confidence")
        if cf is not None:
            try:
                confs.append(float(cf))
            except (TypeError, ValueError):
                pass
        sm = row.get("source_methods") or row.get("extractor_hits") or []
        if isinstance(sm, str):
            sm = [p for p in sm.split(";") if p.strip()]
        if "attribute_relation" in sm:
            attr_export += 1
        if row.get("cluster_id") is not None and not row.get("is_question"):
            cluster_hits += 1
    non_q = max(
        1,
        sum(
            1
            for row in flat
            if row.get("keyword_type") != "intent_phrase"
            and row.get("bucket") != "intent_phrase"
        ),
    )
    out: dict[str, Any] = {
        "raw_candidate_count_pre_consolidation": pre_consolidation_size,
        "candidate_count_post_consolidation": raw_pool_size,
        "final_keyword_count": len(flat),
        "keywords_per_bucket": dict(by_bucket),
        "duplicate_collapse_rate": round(max(0.0, min(1.0, collapse)), 4),
        "junk_rejection_count": junk_rejections,
        "avg_keyword_length_tokens": round(
            sum(lengths) / n if lengths else 0.0, 4
        ),
        "pct_with_heading_support": round(100.0 * heading_sup / n, 2),
        "multi_extractor_support_rate": round(100.0 * multi_src / n, 2),
        "pct_supported_by_multiple_extractors": round(100.0 * multi_src / n, 2),
        "heading_supported_rate": round(100.0 * heading_sup / n, 2),
        "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "attribute_phrase_count": attr_export,
        "cluster_coverage": round(100.0 * cluster_hits / non_q, 2),
    }
    if extended:
        out.update(extended)
    return out


def generate_comparison_report(
    output_dir: str | Path,
    flat_selected: list[dict[str, Any]],
    *,
    pre_consolidation_candidates: int,
    post_consolidation_candidates: int,
    junk_rejections: int,
    baseline_csv: str | Path | None,
    extended_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    after = _metrics_from_flat(
        flat_selected,
        raw_pool_size=post_consolidation_candidates,
        pre_consolidation_size=pre_consolidation_candidates,
        junk_rejections=junk_rejections,
        extended=extended_metrics,
    )
    baseline_path = Path(baseline_csv) if baseline_csv else None
    before_rows = _load_baseline_csv(baseline_path)
    before = _metrics_from_flat(
        before_rows,
        raw_pool_size=len(before_rows),
        pre_consolidation_size=len(before_rows),
        junk_rejections=0,
    )
    gains: dict[str, Any] = {}
    if before_rows:
        gains = {
            "final_keyword_count_delta": after["final_keyword_count"]
            - before["final_keyword_count"],
            "avg_keyword_length_tokens_delta": round(
                after["avg_keyword_length_tokens"]
                - before["avg_keyword_length_tokens"],
                4,
            ),
            "pct_with_heading_support_delta": round(
                after["pct_with_heading_support"]
                - before["pct_with_heading_support"],
                2,
            ),
            "pct_multi_source_delta": round(
                after["pct_supported_by_multiple_extractors"]
                - before["pct_supported_by_multiple_extractors"],
                2,
            ),
        }
    report = {
        "baseline_csv": str(baseline_path) if baseline_path else None,
        "before": before if before_rows else None,
        "after": after,
        "gains": gains if gains else None,
        "run_metrics_only": not bool(before_rows),
    }
    path = out / "comparison_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", path)
    # Optional markdown summary
    md_lines = [
        "# Keyword pipeline comparison",
        "",
        f"- **Final keyword count (after):** {after['final_keyword_count']}",
        f"- **Post-consolidation pool:** {after['candidate_count_post_consolidation']}",
        f"- **Junk/hygiene rejections (this run):** {after['junk_rejection_count']}",
        f"- **% with heading support:** {after['pct_with_heading_support']}",
        f"- **% multi-extractor:** {after['pct_supported_by_multiple_extractors']}",
        "",
        "## Keywords per bucket (after)",
        "",
    ]
    for k, v in sorted(after["keywords_per_bucket"].items()):
        md_lines.append(f"- `{k}`: {v}")
    if before_rows:
        md_lines.extend(
            [
                "",
                "## vs baseline CSV",
                "",
                f"- Baseline final count: {before['final_keyword_count']}",
                f"- Delta: {gains.get('final_keyword_count_delta', 0):+d}",
            ]
        )
    (out / "comparison_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report
