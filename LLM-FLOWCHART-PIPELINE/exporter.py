"""Export ranked terms to keyword-prefixed JSON/CSV files.

Output files (inside each run folder), e.g. for keyword ``small dog breeds...``:

``{slug}_result.json``, ``{slug}_terms.json``, ``{slug}_terms.csv``,
and ``{slug}_topic_signals.json`` (rich signals + consolidated anchors).

``slug`` is derived via :func:`slugify_keyword`.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from signal_models import InferredCategory, TopicSignalSummary

if TYPE_CHECKING:
    from ranker import RankedTerm
    from signal_extractor import SignalBundle

logger = logging.getLogger(__name__)

_MAX_SLUG_LEN = 120


def slugify_keyword(keyword: str) -> str:
    """Filesystem-safe slug from the topic keyword (for output filenames)."""
    s = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in (keyword or "").strip()
    )
    s = "_".join(part for part in s.split("_") if part)
    if not s:
        s = "keyword"
    if len(s) > _MAX_SLUG_LEN:
        s = s[:_MAX_SLUG_LEN].rstrip("_")
    return s


# ---------------------------------------------------------------------------
# Timestamped output layout
# ---------------------------------------------------------------------------


def resolve_timestamped_output_dir(base_output_dir: str | Path) -> Path:
    """Create ``base_output_dir/run_YYYYMMDD_HHMMSS`` (or ``run_..._2`` if needed).

    Each pipeline run writes ``{slug}_result.json``, ``{slug}_terms.json``,
    and ``{slug}_terms.csv`` under this directory so repeated runs do not
    overwrite prior outputs.
    """
    base = Path(base_output_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"run_{stamp}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base / f"run_{stamp}_{suffix}"
    candidate.mkdir(parents=False)
    return candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[exporter] Wrote %s", path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        logger.info("[exporter] Wrote empty %s", path)
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("[exporter] Wrote %s (%d rows)", path, len(rows))


def disposition_from_merged_eval(
    evaluated: list[tuple[Any, Any, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split merge/support/decision triples into rejected vs demoted inspection records."""
    rejected: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for cand, ev, dec in evaluated:
        row = {
            "candidate": cand.as_dict(),
            "support_evaluation": ev.as_dict(),
            "decision": dec.as_dict(),
        }
        if dec.decision == "reject":
            rejected.append(row)
        elif dec.decision == "demote":
            demoted.append(row)
    return rejected, demoted


def build_result_summary(
    ranked: list[Any],
    inferred_categories: list[InferredCategory],
    *,
    corpus_size: int,
    llm_reject_count: int,
    llm_demote_count: int,
) -> str:
    """Short deterministic summary of why the final ranked set is strong."""
    n = len(ranked)
    strong = sum(1 for r in ranked if getattr(r, "support_label", "") == "strong")
    title_n = sum(1 for r in ranked if getattr(r, "title_alignment", False))
    head_n = sum(1 for r in ranked if getattr(r, "heading_alignment", False))
    ndim = len(inferred_categories)
    parts = [
        f"Ranked {n} keyword(s) from {corpus_size} on-topic page(s).",
        (
            f"{strong} term(s) have strong corpus support; "
            f"{title_n} show title alignment and {head_n} heading alignment."
        ),
        f"Inferred {ndim} topic dimension(s) from corpus signals.",
    ]
    if llm_reject_count or llm_demote_count:
        parts.append(
            f"Disposition: {llm_reject_count} candidate(s) rejected and "
            f"{llm_demote_count} demoted for insufficient corpus support; "
            "corpus-derived terms are preferred over generated terms when quality is similar."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_results(
    ranked: list[RankedTerm],
    signals: SignalBundle,
    inferred_categories: list[InferredCategory],
    topic_signal_summary: TopicSignalSummary | None,
    keyword: str,
    corpus_size: int,
    rejected_pages: int,
    output_dir: str | Path,
    *,
    llm_reject_count: int = 0,
    llm_demote_count: int = 0,
    rejected_keywords: list[dict[str, Any]] | None = None,
    demoted_keywords: list[dict[str, Any]] | None = None,
    output_base_dir: str | Path | None = None,
    run_log_file: str | Path | None = None,
    corpus_mode: str = "llm",
) -> dict[str, Any]:
    """Write all output files and return the result dict.

    Parameters
    ----------
    ranked:
        Final ranked list from ranker.rank_terms().
    signals:
        SignalBundle for metadata in result.json.
    inferred_categories:
        Structured categories from category_inferrer (name, confidence, rationale).
    topic_signal_summary:
        Consolidated anchors from signal_consolidator, or None.
    keyword:
        The user-supplied keyword / topic.
    corpus_size:
        Number of documents used after filtering.
    rejected_pages:
        Number of pages excluded by intent-coherence filter.
    output_dir:
        Directory to write files into (typically a per-run folder from
        ``resolve_timestamped_output_dir``).
    llm_reject_count / llm_demote_count:
        Statistics about LLM term disposition.
    rejected_keywords / demoted_keywords:
        Full inspection rows (merged-eval path or classical LLM term dicts).
    output_base_dir:
        Optional user-facing base directory (e.g. ``--output-dir``) recorded
        in the result JSON for traceability.
    run_log_file:
        Path to the per-run detailed log file if one was written.

    Returns
    -------
    dict
        The full result dict also written to ``{slug}_result.json``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = slugify_keyword(keyword)
    path_result = out / f"{slug}_result.json"
    path_terms_json = out / f"{slug}_terms.json"
    path_terms_csv = out / f"{slug}_terms.csv"
    path_topic_signals = out / f"{slug}_topic_signals.json"

    term_rows = [rt.as_dict() for rt in ranked]

    corpus_terms = [r for r in ranked if r.source == "corpus"]
    llm_corpus_ranked = [r for r in ranked if r.source == "llm_corpus"]
    llm_keep = [r for r in ranked if r.source == "llm_generated" and r.status == "keep"]
    llm_demote = [r for r in ranked if r.source == "llm_generated" and r.status == "demote"]

    dimension_names = [c.name for c in inferred_categories]

    rej = list(rejected_keywords or [])
    dem = list(demoted_keywords or [])
    summary = build_result_summary(
        ranked,
        inferred_categories,
        corpus_size=corpus_size,
        llm_reject_count=llm_reject_count,
        llm_demote_count=llm_demote_count,
    )

    topic_payload: dict[str, Any] = {
        "rich": signals.rich.as_dict() if signals.rich else None,
        "topic_signal_summary": topic_signal_summary.as_dict()
        if topic_signal_summary is not None
        else None,
    }
    _write_json(path_topic_signals, topic_payload)

    result: dict[str, Any] = {
        "pipeline": "LLM-FLOWCHART-PIPELINE",
        "corpus_mode": corpus_mode,
        "keyword": keyword,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_run_directory": str(out.resolve()),
        "corpus_size": corpus_size,
        "rejected_pages": rejected_pages,
        "inferred_dimensions": dimension_names,
        "inferred_categories": [c.as_dict() for c in inferred_categories],
        "topic_signal_summary": topic_signal_summary.as_dict()
        if topic_signal_summary is not None
        else None,
        "signals_summary": signals.as_dict(),
        "stats": {
            "total_ranked": len(ranked),
            "corpus_terms": len(corpus_terms),
            "llm_corpus_terms": len(llm_corpus_ranked),
            "llm_kept": len(llm_keep),
            "llm_demoted": len(llm_demote),
            "llm_rejected": llm_reject_count,
        },
        "summary": summary,
        "terms": term_rows,
        "rejected_keywords": rej,
        "demoted_keywords": dem,
        "keyword_slug": slug,
        "export_files": {
            "result_json": str(path_result.resolve()),
            "terms_json": str(path_terms_json.resolve()),
            "terms_csv": str(path_terms_csv.resolve()),
            "topic_signals_json": str(path_topic_signals.resolve()),
        },
    }
    if output_base_dir is not None:
        result["output_base_directory"] = str(Path(output_base_dir).resolve())
    if run_log_file is not None:
        p = str(Path(run_log_file).resolve())
        result["run_log_file"] = p
        result["export_files"]["run_log"] = p

    _write_json(path_result, result)
    _write_json(path_terms_json, term_rows)
    _write_csv(path_terms_csv, term_rows)

    return result
