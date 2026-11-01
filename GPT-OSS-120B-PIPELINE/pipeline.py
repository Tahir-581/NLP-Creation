"""GPT-OSS-120B tier: LLM-primary extraction via an OpenAI-compatible endpoint, aggregation, gap-fill, buckets."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import sys
from pathlib import Path as _Path

# Ensure workspace root is on sys.path for nlp_keyword_pipeline,
# and this directory is on sys.path for local sibling imports.
_here = _Path(__file__).resolve().parent
_workspace = _here.parent
for _p in (_workspace, _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import httpx

from aggregator import aggregate_llm_terms
from gpt_oss_finalize import apply_support_pipeline
from llm_extractor import (
    extract_all_pages_async,
    extract_page_heading_outlines_async,
    generate_gap_terms_async,
)
from bucket_formatter import format_llm_output

# Shared non-LLM utilities from the existing pipeline package
from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.filter import filter_by_intent_coherence, filter_documents
from nlp_keyword_pipeline.loader import load_corpus
from nlp_keyword_pipeline.normalizer import build_clean_documents

logger = logging.getLogger(__name__)


def _attach_run_file_logger(keyword: str, source: str | Path) -> Path:
    """Attach a per-run file handler under logs/ for GPT-OSS-120B runs."""
    keyword_slug = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in keyword.strip()
    ) or "keyword"
    source_stem = Path(source).stem or "corpus"
    source_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = Path("logs") / f"gpt_oss_120b_run_{keyword_slug}_{source_slug}_{timestamp}.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    level = root_logger.level or logging.INFO
    root_logger.setLevel(level)
    if not root_logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root_logger.addHandler(stream)

    for handler in list(root_logger.handlers):
        if getattr(handler, "_gpt_oss_run_file_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(run_log, encoding="utf-8")
    setattr(file_handler, "_gpt_oss_run_file_handler", True)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)
    return run_log


async def run_pipeline(
    keyword: str,
    source: str | Path,
    embedding_model: Any,
    intent_threshold: float = 0.20,
    config: PipelineConfig | None = None,
    *,
    outline_llm_levels: bool = False,
) -> dict[str, Any]:
    """
    Corpus load → standard filter → clean docs → intent coherence →
    parallel LLM extraction (GPT-OSS-120B) → aggregate → gap-fill → bucketed output.
    """
    cfg = config or PipelineConfig()
    anchor = keyword.strip() or cfg.topic_anchor
    cfg = replace(cfg, topic_anchor=anchor)

    page_docs, parse_issues = load_corpus(source, cfg)
    kept, _exclusions = filter_documents(page_docs, cfg)
    clean_docs = build_clean_documents(kept)

    documents, rejected = filter_by_intent_coherence(
        clean_docs,
        query=keyword,
        embedding_model=embedding_model,
        threshold=intent_threshold,
    )
    logger.info(
        "[pipeline] Corpus: %d kept, %d rejected (intent), parse_issues=%d",
        len(documents),
        len(rejected),
        len(parse_issues),
    )

    if not documents:
        raise ValueError(f"No on-topic documents found for keyword: {keyword!r}")

    per_page_results = await extract_all_pages_async(keyword, documents)
    logger.info(
        "[pipeline] Extracted from %d pages",
        len([p for p in per_page_results if p]),
    )

    heading_outline_dicts = await extract_page_heading_outlines_async(
        documents,
        outline_llm_levels=outline_llm_levels,
        config=cfg,
    )
    page_heading_outlines: list[dict[str, Any]] = []
    for doc, od in zip(documents, heading_outline_dicts):
        page_heading_outlines.append(
            {
                "doc_id": doc.doc_id,
                "url": doc.url,
                "h1": od["h1"],
                "h2": od["h2"],
                "h3": od["h3"],
            }
        )

    aggregated = aggregate_llm_terms(
        per_page_results,
        total_docs=len(documents),
    )
    logger.info(
        "[pipeline] %d unique terms after aggregation",
        len(aggregated),
    )

    gap_timeout = httpx.Timeout(300.0, connect=8.0)
    async with httpx.AsyncClient(timeout=gap_timeout) as client:
        gap_terms = await generate_gap_terms_async(
            keyword,
            extracted_terms=[t["term"] for t in aggregated[:60]],
            client=client,
        )
    logger.info("[pipeline] %d gap-fill terms (raw)", len(gap_terms))

    agg_f, gap_f, support_stats = apply_support_pipeline(
        aggregated,
        gap_terms,
        documents,
    )
    logger.info(
        "[pipeline] After shared support gate: aggregated=%d gap=%d stats=%s",
        len(agg_f),
        len(gap_f),
        support_stats,
    )

    buckets = format_llm_output(agg_f, gap_f)

    return {
        "keyword": keyword,
        "model": "GPT-OSS-120B",
        "corpus_size": len(documents),
        "rejected_pages": len(rejected),
        "total_terms": len(agg_f) + len(gap_f),
        "support_gate_stats": support_stats,
        "buckets": buckets,
        "page_heading_outlines": page_heading_outlines,
    }


def run(
    keyword: str,
    embedding_model: Any,
    source: str | Path,
    *,
    intent_threshold: float = 0.20,
    config: PipelineConfig | None = None,
    outline_llm_levels: bool = False,
) -> dict[str, Any]:
    """Synchronous entry point."""
    run_log = _attach_run_file_logger(keyword, source)
    logger.info("Run log: %s", run_log.resolve())
    return asyncio.run(
        run_pipeline(
            keyword,
            source,
            embedding_model,
            intent_threshold=intent_threshold,
            config=config,
            outline_llm_levels=outline_llm_levels,
        )
    )
