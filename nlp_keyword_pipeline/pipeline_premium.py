"""Premium tier: LLM-primary extraction, aggregation, gap-fill, buckets."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from nlp_keyword_pipeline.aggregation.llm_aggregator import aggregate_llm_terms
from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.extractors.llm_extractor import (
    extract_all_pages_async,
    generate_gap_terms_async,
)
from nlp_keyword_pipeline.filter import filter_by_intent_coherence, filter_documents
from nlp_keyword_pipeline.loader import load_corpus
from nlp_keyword_pipeline.normalizer import build_clean_documents
from nlp_keyword_pipeline.output.bucket_formatter import format_llm_output

logger = logging.getLogger(__name__)


def _attach_premium_run_file_logger(keyword: str, source: str | Path) -> Path:
    """Attach a per-run file handler under logs/ for premium runs."""
    keyword_slug = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in keyword.strip()
    ) or "keyword"
    source_stem = Path(source).stem or "corpus"
    source_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = Path("logs") / f"premium_run_{keyword_slug}_{source_slug}_{timestamp}.log"
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
        if getattr(handler, "_premium_run_file_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(run_log, encoding="utf-8")
    setattr(file_handler, "_premium_run_file_handler", True)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)
    return run_log


async def run_premium_pipeline(
    keyword: str,
    source: str | Path,
    embedding_model: Any,
    intent_threshold: float = 0.20,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """
    Corpus load → standard filter → clean docs → intent coherence →
    parallel LLM extraction → aggregate → gap-fill → bucketed output.
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
        "[pipeline_premium] Corpus: %d kept, %d rejected (intent), parse_issues=%d",
        len(documents),
        len(rejected),
        len(parse_issues),
    )

    if not documents:
        raise ValueError(f"No on-topic documents found for keyword: {keyword!r}")

    per_page_results = await extract_all_pages_async(keyword, documents)
    logger.info(
        "[pipeline_premium] Extracted from %d pages",
        len([p for p in per_page_results if p]),
    )

    aggregated = aggregate_llm_terms(
        per_page_results,
        total_docs=len(documents),
    )
    logger.info(
        "[pipeline_premium] %d unique terms after aggregation",
        len(aggregated),
    )

    gap_timeout = httpx.Timeout(300.0, connect=8.0)
    async with httpx.AsyncClient(timeout=gap_timeout) as client:
        gap_terms = await generate_gap_terms_async(
            keyword,
            extracted_terms=[t["term"] for t in aggregated[:60]],
            client=client,
        )
    logger.info("[pipeline_premium] %d gap-fill terms", len(gap_terms))

    buckets = format_llm_output(aggregated, gap_terms)

    return {
        "keyword": keyword,
        "corpus_size": len(documents),
        "rejected_pages": len(rejected),
        "total_terms": len(aggregated) + len(gap_terms),
        "buckets": buckets,
    }


def run(
    keyword: str,
    embedding_model: Any,
    source: str | Path,
    *,
    intent_threshold: float = 0.20,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Synchronous entry point."""
    run_log = _attach_premium_run_file_logger(keyword, source)
    logger.info("Run log: %s", run_log.resolve())
    return asyncio.run(
        run_premium_pipeline(
            keyword,
            source,
            embedding_model,
            intent_threshold=intent_threshold,
            config=config,
        )
    )
