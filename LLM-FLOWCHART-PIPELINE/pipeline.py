"""LLM Flowchart Pipeline — async orchestrator.

Wires every step in the flowchart:

    Title + Sources
        → Signal Extraction
        → Category Inference (LLM call 1)
        ┌─ Corpus Extraction  ──→ Candidate pool
        └─ LLM Term Generation (LLM call 2) ──→ Candidate pool
        → Corpus Support Check
        → Ranking
        → Export

Usage (programmatic):
    from pipeline import run

    result = run(
        keyword="Best guard dog breeds",
        embedding_model=some_sentence_transformer,
        source="path/to/corpus.json",
        output_dir="output-flowchart",
    )

Usage (async):
    result = await run_pipeline(keyword, source, embedding_model, output_dir=...)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure both the workspace root (for nlp_keyword_pipeline) and this
# directory (for sibling modules) are importable regardless of cwd.
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
_workspace = _here.parent
for _p in (_workspace, _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import httpx

from gpt_oss_client import DEFAULT_TIMEOUT, chat_completion

# Shared corpus utilities from nlp_keyword_pipeline
from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.filter import filter_by_intent_coherence, filter_documents
from nlp_keyword_pipeline.loader import load_corpus
from nlp_keyword_pipeline.normalizer import build_clean_documents

# Local pipeline modules
from category_inferrer import category_dimension_names, infer_categories
from corpus_extractor import extract_corpus_terms
from exporter import (
    disposition_from_merged_eval,
    export_results,
    resolve_timestamped_output_dir,
    slugify_keyword,
)
from llm_generator import generate_llm_terms
from ranker import rank_from_merged_pipeline, rank_terms
from signal_consolidator import consolidate
from signal_extractor import extract_signals
from support_checker import check_support

from nlp_keyword_pipeline.llm_keyword_stages import (
    apply_decision,
    evaluate_support_batch,
    merge_candidate_lists,
)
from nlp_keyword_pipeline.llm_keyword_stages.stage_llm import (
    expand_llm_generated_async,
    extract_llm_corpus_terms_async,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-run detailed log file ({keyword_slug}_run.logs in each timestamped folder)
# ---------------------------------------------------------------------------

_SILENCE_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
    "urllib3",
    "urllib3.connectionpool",
    "sentence_transformers",
    "transformers",
    "torch",
    "huggingface_hub",
    "filelock",
    "spacy",
    "preshed",
    "thinc",
)


def _silence_noisy_loggers() -> None:
    for name in _SILENCE_NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _boost_pipeline_loggers() -> None:
    """Ensure our stack emits DEBUG into the run log."""
    for name in (
        "pipeline",
        "category_inferrer",
        "llm_generator",
        "gpt_oss_client",
        "corpus_extractor",
        "support_checker",
        "ranker",
        "signal_extractor",
        "exporter",
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)
    logging.getLogger("nlp_keyword_pipeline").setLevel(logging.DEBUG)


def _attach_run_detail_log(log_path: Path) -> Path:
    """Attach a DEBUG-level file handler; console stays INFO. Detailed format."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    _silence_noisy_loggers()
    _boost_pipeline_loggers()

    has_stream = any(
        isinstance(h, logging.StreamHandler)
        and not getattr(h, "_flowchart_run_detail_file_handler", False)
        for h in root_logger.handlers
    )
    if not has_stream:
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        stream.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root_logger.addHandler(stream)

    for handler in list(root_logger.handlers):
        if getattr(handler, "_flowchart_run_detail_file_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    setattr(file_handler, "_flowchart_run_detail_file_handler", True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] "
            "%(filename)s:%(lineno)d %(funcName)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler) and not getattr(
            h, "_flowchart_run_detail_file_handler", False
        ):
            h.setLevel(logging.INFO)

    return log_path.resolve()


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(
    keyword: str,
    source: str | Path,
    embedding_model: Any,
    *,
    intent_threshold: float = 0.20,
    output_dir: str | Path = "output-flowchart",
    config: PipelineConfig | None = None,
    spacy_model: str = "en_core_web_sm",
    max_ranked_terms: int = 150,
    corpus_mode: str = "llm",
) -> dict[str, Any]:
    """Full async pipeline from corpus to final ranked keywords.

    Parameters
    ----------
    keyword:
        Topic / article title, e.g. "Best guard dog breeds".
    source:
        Path to corpus JSON file or directory.
    embedding_model:
        A loaded SentenceTransformer model for intent-coherence filtering.
    intent_threshold:
        Cosine similarity threshold for intent filtering (default 0.20).
    output_dir:
        Directory to write output files into.
    config:
        Optional PipelineConfig override.
    spacy_model:
        spaCy model used by corpus_extractor and signal_extractor.
    max_ranked_terms:
        Cap on the final ranked list length.
    corpus_mode:
        ``"llm"`` (default): shared ``llm_corpus`` + ``llm_generated`` stages,
        merge, corpus support evaluation, and decisions.
        ``"classical"``: spaCy/YAKE corpus extraction + legacy LLM generator
        + heuristic support checker.

    Returns
    -------
    dict
        Full result dict (also written to result.json).
    """
    cfg = config or PipelineConfig()
    anchor = keyword.strip() or cfg.topic_anchor
    cfg = replace(cfg, topic_anchor=anchor)

    # ------------------------------------------------------------------
    # Output folder + detailed run log (must be first so all steps are logged)
    # ------------------------------------------------------------------
    run_output_dir = resolve_timestamped_output_dir(output_dir)
    kw_slug = slugify_keyword(keyword)
    run_log_path = run_output_dir / f"{kw_slug}_run.logs"
    _attach_run_detail_log(run_log_path)
    logger.info(
        "[pipeline] Run output directory: %s",
        run_output_dir.resolve(),
    )
    logger.info("[pipeline] Detailed run log: %s", run_log_path.resolve())
    logger.info("=" * 72)
    logger.info("LLM-FLOWCHART-PIPELINE run start")
    logger.info("=" * 72)
    logger.debug(
        "run parameters: keyword=%r source=%r intent_threshold=%s "
        "spacy_model=%r max_ranked_terms=%s output_base_dir=%r",
        keyword,
        str(source),
        intent_threshold,
        spacy_model,
        max_ranked_terms,
        str(output_dir),
    )
    logger.debug("PipelineConfig.topic_anchor=%r", cfg.topic_anchor)

    # ------------------------------------------------------------------
    # Step 0: Load and filter corpus
    # ------------------------------------------------------------------
    logger.info("[pipeline] Loading corpus from %s", source)
    page_docs, parse_issues = load_corpus(source, cfg)
    kept, _exclusions = filter_documents(page_docs, cfg)
    clean_docs = build_clean_documents(kept)

    logger.info(
        "[pipeline] Corpus: %d raw docs, %d kept after filter, %d parse issues",
        len(page_docs),
        len(clean_docs),
        len(parse_issues),
    )
    logger.debug(
        "document ids after filter: %s",
        [d.doc_id for d in clean_docs],
    )
    if parse_issues:
        logger.debug("parse_issues (first 20): %s", parse_issues[:20])

    # ------------------------------------------------------------------
    # Step 0b: Intent-coherence filtering (same gate as other pipelines)
    # ------------------------------------------------------------------
    documents, rejected = filter_by_intent_coherence(
        clean_docs,
        query=keyword,
        embedding_model=embedding_model,
        threshold=intent_threshold,
    )
    logger.info(
        "[pipeline] Intent filter: %d kept, %d rejected",
        len(documents),
        len(rejected),
    )
    logger.debug(
        "intent kept doc_ids: %s",
        [d.doc_id for d in documents],
    )
    logger.debug(
        "intent rejected doc_ids: %s",
        [d.doc_id for d in rejected],
    )

    if not documents:
        raise ValueError(
            f"No on-topic documents found for keyword: {keyword!r}. "
            "Try lowering --intent-threshold."
        )

    # ------------------------------------------------------------------
    # Step 1: Signal extraction (H1/H2/H3, repeated nouns, NER, stats)
    # ------------------------------------------------------------------
    logger.info("[pipeline] Extracting corpus signals …")
    signals = extract_signals(keyword, documents)
    logger.info(
        "[pipeline] Signals: %d headings, %d nouns, %d entities, %d stats",
        len(signals.headings),
        len(signals.repeated_nouns),
        len(signals.named_entities),
        len(signals.stats),
    )
    logger.debug("signals.headings (full): %s", signals.headings)
    logger.debug("signals.repeated_nouns: %s", signals.repeated_nouns)
    logger.debug("signals.named_entities: %s", signals.named_entities)
    logger.debug("signals.stats (first 25): %s", signals.stats[:25])

    topic_signal_summary = consolidate(signals.rich) if signals.rich else None
    if topic_signal_summary:
        logger.debug(
            "[pipeline] Topic signal summary anchors: title=%d heading=%d concept=%d entity=%d numeric=%d",
            len(topic_signal_summary.title_anchors),
            len(topic_signal_summary.heading_anchors),
            len(topic_signal_summary.repeated_concept_anchors),
            len(topic_signal_summary.entity_anchors),
            len(topic_signal_summary.numeric_stat_anchors),
        )

    # ------------------------------------------------------------------
    # Step 2: Category inference (LLM call 1)
    # ------------------------------------------------------------------
    logger.info("[pipeline] Inferring topic categories via LLM …")
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as shared_client:
        inferred_categories = await infer_categories(
            signals,
            topic_summary=topic_signal_summary,
            client=shared_client,
        )

    dimensions = category_dimension_names(inferred_categories)
    logger.info("[pipeline] Inferred %d dimensions: %s", len(dimensions), dimensions)

    if not dimensions:
        logger.warning(
            "[pipeline] No dimensions returned by LLM; proceeding with corpus-only path."
        )

    repeated_noun_strs = [n for n, _ in signals.repeated_nouns]

    # ------------------------------------------------------------------
    # Step 3–5: Default llm path — llm_corpus → llm_generated → merge → support → rank
    # ------------------------------------------------------------------
    if corpus_mode == "llm":
        logger.info("[pipeline] corpus_mode=llm: Stage 1 llm_corpus (LLM) …")
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as llm_client:
            llm_corpus_terms = await extract_llm_corpus_terms_async(
                keyword,
                documents,
                headings=list(signals.headings),
                repeated_nouns=repeated_noun_strs,
                entities=list(signals.named_entities),
                stats=list(signals.stats),
                categories=dimensions,
                chat_completion=chat_completion,
                client=llm_client,
            )
            top_corpus = [t.term for t in llm_corpus_terms[:60]]
            logger.info("[pipeline] Stage 2 llm_generated (expansion LLM) …")
            gen_terms = await expand_llm_generated_async(
                keyword,
                headings=list(signals.headings),
                repeated_nouns=repeated_noun_strs,
                entities=list(signals.named_entities),
                stats=list(signals.stats),
                categories=dimensions,
                top_corpus_terms=top_corpus,
                chat_completion=chat_completion,
                client=llm_client,
            )

        merged_candidates = merge_candidate_lists(llm_corpus_terms, gen_terms)
        logger.info(
            "[pipeline] Merged pool: %d (llm_corpus=%d, generated=%d)",
            len(merged_candidates),
            len(llm_corpus_terms),
            len(gen_terms),
        )

        logger.info("[pipeline] Corpus support evaluation (all candidates) …")
        pairs = evaluate_support_batch(merged_candidates, documents)
        evaluated: list = []
        for cand, ev in pairs:
            dec = apply_decision(cand, ev)
            evaluated.append((cand, ev, dec))

        llm_reject = sum(1 for _c, _ev, d in evaluated if d.decision == "reject")
        llm_demote = sum(1 for _c, _ev, d in evaluated if d.decision == "demote")
        logger.info(
            "[pipeline] Decisions → reject=%d, demote=%d, kept=%d",
            llm_reject,
            llm_demote,
            len(evaluated) - llm_reject,
        )

        logger.info("[pipeline] Ranking merged pipeline …")
        ranked = rank_from_merged_pipeline(
            evaluated,
            max_terms=max_ranked_terms,
        )
        logger.info("[pipeline] Final ranked list: %d terms", len(ranked))
        rejected_kw, demoted_kw = disposition_from_merged_eval(evaluated)

    else:
        # ------------------------------------------------------------------
        # Classical: spaCy/YAKE corpus + legacy LLM generator + support_checker
        # ------------------------------------------------------------------
        logger.info("[pipeline] corpus_mode=classical: extractor stack …")
        corpus_terms = extract_corpus_terms(documents, spacy_model=spacy_model)
        logger.info("[pipeline] Corpus extraction: %d terms", len(corpus_terms))

        logger.info("[pipeline] Generating LLM terms (legacy) …")
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as llm_client:
            llm_terms = await generate_llm_terms(
                keyword, dimensions, documents, client=llm_client
            )
        logger.info("[pipeline] LLM generated: %d candidate terms", len(llm_terms))

        logger.info("[pipeline] Running corpus support check …")
        validated_llm = check_support(llm_terms, documents)

        llm_reject = sum(1 for t in validated_llm if t.support_status == "reject")
        llm_demote = sum(1 for t in validated_llm if t.support_status == "demote")
        logger.info(
            "[pipeline] LLM terms → keep=%d, demote=%d, reject=%d",
            sum(1 for t in validated_llm if t.support_status == "keep"),
            llm_demote,
            llm_reject,
        )

        logger.info("[pipeline] Ranking all terms …")
        ranked = rank_terms(corpus_terms, validated_llm, max_terms=max_ranked_terms)
        logger.info("[pipeline] Final ranked list: %d terms", len(ranked))
        rejected_kw = [t.as_dict() for t in validated_llm if t.support_status == "reject"]
        demoted_kw = [t.as_dict() for t in validated_llm if t.support_status == "demote"]

    # ------------------------------------------------------------------
    # Step 6: Export (same timestamped folder as the keyword-prefixed log file)
    # ------------------------------------------------------------------
    logger.info("[pipeline] Writing JSON/CSV to %s", run_output_dir.resolve())
    result = export_results(
        ranked=ranked,
        signals=signals,
        inferred_categories=inferred_categories,
        topic_signal_summary=topic_signal_summary,
        keyword=keyword,
        corpus_size=len(documents),
        rejected_pages=len(rejected),
        output_dir=run_output_dir,
        output_base_dir=output_dir,
        llm_reject_count=llm_reject,
        llm_demote_count=llm_demote,
        rejected_keywords=rejected_kw,
        demoted_keywords=demoted_kw,
        run_log_file=run_log_path,
        corpus_mode=corpus_mode,
    )
    logger.debug(
        "ranked term sample (first 15): %s",
        [r.term for r in ranked[:15]],
    )
    logger.info("=" * 72)
    logger.info(
        "[pipeline] Run complete. terms=%d log=%s",
        len(ranked),
        run_log_path.resolve(),
    )
    logger.info("=" * 72)

    return result


# ---------------------------------------------------------------------------
# Synchronous entry point
# ---------------------------------------------------------------------------


def run(
    keyword: str,
    embedding_model: Any,
    source: str | Path,
    *,
    intent_threshold: float = 0.20,
    output_dir: str | Path = "output-flowchart",
    config: PipelineConfig | None = None,
    spacy_model: str = "en_core_web_sm",
    max_ranked_terms: int = 150,
    corpus_mode: str = "llm",
) -> dict[str, Any]:
    """Synchronous entry point — wraps run_pipeline in asyncio.run()."""
    try:
        return asyncio.run(
            run_pipeline(
                keyword,
                source,
                embedding_model,
                intent_threshold=intent_threshold,
                output_dir=output_dir,
                config=config,
                spacy_model=spacy_model,
                max_ranked_terms=max_ranked_terms,
                corpus_mode=corpus_mode,
            )
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "[pipeline] Run failed (see *_run.logs under the run output dir if created)"
        )
        raise
