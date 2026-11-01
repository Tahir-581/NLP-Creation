"""End-to-end keyword pipeline orchestration."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import sys
from pathlib import Path

import yake

from nlp_keyword_pipeline.classifier import classify_all
from nlp_keyword_pipeline.clustering import assign_semantic_clusters
from nlp_keyword_pipeline.config import PipelineConfig
from nlp_keyword_pipeline.consolidator import (
    apply_fragment_demotions,
    consolidate_registry,
)
from nlp_keyword_pipeline.exporter import export_results
from nlp_keyword_pipeline.extractors.registry_ops import extract_all_for_document
from nlp_keyword_pipeline.filter import filter_documents
from nlp_keyword_pipeline.loader import load_corpus
from nlp_keyword_pipeline.models import Candidate
from nlp_keyword_pipeline.normalizer import build_clean_documents
from nlp_keyword_pipeline.phrase_hygiene import (
    get_rejection_count,
    reset_rejection_counter,
)
from nlp_keyword_pipeline.scorer import score_candidates
from nlp_keyword_pipeline.semantic_service import get_sentence_transformer
from nlp_keyword_pipeline.topic_anchor import build_topic_anchor

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _make_keybert(config: PipelineConfig):
    if not config.enable_keybert:
        return None
    try:
        from keybert import KeyBERT

        st = get_sentence_transformer(config.sentence_transformer_model)
        if st is not None:
            kb = KeyBERT(model=st)
            logger.info(
                "KeyBERT bound to shared SentenceTransformer (%s)",
                config.sentence_transformer_model,
            )
        else:
            kb = KeyBERT(model=config.sentence_transformer_model)
            logger.info(
                "KeyBERT initialized with model string %s",
                config.sentence_transformer_model,
            )
        return kb
    except Exception as exc:
        logger.warning("KeyBERT disabled (%s)", exc)
        return None


def run_pipeline(
    source: str | Path,
    config: PipelineConfig | None = None,
    *,
    output_dir: str | None = None,
) -> dict:
    """
    Run the full pipeline: load → filter → normalize → topic anchor → extract
    → semantic expansion → PAA → consolidate → score → demote → classify
    → cluster → export → comparison report.
    """
    config = config or PipelineConfig()
    if output_dir is not None:
        config.output_dir = output_dir

    page_docs, parse_issues = load_corpus(source, config)
    kept, exclusions = filter_documents(page_docs, config)
    clean_docs = build_clean_documents(kept)

    import spacy

    nlp = spacy.load("en_core_web_md")
    st_model = get_sentence_transformer(config.sentence_transformer_model)
    topic_anchor = build_topic_anchor(config.topic_anchor, config, nlp, st_model)

    yake_kw = yake.KeywordExtractor(
        lan="en",
        n=config.yake_n_max,
        top=config.yake_top,
        dedupLim=0.7,
    )
    keybert_model = _make_keybert(config)

    reset_rejection_counter()
    registry: dict[str, Candidate] = {}
    for doc in clean_docs:
        extract_all_for_document(doc, nlp, yake_kw, keybert_model, registry, config)

    from nlp_keyword_pipeline.extractors import semantic_expansion  # noqa: PLC0415

    corpus_lower = " ".join(d.full_clean_text.lower() for d in clean_docs)
    semantic_expansion.run(
        registry,
        nlp,
        config,
        corpus_lower=corpus_lower,
        n_docs=len(clean_docs),
    )

    from nlp_keyword_pipeline.extractors.questions import generate_intent_phrases  # noqa: PLC0415

    generate_intent_phrases(registry, config)

    layer1_pool = len(registry)
    logger.info("Layer1 high-recall pool: %d candidates (pre-consolidation)", layer1_pool)

    registry = consolidate_registry(registry, config, nlp)

    score_candidates(registry, clean_docs, config, topic_anchor=topic_anchor)
    apply_fragment_demotions(registry, config)
    classify_all(registry, config, topic_anchor=topic_anchor)
    cluster_summaries = assign_semantic_clusters(registry, config)

    out_path = Path(config.output_dir)
    export_stats = export_results(
        registry,
        out_path,
        config,
        parse_issues,
        exclusions,
        semantic_cluster_summaries=cluster_summaries,
        pre_consolidation_candidate_count=layer1_pool,
        junk_rejection_count=get_rejection_count(),
        clean_docs=clean_docs,
    )

    logger.info(
        "Layer2 surfaced keywords (capped export): %d",
        export_stats.get("flat_selected_count", 0),
    )

    return {
        "retained_documents": len(clean_docs),
        "parse_issues": len(parse_issues),
        "filter_exclusions": len(exclusions),
        "layer1_pre_consolidation_candidates": layer1_pool,
        "candidates_post_consolidation": len(registry),
        "semantic_clusters": len(cluster_summaries),
        "flat_selected_count": export_stats.get("flat_selected_count", 0),
        "junk_hygiene_rejections": get_rejection_count(),
        "output_dir": str(out_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLP keyword generation from JSON page corpus (data-already-available workflow)."
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to folder of JSON files or a .zip archive",
    )
    parser.add_argument(
        "--topic-anchor",
        type=str,
        default="best guard dog breeds",
        help="Topic phrase for semantic relevance / fallback",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for exported files",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work",
        help="Working directory for ZIP extraction",
    )
    parser.add_argument(
        "--include-reddit",
        action="store_true",
        help="Do not exclude reddit.com by default",
    )
    parser.add_argument(
        "--no-keybert",
        action="store_true",
        help="Disable KeyBERT extraction",
    )
    parser.add_argument(
        "--no-clustering",
        action="store_true",
        help="Disable semantic subtopic clustering export",
    )
    parser.add_argument(
        "--baseline-csv",
        type=str,
        default=None,
        help="Path to baseline keywords.csv for comparison_report.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    source_stem = Path(args.source).stem or "corpus"
    source_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = Path("logs") / f"standard_run_{source_slug}_{timestamp}.log"
    _setup_logging(args.verbose, run_log)
    logger.info("Run log: %s", run_log.resolve())
    cfg = PipelineConfig(
        topic_anchor=args.topic_anchor,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        exclude_reddit_by_default=not args.include_reddit,
        enable_keybert=not args.no_keybert,
        enable_semantic_clustering=not args.no_clustering,
        baseline_csv_path=args.baseline_csv,
    )
    summary = run_pipeline(args.source, cfg)
    logger.info("Done: %s", summary)


if __name__ == "__main__":
    main()
