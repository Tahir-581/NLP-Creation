"""CLI entry point for the LLM Flowchart Pipeline.

Usage:
    python run.py --keyword "Best guard dog breeds" --source path/to/corpus.json

    # With custom output dir and lower intent threshold:
    python run.py \\
        --keyword "small dog breeds with longest lifespans" \\
        --source  "My_Corpus/small_dogs.json" \\
        --output-dir output-flowchart \\
        --intent-threshold 0.15

Environment variables:
    LLM_API_KEY (and LLM_BASE_URL)   — LLM authentication token
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure workspace root and this directory are importable
_here = Path(__file__).resolve().parent
_workspace = _here.parent
for _p in (_workspace, _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LLM Flowchart Pipeline — signal extraction → category inference "
            "→ dual-source term generation → corpus support check → ranking.\n\n"
            "Outputs per run: under --output-dir, a subfolder run_YYYYMMDD_HHMMSS "
            "containing {keyword_slug}_result.json, {keyword_slug}_terms.json, "
            "{keyword_slug}_terms.csv, {keyword_slug}_topic_signals.json, "
            "{keyword_slug}_run.logs (no overwrites)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--keyword",
        required=True,
        help='Topic / article title, e.g. "Best guard dog breeds in the world".',
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to corpus JSON file or directory containing JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="output-flowchart",
        help="Output base directory; each run creates run_* with {slug}_result.json, {slug}_terms.* (default: output-flowchart).",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for intent-coherence filtering (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--intent-threshold",
        type=float,
        default=0.20,
        help="Cosine similarity threshold for intent filtering (default: 0.20).",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model for corpus extraction / signal extraction (default: en_core_web_sm).",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=150,
        help="Maximum number of terms in the final ranked list (default: 150).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load SentenceTransformer model only from local cache (no download).",
    )
    parser.add_argument(
        "--corpus-mode",
        choices=("llm", "classical"),
        default="llm",
        help='Term sources: "llm" = llm_corpus + llm_generated stages (default); '
        '"classical" = spaCy/YAKE corpus + legacy LLM generator.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Heavy import deferred so --help is instant
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {args.model} …")
    embedding_model = SentenceTransformer(
        args.model, local_files_only=args.local_files_only
    )

    print(f"Running LLM Flowchart Pipeline for: {args.keyword!r}")
    result = run(
        keyword=args.keyword,
        embedding_model=embedding_model,
        source=args.source,
        intent_threshold=args.intent_threshold,
        output_dir=args.output_dir,
        spacy_model=args.spacy_model,
        max_ranked_terms=args.max_terms,
        corpus_mode=args.corpus_mode,
    )

    stats = result.get("stats", {})
    base = Path(args.output_dir)
    run_dir = result.get("output_run_directory") or str(base.resolve())
    print()
    print(f"keyword:           {result.get('keyword')}")
    print(f"corpus_size:       {result.get('corpus_size')}")
    print(f"rejected_pages:    {result.get('rejected_pages')}")
    print(f"dimensions:        {result.get('inferred_dimensions')}")
    ic = result.get("inferred_categories")
    if ic:
        print(f"categories (n):    {len(ic)}")
    print(f"total_ranked:      {stats.get('total_ranked')}")
    print(f"  corpus_terms:    {stats.get('corpus_terms')}")
    if stats.get("llm_corpus_terms") is not None:
        print(f"  llm_corpus_terms: {stats.get('llm_corpus_terms')}")
    print(f"  llm_kept:        {stats.get('llm_kept')}")
    print(f"  llm_demoted:     {stats.get('llm_demoted')}")
    print(f"  llm_rejected:    {stats.get('llm_rejected')}")
    print(f"output_base_dir:   {base.resolve()}")
    print(f"output_run_dir:    {run_dir}")
    if result.get("run_log_file"):
        print(f"run_log_file:      {result['run_log_file']}")
    ef = result.get("export_files") or {}
    if ef.get("terms_csv"):
        print(f"terms_csv:         {ef['terms_csv']}")
    if ef.get("topic_signals_json"):
        print(f"topic_signals_json: {ef['topic_signals_json']}")


if __name__ == "__main__":
    main()
