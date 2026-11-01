# GPT-OSS-120B Pipeline

LLM-based keyword extraction pipeline using the **GPT-OSS-120B** model via an OpenAI-compatible endpoint (`LLM_BASE_URL`).

This pipeline mirrors the existing premium pipeline but replaces the `qwen3-coder-next` model with `GPT-OSS-120B`. All corpus loading, filtering, and intent-coherence logic is shared with `nlp_keyword_pipeline`; only the LLM extraction layer is local to this directory.

## How It Works

```
Corpus → filter_documents → intent_coherence → LLM extraction (GPT-OSS-120B)
       → heading outlines (per page) → aggregate terms (`source: llm_corpus`)
       → gap-fill (`source: llm_generated`, free-form likely_category)
       → shared merge + corpus support + decisions (`nlp_keyword_pipeline/llm_keyword_stages`)
       → bucket output → JSON / CSV
```

1. **Corpus load & filter** — loads pages from your source directory, drops low-quality/off-topic pages
2. **Intent coherence** — uses a SentenceTransformer embedding model to keep only pages relevant to the keyword
3. **LLM extraction** — sends each page to GPT-OSS-120B for entity/term extraction (bounded concurrency)
4. **Heading outlines** — builds `h1` / `h2` / `h3` lists per page from corpus headings (see below); optional LLM pass assigns levels when all crawl headings are default level 2
5. **Aggregation** — merges per-page results, scoring by prevalence + average salience (`llm_corpus`)
6. **Gap-fill** — asks GPT-OSS-120B for additional terms with free-form `likely_category`, `why_supported`, and `supporting_signals` (not a fixed category list)
7. **Shared support gate** — `gpt_oss_finalize.apply_support_pipeline` merges aggregated + gap terms, runs heuristic corpus support and the same keep/demote/reject rules as the LLM Flowchart pipeline, then filters rows
8. **Bucketing** — organises surviving terms into: `anchor_terms`, `core_entities`, `supporting_terms`, `context_phrases`, `questions`, `suggested_additions`

## Directory Structure

```
GPT-OSS-120B-PIPELINE/
├── __init__.py           # package marker
├── heading_outline.py    # h1/h2/h3 bucketing, boilerplate filter, LLM merge helpers
├── llm_extractor.py      # async HTTP calls to GPT-OSS-120B
├── aggregator.py         # merge per-page results into ranked list
├── gpt_oss_finalize.py   # shared merge + corpus support + decisions (calls llm_keyword_stages)
├── bucket_formatter.py   # sort terms into display buckets
├── pipeline.py           # orchestration (load → filter → LLM → export)
├── run.py                # CLI entry point
├── requirements.txt      # Python dependencies
└── README.md             # this file
```

## Installation

Install dependencies (from the workspace root):

```bash
pip install -r GPT-OSS-120B-PIPELINE/requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Run from the workspace root (`d:\Tahir\NLP Creation`):

```bash
python GPT-OSS-120B-PIPELINE/run.py \
  --keyword "Best guard dog breeds" \
  --source "examples/synthetic_corpus" \
  --output-dir output-gpt-oss-120b
```

### All Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--keyword` | Yes | — | Topic/keyword phrase, e.g. `"Best guard dog breeds"` |
| `--source` | Yes | — | Corpus directory name or path |
| `--output-dir` | No | `output-gpt-oss-120b` | Where to write output files |
| `--model` | No | `all-MiniLM-L6-v2` | SentenceTransformer model for intent filtering |
| `--intent-threshold` | No | `0.20` | Cosine similarity threshold for intent coherence |
| `--local-files-only` | No | `False` | Load embedding model from local cache only |
| `--outline-llm-levels` | No | `False` | When every remaining heading is crawl-default level 2, call GPT-OSS-120B once per page to assign levels 1–3 per **verbatim** heading (validated; invalid responses fall back to all `h2`) |

## Output Files

All output goes to `--output-dir/`:

| File | Description |
|---|---|
| `result.json` | Full pipeline result with metadata, `support_gate_stats`, `page_heading_outlines`, and bucketed terms |
| `heading_outlines.json` | List of `{ doc_id, url, h1, h2, h3 }` per retained page |
| `terms.json` | Flat list of all terms with bucket name |
| `terms.csv` | Same as `terms.json` in CSV format |

## Heading hierarchy (`h1` / `h2` / `h3`)

- Outlines are derived from the corpus `headings` array after normalization in `nlp_keyword_pipeline` (`CleanDocument.heading_hierarchy`).
- **Deterministic levels:** If a heading line in the JSON corpus uses the prefix form `H1: Title`, `H2: Section`, … (see `normalizer._heading_hierarchy`), levels are respected; levels 4–6 are capped to `h3`.
- **Flat crawls:** If no level prefix is present, every heading is stored as level 2. Without `--outline-llm-levels`, those strings appear under **`h2`** only (after boilerplate/nav filtering). With `--outline-llm-levels`, the model must echo each heading string **exactly** in JSON; Python rejects paraphrases or missing rows and falls back to all-`h2` for that page.
- **Boilerplate:** Nav-style lines, promo blurbs, and phrases matching `PipelineConfig.junk_phrases` are dropped before bucketing.

## Authentication

Set the API key and endpoint via environment variables (see `.env.example`):

```bash
export LLM_BASE_URL="http://localhost:8000/v1/chat/completions"
export LLM_API_KEY="your-key-here"
```

The `Authorization: Bearer <key>` header is always sent with every request.

## Logs

Each run writes a timestamped log to `logs/gpt_oss_120b_run_<keyword>_<source>_<timestamp>.log`.
