# LLM Flowchart Pipeline

A keyword generation pipeline that combines **corpus-driven extraction** with **LLM-based term generation**, using a strict corpus-support gate to ensure every LLM output is grounded in real evidence.

## Flowchart

```
Title + Sources
    │
    ▼
Signal Extraction  ──  H1/H2/H3 headings
                   ──  Repeated nouns (spaCy)
                   ──  Named entities (NER)
                   ──  Stats / numbers
    │
    ▼
Category Inference (LLM call 1 — **GPT-OSS-120B** `gptoss120`)
  → Infer topic DIMENSIONS from signals only
  → No predefined categories
    │
    ├────────────────────────────┐
    ▼                            ▼
Stage 1: `llm_corpus`       Stage 2: `llm_generated`
(LLM — signals + corpus)    (LLM — expansion from top
 from `nlp_keyword_pipeline/   corpus terms + signals;
 llm_keyword_stages`)           no fixed categories)
    │                            │
    └────────────┬───────────────┘
                 ▼
          Merge + dedupe (`llm_corpus` wins)
                 │
                 ▼
    Corpus support eval (all candidates, heuristics)
                 │
                 ▼
    Decisions (`llm_generated`: strict; `llm_corpus`: keep unless unsupported)
                 │
                 ▼
            Final ranked terms

**Classical mode** (`--corpus-mode classical`): spaCy/YAKE `corpus_extractor` + legacy `llm_generator` + `support_checker` instead of the stages above.
```

## Rules

1. Do NOT rely on predefined categories.
2. Do NOT force the topic into generic buckets.
3. Infer topic dimensions from the title and source signals only.
4. Prefer terms explicitly present in the corpus.
5. Corpus-derived terms are treated as more reliable than free-generated terms.
6. Any LLM-generated term is checked for corpus support before inclusion.

## Directory Structure

```
LLM-FLOWCHART-PIPELINE/
├── gpt_oss_client.py     # GPT-OSS-120B HTTP client (OpenAI-compatible endpoint, LLM_BASE_URL)
├── signal_extractor.py   # Step 1: H1/H2/H3, nouns, NER, stats
├── category_inferrer.py  # Step 2: LLM → inferred dimensions (no predefined)
├── corpus_extractor.py   # Classical mode: nlp_keyword_pipeline extractor stack
├── llm_generator.py      # Classical mode: LLM anchored to dimensions
├── support_checker.py    # Classical mode: fuzzy/exact support gate
├── ranker.py             # Merge & score (incl. `rank_from_merged_pipeline`)
├── exporter.py           # Write result.json / terms.json / terms.csv
├── pipeline.py           # Async orchestrator + sync run()
├── run.py                # CLI entry point
└── requirements.txt

Shared library (workspace): `nlp_keyword_pipeline/llm_keyword_stages/` — prompts, merge,
corpus support evaluation, decisions, and async Stage 1/2 LLM calls (default path).
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set your LLM API key:

```bash
export LLM_BASE_URL=http://localhost:8000/v1/chat/completions
export LLM_API_KEY=your-key-here
```

## Usage

```bash
python run.py \
    --keyword "Best guard dog breeds" \
    --source  "path/to/corpus.json" \
    --output-dir output-flowchart
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--keyword` | *(required)* | Topic / article title |
| `--source` | *(required)* | Corpus JSON file or directory |
| `--output-dir` | `output-flowchart` | Output directory |
| `--model` | `all-MiniLM-L6-v2` | SentenceTransformer model for intent filtering |
| `--intent-threshold` | `0.20` | Cosine similarity cutoff for intent filtering |
| `--spacy-model` | `en_core_web_sm` | spaCy model for extraction |
| `--max-terms` | `150` | Max terms in ranked output |
| `--corpus-mode` | `llm` | `llm` = shared `llm_corpus` + `llm_generated` stages + merge + support + decisions; `classical` = spaCy/YAKE + legacy LLM + `support_checker` |
| `--local-files-only` | `false` | Use only locally cached models |

## Output Files

Each run creates a **timestamped subfolder** under your `--output-dir`, e.g. `output-flowchart/run_20260327_192530/`, so files are never overwritten. Inside that folder:

| File | Description |
|---|---|
| `{keyword_slug}_run.logs` | **Detailed DEBUG-level run log** (`keyword_slug` = filesystem-safe form of `--keyword`, e.g. `small_dog_breeds_with_longest_lifespans`). |
| `{keyword_slug}_result.json` | Full result: metadata, signals, dimensions, all ranked terms |
| `{keyword_slug}_terms.json` | Flat list of ranked term objects |
| `{keyword_slug}_terms.csv` | CSV version of terms.json (dynamic columns) |

Example for `--keyword "small dog breeds with longest lifespans"`:  
`small_dog_breeds_with_longest_lifespans_terms.csv`, `small_dog_breeds_with_longest_lifespans_result.json`, etc.

`result.json` also includes `keyword_slug`, `export_files` (paths to each written file), `output_base_directory`, `output_run_directory`, and `run_log_file`.

## Support Thresholds

Adjust in `support_checker.py`:

| Constant | Default | Meaning |
|---|---|---|
| `STRONG_DOC_COUNT` | `2` | Exact match in N+ docs → Strong |
| `FUZZY_WEAK_THRESHOLD` | `82` | RapidFuzz partial_ratio ≥ this → Weak |
| `FUZZY_STRONG_THRESHOLD` | `95` | RapidFuzz partial_ratio ≥ this → Strong |

## Scoring Formulas

**Corpus terms** (always Keep):
```
score = 0.65 * prevalence_rate + 0.35 * zone_bonus
```

**LLM terms — Strong (Keep)**:
```
score = 0.50 * salience + 0.40 * support_score + 0.10 * zone_bonus
```

**LLM terms — Weak (Demote)**:
```
score = (0.50 * salience + 0.40 * support_score + 0.10 * zone_bonus) * 0.55
```

## LLM backend

Both **Infer Categories** and **LLM Term Generation** call **GPT-OSS-120B** at the OpenAI-compatible endpoint in `LLM_BASE_URL` with model name `gptoss120` (same as [GPT-OSS-120B-PIPELINE](../GPT-OSS-120B-PIPELINE/llm_extractor.py)). Requests use `response_format: json_object` and `reasoning_effort: low`.

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible chat completions URL serving GPT-OSS-120B |
| `LLM_API_KEY` | API key (fallback: `OPENAI_API_KEY`) |
| `LLM_MODEL` | Model name (default `gptoss120`) |
