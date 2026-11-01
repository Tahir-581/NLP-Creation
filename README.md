# NLP-Creation: SEO keyword and term generation from a competitor corpus

These pipelines take a folder (or ZIP) of competitor-page JSON documents and produce SEO-oriented keywords, phrases, entities, headings and questions, grouped into typed buckets for content planning.

## Pipelines

| Folder | What it does |
|---|---|
| `nlp_keyword_pipeline/` | Corpus-driven extraction. Load → filter (thin pages, TF-IDF near-duplicates) → normalize zones → extract (n-grams, spaCy noun chunks/NER, headings, YAKE, optional KeyBERT, questions, POS patterns) → RapidFuzz consolidation → scoring with sentence-transformer relevance → classification → export. Includes a "premium" variant with LLM keyword stages. |
| `GPT-OSS-120B-PIPELINE/` | LLM-primary extraction with GPT-OSS-120B on an OpenAI-compatible endpoint, plus heading outlines, aggregation, gap-fill and bucket formatting. Reuses the loader and filters from `nlp_keyword_pipeline`. |
| `LLM-FLOWCHART-PIPELINE/` | Signal extraction (headings, repeated nouns, NER, stats) → LLM category inference → LLM term generation → a corpus-support gate that keeps only terms backed by evidence in the sources → ranking and export. |
| `data_available_refactor/` | Design notes, a scoring spec and a pseudo-pipeline for the "data already available" workflow. |

## Setup

```bash
pip install -r nlp_keyword_pipeline/requirements.txt
python -m spacy download en_core_web_md
cp .env.example .env   # or export LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
```

## Run on the synthetic sample

`examples/synthetic_corpus/` contains two hand-written **synthetic** pages in the expected JSON format.

```bash
python -m nlp_keyword_pipeline.pipeline examples/synthetic_corpus --topic-anchor "best guard dog breeds" --output-dir output
python run_premium_pipeline_export.py --keyword "best guard dog breeds" --source examples/synthetic_corpus
```

The LLM pipelines (`GPT-OSS-120B-PIPELINE/run.py`, `LLM-FLOWCHART-PIPELINE/run.py`) need `LLM_BASE_URL` and `LLM_API_KEY` set. See each folder's README.

## Tests

```bash
pytest nlp_keyword_pipeline/tests LLM-FLOWCHART-PIPELINE/tests
```

## License

MIT
