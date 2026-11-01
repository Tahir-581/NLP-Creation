# NLP Keyword Generation Pipeline

Corpus-driven keyword extraction from **pre-collected** competitor page JSON (ZIP or folder). Implements the **data-already-available** workflow from `data_available_refactor/`.

## Architecture (two layers)

1. **Layer 1 — high-recall candidate pool**  
   All extractors register candidates (target roughly **hundreds to ~2.5k** unique keys before merge, tunable via `config.py`: body char limits, `pos_pattern_max_per_doc`, heading fragment caps).  
   Each hit passes **`phrase_hygiene`** (junk, malformed text, stopword-heavy fragments) unless `skip_phrase_hygiene` (e.g. question phrases).

2. **Layer 2 — consolidation, scoring, classification**  
   **RapidFuzz** merge with optional **lemma fold** near the merge threshold, **fragment demotion** for single-token substrings of stronger phrases, weighted **scoring** (incl. **source consensus**, **heading+title combo**, stronger **junk penalty**), **classification** into typed buckets, capped export.

## Pipeline stages

1. Load / validate JSON  
2. Filter (domains, blocked titles, thin pages, TF–IDF near-dupes, optional body-prefix fuzzy dupes)  
3. Normalize (zones: title, description, headings, body); registry keys strip leading **a/an/the**  
4. **Extract:** n-grams (body slice), noun chunks (char cap), NER + **domain lexicon**, **heading** miner (normalize, split, capped fragments), YAKE (body cap), **KeyBERT (MMR)** optional, questions, **`pos_pattern`** (ADJ+NOUN, NOUN+NOUN, …)  
5. Consolidate (RapidFuzz + selective lemma fold)  
6. Score: rebalanced weights + fused semantic relevance (top-N phrases embedded; rest lexical fallback) + penalties  
7. **Fragment demotion** (substring orphans)  
8. Classify: `primary_topic_term`, `core_entity`, `supporting_phrase`, `subtopic`, `heading_candidate`, `question_phrase`, **`overused_terms`**, **`ignore_or_noise_terms`**  
9. Optional **semantic clustering** of heading-backed phrases  
10. Export legacy + **`improved_*`** artifacts + **`comparison_report`**

## Setup

From the repository root (`NLP Creation`):

```bash
cd nlp_keyword_pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## Run

```bash
# From repo root (parent of nlp_keyword_pipeline)
python -m nlp_keyword_pipeline.pipeline "examples/synthetic_corpus" --topic-anchor "best guard dog breeds" --output-dir output
```

Options:

- `--include-reddit` — keep `reddit.com` pages (default: excluded).  
- `--no-keybert` — disable KeyBERT extraction.  
- `--no-clustering` — skip semantic cluster assignment / `semantic_subtopic_clusters.json`.  
- `--baseline-csv path\to\old_keywords.csv` — fills `comparison_report.json` / `.md` with before/after metrics.  
- `--output-dir`, `--work-dir`, `-v` — as before.

## Outputs (under `--output-dir`)

| File | Description |
|------|-------------|
| `keywords.json`, `keywords_grouped.json`, `keywords.csv` | Legacy flat/grouped/CSV export. |
| `keywords_full.json` | All scored candidates (uncapped). |
| **`improved_keywords.json`**, **`improved_keywords_grouped.json`**, **`improved_keywords.csv`** | Rich records: `bucket`, `confidence`, `supporting_urls`, `document_prevalence`, `notes`, etc. |
| **`comparison_report.json`**, **`comparison_report.md`** | Pool sizes, per-bucket counts, hygiene rejection count, optional delta vs `--baseline-csv`. |
| `semantic_subtopic_clusters.json` | Heading-backed semantic clusters (if enabled). |
| `exclusion_report.json` | Parse failures and filtered URLs + reasons. |

## Modules (`nlp_keyword_pipeline/`)

| Module | Role |
|--------|------|
| `phrase_hygiene.py` | Pre-registration accept/reject + rejection counter for metrics. |
| `extractors/phrase_patterns.py` | spaCy POS patterns (`pos_pattern` method). |
| `extractors/headings.py` | Normalized headings, splits, capped sliding-window fragments. |
| `consolidator.py` | Merge + `apply_fragment_demotions`. |
| `scorer.py` | Weights, semantic cap (`semantic_max_phrases_to_encode`), audit trail. |
| `classifier.py` | Buckets incl. `overused_terms`, `ignore_or_noise_terms`. |
| `evaluator.py` | Comparison metrics vs baseline CSV. |
| `exporter.py` | Legacy + improved exports + comparison report. |

## Configuration highlights (`config.py`)

- **Layer1 size:** `ngram_body_max_chars`, `noun_chunk_max_chars`, `yake_body_max_chars`, `keybert_body_max_chars`, `entity_match_max_chars`, `pos_pattern_max_per_doc`, `heading_fragment_*`, `yake_top`, `keybert_top_n`.  
- **Layer2:** `rapidfuzz_merge_threshold`, `semantic_max_phrases_to_encode`, `weights` (incl. `source_consensus`, `heading_title_combo`, `junk_penalty`), `bucket_caps` (incl. `overused_terms`), `min_absolute_score`, `source_consensus_min_methods`, `overused_prevalence_threshold`.  
- `junk_phrases`, `blocked_title_patterns`, body-prefix duplicate settings.  
- Phase 3: KeyBERT, semantic clustering, authority support, `override_*` bucket thresholds.

## Library use

```python
from pathlib import Path
from nlp_keyword_pipeline.pipeline import run_pipeline
from nlp_keyword_pipeline.config import PipelineConfig

cfg = PipelineConfig(topic_anchor="best guard dog breeds", output_dir="out")
run_pipeline(Path("examples/synthetic_corpus"), cfg)
```

## Notes

- If `sentence-transformers` cannot load, semantic similarity falls back to **lexical Jaccard** overlap with `topic_anchor`.  
- First run may download the embedding model from Hugging Face.  
- Logs include **Layer1** pre-consolidation count and **Layer2** surfaced count after caps.
