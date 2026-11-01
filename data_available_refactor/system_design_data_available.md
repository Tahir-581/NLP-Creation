# System Design — Keyword Generation from Existing JSON Documents

## System objective

Build a pipeline that accepts a ZIP of already-collected page JSON files and outputs a structured set of keyword suggestions grouped by type. The system should be practical to implement, robust to noisy files, and explainable enough for later editor or scoring integration.

## Input assumptions

The input is a ZIP file containing topic-specific JSON page documents. A typical document includes:

- `url`
- `title`
- `description`
- `headings`
- `paragraphs`
- `word_count`
- `heading_count`
- `para_count`
- `authority`
- `content`

Some documents may instead include:

- `error`
- incomplete fields
- blocked/access-denied content
- social or video pages

## Top-level pipeline

1. ZIP ingestion
2. JSON validation
3. document filtering
4. normalization and structural parsing
5. candidate generation
6. candidate consolidation
7. candidate scoring
8. candidate classification
9. final selection and export

## Data structures

### RawDocument

```python
RawDocument = {
  "file_path": str,
  "payload": dict
}
```

### PageDocument

```python
PageDocument = {
  "doc_id": str,
  "url": str,
  "domain": str,
  "title": str,
  "description": str,
  "headings": list[str],
  "paragraphs": list[str],
  "content": str,
  "word_count": int | None,
  "authority": int | None,
}
```

### CleanDocument

```python
CleanDocument = {
  "doc_id": str,
  "url": str,
  "domain": str,
  "title_text": str,
  "description_text": str,
  "heading_texts": list[str],
  "paragraph_texts": list[str],
  "intro_text": str,
  "body_text": str,
  "full_clean_text": str,
  "metadata": {...}
}
```

### Candidate

```python
Candidate = {
  "surface_form": str,
  "normalized_form": str,
  "source_methods": list[str],
  "source_docs": list[str],
  "title_hits": int,
  "heading_hits": int,
  "body_hits": int,
  "doc_prevalence": float,
  "semantic_score": float | None,
  "score": float,
  "candidate_type": str | None,
}
```

## Stage 1 — ZIP ingestion

### Responsibilities
- unpack ZIP to a working directory
- enumerate `.json` files
- create file-level metadata

### Output
A list of raw JSON file payloads.

## Stage 2 — JSON validation

### Responsibilities
- parse JSON safely
- validate required fields
- detect documents with only an `error` field
- log invalid or partial documents

### Required minimum for continued processing
A document must provide at least:
- `url`
- one meaningful text source among `title`, `headings`, `paragraphs`, or `content`

### Output
A set of normalized `PageDocument` objects plus a validation log.

## Stage 3 — document filtering

This is a first-class stage.

### Filtering rules
Exclude a document if any of the following are true:

- domain is social/video by default (`facebook.com`, `youtube.com`)
- payload contains only an error or blocked-page pattern
- title contains known blocked/access-denied markers
- word count is below a threshold (recommended default: 250)
- there are fewer than 2 meaningful headings and very little body text
- content is mostly boilerplate or platform UI text
- the document is a near-duplicate of a stronger retained document

### Near-duplicate handling
Use a two-step approach:
1. normalized-content hash for exact duplicates
2. TF-IDF cosine or MinHash similarity for near-duplicates

### Output
A retained competitor corpus plus an exclusion report.

## Stage 4 — normalization and structural parsing

### Responsibilities
- preserve title, description, headings, and paragraphs as separate zones
- normalize whitespace and punctuation
- create lowercase and original-form views
- tokenize and lemmatize where useful
- keep multi-word phrases intact when possible
- build an intro/body split from paragraphs

### Important rule
Do **not** over-normalize proper nouns or multi-word entities.

### Output
`CleanDocument` objects with structured text zones.

## Stage 5 — candidate generation

Use multiple generators in parallel.

### 5.1 N-gram generator
- unigrams, bigrams, trigrams
- stopword and punctuation filters
- minimum phrase-quality rules

### 5.2 Noun chunk generator
- spaCy noun chunks
- good for descriptive and explanatory phrases

### 5.3 Named entity generator
- spaCy NER
- enhanced with domain lexicon support where generic NER misses topic entities

### 5.4 Heading phrase miner
- extract heading strings
- derive recurring subtopic phrases
- detect heading-level question forms

### 5.5 YAKE keyphrase generator
- run per document or over weighted text slices
- aggregate retained phrases across documents

### 5.6 Question extractor
- detect interrogative headings and paragraph sentences
- useful for question-phrase output

### Output
A raw candidate pool with provenance by method and document.

## Stage 6 — candidate consolidation

### Responsibilities
- canonicalize surface variants
- merge singular/plural variants
- merge whitespace/punctuation variants
- retain best display form
- accumulate provenance across extraction methods
- optionally merge fuzzy variants with RapidFuzz

### Examples
- `guard dog` and `guard dogs`
- `German Shepherd` and `German Shepherds`
- closely overlapping heading phrases

### Output
A consolidated candidate registry.

## Stage 7 — candidate scoring

### Positive signals
- document prevalence
- heading presence
- title presence
- phrase quality
- entity bonus
- semantic relevance to topic
- source-method agreement

### Negative signals
- redundancy penalty
- junk penalty
- overly generic phrase penalty

### Recommended formula

```text
score =
  0.25 * document_prevalence
+ 0.20 * heading_presence
+ 0.15 * title_presence
+ 0.15 * phrase_quality
+ 0.10 * entity_bonus
+ 0.10 * semantic_relevance
+ 0.05 * method_agreement
- 0.05 * redundancy_penalty
- 0.05 * junk_penalty
```

Exact weights can be tuned, but the design should remain interpretable.

## Stage 8 — candidate classification

After scoring, classify candidates into typed outputs.

### Recommended buckets
- `primary_topic_term`
- `core_entity`
- `supporting_phrase`
- `subtopic`
- `heading_candidate`
- `question_phrase`
- `ignore_noise`

### Classification signals
- phrase content
- structural position
- heading frequency
- question form
- entity flags
- prevalence and specificity

## Stage 9 — final selection and export

### Selection policy
- enforce minimum document prevalence thresholds by bucket
- cap output counts by bucket
- remove near-duplicates
- keep example source docs for traceability

### Export formats
Recommended outputs:
- JSON for downstream systems
- Markdown summary for human review

## Cacheable boundaries

Even in this narrower workflow, some boundaries are worth caching:

- parsed JSON documents
- filtered corpus list
- spaCy doc objects
- candidate registry before final scoring
- final export artifacts

These caches help re-run scoring and tuning without repeating expensive extraction steps.

## Synchronous vs deferred work

### Synchronous in the first implementation
- unzip and parse
- filtering
- normalization
- extraction
- consolidation
- scoring
- export

### Deferred / optional
- heavy semantic clustering
- expensive model-based expansion
- live UI rescoring loop
- multi-topic batch orchestration

## Success criteria

The system is successful when it:

1. reliably excludes unusable pages
2. produces clean typed keyword buckets
3. surfaces topic-relevant phrases and entities with traceable reasons
4. avoids generic junk and platform contamination
5. can be rerun and tuned easily during implementation
