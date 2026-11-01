# Implementation Plan — Data Already Available Workflow

## Goal

Implement a practical keyword-generation pipeline that starts from the provided ZIP corpus and produces structured keyword outputs suitable for immediate use in development.

## Build strategy

The implementation should be staged by quality risk, not by product ambition. Because the data already exists, the highest-value work is to make extraction reliable, explainable, and easy to tune.

## Phase 1 — Core working pipeline

### Objective
Build the smallest complete pipeline that can process the ZIP end to end and export useful typed keyword outputs.

### Deliverables
- ZIP loader
- JSON parser and validator
- document exclusion report
- cleaned retained corpus
- candidate extraction from multiple methods
- candidate consolidation
- scoring and classification
- final JSON and Markdown export

### Recommended stack
- Python 3.11+
- `spacy`
- `scikit-learn`
- `yake`
- `rapidfuzz`
- `orjson` or built-in `json`

### Step-by-step implementation

1. **Create corpus loader**
   - unzip to workspace
   - enumerate JSON files
   - parse safely
   - attach file metadata

2. **Implement schema validation**
   - validate expected fields
   - flag malformed files
   - keep partial-but-usable docs when possible

3. **Implement filtering rules**
   - exclude social/video/blocked/error pages
   - exclude thin docs
   - exclude duplicate/near-duplicate docs
   - write exclusions to a report for review

4. **Implement structured normalization**
   - preserve title, description, headings, paragraphs, content separately
   - build clean text zones
   - build lowercase + display-form views

5. **Implement extraction methods**
   - n-grams
   - noun chunks
   - NER
   - heading mining
   - YAKE
   - question detection

6. **Implement candidate registry and consolidation**
   - canonicalize variants
   - merge surface forms
   - accumulate provenance

7. **Implement scoring**
   - start with the JSON scoring spec in this pack
   - compute prevalence and structural-position metrics
   - apply penalties

8. **Implement classification and export**
   - assign bucket types
   - apply selection caps
   - export machine-readable results and a human-readable review file

### Manual validation required after Phase 1
- inspect exclusion report
- inspect top candidates by bucket
- verify that junk terms are excluded
- verify that real entities and heading phrases are surfaced
- inspect near-duplicate handling

## Phase 2 — Quality layer

### Objective
Improve candidate quality without radically changing the architecture.

### Additions
- domain lexicon support for breed names / topic entities
- better boilerplate blacklist rules
- stronger duplicate detection thresholds
- bucket-specific thresholds
- optional authority-aware tie-breaking
- per-candidate audit trails showing exactly why a term ranked highly

### Why this phase matters
The difference between a mediocre keyword generator and a useful one often comes from consolidation and filtering quality, not from adding bigger models.

## Phase 3 — Semantic enhancement layer

### Objective
Improve relevance and grouping where lexical methods alone are too shallow.

### Additions
- sentence-transformer similarity to a topic anchor phrase
- semantic grouping of heading/subtopic candidates
- optional KeyBERT for additional phrase recall
- optional cluster summaries for subtopics

### Recommended priority
Pull in sentence-transformer similarity early if team bandwidth allows. It is the most useful “v2-style” feature to promote because it improves ranking quality without requiring major infrastructure.

## Optional future extensions

These are useful later, but are not required for the current corpus-first objective:

- live SERP collection
- crawling/scraping
- interactive editor rescoring
- web frontend
- multi-topic batch pipelines
- multilingual support
- large-scale distributed processing

## Dependencies by phase

### Phase 1
```bash
pip install spacy scikit-learn yake rapidfuzz orjson
python -m spacy download en_core_web_md
```

### Phase 3 optional
```bash
pip install sentence-transformers keybert
```

## Top implementation priorities

1. schema validation
2. hard filtering rules
3. duplicate control
4. structural extraction quality
5. candidate consolidation

## Definition of done

The implementation is ready for downstream use when:

- bad documents are reliably excluded
- the retained corpus is stable and reviewable
- keyword outputs are typed and explainable
- top results are visibly relevant to the topic
- reruns are reproducible and easy to tune
