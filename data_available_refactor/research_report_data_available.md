# Research Report — NLP Keyword Generation from an Existing Competitor Corpus

## Scope

This document defines the implementation strategy for generating SEO-oriented keywords, phrases, entities, and subtopics **from an already-available dataset of competitor page JSON files**. The goal is not to build a full Surfer-like platform in one step. The goal is to build a high-quality, explainable keyword-generation pipeline that starts from the provided corpus and produces structured outputs suitable for content planning or downstream scoring.

## What we are building in this scenario

We are building a corpus-driven extraction system that:

1. loads a ZIP of competitor/page JSON documents
2. filters unusable or misleading documents
3. normalizes structure and text
4. extracts candidate terms, phrases, entities, headings, and questions
5. consolidates overlapping candidates
6. scores and classifies them into useful buckets
7. exports structured keyword suggestions

## What “keywords” mean here

In this context, “keywords” are **not** just single words with high frequency. They are a mix of topic signals that matter for content coverage. The system should produce typed outputs such as:

- **primary topic terms** — high-level phrases that define the topic
- **core entities** — named entities or domain-specific proper nouns
- **supporting phrases** — useful descriptive or intent-relevant phrases
- **subtopics** — repeated conceptual clusters from headings and content
- **heading candidates** — phrases suitable for H2/H3 sections
- **question phrases** — user-oriented question patterns found in the corpus
- **ignore/noise terms** — frequent but low-value or irrelevant phrases

That framing keeps the system aligned with useful content planning instead of keyword stuffing.

## Why the existing-data case changes the design

Because the data is already available, the highest-leverage engineering work is no longer collection. It is:

- data validation
- filtering
- normalization
- candidate extraction quality
- consolidation
- scoring
- output design

This makes the system both narrower and more practical to implement.

## Practical design principles

### 1. Filter aggressively before extraction
A bad document can pollute the output more than a missing document can hurt it. Any blocked page, social page, video page, forum page, or thin page should be excluded unless it serves a deliberate purpose.

### 2. Preserve structure
Titles, headings, and body paragraphs should not be collapsed too early. Terms appearing in titles and headings are stronger signals than terms that only appear deep in the body.

### 3. Use multiple extraction methods
No single method is sufficient.

Recommended combination for this dataset:

- phrase extraction from n-grams
- noun chunk extraction via spaCy
- named-entity extraction via spaCy plus domain lexicon support
- YAKE for local keyphrase salience
- heading phrase mining
- optional semantic relevance via sentence-transformer embeddings

### 4. Consolidate before scoring
Many candidate generators produce overlapping variants. Scoring before consolidation leads to duplicated or misleading outputs.

### 5. Score with interpretable signals
The output should be explainable. A keyword should rank highly because it:

- appears across retained documents
- appears in important structural positions
- is phrased cleanly
- is semantically close to the topic
- is not junk or a near-duplicate of a better candidate

### 6. Output types, not just lists
Users should understand whether a candidate is a central topic term, an entity, a heading idea, a question, or something to ignore.

## Recommended extraction stack

### Baseline stack
- Python
- `orjson` or built-in `json`
- `spaCy` (`en_core_web_md` or `en_core_web_lg`)
- `scikit-learn`
- `yake`
- `rapidfuzz`
- `sentence-transformers` (optional but recommended)

### Core extractors
- n-gram extraction (uni/bi/trigrams)
- noun chunks
- NER
- heading phrase extraction
- YAKE phrases
- question detection from headings and paragraphs

### Optional semantic layer
- sentence embeddings for topic similarity
- clustering for heading/subtopic grouping

## Recommended output model

Every exported keyword candidate should include at least:

- `keyword`
- `normalized_form`
- `type`
- `score`
- `source_methods`
- `document_prevalence`
- `heading_presence`
- `title_presence`
- `example_sources`
- `notes`

## What to avoid

- treating raw frequency as the main signal
- merging all text fields before preserving structure
- keeping social/video/forum pages by default
- returning a flat top-100 keyword list with no classification
- over-normalizing multi-word phrases until they lose meaning
- claiming this pipeline reproduces Surfer internals

## Most important quality controls for this dataset

1. **Schema validation** — some files do not match the full JSON structure.
2. **Noise exclusion** — blocked pages and platform pages should be removed.
3. **Duplicate control** — similar listicles can dominate phrase prevalence.
4. **Phrase canonicalization** — e.g. singular/plural and variant merging.
5. **Heading weighting** — key ideas often surface first in headings.
6. **Explainability** — every kept keyword should have a traceable reason.

## Recommended first implementation target

The first version should be able to:

1. ingest the ZIP
2. validate and filter documents
3. extract candidates from titles, headings, paragraphs, and content
4. consolidate overlapping candidates
5. score and classify them
6. export a structured JSON/Markdown result

That is enough to support real implementation work and later scoring integration.

## Open questions to resolve during implementation

- how strict duplicate filtering should be for repeated list-style pages
- whether Reddit/forum pages should ever be retained as a separate “voice-of-user” source
- how much authority should influence candidate weighting, if at all
- whether semantic similarity should be mandatory in v1 or optional
- how to maintain domain-specific lexicons for breed names or similar topic entities

## Recommended conclusion

For the current scenario, the best system is a **document-first, structure-aware, typed keyword generator**. It should be built for explainability and data quality, not for imitation of proprietary SEO scoring logic. Once the keyword-generation layer is stable, it can feed a larger optimization product.
