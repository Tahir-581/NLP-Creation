# Diagnostic Summary — Data Already Available

## What the original pack was trying to build

The original documents aimed to design a Surfer-like SEO optimization system that could collect competitor pages, extract terms and entities, compare a draft against the competitor corpus, and return suggestions plus a content score. The research direction was strong, but the pack assumed a broader platform workflow than we need right now.

## Why that is too broad for the current scenario

For the current implementation, the competitor data already exists as a ZIP of JSON page documents. That changes the immediate engineering problem:

- We do **not** need live SERP fetching or scraping orchestration.
- We do **not** need a broad product architecture first.
- We **do** need a robust pipeline for loading, filtering, normalizing, and extracting structured keyword candidates from an existing corpus.

The highest-risk problems are no longer “how do we get pages?” but:

- which pages should be excluded
- how to clean the text without destroying useful phrases
- how to generate typed keyword suggestions instead of a flat term list
- how to score and filter candidates so the output is useful for implementation

## What the ZIP introduces as first-class assumptions

The attached ZIP contains topic-specific JSON documents with fields such as:

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

It also contains clear noise/outlier cases:

- social pages like Facebook
- video pages like YouTube
- community pages like Reddit
- blocked/access-denied pages (Cloudflare)
- at least one file with only an `error` field

This means the refactored system must treat **document quality filtering** as a mandatory stage before keyword generation.

## What remains useful from the original pack

These ideas remain strong and should be preserved:

- use a competitor corpus instead of generic keyword tools
- avoid raw keyword-density thinking
- combine multiple candidate generators instead of relying on one method
- keep the system explainable
- use headings and structural positions as strong signals
- use a weighted score instead of a single frequency heuristic

## What needed to change

### 1. Remove live collection from the core flow
The system now starts from ZIP ingestion, not from SERP fetch or crawler design.

### 2. Narrow the scope to keyword generation from existing documents
The refactor focuses on **keyword / term / entity / subtopic extraction** from the corpus. Anything about platform productization is secondary.

### 3. Make filtering first-class
The original docs treated filtering mostly as a competitor-selection concern. In this scenario it must become a concrete document-cleaning stage with hard rules.

### 4. Replace a flat “terms” mindset with typed outputs
The output must be grouped into useful buckets:

- primary topic terms
- breed entities / named entities
- supporting phrases
- subtopics
- heading candidates
- question phrases
- ignore/noise terms

### 5. Reframe scoring around corpus evidence
Scoring now means ranking candidates from the existing document set using signals like:

- document prevalence
- heading presence
- title presence
- phrase quality
- entity bonus
- semantic relevance
- redundancy penalty
- junk penalty

## What is out of scope for this refactor

These may be added later, but they are not central to this pack:

- live SERP fetching
- scraping infrastructure
- distributed workers
- frontend editor UX
- multi-tenant SaaS architecture
- multilingual rollout

## Top 10 refactor decisions

1. Start the architecture from **ZIP ingestion**.
2. Treat **JSON validation** as a formal stage.
3. Make **document filtering** mandatory before extraction.
4. Keep **headings**, **paragraphs**, and **titles** as separate weighted zones.
5. Use multiple candidate generators: n-grams, noun chunks, NER, YAKE, heading mining.
6. Merge candidates through **canonicalization and deduplication**.
7. Score keywords using **multiple interpretable signals**, not raw counts alone.
8. Produce **typed keyword buckets** instead of a single list.
9. Keep the implementation plan focused on what a developer can build now from the provided corpus.
10. Defer platform-scale concerns until the extraction pipeline is stable and validated.
