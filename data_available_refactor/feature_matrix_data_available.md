# Feature Matrix — Data Already Available Keyword Pipeline

| Method | Role in pipeline | Value for this dataset | Implementation complexity | Runtime cost | Explainability | Use now / later / optional | Notes |
|---|---|---:|---:|---:|---:|---|---|
| JSON schema validation | ingestion | Very high | Low | Low | High | **Use now** | Required because some files have missing fields or only an error payload |
| Hard document filtering rules | filtering | Very high | Low | Low | High | **Use now** | Exclude social/video/blocked/thin pages before extraction |
| Duplicate / near-duplicate detection | filtering | High | Medium | Low-Medium | High | **Use now** | Prevent repeated listicles from dominating prevalence |
| Text normalization | preprocessing | Very high | Low | Low | High | **Use now** | Required before all candidate generators |
| Field separation (title/headings/paragraphs/content) | preprocessing / structure | Very high | Low | Low | High | **Use now** | Keeps structural weighting possible |
| N-grams (1-3) | candidate generation | High | Low | Low | High | **Use now** | Strong baseline for recurring lexical phrases |
| Noun chunk extraction (spaCy) | candidate generation | Very high | Low-Medium | Medium | High | **Use now** | One of the best sources of useful multi-word phrases |
| Named entity extraction (spaCy) | candidate generation | High | Low-Medium | Medium | High | **Use now** | Good for breed names / proper nouns; better with domain lexicon |
| Domain lexicon support | candidate generation | High | Medium | Low | High | **Use now** | Important when generic NER misses topic-specific entities |
| Heading phrase mining | candidate generation | Very high | Low | Low | High | **Use now** | Strong signal for subtopics and heading candidates |
| YAKE | candidate generation | High | Low | Low-Medium | Medium-High | **Use now** | Good local phrase salience; complements noun chunks |
| TF-IDF prevalence signals | scoring | High | Low | Low | High | **Use now** | Useful for prevalence-aware lexical weighting |
| BM25 | scoring | Medium | Low-Medium | Low | Medium | Optional | Helpful if lexical relevance ranking is needed later |
| Sentence-transformer semantic relevance | scoring / grouping | High | Medium | Medium | Medium | **Use now if feasible** | Best pull-forward feature from the broader v2 plan |
| KeyBERT | candidate generation | Medium-High | Medium | Medium-High | Medium | Optional | Good upgrade, not required for first pass |
| Heading/subtopic clustering | grouping | High | Medium | Medium | Medium | **Use now if feasible** | Valuable for converting phrases into subtopic groups |
| Question detection from headings/paragraphs | classification | Medium-High | Low | Low | High | **Use now** | Useful when headings already expose question forms |
| Authority weighting | scoring | Medium | Low | Low | Medium | Optional | Can be added cautiously; should not dominate outputs |
| Junk / boilerplate blacklist | filtering / penalties | High | Low | Low | High | **Use now** | Needed for platform/UI text contamination |
| RapidFuzz variant merging | consolidation | High | Low | Low | High | **Use now** | Useful for canonicalizing plural/surface variants |

## Recommendation summary

### Must ship in the first implementation
- schema validation
- hard filtering rules
- duplicate control
- field separation
- normalization
- n-grams
- noun chunks
- NER + domain lexicon support
- heading phrase mining
- YAKE
- TF-IDF-style prevalence
- candidate consolidation
- typed output buckets

### Pull forward if feasible
- sentence-transformer semantic relevance
- heading/subtopic grouping

### Optional or later
- KeyBERT
- BM25
- authority weighting
