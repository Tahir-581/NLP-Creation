# Source Notes — Data Available Workflow

## Usage note

These sources support the narrower workflow of extracting high-quality keyword candidates from an existing corpus of page documents. The emphasis is on filtering, extraction, scoring, and classification.

| Source | URL | Type | Why it matters here | Confidence | Supports |
|---|---|---|---|---|---|
| Google Search Essentials / helpful content guidance | https://developers.google.com/search/docs/fundamentals/creating-helpful-content | Primary | Reinforces that we should favor topical completeness and natural usage over density; helps define what the system should avoid rewarding | High | output philosophy, scoring guardrails |
| Google spam policies | https://developers.google.com/search/docs/essentials/spam-policies | Primary | Supports anti-stuffing and anti-manipulation constraints in keyword selection and any later scoring | High | scoring guardrails, penalties |
| spaCy linguistic features and NER docs | https://spacy.io/usage/linguistic-features | Primary / implementation | Practical reference for noun chunks, POS patterns, lemmatization, and entity extraction | High | preprocessing, extraction |
| YAKE project/docs | https://github.com/LIAAD/yake | Implementation reference | Useful for local statistical keyphrase extraction without heavy model dependencies | High | extraction, candidate generation |
| scikit-learn TfidfVectorizer docs | https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html | Primary / implementation | Useful for prevalence-aware lexical weighting and duplicate analysis | High | extraction, scoring, duplicate detection |
| sentence-transformers docs | https://www.sbert.net/ | Primary / implementation | Supports semantic relevance scoring and optional clustering without building custom models | High | semantic scoring, grouping |
| KeyBERT repo/docs | https://github.com/MaartenGr/KeyBERT | Implementation reference | Optional semantic keyphrase layer; useful but not required for first pass | Medium-High | optional extraction enhancement |
| RapidFuzz docs | https://rapidfuzz.github.io/RapidFuzz/ | Implementation reference | Practical for canonicalization, string similarity, and variant merging | High | consolidation |
| Trafilatura docs | https://trafilatura.readthedocs.io/ | Implementation reference | Less central here because the dataset is already extracted, but useful if some fields need re-cleaning or fallback parsing later | Medium | fallback cleaning |
| rank_bm25 docs | https://github.com/dorianbrown/rank_bm25 | Implementation reference | Useful if we want lexical relevance signals beyond TF-IDF; not essential for first pass | Medium | optional scoring |

## Notes on relevance

### Keep as high-priority references

These are the most directly useful for implementation now:

- Google helpful content guidance
- Google spam policies
- spaCy docs
- YAKE docs
- scikit-learn TF-IDF docs
- sentence-transformers docs
- RapidFuzz docs

### Downgraded for this scenario

The original pack included sources that matter more for a full product or live competitive analysis system. Those are still useful conceptually, but they are not central to the current “data already available” workflow.

### Confirmed fact vs inference usage

- **Confirmed fact:** Google does not want keyword stuffing and favors helpful, people-first content.
- **Confirmed fact:** spaCy, YAKE, TF-IDF, and sentence-transformers can be used directly in this pipeline.
- **Inference / design choice:** the final candidate scoring formula and bucket definitions are implementation choices, not facts from an external source.
- **Inference / design choice:** how heavily to weight headings, titles, and semantic similarity is a product decision.
