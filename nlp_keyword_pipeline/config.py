"""Pipeline configuration — thresholds, weights, and paths."""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_extractor_trust_weights() -> dict[str, float]:
    return {
        "heading": 1.0,
        "heading_fragment": 0.85,
        "heading_structural": 1.0,
        "entity": 1.0,
        "lexicon": 0.95,
        "keybert": 0.90,
        "noun_chunk": 0.80,
        "context_phrase": 0.75,
        "attribute_relation": 0.80,
        "yake": 0.65,
        "pos_pattern": 0.65,
        "dep_pattern": 0.70,
        "ngram": 0.45,
        "semantic_expansion": 0.40,
        "intent_heading": 0.95,
        "intent_phrase_body": 0.90,
        "intent_corpus_backed": 0.75,
        "intent_template": 0.45,
    }


@dataclass
class ScoringWeights:
    """Rebalanced weights: coverage + structure + semantics + consensus; stronger noise penalty."""

    document_prevalence: float = 0.25
    heading_presence: float = 0.20
    title_presence: float = 0.15
    phrase_quality: float = 0.12
    entity_bonus: float = 0.10
    semantic_relevance: float = 0.08
    source_consensus: float = 0.05
    method_agreement: float = 0.03
    heading_title_combo: float = 0.02
    authority_support: float = 0.02
    intro_presence: float = 0.04
    source_relevance_weighted_prevalence: float = 0.06
    extractor_consensus_weighted: float = 0.04
    redundancy_penalty: float = 0.05
    junk_penalty: float = 0.10
    genericity_penalty: float = 0.03
    semantic_drift_penalty: float = 0.04
    hygiene_soft_penalty: float = 0.05


@dataclass
class BucketCaps:
    """default_bucket_caps from selection_policy + overused_terms."""

    primary_topic_term: int = 12
    core_entity: int = 20
    supporting_phrase: int = 50
    subtopic: int = 40
    heading_candidate: int = 20
    intent_phrase: int = 25
    overused_terms: int = 10


@dataclass
class PipelineConfig:
    """All tunable pipeline parameters."""

    # Paths
    topic_anchor: str = "best guard dog breeds"
    output_dir: str = "output"
    work_dir: str = "work"

    # Filtering
    excluded_domains: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"facebook.com", "youtube.com", "youtu.be"}
        )
    )
    blocked_title_patterns: tuple[str, ...] = (
        "attention required",
        "access denied",
        "just a moment",
        "sorry, you have been blocked",
        "enable javascript",
        "checking your browser",
        "verify you are human",
        "cf-ray id",
        "error 1020",
        "access to this page has been denied",
    )
    min_word_count: int = 250
    min_meaningful_headings: int = 2
    min_content_words_if_few_headings: int = 250
    exclude_reddit_by_default: bool = True
    near_dup_cosine_threshold: float = 0.93
    # Layer B: soft-exclude pages below this relevance (after Layer A)
    source_relevance_min_threshold: float = 0.15
    topic_anchor_content_type: str = "auto"
    # Phase 2: second-pass body similarity (RapidFuzz ratio on normalized prefix)
    enable_body_prefix_duplicate_filter: bool = True
    body_prefix_duplicate_chars: int = 900
    body_prefix_duplicate_ratio: float = 96.0

    # Junk / boilerplate (lowercase substrings)
    junk_phrases: tuple[str, ...] = (
        "cookie policy",
        "privacy policy",
        "terms of use",
        "terms of service",
        "subscribe to",
        "sign up",
        "newsletter",
        "read more",
        "advertisement",
        "skip to content",
        "accept all cookies",
        "manage preferences",
        "we value your privacy",
        "this website uses cookies",
        "click here",
        "follow us on",
        "share this article",
        "related articles",
        "table of contents",
        "last updated",
        "written by",
        "photo by",
        "all rights reserved",
        "copyright ©",
        # CTA / ecommerce / promo / widget
        "complimentary vet consult",
        "vet consult",
        "pet pharmacy",
        "pet insurance",
        "meow pet",
        "what's in the bag",
        "whats in the bag",
        "want a clever dog",
        "shop now",
        "add to cart",
        "free shipping",
        "filter breed",
        "filter breeds",
        "categories guard",
        "browse categories",
        "sponsored content",
        "affiliate disclosure",
        "disclosure:",
        "quiz:",
        "take the quiz",
        "sign in",
        "log in",
        "create account",
        "download app",
        "get the app",
        "limited time",
        "buy now",
        "order now",
        "checkout",
        "shopping cart",
        "widget",
        "cookie consent",
        "inside the bag",
        "what's inside",
        "whats inside",
        "thinking of getting a dog",
        "prefer a chilled-out",
        "chilled-out companion",
        "breeds to lovingly",
        "lovingly guard",
        "compare breeds",
        "compare breed",
        "breeds for ultimate",
        "dog breeds for ultimate",
        "breeds to patrol",
        "breeds protect",
        "breeds they protect",
        # Additional ad/nav/affiliate noise
        "shop by breed",
        "sort by",
        "price range",
        "view all",
        "best seller",
        "top pick",
        "editor's choice",
        "editors choice",
        "as an amazon associate",
        "affiliate link",
        "we may earn",
        "sponsored by",
        "i may earn",
        "commission at no",
        "at no extra cost",
        "fact checked by",
        "medically reviewed",
        "review policy",
        "advertiser disclosure",
        # Residual brand / corpus-specific noise (from first optimised run)
        "bow wow meow",
        "bow wow",
        "diamond pet",
        "fromdiamond",
        "spruce pets",
        "the spruce",
        "world animal foundation",
        "animal hospitalis",
        "dog daycare",
        "diy cat",
        "easy diy",
        "paws and love",
        "home with paws",
        "favorite things",
        "key takeaways",
        "latest from",
        "vet advice",
        "address enter",
        "vaccination is it necessary",
        "grass eating to induce",
        "puppies during early development",
        "breeds needproper",
        "needproper training",
        "themost protective",
    )
    generic_tokens: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "dog",
                "dogs",
                "breed",
                "breeds",
                "pet",
                "pets",
                "animal",
                "animals",
                "puppy",
                "puppies",
            }
        )
    )

    # Phrase hygiene (pre-registration gate)
    phrase_hygiene_max_stopword_ratio: float = 0.6
    phrase_hygiene_min_token_len: int = 2
    phrase_hygiene_max_leading_junk_chars: int = 2
    phrase_hygiene_reject_single_token_not_lexicon: bool = True
    # Opt-2: allow single-word lexicon terms and reasonably long single words
    phrase_hygiene_allow_lexicon_singles: bool = True
    phrase_hygiene_single_word_min_len: int = 5
    phrase_hygiene_max_tokens: int = 12
    phrase_hygiene_soft_stopword_ratio: float = 0.45
    source_consensus_min_methods: int = 2

    # N-grams (cap body slice to keep Layer1 pool near 300–800 unique keys)
    ngram_min_len: int = 2
    ngram_max_len: int = 3
    ngram_min_doc_freq: int = 1
    ngram_body_max_chars: int = 2200
    noun_chunk_max_chars: int = 4000

    # YAKE
    yake_n_max: int = 4
    yake_top: int = 32
    yake_body_max_chars: int = 4500

    # Heading fragment miner (sliding windows); cap per heading to control Layer1 pool size
    heading_fragment_max_per_heading: int = 10
    heading_fragment_min_heading_tokens: int = 8
    pos_pattern_max_per_doc: int = 80
    entity_match_max_chars: int = 5000
    # Opt-3: context/lifestyle phrase extractor
    enable_context_phrase_extractor: bool = True
    context_phrase_max_per_doc: int = 60
    attribute_relation_max_per_doc: int = 40
    # Opt-5: semantic synonym expansion post-pass
    enable_semantic_expansion: bool = True
    semantic_expansion_similarity_threshold: float = 0.72
    semantic_expansion_max_per_seed: int = 4
    semantic_expansion_min_seed_score: float = 0.35
    semantic_expansion_min_seed_heading_hits: int = 1
    semantic_expansion_min_corpus_presence: int = 1
    semantic_expansion_min_seed_doc_fraction: float = 0.12
    # Opt-6: PAA question generation post-pass
    enable_paa_generation: bool = True
    paa_max_questions: int = 30

    # Consolidation
    rapidfuzz_merge_threshold: int = 88

    # Scoring / selection
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    bucket_caps: BucketCaps = field(default_factory=BucketCaps)
    min_absolute_score: float = 0.165
    remove_near_duplicates_after_ranking: bool = True
    # Fragment demotion after scoring (consolidator post-pass uses score later in pipeline)
    fragment_demotion_score_threshold: float = 0.4
    overused_prevalence_threshold: float = 0.7

    # Bucket rule thresholds (from bucket_rules); Phase 2 optional overrides
    primary_min_prevalence: float = 0.25
    primary_phrase_len: tuple[int, int] = (2, 4)
    core_entity_min_prevalence: float = 0.10
    supporting_min_prevalence: float = 0.15
    subtopic_min_heading_docs: int = 2
    heading_candidate_len: tuple[int, int] = (3, 8)
    # If set, override the corresponding threshold above
    override_primary_min_prevalence: float | None = None
    override_core_entity_min_prevalence: float | None = None
    override_supporting_min_prevalence: float | None = None
    override_subtopic_min_heading_docs: int | None = None

    # Semantic (Phase 3 — primary path)
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    semantic_anchor_weight: float = 0.62
    semantic_corpus_weight: float = 0.38
    semantic_batch_size: int = 256
    semantic_max_phrases_to_encode: int = 700
    # Phase 3: KeyBERT (same embedding backbone by default)
    enable_keybert: bool = True
    keybert_top_n: int = 25
    keybert_ngram_range: tuple[int, int] = (1, 3)
    keybert_mmr_diversity: float = 0.52
    keybert_body_max_chars: int = 6000
    # Phase 3: subtopic / heading semantic clustering
    enable_semantic_clustering: bool = True
    cluster_min_candidates: int = 2
    cluster_max_candidates: int = 600
    cluster_distance_threshold: float = 0.45
    cluster_min_heading_docs: int = 1

    # Per-extractor trust (used in weighted consensus / confidence)
    extractor_trust_weights: dict[str, float] = field(
        default_factory=_default_extractor_trust_weights
    )

    # Authority normalization (typical JSON scale 1–10)
    authority_min_scale: int = 1
    authority_max_scale: int = 10

    # Evaluation: optional baseline CSV for comparison_report
    baseline_csv_path: str | None = None

    def bucket_caps_dict(self) -> dict[str, int]:
        return {
            "primary_topic_term": self.bucket_caps.primary_topic_term,
            "core_entity": self.bucket_caps.core_entity,
            "supporting_phrase": self.bucket_caps.supporting_phrase,
            "subtopic": self.bucket_caps.subtopic,
            "heading_candidate": self.bucket_caps.heading_candidate,
            "intent_phrase": self.bucket_caps.intent_phrase,
            "overused_terms": self.bucket_caps.overused_terms,
        }

    def effective_primary_min_prevalence(self) -> float:
        return (
            self.override_primary_min_prevalence
            if self.override_primary_min_prevalence is not None
            else self.primary_min_prevalence
        )

    def effective_core_entity_min_prevalence(self) -> float:
        return (
            self.override_core_entity_min_prevalence
            if self.override_core_entity_min_prevalence is not None
            else self.core_entity_min_prevalence
        )

    def effective_supporting_min_prevalence(self) -> float:
        return (
            self.override_supporting_min_prevalence
            if self.override_supporting_min_prevalence is not None
            else self.supporting_min_prevalence
        )

    def effective_subtopic_min_heading_docs(self) -> int:
        return (
            self.override_subtopic_min_heading_docs
            if self.override_subtopic_min_heading_docs is not None
            else self.subtopic_min_heading_docs
        )
