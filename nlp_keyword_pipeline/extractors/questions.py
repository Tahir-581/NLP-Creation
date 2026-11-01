"""Intent phrase detection from headings and sentences."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from nlp_keyword_pipeline.extractors.registry_ops import register_hit

if TYPE_CHECKING:
    from spacy.language import Language

    from nlp_keyword_pipeline.config import PipelineConfig
    from nlp_keyword_pipeline.models import CleanDocument

# Headings that start with an intent signal word are treated as intent phrases
_INTENT_START = re.compile(
    r"^(best|top|most|how to|training|choosing|types of|guide to|tips for|"
    r"benefits of|importance of|effective|common|understanding|selecting|"
    r"picking|finding|comparing|recommended|ideal|proven|essential)\b",
    re.I,
)

# Body sentences that contain inline intent patterns (non-question, actionable)
_INTENT_BODY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bbest\s+\w[\w\s]{3,40}(?:for|to)\b", re.I),
    re.compile(r"\btraining\s+(?:a|an|your|the)\s+\w[\w\s]{2,35}\b", re.I),
    re.compile(r"\bchoosing\s+(?:a|an|the|the right)\s+\w[\w\s]{2,35}\b", re.I),
    re.compile(r"\bselecting\s+(?:a|an|the)\s+\w[\w\s]{2,35}\b", re.I),
    re.compile(r"\btop\s+\d*\s*\w[\w\s]{3,40}\bbreeds?\b", re.I),
    re.compile(r"\bmost\s+(?:effective|loyal|protective|reliable|suitable)\s+\w[\w\s]{2,30}\b", re.I),
    re.compile(r"\b(?:guard|protection|protective)\s+dog\s+\w[\w\s]{2,35}\b", re.I),
    re.compile(r"\bdog\s+(?:training|breed|breeds)\s+(?:for|that|to)\s+\w[\w\s]{2,30}\b", re.I),
]


def _is_intent_heading(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # Exclude anything ending with ? (questions remain excluded)
    if s.endswith("?"):
        return False
    return bool(_INTENT_START.match(s))


def _extract_intent_spans(sentence: str) -> list[str]:
    """Return substrings of a body sentence that match intent patterns."""
    spans: list[str] = []
    for pat in _INTENT_BODY_PATTERNS:
        for m in pat.finditer(sentence):
            span = m.group(0).strip()
            if 10 <= len(span) <= 120:
                spans.append(span)
    return spans


def extract(
    doc: CleanDocument,
    nlp: Language,
    registry: dict,
    config: PipelineConfig,
) -> None:
    # --- Heading-level intent phrases ---
    for h in doc.heading_texts:
        if _is_intent_heading(h):
            register_hit(
                registry,
                h.strip(),
                doc.doc_id,
                "intent_heading",
                {"heading"},
                is_question=True,
                config=config,
                skip_phrase_hygiene=True,
            )

    # --- Body-sentence intent spans ---
    body = doc.body_text
    if not body.strip():
        return
    spacy_doc = nlp(body[: min(len(body), nlp.max_length)])
    for sent in spacy_doc.sents:
        t = sent.text.strip()
        if len(t) < 15 or len(t) > 300:
            continue
        # Skip question sentences
        if t.endswith("?"):
            continue
        spans = _extract_intent_spans(t)
        for span in spans:
            register_hit(
                registry,
                span,
                doc.doc_id,
                "intent_phrase_body",
                {"body"},
                is_question=True,
                config=config,
                skip_phrase_hygiene=False,
            )


# ---------------------------------------------------------------------------
# Intent phrase generation post-pass (replaces PAA question generation)
# ---------------------------------------------------------------------------

# Breed names for per-breed intent template expansion
_BREED_SINGULAR_MAP: dict[str, str] = {
    "german shepherd": "German Shepherd",
    "rottweiler": "Rottweiler",
    "doberman pinscher": "Doberman Pinscher",
    "doberman": "Doberman",
    "bullmastiff": "Bullmastiff",
    "tibetan mastiff": "Tibetan Mastiff",
    "cane corso": "Cane Corso",
    "akita": "Akita",
    "belgian malinois": "Belgian Malinois",
    "great dane": "Great Dane",
    "rhodesian ridgeback": "Rhodesian Ridgeback",
    "giant schnauzer": "Giant Schnauzer",
    "komondor": "Komondor",
    "anatolian shepherd dog": "Anatolian Shepherd",
    "staffordshire bull terrier": "Staffordshire Bull Terrier",
    "boxer": "Boxer",
    "kangal": "Kangal",
    "boerboel": "Boerboel",
    "great pyrenees": "Great Pyrenees",
    "chow chow": "Chow Chow",
    "neapolitan mastiff": "Neapolitan Mastiff",
}

# Topic-level intent phrases (non-question, actionable)
_TOPIC_INTENT_PHRASES: list[str] = [
    "best guard dog breeds for families",
    "choosing the right guard dog breed",
    "top guard dog breeds for home protection",
    "most loyal guard dog breeds",
    "guard dog training tips for beginners",
    "effective guard dog training methods",
    "best guard dogs for apartments",
    "guard dog breeds suitable for first-time owners",
    "most protective dog breeds",
    "guard dogs good with children",
    "guard dog breeds that are easy to train",
    "guard dogs versus watchdogs differences",
    "calmest guard dog breeds for families",
    "guard dog exercise requirements",
    "strongest guard dog breeds worldwide",
    "guard dogs as family pets",
]

# Per-breed intent templates (use {breed} placeholder, no question marks)
_BREED_INTENT_TEMPLATES: list[str] = [
    "{breed} as a family guard dog",
    "training a {breed} for protection",
    "{breed} for home security",
    "{breed} temperament and guarding instincts",
    "{breed} exercise needs and activity level",
]


def generate_intent_phrases(
    registry: dict,
    config: PipelineConfig,
) -> None:
    """Corpus-backed intent phrase templates (requires ≥2 source docs for breed entity)."""
    if not getattr(config, "enable_paa_generation", True):
        return
    max_phrases = getattr(config, "paa_max_questions", 30)
    emitted = 0
    synthetic_doc_id = "__intent_generated__"

    # Emit topic-level intent phrases first (no breed gating needed)
    for phrase in _TOPIC_INTENT_PHRASES:
        if emitted >= max_phrases:
            return
        register_hit(
            registry,
            phrase,
            synthetic_doc_id,
            "intent_corpus_backed",
            {"body"},
            is_question=True,
            config=config,
            skip_phrase_hygiene=False,
        )
        emitted += 1

    # Per-breed intent phrases gated by corpus evidence (≥2 source docs)
    for norm, display in _BREED_SINGULAR_MAP.items():
        cand = registry.get(norm)
        if cand is None or len(cand.source_docs) < 2:
            continue
        for tmpl in _BREED_INTENT_TEMPLATES:
            if emitted >= max_phrases:
                return
            phrase = tmpl.format(breed=display)
            register_hit(
                registry,
                phrase,
                synthetic_doc_id,
                "intent_corpus_backed",
                {"body"},
                is_question=True,
                config=config,
                skip_phrase_hygiene=False,
            )
            emitted += 1
