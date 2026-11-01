"""
pseudo_pipeline_data_available.py
---------------------------------
Implementation skeleton for generating structured keyword outputs from an
already-available ZIP of page JSON documents.
"""

from __future__ import annotations

import json
import zipfile
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections import defaultdict, Counter

import yake
import spacy
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PageDocument:
    doc_id: str
    file_path: str
    url: str
    domain: str
    title: str = ""
    description: str = ""
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    content: str = ""
    word_count: int | None = None
    heading_count: int | None = None
    para_count: int | None = None
    authority: int | None = None


@dataclass
class CleanDocument:
    doc_id: str
    url: str
    domain: str
    title_text: str
    description_text: str
    heading_texts: list[str]
    paragraph_texts: list[str]
    intro_text: str
    body_text: str
    full_clean_text: str


@dataclass
class Candidate:
    surface_form: str
    normalized_form: str
    source_methods: set[str] = field(default_factory=set)
    source_docs: set[str] = field(default_factory=set)
    title_hits: int = 0
    heading_hits: int = 0
    body_hits: int = 0
    score: float = 0.0
    candidate_type: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ingestion and validation
# ---------------------------------------------------------------------------

REQUIRED_MIN_TEXT_FIELDS = ["title", "headings", "paragraphs", "content"]
BLOCKED_PATTERNS = [
    "attention required",
    "access denied",
    "sorry, you have been blocked",
]
EXCLUDED_DOMAINS = {"facebook.com", "youtube.com"}


def unzip_dataset(zip_path: str, out_dir: str) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_path)
    return list(out_path.rglob("*.json"))


def parse_json_files(paths: list[Path]) -> tuple[list[PageDocument], list[dict[str, Any]]]:
    docs: list[PageDocument] = []
    issues: list[dict[str, Any]] = []

    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            issues.append({"file": str(path), "stage": "parse", "error": str(exc)})
            continue

        if not is_payload_usable(payload):
            issues.append({"file": str(path), "stage": "validate", "error": "No usable text fields"})
            continue

        docs.append(to_page_document(path, payload))

    return docs, issues


def is_payload_usable(payload: dict[str, Any]) -> bool:
    if "error" in payload and len(payload.keys()) <= 2:
        return False
    if not payload.get("url"):
        return False
    return any(payload.get(field) for field in REQUIRED_MIN_TEXT_FIELDS)


def to_page_document(path: Path, payload: dict[str, Any]) -> PageDocument:
    doc_id = hashlib.md5(str(path).encode()).hexdigest()[:12]
    url = payload.get("url", "")
    domain = extract_domain(url)
    return PageDocument(
        doc_id=doc_id,
        file_path=str(path),
        url=url,
        domain=domain,
        title=payload.get("title", "") or "",
        description=payload.get("description", "") or "",
        headings=payload.get("headings", []) or [],
        paragraphs=payload.get("paragraphs", []) or [],
        content=payload.get("content", "") or "",
        word_count=payload.get("word_count"),
        heading_count=payload.get("heading_count"),
        para_count=payload.get("para_count"),
        authority=payload.get("authority"),
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_documents(docs: list[PageDocument]) -> tuple[list[PageDocument], list[dict[str, Any]]]:
    kept: list[PageDocument] = []
    excluded: list[dict[str, Any]] = []

    for doc in docs:
        reason = exclusion_reason(doc)
        if reason:
            excluded.append({"doc_id": doc.doc_id, "url": doc.url, "reason": reason})
            continue
        kept.append(doc)

    kept = remove_near_duplicates(kept, excluded)
    return kept, excluded


def exclusion_reason(doc: PageDocument) -> str | None:
    if doc.domain in EXCLUDED_DOMAINS:
        return "excluded_domain"
    title_lower = (doc.title or "").lower()
    if any(pat in title_lower for pat in BLOCKED_PATTERNS):
        return "blocked_page"
    if (doc.word_count or 0) < 250:
        return "thin_content"
    if len([h for h in doc.headings if h.strip()]) < 2 and len(doc.content.split()) < 250:
        return "insufficient_structure"
    return None


def remove_near_duplicates(docs: list[PageDocument], excluded: list[dict[str, Any]]) -> list[PageDocument]:
    if len(docs) < 2:
        return docs

    corpus = [normalize_text(doc.content or " ".join(doc.paragraphs)) for doc in docs]
    vec = TfidfVectorizer(min_df=1).fit_transform(corpus)
    sims = cosine_similarity(vec)

    keep = [True] * len(docs)
    for i in range(len(docs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(docs)):
            if sims[i, j] > 0.93:
                keep[j] = False
                excluded.append({"doc_id": docs[j].doc_id, "url": docs[j].url, "reason": "near_duplicate"})

    return [doc for doc, flag in zip(docs, keep) if flag]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def build_clean_documents(docs: list[PageDocument]) -> list[CleanDocument]:
    clean_docs: list[CleanDocument] = []
    for doc in docs:
        headings = [clean_line(h) for h in doc.headings if clean_line(h)]
        paragraphs = [clean_line(p) for p in doc.paragraphs if clean_line(p)]
        intro = " ".join(paragraphs[:3])
        body = " ".join(paragraphs)
        full_text = "\n".join([doc.title, doc.description, *headings, *paragraphs])
        clean_docs.append(CleanDocument(
            doc_id=doc.doc_id,
            url=doc.url,
            domain=doc.domain,
            title_text=clean_line(doc.title),
            description_text=clean_line(doc.description),
            heading_texts=headings,
            paragraph_texts=paragraphs,
            intro_text=intro,
            body_text=body,
            full_clean_text=normalize_text(full_text, keep_case=True),
        ))
    return clean_docs


def clean_line(text: str) -> str:
    text = (text or "").replace("\xa0", " ").strip()
    if text.lower() in {"advertisement", "read more", "subscribe"}:
        return ""
    return " ".join(text.split())


def normalize_text(text: str, keep_case: bool = False) -> str:
    text = clean_line(text)
    return text if keep_case else text.lower()


def extract_domain(url: str) -> str:
    if not url:
        return ""
    domain = url.replace("https://", "").replace("http://", "").split("/")[0].lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def build_candidate_registry(clean_docs: list[CleanDocument], topic_anchor: str) -> dict[str, Candidate]:
    registry: dict[str, Candidate] = {}
    nlp = spacy.load("en_core_web_md")
    yake_extractor = yake.KeywordExtractor(lan="en", n=3, top=30)

    for doc in clean_docs:
        zone_docs = {
            "title": doc.title_text,
            "headings": " ".join(doc.heading_texts),
            "body": doc.body_text,
        }

        add_ngram_candidates(registry, doc, zone_docs)
        add_noun_chunk_candidates(registry, doc, nlp(doc.full_clean_text))
        add_entity_candidates(registry, doc, nlp(doc.full_clean_text))
        add_heading_candidates(registry, doc)
        add_yake_candidates(registry, doc, yake_extractor)
        add_question_candidates(registry, doc)

    consolidate_registry(registry)
    score_candidates(registry, clean_docs, topic_anchor, nlp)
    classify_candidates(registry)
    return registry


def add_ngram_candidates(registry: dict[str, Candidate], doc: CleanDocument, zone_docs: dict[str, str]) -> None:
    # Placeholder: implement tokenization and n-gram phrase filtering.
    pass


def add_noun_chunk_candidates(registry: dict[str, Candidate], doc: CleanDocument, spacy_doc) -> None:
    # Placeholder: add noun chunks as candidate phrases.
    pass


def add_entity_candidates(registry: dict[str, Candidate], doc: CleanDocument, spacy_doc) -> None:
    # Placeholder: add named entities and lexicon-backed entities.
    pass


def add_heading_candidates(registry: dict[str, Candidate], doc: CleanDocument) -> None:
    # Placeholder: add heading phrases and repeated subtopic patterns.
    pass


def add_yake_candidates(registry: dict[str, Candidate], doc: CleanDocument, extractor) -> None:
    # Placeholder: run YAKE on body or weighted text.
    pass


def add_question_candidates(registry: dict[str, Candidate], doc: CleanDocument) -> None:
    # Placeholder: detect interrogative headings or sentences.
    pass


# ---------------------------------------------------------------------------
# Consolidation, scoring, classification
# ---------------------------------------------------------------------------


def consolidate_registry(registry: dict[str, Candidate]) -> None:
    # Placeholder: canonicalize plurals/variants and merge fuzzy duplicates.
    pass


def score_candidates(registry: dict[str, Candidate], clean_docs: list[CleanDocument], topic_anchor: str, nlp) -> None:
    # Placeholder: compute prevalence, heading/title presence, phrase quality,
    # semantic relevance, penalties, and final score.
    pass


def classify_candidates(registry: dict[str, Candidate]) -> None:
    # Placeholder: assign buckets like primary_topic_term, core_entity,
    # supporting_phrase, subtopic, heading_candidate, question_phrase, ignore_noise.
    pass


# ---------------------------------------------------------------------------
# Selection and export
# ---------------------------------------------------------------------------


def export_results(registry: dict[str, Candidate], output_path: str) -> None:
    # Placeholder: rank, cap by bucket, and export to JSON.
    pass


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(zip_path: str, work_dir: str, topic_anchor: str, output_path: str) -> None:
    json_files = unzip_dataset(zip_path, f"{work_dir}/unzipped")
    page_docs, parse_issues = parse_json_files(json_files)
    kept_docs, exclusions = filter_documents(page_docs)
    clean_docs = build_clean_documents(kept_docs)
    registry = build_candidate_registry(clean_docs, topic_anchor)
    export_results(registry, output_path)

    print("Parse issues:", len(parse_issues))
    print("Excluded docs:", len(exclusions))
    print("Retained docs:", len(clean_docs))


if __name__ == "__main__":
    run_pipeline(
        zip_path="examples/synthetic_corpus.zip",
        work_dir="./work",
        topic_anchor="best guard dog breeds",
        output_path="./keyword_output.json",
    )
