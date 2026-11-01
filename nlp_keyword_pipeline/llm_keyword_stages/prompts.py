"""Prompt builders for Stage 1 (llm_corpus) and Stage 2 (llm_generated)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlp_keyword_pipeline.models import CleanDocument


def _truncate(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 20] + f"... [{len(t) - limit + 20} more chars]"


MAX_CORPUS_DOCS = 4
MAX_CHARS_PER_DOC = 2500


def build_corpus_context(docs: list[CleanDocument]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs[:MAX_CORPUS_DOCS]):
        snippet = (doc.intro_text + " " + doc.body_text).strip()[:MAX_CHARS_PER_DOC]
        parts.append(f"[Source {i + 1}: {doc.title_text}]\n{_truncate(snippet, MAX_CHARS_PER_DOC)}")
    return "\n\n".join(parts)


SYSTEM_LLM_CORPUS = """You extract SEO keyword candidates strictly from the given evidence.
Return ONLY valid JSON. No markdown fences. The response MUST be a JSON object with a "terms" key."""

USER_LLM_CORPUS_TEMPLATE = """Extract high-quality keyword terms that are explicitly present in, or directly grounded in, the corpus.

You are given:
- title: {title}
- cleaned corpus excerpts (below)
- headings: {headings}
- repeated noun phrases: {repeated_nouns}
- named entities: {entities}
- stats: {stats}
- inferred categories (dimensions): {categories}

Instructions:
1. Prefer terms that appear in headings, repeated phrases, named entities, and important factual statements.
2. Prefer specific, meaningful phrases over generic single words.
3. Keep terms strongly aligned with the title and inferred categories.
4. Include only terms with strong textual grounding.
5. Mark conceptual source as llm_corpus (each term in JSON includes source: "llm_corpus").

Do not invent new concepts that are not supported by the text.
Do not include generic broad-topic filler unless clearly central to the page.

Return JSON:
{{
  "terms": [
    {{
      "term": "...",
      "source": "llm_corpus",
      "likely_category": "short free-form label",
      "evidence_snippet": "short quote from corpus",
      "corpus_support_estimate": 0.0,
      "reason_for_inclusion": "one sentence"
    }}
  ]
}}

corpus excerpts:
{corpus_context}
"""


SYSTEM_LLM_GENERATED = """You propose a SMALL set of additional keyword candidates supported by the evidence.
Do not use any predefined category taxonomy — use free-form likely_category phrases.
Return ONLY valid JSON with a "terms" key."""

USER_LLM_GENERATED_TEMPLATE = """Generate additional keyword candidates using the provided evidence.

You are given:
- title: {title}
- headings: {headings}
- repeated noun phrases: {repeated_nouns}
- named entities: {entities}
- stats: {stats}
- inferred categories: {categories}
- top corpus-derived terms: {top_corpus_terms}

Rules:
1. Do NOT use predefined categories (use short free-form likely_category only).
2. Do NOT generate broad filler terms just because they are common in the niche.
3. Only suggest terms that are tightly related to the title and source signals.
4. Prefer terms that are semantically supported by multiple signals.
5. These are expansion candidates, not primary truth — mark source as "llm_generated".

Return JSON:
{{
  "terms": [
    {{
      "term": "...",
      "source": "llm_generated",
      "likely_category": "...",
      "why_supported": "sentence",
      "supporting_signals": ["heading", "entity", ...],
      "salience": 0.0
    }}
  ]
}}
"""


def build_llm_corpus_user_prompt(
    title: str,
    *,
    headings: list[str],
    repeated_nouns: list[str],
    entities: list[str],
    stats: list[str],
    categories: list[str],
    corpus_context: str,
) -> str:
    def _join(xs: list[str], n: int = 80) -> str:
        items = [x for x in xs if x][:n]
        return "; ".join(items) if items else "(none)"

    return USER_LLM_CORPUS_TEMPLATE.format(
        title=title,
        headings=_join(headings, 60),
        repeated_nouns=_join(repeated_nouns, 60),
        entities=_join(entities, 60),
        stats=_join(stats, 40),
        categories=_join(categories, 30),
        corpus_context=corpus_context,
    )


def build_llm_generated_user_prompt(
    title: str,
    *,
    headings: list[str],
    repeated_nouns: list[str],
    entities: list[str],
    stats: list[str],
    categories: list[str],
    top_corpus_terms: list[str],
) -> str:
    def _join(xs: list[str], n: int = 80) -> str:
        items = [x for x in xs if x][:n]
        return "; ".join(items) if items else "(none)"

    top = "; ".join(top_corpus_terms[:60]) if top_corpus_terms else "(none)"
    return USER_LLM_GENERATED_TEMPLATE.format(
        title=title,
        headings=_join(headings, 60),
        repeated_nouns=_join(repeated_nouns, 60),
        entities=_join(entities, 60),
        stats=_join(stats, 40),
        categories=_join(categories, 30),
        top_corpus_terms=top,
    )
