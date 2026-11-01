"""Async LLM calls for Stage 1 (llm_corpus) and Stage 2 (llm_generated)."""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, TYPE_CHECKING

import httpx

from nlp_keyword_pipeline.llm_keyword_stages.json_utils import parse_json_object
from nlp_keyword_pipeline.llm_keyword_stages.models import LLMCorpusTerm, LLMGeneratedTerm
from nlp_keyword_pipeline.llm_keyword_stages.prompts import (
    SYSTEM_LLM_CORPUS,
    SYSTEM_LLM_GENERATED,
    build_corpus_context,
    build_llm_corpus_user_prompt,
    build_llm_generated_user_prompt,
)

if TYPE_CHECKING:
    from nlp_keyword_pipeline.models import CleanDocument

logger = logging.getLogger(__name__)


class ChatFn(Protocol):
    async def __call__(
        self,
        client: httpx.AsyncClient,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _unwrap_terms_list(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("terms", "keywords", "results", "items"):
        v = data.get(key)
        if isinstance(v, list):
            return v
    for v in data.values():
        if isinstance(v, list):
            return v
    return []


def _parse_corpus_terms(raw: str) -> list[LLMCorpusTerm]:
    data = parse_json_object(raw)
    if data is None:
        return []
    items = _unwrap_terms_list(data)
    out: list[LLMCorpusTerm] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        term = (item.get("term") or item.get("keyword") or "").strip()
        if not term:
            continue
        norm = _normalize(term)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            cse = float(item.get("corpus_support_estimate") or 0.7)
        except (TypeError, ValueError):
            cse = 0.7
        cse = max(0.0, min(1.0, cse))
        out.append(
            LLMCorpusTerm(
                term=term,
                normalized=norm,
                likely_category=str(item.get("likely_category") or "").strip(),
                evidence_snippet=str(item.get("evidence_snippet") or "").strip()[:500],
                corpus_support_estimate=cse,
                reason_for_inclusion=str(item.get("reason_for_inclusion") or "").strip()[:500],
            )
        )
    return out


def _parse_generated_terms(raw: str) -> list[LLMGeneratedTerm]:
    data = parse_json_object(raw)
    if data is None:
        return []
    items = _unwrap_terms_list(data)
    out: list[LLMGeneratedTerm] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        term = (item.get("term") or item.get("keyword") or "").strip()
        if not term:
            continue
        norm = _normalize(term)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        sigs = item.get("supporting_signals")
        if isinstance(sigs, str):
            supporting = [sigs]
        elif isinstance(sigs, list):
            supporting = [str(x) for x in sigs if x]
        else:
            supporting = []
        try:
            sal = float(item.get("salience") or 0.5)
        except (TypeError, ValueError):
            sal = 0.5
        sal = max(0.0, min(1.0, sal))
        out.append(
            LLMGeneratedTerm(
                term=term,
                normalized=norm,
                likely_category=str(item.get("likely_category") or "").strip(),
                why_supported=str(item.get("why_supported") or "").strip()[:500],
                supporting_signals=supporting,
                salience=sal,
            )
        )
    return out


async def extract_llm_corpus_terms_async(
    title: str,
    docs: list[CleanDocument],
    *,
    headings: list[str],
    repeated_nouns: list[str],
    entities: list[str],
    stats: list[str],
    categories: list[str],
    chat_completion: ChatFn,
    client: httpx.AsyncClient | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.25,
    max_retries: int = 2,
) -> list[LLMCorpusTerm]:
    ctx = build_corpus_context(docs)
    user = build_llm_corpus_user_prompt(
        title,
        headings=headings,
        repeated_nouns=repeated_nouns,
        entities=entities,
        stats=stats,
        categories=categories,
        corpus_context=ctx,
    )
    owns = client is None
    _client = client or httpx.AsyncClient(timeout=90.0, connect=8.0)
    try:
        for attempt in range(max_retries + 1):
            try:
                raw = await chat_completion(
                    _client,
                    system=SYSTEM_LLM_CORPUS,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                parsed = _parse_corpus_terms(raw)
                if parsed:
                    logger.info("[stage_llm] llm_corpus: parsed %d terms", len(parsed))
                    return parsed
                logger.warning("[stage_llm] llm_corpus: empty parse attempt %s", attempt + 1)
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("[stage_llm] llm_corpus HTTP error: %s", exc)
                if attempt == max_retries:
                    return []
        return []
    finally:
        if owns:
            await _client.aclose()


async def expand_llm_generated_async(
    title: str,
    *,
    headings: list[str],
    repeated_nouns: list[str],
    entities: list[str],
    stats: list[str],
    categories: list[str],
    top_corpus_terms: list[str],
    chat_completion: ChatFn,
    client: httpx.AsyncClient | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    max_retries: int = 2,
) -> list[LLMGeneratedTerm]:
    user = build_llm_generated_user_prompt(
        title,
        headings=headings,
        repeated_nouns=repeated_nouns,
        entities=entities,
        stats=stats,
        categories=categories,
        top_corpus_terms=top_corpus_terms,
    )
    owns = client is None
    _client = client or httpx.AsyncClient(timeout=90.0, connect=8.0)
    try:
        for attempt in range(max_retries + 1):
            try:
                raw = await chat_completion(
                    _client,
                    system=SYSTEM_LLM_GENERATED,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                parsed = _parse_generated_terms(raw)
                if parsed:
                    logger.info("[stage_llm] llm_generated: parsed %d terms", len(parsed))
                    return parsed
                logger.warning("[stage_llm] llm_generated: empty parse attempt %s", attempt + 1)
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("[stage_llm] llm_generated HTTP error: %s", exc)
                if attempt == max_retries:
                    return []
        return []
    finally:
        if owns:
            await _client.aclose()
