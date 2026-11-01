"""Loose JSON extraction from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def strip_json_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?|```", "", raw, flags=re.IGNORECASE).strip()


def extract_first_json(raw: str) -> str:
    text = strip_json_fences(raw or "")
    if not text:
        return ""
    decoder = json.JSONDecoder()
    for ch in ("{", "["):
        idx = text.find(ch)
        if idx == -1:
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
            return text[idx : idx + end]
        except json.JSONDecodeError:
            continue
    return text


def parse_json_object(raw: str) -> Any | None:
    candidate = extract_first_json(raw)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    return None
