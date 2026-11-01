"""Phrase hygiene: pass / soft_penalty / hard_reject with structured diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spacy.lang.en.stop_words import STOP_WORDS

from nlp_keyword_pipeline.domain_lexicon import DOMAIN_LEXICON

if TYPE_CHECKING:
    from nlp_keyword_pipeline.config import PipelineConfig

_MALFORMED_BOUNDARY = re.compile(r"[a-z][A-Z]|[a-z]{2,}[A-Z]")
_LEADING_JUNK = re.compile(r"^[\s#+•*·\d.\-–—:@$+]+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']*")
_PUNCT_RATIO = re.compile(r"[^\w\s]")
_ORPHAN_NATIONALITY = frozenset(
    {"german", "belgian", "american", "british", "french", "italian", "russian", "chinese"}
)

_rejection_count = 0


@dataclass
class HygieneResult:
    verdict: str  # "pass" | "soft_penalty" | "hard_reject"
    reason: str
    penalty_weight: float  # 0.0–1.0 for scorer when soft_penalty


def reset_rejection_counter() -> None:
    global _rejection_count
    _rejection_count = 0


def get_rejection_count() -> int:
    return _rejection_count


def evaluate_phrase_hygiene(
    surface: str,
    norm: str,
    config: PipelineConfig,
) -> HygieneResult:
    """Structured hygiene verdict; hard_reject blocks registration."""
    global _rejection_count

    if not surface or not norm or len(surface.strip()) < 2:
        _rejection_count += 1
        return HygieneResult("hard_reject", "empty", 1.0)

    s = surface.strip()
    low = norm.lower().strip()
    stripped = _LEADING_JUNK.sub("", s).strip()
    if not stripped or len(stripped) < 2:
        _rejection_count += 1
        return HygieneResult("hard_reject", "leading_junk", 1.0)

    first_char = stripped[0]
    if not first_char.isalpha():
        _rejection_count += 1
        return HygieneResult("hard_reject", "starts_non_alpha", 1.0)

    if _MALFORMED_BOUNDARY.search(stripped):
        _rejection_count += 1
        return HygieneResult("hard_reject", "malformed_concat", 1.0)

    _run_together_prefixes = ("the", "our", "my", "your", "best", "top")
    if " " not in low and len(low) >= 8:
        for prefix in _run_together_prefixes:
            if low.startswith(prefix) and len(low) > len(prefix) + 2:
                rest = low[len(prefix) :]
                if rest.isalpha() and len(rest) >= 4:
                    _rejection_count += 1
                    return HygieneResult("hard_reject", "run_together", 1.0)
    if " " in low:
        for tok_candidate in low.split():
            if len(tok_candidate) >= 8:
                for prefix in _run_together_prefixes:
                    if tok_candidate.startswith(prefix) and len(tok_candidate) > len(prefix) + 2:
                        rest = tok_candidate[len(prefix) :]
                        if rest.isalpha() and len(rest) >= 4:
                            _rejection_count += 1
                            return HygieneResult("hard_reject", "run_together_token", 1.0)

    low_full = (surface + " " + norm).lower()
    for j in config.junk_phrases:
        if j in low_full:
            _rejection_count += 1
            return HygieneResult("hard_reject", "junk_substring", 1.0)

    toks = _TOKEN.findall(low)
    if not toks:
        _rejection_count += 1
        return HygieneResult("hard_reject", "no_tokens", 1.0)

    # Repeated token (e.g. "dog dog breeds")
    if len(toks) >= 2 and len(set(t.lower() for t in toks)) < len(toks):
        _rejection_count += 1
        return HygieneResult("hard_reject", "repeated_token", 1.0)

    max_tok = getattr(config, "phrase_hygiene_max_tokens", 12)
    if len(toks) > max_tok:
        _rejection_count += 1
        return HygieneResult("hard_reject", "too_many_tokens", 1.0)

    punct_hits = len(_PUNCT_RATIO.findall(surface))
    if punct_hits >= 3 or (len(surface) > 0 and punct_hits / max(len(surface), 1) > 0.25):
        return HygieneResult("soft_penalty", "punctuation_heavy", 0.35)

    if len(toks) == 1 and toks[0].istitle() and toks[0].lower() in _ORPHAN_NATIONALITY:
        if toks[0].lower() not in DOMAIN_LEXICON:
            _rejection_count += 1
            return HygieneResult("hard_reject", "orphan_nationality_token", 1.0)

    stop_hits = sum(1 for t in toks if t.lower() in STOP_WORDS)
    ratio = stop_hits / max(len(toks), 1)
    soft_thresh = getattr(config, "phrase_hygiene_soft_stopword_ratio", 0.45)
    if ratio > config.phrase_hygiene_max_stopword_ratio:
        _rejection_count += 1
        return HygieneResult("hard_reject", "stopword_heavy", 1.0)
    if ratio > soft_thresh:
        return HygieneResult("soft_penalty", "stopword_elevated", 0.25)

    if config.phrase_hygiene_reject_single_token_not_lexicon and len(toks) == 1:
        tok = toks[0]
        in_lexicon = tok.lower() in DOMAIN_LEXICON
        allow_lexicon = getattr(config, "phrase_hygiene_allow_lexicon_singles", True)
        min_len = getattr(config, "phrase_hygiene_single_word_min_len", 5)
        if in_lexicon and allow_lexicon:
            pass
        elif len(tok) >= min_len:
            pass
        elif tok.lower() not in config.generic_tokens:
            if len(tok) < 4:
                _rejection_count += 1
                return HygieneResult("hard_reject", "single_token_short", 1.0)

    return HygieneResult("pass", "", 0.0)


def is_phrase_acceptable(
    surface: str,
    norm: str,
    config: PipelineConfig,
) -> tuple[bool, str]:
    """Backward compatibility: False only for hard_reject."""
    r = evaluate_phrase_hygiene(surface, norm, config)
    if r.verdict == "hard_reject":
        return False, r.reason
    return True, r.reason
