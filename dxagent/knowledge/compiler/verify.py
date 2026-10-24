"""Deterministic verification of extracted claims (v0.4). No LLM.

A claim extracted from a document survives only if all checks pass:

    1. span      the quote occurs verbatim in the source text (after normalizing case, whitespace, dashes, quotes,
                 and PDF line-break hyphenation), and is at least 6 words long
    2. numbers   every number the extractor reported (percent, n, m, sensitivity, specificity, LRs) appears in the
                 quote itself; percentages may be written as fractions (96% / 0.96)
    3. subject   the quote mentions the finding (or test) it is about: a shared content concept with the finding
                 label, its ontology synonyms, or the test's name and aliases
    4. value     a value can be computed from the stated numbers, or, with no number, from exactly one
                 unnegated frequency word in the quote (claims.qualitative_value)

Rejections are returned with a reason so the compile report can show what was dropped and why.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from dxagent.knowledge.compiler.claims import numeric_value, qualitative_value
from dxagent.knowledge.lexicon import concepts

_DASH = re.compile(r"[\u2010-\u2015\u2212]")
_QUOTE = re.compile(r"[\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f`´]")


def norm_text(text: str) -> str:
    t = text.replace("\u00ad", "")  # soft hyphen
    t = _DASH.sub("-", t)
    t = _QUOTE.sub("'", t)
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)  # PDF hyphenation across a line break
    t = t.lower()
    t = re.sub(r"(\d),(\d{3})\b", r"\1\2", t)  # 1,234 -> 1234
    t = re.sub(r"[^a-z0-9.%<>=/+-]+", " ", t)
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", t)  # keep decimal points only
    return re.sub(r"\s+", " ", t).strip()


def span_found(quote: str, source_norm: str, min_words: int = 6) -> bool:
    q = norm_text(quote)
    if len(q.split()) < min_words:
        return False
    return f" {q} " in f" {source_norm} "


_NUM = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)")


def numbers_in(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(norm_text(text))]


def number_stated(value: float, quote_numbers: Iterable[float], percent_like: bool = False) -> bool:
    for q in quote_numbers:
        if abs(q - value) <= max(0.051, abs(value) * 0.005):
            return True
        if percent_like and (abs(q * 100 - value) <= 0.51 or abs(q - value * 100) <= 0.51):
            return True
    return False


def subject_mentioned(quote: str, names: Iterable[str]) -> bool:
    qc = set(concepts(quote))
    for n in names:
        nc = {c for c in concepts(n) if len(c) > 2}
        if nc and nc & qc:
            return True
    return False


def verify_numbers(quote: str, fields: dict[str, Optional[float]]) -> Optional[str]:
    """None if every reported number appears in the quote, else the reason."""
    nums = numbers_in(quote)
    for key, val in fields.items():
        if val is None:
            continue
        pct = key in ("percent", "sensitivity", "specificity")
        if not number_stated(float(val), nums, percent_like=pct):
            return f"{key}={val:g} not stated in the quote"
    return None


def derive(quote: str, percent=None, n=None, m=None, sensitivity=None, specificity=None, lr_pos=None,
           lr_neg=None, subject_names: Optional[list[str]] = None) -> tuple[Optional[float], Optional[float], str, Optional[int], str]:
    """(p, background, precision, sample size, reason). Numbers win; else a frequency word in the clause that
    mentions the subject; else rejected."""
    p, bg, how, size = numeric_value(percent, n, m, sensitivity, specificity, lr_pos, lr_neg)
    if p is not None:
        return p, bg, "numeric", size, how
    v, why = qualitative_value(quote, subject_names)
    if v is not None:
        return v, None, "qualitative", None, "frequency word"
    return None, None, "", None, why


def verify_claim(quote: str, source_norm: str, subject_names: list[str], numbers: dict[str, Optional[float]],
                 min_words: int = 6) -> Optional[str]:
    """Checks 1-3. Returns None when the claim passes, else the rejection reason."""
    if not quote or not quote.strip():
        return "empty quote"
    if not span_found(quote, source_norm, min_words):
        return "quote not found verbatim in the source"
    why = verify_numbers(quote, numbers)
    if why:
        return why
    if subject_names and not subject_mentioned(quote, subject_names):
        return "quote does not mention the finding or test"
    return None
