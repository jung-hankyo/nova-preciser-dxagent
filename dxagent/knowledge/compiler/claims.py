"""Claims: one sourced statement about one disease, before reconciliation into a card (v0.4).

A claim's VALUE is always computed here from what the source states, never chosen by an LLM:

    percent           p = percent / 100
    n of m            p = (n + 0.5) / (m + 1)            (Jeffreys-style shrink; 1/1 never becomes certainty)
    sensitivity       p = sensitivity / 100; background = 1 - specificity / 100 when specificity is given
    LR+ and LR-       specificity = (LR+ - 1) / (LR+ - LR-), sensitivity = LR+ (1 - specificity)
    HPO class         Obligate 0.99, Very frequent 0.90, Frequent 0.55, Occasional 0.17, Very rare 0.025, Excluded 0.01
    frequency word    fixed lexicon below; a quote with conflicting or negated frequency words is rejected
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# HPO frequency terms (HP:0040279 subtree) -> representative P(finding | disease).
HPO_FREQUENCY = {
    "HP:0040280": 0.99,  # Obligate (100%)
    "HP:0040281": 0.90,  # Very frequent (99-80%)
    "HP:0040282": 0.55,  # Frequent (79-30%)
    "HP:0040283": 0.17,  # Occasional (29-5%)
    "HP:0040284": 0.025,  # Very rare (<4-1%)
    "HP:0040285": 0.01,  # Excluded (0%)
}

# Ordered: longer and more specific phrases first. (pattern, value)
QUALITATIVE = [
    (r"\b(almost always|nearly always|virtually always|invariably|virtually all|nearly all|almost all)\b", 0.95),
    (r"\b(very rarely|very rare|extremely rare|exceptionally|exceptional)\b", 0.01),
    (r"\b(less commonly|less common|less frequently|less frequent|uncommonly|uncommon|infrequently|infrequent|minority)\b", 0.15),
    (r"\b(most commonly|most common)\b", 0.60),  # "the most common symptom" says rank, not >80%
    (r"\b(usually|majority|most patients|most cases|typically|predominantly)\b", 0.80),
    (r"\b(commonly|common|frequently|frequent|often)\b", 0.60),
    (r"\b(sometimes|occasionally|occasional|variably|in some patients|in some cases)\b", 0.30),
    (r"\b(rarely|rare|seldom|unusual)\b", 0.05),
    (r"\balways\b", 0.95),
    (r"\b(never|absent in all)\b", 0.01),
]
_NEG = re.compile(r"\b(not|no|without|neither|nor)\s+(?:\w+\s+){0,1}$")


_CLAUSE = re.compile(r"[;,:]|\b(?:while|whereas|but|however|although)\b", re.I)


def subject_clause(quote: str, subject_names: Optional[list[str]]) -> str:
    """The single clause of the quote that mentions the subject, or the whole quote if none or several do.
    "Pleuritic pain occurs frequently in pneumonia, while hemoptysis is rare" -> "hemoptysis is rare" for hemoptysis."""
    if not subject_names:
        return quote
    from dxagent.knowledge.lexicon import concepts

    names = [{c for c in concepts(n) if len(c) > 2} for n in subject_names]
    clauses = [c for c in _CLAUSE.split(quote) if c and c.strip()]
    hits = [c for c in clauses if any(n and n & set(concepts(c)) for n in names)]
    return hits[0] if len(hits) == 1 else quote


def qualitative_value(quote: str, subject_names: Optional[list[str]] = None) -> tuple[Optional[float], str]:
    """(value, reason). Evaluated on the clause that mentions the subject. None when no frequency word, when two
    different values are stated, or when negated."""
    q = " " + subject_clause(quote, subject_names).lower() + " "
    found: list[float] = []
    spans: list[tuple[int, int]] = []
    for pat, val in QUALITATIVE:
        for m in re.finditer(pat, q):
            if any(a <= m.start() < b for a, b in spans):
                continue  # already covered by a longer phrase
            if _NEG.search(q[: m.start()]):
                return None, f"negated frequency word '{m.group(0)}'"
            spans.append((m.start(), m.end()))
            found.append(val)
    vals = sorted(set(found))
    if not vals:
        return None, "no frequency word"
    if len(vals) > 1:
        return None, f"conflicting frequency words {vals}"
    return vals[0], "ok"


def sens_spec_from_lr(lr_pos: float, lr_neg: float) -> Optional[tuple[float, float]]:
    if lr_pos is None or lr_neg is None or lr_pos <= 1 or lr_neg >= 1 or lr_neg < 0:
        return None
    spec = (lr_pos - 1) / (lr_pos - lr_neg)
    sens = lr_pos * (1 - spec)
    if not (0 < sens < 1 and 0 < spec < 1):
        return None
    return sens, spec


def numeric_value(percent: Optional[float] = None, n: Optional[int] = None, m: Optional[int] = None,
                  sensitivity: Optional[float] = None, specificity: Optional[float] = None,
                  lr_pos: Optional[float] = None, lr_neg: Optional[float] = None) -> tuple[Optional[float], Optional[float], str, Optional[int]]:
    """(p, background or None, how, sample size). Order of preference: n of m, percent, sensitivity, LR pair."""
    if n is not None and m is not None and m > 0 and 0 <= n <= m:
        return (n + 0.5) / (m + 1), None, "n_of_m", m
    if percent is not None and 0 <= percent <= 100:
        return percent / 100.0, None, "percent", None
    if sensitivity is not None and 0 <= sensitivity <= 100:
        bg = (1 - specificity / 100.0) if specificity is not None and 0 <= specificity <= 100 else None
        return sensitivity / 100.0, bg, "sensitivity", None
    ss = sens_spec_from_lr(lr_pos, lr_neg) if lr_pos is not None and lr_neg is not None else None
    if ss:
        return ss[0], 1 - ss[1], "lr_pair", None
    return None, None, "none", None


@dataclass
class Claim:
    disease: str  # ontology id of the disease the card is for (e.g. "MONDO:0005249")
    kind: str  # finding | test | harm | mimic
    source_id: str
    source_kind: str
    tier: int
    edition: str = ""
    precision: str = "qualitative"  # numeric | class | qualitative
    locator: str = ""
    quote: str = ""
    population: str = "unknown"
    directness: float = 1.0
    n: Optional[int] = None
    # finding
    label: str = ""
    category: str = "symptom"
    hpo_id: str = ""
    code: str = ""  # finding code, assigned when the claim is mapped onto the vocabulary
    p: Optional[float] = None
    background: Optional[float] = None  # 1 - specificity, when the source states it
    # test
    test_id: str = ""
    positive: list[str] = field(default_factory=list)
    # harm / mimic
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple:
        if self.kind == "finding":
            return ("finding", self.code)
        if self.kind == "test":
            return ("test", self.test_id, tuple(sorted(self.positive)))
        if self.kind == "mimic":
            return ("mimic", self.extra.get("name", "").lower())
        return (self.kind,)

    def citation(self) -> str:
        loc = f" {self.locator}" if self.locator else ""
        return f"{self.source_id}{(' ' + self.edition) if self.edition else ''}{loc}"


def truncate_words(text: str, n: int = 30) -> str:
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")
