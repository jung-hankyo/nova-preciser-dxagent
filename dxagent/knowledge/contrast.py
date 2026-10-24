"""Contrast cards (A vs B), elicited by the LLM for two live confusable diseases.

This is the one knowledge path in v0.4 that is still LLM-only, kept because the contrast step works well and is
cheap. Its reach is bounded so it cannot produce confident errors on its own:

    vocabulary   findings must map onto EXISTING finding codes (registry or compiled), tests onto EXISTING
                 registry tests and outcomes; anything else is dropped, never added to a registry
    confidence   self-consistency over K samples, then capped at `contrast_conf_cap` (default 0.55)
    precedence   a contrast entry is used only when neither disease card (own or inherited) has the entry
    index        contrast knowledge never admits a disease through the reverse index
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from dxagent import usage
from typing import Optional

from dxagent.config import FREQ, FREQ_LEVELS, Settings
from dxagent.knowledge.cards import CardStore, ContrastCard, ContrastFinding, ContrastTest, Provenance, normalize, slug
from dxagent.llm import LLMLike
from dxagent.prompts import CONTRAST_SYSTEM, contrast_user
from dxagent.schemas import ContrastDraft, OutcomeLevel

log = logging.getLogger(__name__)


def _median_level(levels: list[str]) -> str:
    idx = sorted(FREQ_LEVELS.index(l) for l in levels)
    return FREQ_LEVELS[int(statistics.median_low(idx))]


def _agreement(levels: list[str], k: int) -> float:
    """Share of all K samples that produced the entry within one level of the median."""
    med = FREQ_LEVELS.index(_median_level(levels))
    return sum(1 for l in levels if abs(FREQ_LEVELS.index(l) - med) <= 1) / k


def _conf(agreement: float, k: int) -> float:
    return 0.55 if k == 1 else 0.4 + 0.5 * agreement


def _quorum(k: int, strict: bool) -> int:
    return k // 2 + 1 if strict else math.ceil(k / 2)


def _dist(levels: list[OutcomeLevel], allowed: list[str]) -> dict[str, float]:
    d: dict[str, float] = {}
    for ol in levels:
        o = slug(ol.outcome)
        if o in allowed:
            d[o] = d.get(o, 0.0) + FREQ[ol.level]
    return normalize(d) if d else {}


class ContrastService:
    def __init__(self, llm: LLMLike, store: CardStore, settings: Settings):
        self.llm = llm
        self.store = store
        self.s = settings

    def _sample(self, user: str, k: int) -> list[ContrastDraft]:
        def one(_):
            try:
                return self.llm.structured(ContrastDraft, CONTRAST_SYSTEM, user)
            except Exception as exc:
                log.warning("Contrast sample failed: %s", exc)
                return None

        with ThreadPoolExecutor(max_workers=max(1, k)) as pool:
            return [r for r in pool.map(usage.bind(one), range(k)) if r is not None]

    def existing_code(self, code: str, label: str) -> Optional[str]:
        """Map a drafted (code, label) onto an existing finding code, or None (never creates one)."""
        code = slug(code)
        canon = self.store.canonical(code)
        if self.store.finding_def(canon) is not None:
            return canon
        sims = self.store.lexicon().similar(code, label, k=1)
        if sims and sims[0][1] >= self.s.dedup_auto:
            return self.store.canonical(sims[0][0])
        return None

    def _vocab_lines(self, a_id: str, b_id: str) -> list[str]:
        codes = self.store.codes_for(a_id) | self.store.codes_for(b_id)
        lines = [f"{c}: {self.store.finding_def(c).label}" for c in sorted(codes) if self.store.finding_def(c)]
        if len(lines) < 120:
            lines += [f"{c}: {v['label']}" for c, v in self.store.vocabulary().items() if c not in codes][: 300 - len(lines)]
        return lines

    def draft_contrast(self, a_id: str, b_id: str, presentation: str, samples: Optional[int] = None) -> Optional[ContrastCard]:
        a, b = self.store.get(a_id), self.store.get(b_id)
        if a is None or b is None or self.store.has_contrast(a.id, b.id):
            return None
        k = samples or self.s.contrast_samples
        test_lines = [f"{t.id} ({t.name}, {t.type}): outcomes {', '.join(t.outcomes)}" for t in self.store.tests()]
        user = contrast_user(a.name, b.name, presentation, self._vocab_lines(a.id, b.id), test_lines)
        drafts = self._sample(user, k)
        if not drafts:
            return None
        k = len(drafts)
        cap = self.s.contrast_conf_cap
        by_code: dict[str, list] = {}
        dropped = 0
        for d in drafts:
            seen = set()
            for f in d.findings:
                code = self.existing_code(f.code, f.label)
                if code is None:
                    dropped += 1
                    continue
                if code not in seen:
                    seen.add(code)
                    by_code.setdefault(code, []).append(f)
        findings = []
        for code, rows in by_code.items():
            if len(rows) < _quorum(k, self.s.strict_quorum):
                continue
            la, lb = [r.freq_in_a for r in rows], [r.freq_in_b for r in rows]
            conf = min(_conf(min(_agreement(la, k), _agreement(lb, k)), k), cap)
            fd = self.store.finding_def(code)
            findings.append(ContrastFinding(code=code, label=fd.label if fd else rows[0].label, source=fd.source if fd else rows[0].source,
                                            p_a=FREQ[_median_level(la)], p_b=FREQ[_median_level(lb)], conf=conf,
                                            cluster=fd.cluster if fd else code))
        tests: dict[str, ContrastTest] = {}
        counts = Counter(slug(t.test_id) for d in drafts for t in d.tests)
        for tid, n in counts.items():
            td = self.store.test_def(tid)
            if td is None or n < _quorum(k, self.s.strict_quorum):
                continue
            rows = [t for d in drafts for t in d.tests if slug(t.test_id) == tid]
            da = [x for x in (_dist(r.outcomes_a, td.outcomes) for r in rows) if x]
            db = [x for x in (_dist(r.outcomes_b, td.outcomes) for r in rows) if x]
            if not da or not db:
                continue
            avg = lambda ds: normalize({o: sum(d.get(o, 0.0) for d in ds) / len(ds) for o in td.outcomes})  # noqa: E731
            tests[tid] = ContrastTest(dist_a=avg(da), dist_b=avg(db), conf=min(_conf(n / k, k), cap))
        if not findings and not tests:
            return None
        cc = ContrastCard(a=a.id, b=b.id, findings=findings, tests=tests,
                          management_differs=Counter(d.management_differs for d in drafts).most_common(1)[0][0],
                          provenance=Provenance(status="llm_contrast",
                                                source=f"LLM contrast, {k} samples, existing codes only ({dropped} dropped), conf cap {cap}"))
        self.store.add_contrast(cc, persist=True)
        return cc
