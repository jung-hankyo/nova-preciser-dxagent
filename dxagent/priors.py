"""Prior context and prior assignment (v0.3.1, P1).

The prior may condition ONLY on S = {age, sex, care setting, complaint}. Everything else the patient says is
evidence, scored by the deterministic update against the cards. This module makes that rule structural:

    1. Complaint selection is a deterministic rule over LLM-normalized complaints (stated reason for the visit,
       then the most urgent typical acuity, then the earliest mention). The selected complaint's finding is the
       one finding excluded from the update.
    2. Every bucket, whichever route proposed the disease (generator, critic, veto, report dissent), is assigned
       here by a prior assessor that is SHOWN ONLY S and the disease names. It cannot use findings it never sees.
       The generator still proposes WHICH diseases to consider; its own bucket is logged, never used.
    3. The assessor is sampled K times and the median bucket is taken (self-consistency, as for cards).
    4. If the card has a prevalence for S's complaint and it differs from the assessor by >= prior_reconcile_steps
       buckets, the two meet in the middle (rounded toward "uncommon"). Otherwise the assessor wins: it alone
       sees age, sex and setting.
    5. Buckets are reassigned whenever S changes (for example when intake supplies age and sex). A disease's
       prior is a function of S, never of the evidence.

The reverse index (reasoning/belief.py) stays LLM-free: it uses the card's complaint-keyed prevalence only.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from dxagent import usage
from typing import Any, Optional

from dxagent.config import PRIOR_BUCKET, Settings

log = logging.getLogger(__name__)

BUCKETS = list(PRIOR_BUCKET)  # very_common (0) ... very_rare (4)
MIDDLE = BUCKETS.index("uncommon")

PRIOR_ASSESS_SYSTEM = """You estimate pretest probabilities for a diagnostic agent.

You are shown ONLY who the patient is and why they came: age, sex, care setting, and one presenting complaint.
Nothing else about the case is shown to you, by design; do not imagine other findings.

For each listed disease, choose the bucket for P(disease | these facts) among patients like this one:
  very_common  about 1 in 4
  common       about 1 in 12
  uncommon     about 1 in 50
  rare         about 1 in 200
  very_rare    about 1 in 1000
Judge each disease on its own (the buckets need not sum to 1). If age or sex is unknown, average over adults.
Return every listed disease_id exactly once."""


def prior_assess_user(S: dict[str, Any], diseases: list[tuple[str, str]]) -> str:
    def v(x: Any) -> str:
        return "unknown" if x in (None, "") else str(x)

    lines = "\n".join(f"- {i}: {n}" for i, n in diseases)
    return (f"Age: {v(S.get('age'))}\nSex: {v(S.get('sex'))}\nSetting: {v(S.get('setting'))}\n"
            f"Presenting complaint: {v(S.get('complaint_label'))}\n\nDiseases (id: name):\n{lines}")


def signature(S: Optional[dict[str, Any]]) -> list:
    """The part of S a prior depends on. A change here reassigns every bucket."""
    S = S or {}
    return [S.get("age"), S.get("sex"), S.get("setting"), S.get("complaint")]


def select_complaint(cands: list[dict[str, Any]]) -> tuple[int, str]:
    """Deterministic primary-complaint rule over resolved candidates (in order of mention).

    Each candidate: {stated_reason: bool, acuity: int (1 most urgent), pos: int (position in the text)}.
    Returns (index, rule) with rule in {only, stated, acuity, earliest}.
    """
    if not cands:
        return -1, "none"
    stated = [i for i, c in enumerate(cands) if c.get("stated_reason")]
    pool = stated or list(range(len(cands)))
    if len(pool) == 1:
        return pool[0], ("stated" if stated and len(cands) > 1 else "only")
    best = min(cands[i].get("acuity", 3) for i in pool)
    top = [i for i in pool if cands[i].get("acuity", 3) == best]
    if len(top) == 1:
        return top[0], "acuity"
    return min(top, key=lambda i: (cands[i].get("pos", 10**6), i)), "earliest"


def _toward_middle(x: float) -> int:
    if x == int(x):
        return int(x)
    return int(x) + 1 if x < MIDDLE else int(x)  # e.g. 0.5 -> 1, 3.5 -> 3, 1.5 -> 2, 2.5 -> 2


def median_bucket(buckets: list[str]) -> Optional[str]:
    idx = sorted(BUCKETS.index(b) for b in buckets if b in PRIOR_BUCKET)
    if not idx:
        return None
    n = len(idx)
    mid = idx[n // 2] if n % 2 else (idx[n // 2 - 1] + idx[n // 2]) / 2
    return BUCKETS[_toward_middle(mid)]


def reconcile(card: Optional[str], assessor: Optional[str], steps: int) -> tuple[Optional[str], str]:
    if card is None and assessor is None:
        return None, "none"
    if card is None:
        return assessor, "assessor"
    if assessor is None:
        return card, "card"
    i, j = BUCKETS.index(card), BUCKETS.index(assessor)
    if abs(i - j) < steps:
        return assessor, "assessor"
    return BUCKETS[_toward_middle((i + j) / 2)], "reconciled"


class PriorService:
    """Assigns pretest buckets from S alone. `llm` may be None (offline without a handler): card or fallback."""

    def __init__(self, llm, store, settings: Settings):
        self.llm, self.store, self.s = llm, store, settings

    def assess(self, S: dict[str, Any], diseases: list[tuple[str, str]]) -> tuple[dict[str, str], dict[str, list[str]]]:
        """(median bucket per disease, the K sampled buckets per disease). The votes measure bucket stability."""
        if self.llm is None or not diseases:
            return {}, {}
        from dxagent.schemas import PriorAssessment

        user = prior_assess_user(S, diseases)

        def one(_):
            try:
                return self.llm.structured(PriorAssessment, PRIOR_ASSESS_SYSTEM, user)
            except Exception as exc:
                log.warning("Prior assessment failed: %s", exc)
                return None

        k = max(1, self.s.prior_samples)
        with ThreadPoolExecutor(max_workers=k) as pool:
            outs = [o for o in pool.map(usage.bind(one), range(k)) if o is not None]
        votes: dict[str, list[str]] = {}
        for o in outs:
            for it in o.items:
                votes.setdefault(it.disease_id, []).append(it.prior_bucket)
        return {d: b for d, v in votes.items() if (b := median_bucket(v))}, votes

    def card_bucket(self, disease_id: str, S: dict[str, Any]) -> Optional[str]:
        card = self.store.get(disease_id)
        return card.prevalence_for(S.get("complaint") or "") if card else None

    def assign(self, S: dict[str, Any], hyps: list[dict[str, Any]], turn: int) -> tuple[list[dict[str, Any]], list[dict]]:
        """Reassign the bucket of every hypothesis whose prior was set under a different S (or never)."""
        sig = signature(S)
        todo = [h for h in hyps if h.get("prior_sig") != sig]
        if not todo:
            return hyps, []
        assessed, votes = self.assess(S, [(h["disease_id"], h.get("name") or h["disease_id"]) for h in todo])
        logs = []
        for h in todo:
            card_b = self.card_bucket(h["disease_id"], S)
            final, route = reconcile(card_b, assessed.get(h["disease_id"]), self.s.prior_reconcile_steps)
            if final is None:
                final, route = h.get("proposed_bucket") or h.get("prior_bucket") or "uncommon", "fallback"
            logs.append({"turn": turn, "disease_id": h["disease_id"], "entry": h.get("prior_source", ""),
                         "proposed": h.get("proposed_bucket"), "card": card_b, "assessor": assessed.get(h["disease_id"]),
                         "final": final, "route": route, "reassigned": h.get("prior_sig") is not None,
                         "votes": votes.get(h["disease_id"], [])})
            h["prior"], h["prior_bucket"], h["prior_route"], h["prior_sig"] = PRIOR_BUCKET[final], final, route, sig
            h["assessor_bucket"] = assessed.get(h["disease_id"])
        return hyps, logs

    def refresh_cards(self, S: dict[str, Any], hyps: list[dict[str, Any]], turn: int) -> list[dict]:
        """After cards were drafted, re-reconcile with the new card buckets. Deterministic, no LLM call."""
        sig, logs = signature(S), []
        for h in hyps:
            if h.get("prior_sig") != sig or h.get("prior_route") not in ("assessor", "fallback"):
                continue
            card_b = self.card_bucket(h["disease_id"], S)
            if card_b is None:
                continue
            final, route = reconcile(card_b, h.get("assessor_bucket"), self.s.prior_reconcile_steps)
            if final and (final != h.get("prior_bucket") or route != h.get("prior_route")):
                logs.append({"turn": turn, "disease_id": h["disease_id"], "entry": h.get("prior_source", ""),
                             "proposed": h.get("proposed_bucket"), "card": card_b, "assessor": h.get("assessor_bucket"),
                             "final": final, "route": route, "reassigned": True})
                h["prior"], h["prior_bucket"], h["prior_route"] = PRIOR_BUCKET[final], final, route
        return logs
