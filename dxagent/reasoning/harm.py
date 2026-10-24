"""Pairwise confusion weights (plan section "Harm, reconsidered").

Harm is a property of a CONFUSION, not of a disease: the cost of acting as if i when the truth is j.

    L(i -> j) = miss(j) * urgency(j) * (1 - overlap(i, j)) + overtreat(i)

    miss(j)          untreated course of j: lethal / serious / self-limited
    urgency(j)       how fast harm accrues: hours / days / weeks
    overlap(i, j)    fraction of j's treatments that i's treatment also delivers (Jaccard of treatment sets)
    overtreat(i)     harm of giving i's treatment to someone without i: high / moderate / low

Pair weight used by the planner:
    objective "accuracy":  w_ij = 1 for every pair (every confusion costs one misdiagnosis; benchmark)
    objective "clinical":  w_ij = 1 + safety_weight * (L(i->j) + L(j->i)) / 2
"""

from __future__ import annotations

from typing import Optional

from dxagent.config import MISS_HARM, OVERTREAT_HARM, URGENCY, Settings
from dxagent.knowledge.cards import CardStore, HarmProfile


def _profile(store: CardStore, disease_id: str) -> Optional[HarmProfile]:
    """The disease's harm profile: its own card, else the nearest carded ancestor (v0.4). A compiled card whose
    harm profile no source stated (empty `evidence`) counts as unknown, like a missing card."""
    lineage = store.lineage(disease_id) if hasattr(store, "lineage") else []
    card = store.get(lineage[0][0]) if lineage else store.get(disease_id)
    if card is None:
        return None
    if card.provenance.status in ("compiled", "on_demand", "provisional") and not card.harm.evidence:
        return None
    return card.harm


def overlap(a: HarmProfile, b: HarmProfile) -> float:
    ta, tb = set(a.treatments), set(b.treatments)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def confusion_loss(store: CardStore, i: str, j: str) -> float:
    pi, pj = _profile(store, i), _profile(store, j)
    if pi is None or pj is None:
        return MISS_HARM["serious"]
    return MISS_HARM[pj.untreated_course] * URGENCY[pj.urgency] * (1 - overlap(pi, pj)) + OVERTREAT_HARM[pi.treatment_risk]


def pair_weights(store: CardStore, ids: list[str], settings: Settings) -> list[list[float]]:
    n = len(ids)
    w = [[1.0] * n for _ in range(n)]
    if settings.objective != "clinical":
        return w
    for a in range(n):
        for b in range(a + 1, n):
            v = 1.0 + settings.safety_weight * 0.5 * (confusion_loss(store, ids[a], ids[b]) + confusion_loss(store, ids[b], ids[a]))
            w[a][b] = w[b][a] = v
    return w
