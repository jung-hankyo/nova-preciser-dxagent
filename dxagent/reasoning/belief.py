"""Deterministic belief update. No LLM calls.

For every hypothesis d and every observation:

    binary finding f (present / absent):   LLR = log P(obs | d) - log P(obs | background)
    test t with outcome o:                  LLR = log P(o | d)  - log P(o | background)

P(. | d) comes from the disease card (or a contrast card), shrunk toward the background by
the entry's confidence; diseases with no entry for f use the background, i.e. LLR = 0.
Each LLR is multiplied by the observation's confidence (a vague patient answer moves beliefs
less than a lab report). Within a correlation cluster only the strongest supporting and the
strongest opposing LLR count. A residual "other" hypothesis always uses the background.

Because every disease is scored against the same background, the ratio between two diseases
after an observation is P(obs | A) / P(obs | B): the likelihood ratio is formed at run time
between the diseases actually on the differential.

v0.4 (ontology tree). When the list holds a disease and one of its ontology descendants (pneumonia and
Legionnaires' disease), they are not rivals: the ancestor's entry stands for "this disease, any other
subtype", and the subtree shares the ancestor's prior bucket. Every listed node is still one mutually
exclusive hypothesis, so the update itself is unchanged; only the priors are made consistent
(`tree_priors`), and the differential reports each node's tree parent and subtree mass for the planner,
which separates siblings only once their parent leads (reasoning/planner.py `view_groups`).
"""

from __future__ import annotations

import math
from typing import Any

from dxagent.config import PRIOR_BUCKET, Settings
from dxagent.knowledge.cards import CardStore, norm_key


def merge_findings(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Latest observation per code wins; within the same turn the more confident one wins."""
    by_code: dict[str, dict[str, Any]] = {f["code"]: f for f in existing}
    for f in new:
        old = by_code.get(f["code"])
        if old is None or f["turn"] > old["turn"] or (f["turn"] == old["turn"] and f["confidence"] >= old["confidence"]):
            if old is not None and old.get("chief_complaint") and not f.get("chief_complaint"):
                f = {**f, "chief_complaint": True}
            by_code[f["code"]] = f
    return list(by_code.values())


def merge_tests(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {t["test_id"]: t for t in existing}
    for t in new:
        old = by_id.get(t["test_id"])
        if old is None or t["turn"] > old["turn"] or (t["turn"] == old["turn"] and t["confidence"] >= old["confidence"]):
            by_id[t["test_id"]] = t
    return list(by_id.values())


def apply_numeric_rules(lab_values: list[dict[str, Any]], store: CardStore, turn: int) -> list[dict[str, Any]]:
    """Map numeric lab values to test outcomes using registry thresholds, not the LLM."""
    out = []
    for lv in lab_values:
        key = norm_key(lv.get("analyte", ""))
        for test_id, rule in store.numeric_rules():
            if norm_key(rule.analyte) != key:
                continue
            if rule.unit and lv.get("unit") and norm_key(rule.unit) != norm_key(lv["unit"]):
                continue  # unit mismatch: never guess a conversion
            outcome = rule.evaluate(float(lv["value"]))
            if outcome:
                out.append({"test_id": test_id, "outcome": outcome, "confidence": 0.97, "turn": turn,
                            "evidence": f"{lv.get('analyte')} = {lv['value']} {lv.get('unit', '')}".strip()})
    return out


def apply_vital_rules(values: list[dict[str, Any]], store: CardStore, turn: int) -> list[dict[str, Any]]:
    """v0.3.1: map numeric vital signs to findings with registry thresholds, never by the LLM (as for labs)."""
    out = []
    for v in values:
        key = norm_key(v.get("analyte", ""))
        for code, rule in store.finding_rules():
            if norm_key(rule.analyte) != key:
                continue
            if rule.unit and v.get("unit") and norm_key(rule.unit) != norm_key(v["unit"]):
                continue  # unit mismatch: never guess a conversion
            outcome = rule.evaluate(float(v["value"]))
            if outcome in ("present", "absent"):
                fd = store.finding_def(code)
                out.append({"code": code, "label": fd.label if fd else code, "present": outcome == "present",
                            "confidence": 0.97, "source": fd.source if fd else "exam", "turn": turn,
                            "evidence": f"{v.get('analyte')} = {v['value']} {v.get('unit', '')}".strip(),
                            "chief_complaint": False})
    return out


def _normalized_entropy(probs: list[float]) -> float:
    ps = [p for p in probs if p > 0]
    if len(ps) <= 1:
        return 0.0
    return -sum(p * math.log(p) for p in ps) / math.log(len(ps))


def disease_llrs(
    disease_id: str, findings: list[dict[str, Any]], tests: list[dict[str, Any]], store: CardStore
) -> tuple[float, list[str]]:
    pos: dict[str, float] = {}
    neg: dict[str, float] = {}
    notes: list[str] = []
    for f in findings:
        code = f["code"]
        if store.finding_p(disease_id, code) is None:
            continue
        bg = store.background(code)
        p = store.finding_p_eff(disease_id, code)
        llr = math.log(p / bg) if f["present"] else math.log((1 - p) / (1 - bg))
        llr *= float(f["confidence"])
        cl = store.cluster(code)
        if llr >= 0:
            pos[cl] = max(pos.get(cl, 0.0), llr)
        else:
            neg[cl] = min(neg.get(cl, 0.0), llr)
        notes.append(f"{'+' if f['present'] else '-'}{code}: {llr:+.2f}")
    for t in tests:
        td = store.test_def(t["test_id"])
        if td is None or t["outcome"] not in td.outcomes or store.test_dist(disease_id, t["test_id"]) is None:
            continue
        p = store.test_dist_eff(disease_id, t["test_id"])[t["outcome"]]
        llr = math.log(p / td.background[t["outcome"]]) * float(t["confidence"])
        cl = f"test:{t['test_id']}"
        if llr >= 0:
            pos[cl] = max(pos.get(cl, 0.0), llr)
        else:
            neg[cl] = min(neg.get(cl, 0.0), llr)
        notes.append(f"{t['test_id']}={t['outcome']}: {llr:+.2f}")
    return sum(pos.values()) + sum(neg.values()), notes


def evidence_breakdown(disease_id: str, findings: list[dict[str, Any]], tests: list[dict[str, Any]], store: CardStore,
                       limit: int = 12) -> list[dict[str, Any]]:
    """v0.4.1 diagnostics: each observation's tempered log-likelihood ratio for one disease and the knowledge it came
    from (own card status / tier / precision, inherited, contrast). Largest |llr| first. Cluster pooling is not
    applied here, so the sum can exceed the LLR the update actually used."""
    rows: list[dict[str, Any]] = []
    for f in findings:
        if f.get("chief_complaint") or store.finding_p(disease_id, f["code"]) is None:
            continue
        bg = store.background(f["code"])
        p = store.finding_p_eff(disease_id, f["code"])
        llr = (math.log(p / bg) if f["present"] else math.log((1 - p) / (1 - bg))) * float(f["confidence"])
        rows.append({"obs": ("+" if f["present"] else "-") + f["code"], "llr": round(llr, 3),
                     **store.entry_source(disease_id, code=f["code"])})
    for t in tests:
        td = store.test_def(t["test_id"])
        if td is None or t["outcome"] not in td.outcomes or store.test_dist(disease_id, t["test_id"]) is None:
            continue
        p = store.test_dist_eff(disease_id, t["test_id"])[t["outcome"]]
        llr = math.log(p / td.background[t["outcome"]]) * float(t["confidence"])
        rows.append({"obs": f"{t['test_id']}={t['outcome']}", "llr": round(llr, 3), **store.entry_source(disease_id, test_id=t["test_id"])})
    rows.sort(key=lambda r: -abs(r["llr"]))
    return rows[:limit]


def tree_parents(hypotheses: list[dict[str, Any]], store: CardStore, enabled: bool = True) -> dict[str, str | None]:
    """{disease id: nearest LISTED ontology ancestor or None}. Empty structure when the tree is off."""
    ids = [h["disease_id"] for h in hypotheses]
    out: dict[str, str | None] = {i: None for i in ids}
    if not enabled or len(ids) < 2:
        return out
    listed = set(ids)
    for i in ids:
        anc = store.ancestors(i)
        cands = [(d, a) for a, d in anc.items() if a in listed and a != i]
        if cands:
            out[i] = min(cands)[1]
    return out


def tree_priors(hypotheses: list[dict[str, Any]], parents: dict[str, str | None], floor: float = 0.1) -> dict[str, float]:
    """Consistent priors over a tree of listed hypotheses.

    A root keeps its bucket as the prior of its whole subtree. Each child's subtree prior is its own bucket,
    clipped so that the children together leave at least `floor` of the parent's subtree prior to the parent's
    own node ("any other subtype"). Without listed ancestors this returns the buckets unchanged.
    """
    bucket = {h["disease_id"]: float(h["prior"]) for h in hypotheses}
    children: dict[str, list[str]] = {}
    for c, p in parents.items():
        if p is not None:
            children.setdefault(p, []).append(c)
    if not children:
        return dict(bucket)
    out: dict[str, float] = {}

    def assign(node: str, total: float) -> None:
        kids = sorted(children.get(node, []))
        want = {k: min(bucket[k], total) for k in kids}
        room = (1.0 - floor) * total
        z = sum(want.values())
        if z > room > 0:
            want = {k: v * room / z for k, v in want.items()}
        out[node] = max(total - sum(want.values()), 0.0)
        for k in kids:
            assign(k, want[k])

    for root in [i for i, p in parents.items() if p is None]:
        assign(root, bucket[root])
    for i in bucket:  # cycles cannot occur in an is_a DAG, but never drop a hypothesis
        out.setdefault(i, bucket[i])
    return out


def compute_posterior(
    hypotheses: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    store: CardStore,
    settings: Settings,
    tests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return {differential, other_prob, entropy, unexplained, unexplained_codes, trace}.

    Prior scaling: each hypothesis keeps its bucket, but at every update the listed priors are rescaled
    TOGETHER so that "other" stays within [other_mass, 0.3]. The factor is common to all listed diseases,
    so their ranking never depends on what else is listed; the absolute probabilities, and the split
    between the list and "other", do. A disease's effective prior is therefore not fixed on entry. The
    size of this effect is logged per case (`list_effect_log`, v0.3 B1 diagnostic).
    """
    tests = tests or []
    if not hypotheses:
        return {"differential": [], "other_prob": 1.0, "entropy": 1.0, "unexplained": [], "trace": {}}
    used = [f for f in findings if not f.get("chief_complaint")]
    parents = tree_parents(hypotheses, store, getattr(settings, "tree", True))
    priors = tree_priors(hypotheses, parents, getattr(settings, "tree_residual_floor", 0.1))

    prior_sum = sum(priors.values())
    other_prior = min(max(1.0 - prior_sum, settings.other_mass), 0.3)
    scale = (1.0 - other_prior) / prior_sum if prior_sum > 0 else 1.0

    log_scores: dict[str, float] = {}
    trace: dict[str, list[str]] = {}
    for h in hypotheses:
        llr, notes = disease_llrs(h["disease_id"], used, tests, store)
        log_scores[h["disease_id"]] = math.log(max(priors[h["disease_id"]] * scale, 1e-9)) + llr
        known = store.has_knowledge(h["disease_id"])
        via = store.inherited_from(h["disease_id"])
        trace[h["disease_id"]] = ([f"prior {h['prior']:.3f}" + (f" (tree {priors[h['disease_id']]:.3f})" if parents.get(h["disease_id"]) or
                                                                 priors[h["disease_id"]] != h["prior"] else "")]
                                  + ([f"inherits {via}"] if via else []) + (notes if known else ["no card: prior only"]))

    log_other = math.log(other_prior)
    m = max(max(log_scores.values()), log_other)
    z = sum(math.exp(v - m) for v in log_scores.values()) + math.exp(log_other - m)

    differential = []
    for h in hypotheses:
        card = store.get(h["disease_id"])
        via = None if card else store.inherited_from(h["disease_id"])
        differential.append({
            "disease_id": h["disease_id"],
            "name": card.name if card else h["name"],
            "prob": math.exp(log_scores[h["disease_id"]] - m) / z,
            "cant_miss": bool(card.cant_miss if card else h.get("cant_miss")),
            "has_card": card is not None,
            "card_status": card.provenance.status if card else (f"inherited:{via}" if via else "none"),
            "tree_parent": parents.get(h["disease_id"]),
        })
    _subtree_mass(differential)
    differential.sort(key=lambda d: d["prob"], reverse=True)
    other_prob = math.exp(log_other - m) / z

    top_codes: set[str] = set()
    for d in differential[:5]:
        top_codes |= store.codes_for(d["disease_id"])
    unexplained_f = [f for f in used if f["present"] and f["code"] not in top_codes and f["confidence"] >= 0.6]
    unexplained = sorted({f["label"] or f["code"] for f in unexplained_f})
    entropy = _normalized_entropy([d["prob"] for d in differential] + [other_prob])
    return {"differential": differential, "other_prob": other_prob, "entropy": entropy, "unexplained": unexplained,
            "unexplained_codes": sorted({f["code"] for f in unexplained_f}), "trace": trace}


def _subtree_mass(differential: list[dict[str, Any]]) -> None:
    """Annotate each entry with `subtree_prob`: its own mass plus all listed descendants' (display and planner)."""
    probs = {d["disease_id"]: d["prob"] for d in differential}
    parent = {d["disease_id"]: d.get("tree_parent") for d in differential}
    total = dict(probs)
    for i, p in probs.items():
        seen = set()
        a = parent.get(i)
        while a is not None and a not in seen:
            seen.add(a)
            total[a] = total.get(a, 0.0) + p
            a = parent.get(a)
    for d in differential:
        d["subtree_prob"] = total[d["disease_id"]]


def log_mixture_evidence(
    hypotheses: list[dict[str, Any]], findings: list[dict[str, Any]], store: CardStore, settings: Settings,
    tests: list[dict[str, Any]] | None = None,
) -> float:
    """log sum_j pi_j exp(LLR_j) over the current list plus "other" (LLR 0), with the same prior scaling
    as compute_posterior: how well the current differential as a whole explains the evidence."""
    tests = tests or []
    used = [f for f in findings if not f.get("chief_complaint")]
    if not hypotheses:
        return 0.0
    priors = tree_priors(hypotheses, tree_parents(hypotheses, store, getattr(settings, "tree", True)),
                         getattr(settings, "tree_residual_floor", 0.1))
    prior_sum = sum(priors.values())
    other_prior = min(max(1.0 - prior_sum, settings.other_mass), 0.3)
    scale = (1.0 - other_prior) / prior_sum if prior_sum > 0 else 1.0
    terms = [math.log(max(priors[h["disease_id"]] * scale, 1e-9)) + disease_llrs(h["disease_id"], used, tests, store)[0]
             for h in hypotheses]
    terms.append(math.log(other_prior))
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def prune_hypotheses(hypotheses: list[dict[str, Any]], differential: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    keep: set[str] = set()
    for i, d in enumerate(differential):
        if i < settings.max_differential and d["prob"] >= settings.drop_below:
            keep.add(d["disease_id"])
        elif d["cant_miss"] and d["prob"] >= settings.drop_below / 2:
            keep.add(d["disease_id"])
    # v0.4: a kept node keeps its listed ancestors, so a subtree never loses the parent that sets its prior.
    parent = {d["disease_id"]: d.get("tree_parent") for d in differential}
    for i in list(keep):
        a, seen = parent.get(i), set()
        while a is not None and a not in seen:
            seen.add(a)
            keep.add(a)
            a = parent.get(a)
    return [h for h in hypotheses if h["disease_id"] in keep]


def _card_prior(card, settings: Settings, complaint: str | None) -> tuple[float, str]:
    """(prior, source) for an index-added disease. v0.3.1 (P1): the card's prevalence for S's complaint only;
    v0.4: else the card's complaint-independent floor (`base_prevalence`, e.g. very_rare for Orphanet diseases);
    v0.3 (C3): the card's single bucket; v0.2: the fixed default."""
    if getattr(settings, "prior_context", False):
        b = card.prevalence_for(complaint or "") or getattr(card, "base_prevalence", None)
        return (PRIOR_BUCKET[b], f"reverse_index:{b}") if b else (settings.reverse_prior, "reverse_index:default")
    if getattr(settings, "reverse_prevalence_prior", False) and card.prevalence:
        return PRIOR_BUCKET[card.prevalence], f"reverse_index:{card.prevalence}"
    return settings.reverse_prior, "reverse_index"


def reverse_index_candidates(
    hypotheses: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    store: CardStore,
    settings: Settings,
    tests: list[dict[str, Any]] | None = None,
    complaint: str | None = None,
) -> list[dict[str, Any]]:
    """Known diseases, absent from the differential, that the evidence would make at least
    `reverse_min_prob` likely if added (computed exactly by re-running the update with each one).

    Deterministic complement to LLM hypothesis generation: a disease that is never proposed can
    never be diagnosed, and the LLM's blind spots tend to persist across calls. Only cards with at
    least one observation they actually speak to are evaluated, so evidence-free cards never enter.
    v0.2: added diseases get a fixed prior (`reverse_prior`), since cards carried no prior of their own.

    v0.3 (C3, `reverse_evidence_rule`): a candidate starts from its card's own prevalence bucket, and is
    admitted when the evidence favors it over the current differential as a whole by at least `reverse_bf`
    (a Bayes factor, independent of how the list happens to be composed) and its posterior would reach
    `reverse_floor` (can't-miss diseases are exempt from the floor). Candidates are ranked by that factor.
    """
    tests = tests or []
    present = {h["disease_id"] for h in hypotheses}
    used = [f for f in findings if not f.get("chief_complaint")]
    evidence_rule = getattr(settings, "reverse_evidence_rule", False)
    log_mix = log_mixture_evidence(hypotheses, findings, store, settings, tests) if evidence_rule else 0.0
    scored = []
    # v0.4: only cards with an entry for some observation are candidates (inverted index), only compiled
    # (or fixture) cards may enter, and a cheap screen against the current normalizer skips hopeless ones
    # before the exact re-computation.
    speaking = store.cards_speaking_to({f["code"] for f in used}, {t["test_id"] for t in tests})
    log_z = log_mixture_evidence(hypotheses, findings, store, settings, tests) if not evidence_rule else log_mix
    for cid in sorted(speaking):
        card = store.card_meta(cid) if hasattr(store, "card_meta") else store.get(cid)
        if card is None or card.id in present or not store.index_eligible(card):
            continue
        llr, _ = disease_llrs(card.id, used, tests, store)
        if llr == 0.0:
            continue  # the card says nothing about any observation
        if not evidence_rule:
            prior0, _ = _card_prior(card, settings, complaint)
            approx = math.exp(math.log(prior0) + llr - log_z)
            if approx / (1 + approx) < settings.reverse_min_prob / 10:
                continue
        if evidence_rule:
            log_bf = llr - log_mix
            if log_bf < math.log(settings.reverse_bf):
                continue
            if getattr(settings, "prior_context", False):
                prior, src = _card_prior(card, settings, complaint)
            else:
                prior = PRIOR_BUCKET.get(card.prevalence or "", settings.reverse_prior)
                src = f"reverse_index:{card.prevalence or 'default'}"
            cand = {"disease_id": card.id, "name": card.name, "prior": prior,
                    "prior_source": src, "cant_miss": card.cant_miss}
            res = compute_posterior(hypotheses + [cand], findings, store, settings, tests)
            p = next(d["prob"] for d in res["differential"] if d["disease_id"] == card.id)
            if p >= settings.reverse_floor or card.cant_miss:
                scored.append((log_bf, cand))
            continue
        prior, src = _card_prior(card, settings, complaint)
        cand = {"disease_id": card.id, "name": card.name, "prior": prior, "prior_source": src, "cant_miss": card.cant_miss}
        res = compute_posterior(hypotheses + [cand], findings, store, settings, tests)
        p = next(d["prob"] for d in res["differential"] if d["disease_id"] == card.id)
        if p >= settings.reverse_min_prob:
            scored.append((p, cand))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[: settings.reverse_max_per_turn]]


# --------------------------------------------------------------------------- v0.3.1 diagnostics (P1, step 1)


def prior_sensitivity(
    hypotheses: list[dict[str, Any]], findings: list[dict[str, Any]], store: CardStore, settings: Settings,
    tests: list[dict[str, Any]] | None = None, alpha: float = 0.5,
) -> dict[str, Any]:
    """How much the prior decides the answer: recompute with flat and with tempered (pi^alpha) priors.

    Both keep the sum of listed priors, so "other" is unchanged and only the split among listed diseases moves.
    Pure function of the stored state; no LLM call.
    """
    if not hypotheses:
        return {}
    total = sum(h["prior"] for h in hypotheses)

    def run(weights: list[float]) -> dict[str, Any]:
        z = sum(weights) or 1.0
        hs = [{**h, "prior": total * w / z} for h, w in zip(hypotheses, weights)]
        diff = compute_posterior(hs, findings, store, settings, tests)["differential"]
        return {"leader": diff[0]["disease_id"], "p": round(diff[0]["prob"], 4)} if diff else {"leader": None, "p": 0.0}

    base = run([h["prior"] for h in hypotheses])
    flat = run([1.0] * len(hypotheses))
    temp = run([h["prior"] ** alpha for h in hypotheses])
    return {"leader": base["leader"], "p_leader": base["p"], "leader_flat": flat["leader"], "p_flat": flat["p"],
            "leader_tempered": temp["leader"], "p_tempered": temp["p"], "alpha": alpha,
            "changed_flat": flat["leader"] != base["leader"], "changed_tempered": temp["leader"] != base["leader"]}


def context_coverage(
    findings: list[dict[str, Any]], differential: list[dict[str, Any]], store: CardStore, intake_turns: int = 3,
    top: int = 5, min_conf: float = 0.6,
) -> dict[str, Any]:
    """Present findings (primary complaint excluded) that no top card has an entry for, so they moved nothing.

    Split by when they arrived (opening statement and intake vs later) and by channel. A high uncovered share in
    the opening/intake group is the signal that context is being lost because the cards lack risk factors,
    exposures, medications, or vital signs.
    """
    codes: set[str] = set()
    for d in differential[:top]:
        codes |= store.codes_for(d["disease_id"])
    out: dict[str, Any] = {"opening_intake": [0, 0], "later": [0, 0], "by_source": {}, "by_category": {}, "uncovered": []}
    for f in findings:
        if f.get("chief_complaint") or not f["present"] or f["confidence"] < min_conf:
            continue
        grp = "opening_intake" if f.get("turn", 0) < intake_turns else "later"
        miss = f["code"] not in codes
        out[grp][0] += 1
        out[grp][1] += int(miss)
        src = out["by_source"].setdefault(f.get("source", "history"), [0, 0])
        src[0] += 1
        src[1] += int(miss)
        fd = store.finding_def(f["code"])
        cat = out["by_category"].setdefault((fd.category or "uncategorized") if fd else "unregistered", [0, 0])
        cat[0] += 1
        cat[1] += int(miss)
        if miss:
            out["uncovered"].append(f["code"])
    return out
