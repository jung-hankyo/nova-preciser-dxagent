"""Deterministic action planner: choose the move that best separates the diseases still confusable.

For a candidate action a with possible observations o:

    gain(a) = M * [P_err(b) - E_o P_err(b|o)]          accuracy: chance the current leader is wrong
            + lambda * [C(b) - E_o C(b|o)]             pairwise separation

    C(b) = sum_{i<j} w_ij * sqrt(b_i * b_j)            over the differential plus "other"

    VOI(a) = gain(a) - cost_weight * cost(a) - risk(a)

sqrt(b_i b_j) is the Bhattacharyya bound on confusing i with j; it is large only when BOTH are
plausible, so the planner looks for the question or test that splits live pairs, the way a
clinician asks "is it A or B?". The geometric mean is concave and posteriors are a martingale,
so E[C(b|o)] <= C(b): information can never score negative. w_ij comes from reasoning/harm.py
(all 1 in the default "accuracy" objective).

Observation models:
    history / exam finding   binary, P(report present | d) = r * p_d + (1 - r) * (1 - p_d)
    test                     categorical outcomes, P(o | d) = r * dist_d(o) + (1 - r) * background(o)
where r is the channel reliability and p_d, dist_d are confidence-shrunk card values.

Options are compared per TURN (up to k questions, exams, or cheap tests together; imaging and
procedures alone). Advanced imaging and procedures are gated until a disease they discriminate
is reasonably likely. Stop rules are in `decide`.

v0.4 (ontology tree). P_err and C are computed on a VIEW of the differential, not on its raw entries.
Listed diseases related by the ontology form subtrees; the view starts with one node per subtree (its
mass summed) and expands a subtree into its parent's own entry and its children only while that subtree
leads. Siblings are therefore separated only once their parent leads, and a non-leading subtree is
treated as one hypothesis when deciding what to ask. Without ontology relations the view is the
differential itself, so the objective is unchanged.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from dxagent.config import CHANNEL_RELIABILITY, MIN_TARGET_PROB, TEST_TYPES, Settings
from dxagent.knowledge.cards import ActionSpec, CardStore
from dxagent.reasoning.harm import pair_weights


@dataclass
class ScoredAction:
    action: ActionSpec
    voi: float
    gain: float
    reveals: list[str]  # finding codes, or [test_id] for a test


@dataclass
class Option:
    kind: str
    actions: list[ScoredAction]
    value: float


@dataclass
class Decision:
    commit: bool
    reason: str
    actions: list[ScoredAction] = field(default_factory=list)
    best_voi: float = 0.0
    scored: list[ScoredAction] = field(default_factory=list)


def p_error(probs: list[float]) -> float:
    return 1.0 - (max(probs) if probs else 0.0)


def confusion(probs: list[float], other_prob: float, w: list[list[float]]) -> float:
    """C(b) = sum_{i<j} w_ij sqrt(b_i b_j), with "other" as an extra member (weight 1)."""
    n = len(probs)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += w[i][j] * math.sqrt(probs[i] * probs[j])
        total += math.sqrt(probs[i] * other_prob)
    return total


def view_groups(differential: list[dict[str, Any]]) -> list[list[int]]:
    """Partition differential indices into the planner's view (see module docstring).

    Each group is a list of indices whose first element is the group's node. A group with more than one member
    is a collapsed subtree; the leading group is expanded (node alone, then each child's subtree) until the
    leader is a single entry. Entries without `tree_parent` links give singleton groups (the v0.3 behavior).
    """
    n = len(differential)
    pos = {d["disease_id"]: i for i, d in enumerate(differential)}
    parent = {i: pos.get(differential[i].get("tree_parent")) for i in range(n)}
    children: dict[int, list[int]] = {}
    for i, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(i)

    def subtree(i: int) -> list[int]:
        out, stack = [i], list(children.get(i, []))
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(children.get(j, []))
        return out

    groups = [subtree(i) for i in range(n) if parent[i] is None]
    probs = [d["prob"] for d in differential]
    for _ in range(n):  # each expansion strictly refines; bounded by n
        if not groups:
            break
        masses = [sum(probs[k] for k in g) for g in groups]
        lead = max(range(len(groups)), key=lambda g: (masses[g], -groups[g][0]))
        g = groups[lead]
        if len(g) == 1:
            break
        node = g[0]
        groups = groups[:lead] + [[node]] + [subtree(c) for c in children.get(node, [])] + groups[lead + 1:]
    return groups


class _Model:
    """Precomputed observation likelihoods for the current differential."""

    def __init__(self, differential: list[dict[str, Any]], other_prob: float, store: CardStore, settings: Settings):
        self.ids = [d["disease_id"] for d in differential]
        self.probs = [d["prob"] for d in differential]
        self.other = other_prob
        self.store = store
        self.s = settings
        w = pair_weights(store, self.ids, settings)
        self.groups = view_groups(differential)
        # Pair weight between two view nodes: the largest weight between their members (all 1 in accuracy mode).
        self.w = [[max(w[a][b] for a in ga for b in gb) if gi != gj else 0.0 for gj, gb in enumerate(self.groups)]
                  for gi, ga in enumerate(self.groups)]
        view = self.view(self.probs)
        self.base_err = p_error(view)
        self.base_c = confusion(view, other_prob, self.w)

    def view(self, post: list[float]) -> list[float]:
        return [sum(post[k] for k in g) for g in self.groups]

    def _gain(self, outcome_liks: list[tuple[list[float], float]]) -> float:
        """outcome_liks: for each outcome, (likelihood per disease, likelihood under 'other')."""
        exp_err = exp_c = 0.0
        for liks, lik_o in outcome_liks:
            joint = [p * l for p, l in zip(self.probs, liks)]
            joint_other = self.other * lik_o
            z = sum(joint) + joint_other
            if z <= 0:
                continue
            post = self.view([j / z for j in joint])
            exp_err += z * p_error(post)
            exp_c += z * confusion(post, joint_other / z, self.w)
        return self.s.misdiagnosis_loss * (self.base_err - exp_err) + self.s.pair_weight * (self.base_c - exp_c)

    def score_findings(self, action: ActionSpec, codes: list[str]) -> ScoredAction:
        r = CHANNEL_RELIABILITY[action.type]

        def noisy(p: float) -> float:
            return r * p + (1 - r) * (1 - p)

        def spread(c: str) -> float:
            ps = [self.store.finding_p_eff(i, c) for i in self.ids]
            mean = sum(p * q for p, q in zip(ps, self.probs)) + self.other * self.store.background(c)
            return sum(q * (p - mean) ** 2 for p, q in zip(ps, self.probs))

        codes = sorted(codes, key=spread, reverse=True)[:3]
        pd = {c: [noisy(self.store.finding_p_eff(i, c)) for i in self.ids] for c in codes}
        po = {c: noisy(self.store.background(c)) for c in codes}
        outcomes = []
        for pattern in itertools.product([True, False], repeat=len(codes)):
            liks = [1.0] * len(self.ids)
            lo = 1.0
            for c, present in zip(codes, pattern):
                liks = [l * (pd[c][k] if present else 1 - pd[c][k]) for k, l in enumerate(liks)]
                lo *= po[c] if present else 1 - po[c]
            outcomes.append((liks, lo))
        gain = self._gain(outcomes)
        return ScoredAction(action, gain - self.s.cost_weight * action.cost - action.risk, gain, codes)

    # -- v0.3 (D1): joint value of several actions, with the belief update's cluster pooling ------------

    def _table(self, sa: "ScoredAction") -> list[tuple[list[float], float, str]]:
        """Per outcome of one action: (likelihood per disease, likelihood under 'other', cluster key)."""
        a = sa.action
        r = CHANNEL_RELIABILITY[a.type]
        if a.test:
            td = self.store.test_def(a.test)
            dists = [self.store.test_dist_eff(i, td.id) for i in self.ids]
            key = f"test:{td.id}"
            return [([r * d[o] + (1 - r) * td.background[o] for d in dists], td.background[o], key) for o in td.outcomes]
        rows: list[tuple[list[float], float, str]] = []
        code = sa.reveals[0]
        noisy = lambda p: r * p + (1 - r) * (1 - p)  # noqa: E731
        pd = [noisy(self.store.finding_p_eff(i, code)) for i in self.ids]
        po = noisy(self.store.background(code))
        key = self.store.cluster(code)
        return [(pd, po, key), ([1 - x for x in pd], 1 - po, key)]

    def joint_gain(self, actions: list["ScoredAction"]) -> float:
        """Expected gain of observing all `actions` together.

        Outcome combinations are weighted by their predictive probability; the posterior after each combination
        is the one the belief update will compute: per disease, within each correlation cluster only the strongest
        supporting and the strongest opposing log-likelihood ratio count. Two questions about the same cluster
        are therefore worth little more than one.
        """
        if len(actions) == 1 and not actions[0].action.test and len(actions[0].reveals) > 1:
            return actions[0].gain
        tables = [self._table(a) for a in actions]
        n = len(self.ids)
        exp_err = exp_c = 0.0
        for combo in itertools.product(*tables):
            z_d = list(self.probs)
            z_o = self.other
            pos: list[dict[str, float]] = [{} for _ in range(n)]
            neg: list[dict[str, float]] = [{} for _ in range(n)]
            for liks, lo, key in combo:
                z_o *= lo
                for k in range(n):
                    z_d[k] *= liks[k]
                    llr = math.log(max(liks[k], 1e-12) / max(lo, 1e-12))
                    if llr >= 0:
                        pos[k][key] = max(pos[k].get(key, 0.0), llr)
                    else:
                        neg[k][key] = min(neg[k].get(key, 0.0), llr)
            weight = sum(z_d) + z_o  # predictive probability of this combination
            if weight <= 0:
                continue
            logs = [math.log(max(self.probs[k], 1e-12)) + sum(pos[k].values()) + sum(neg[k].values()) for k in range(n)]
            lo_other = math.log(max(self.other, 1e-12))
            m = max(logs + [lo_other])
            e = [math.exp(v - m) for v in logs]
            eo = math.exp(lo_other - m)
            zz = sum(e) + eo
            post = self.view([v / zz for v in e])
            exp_err += weight * p_error(post)
            exp_c += weight * confusion(post, eo / zz, self.w)
        return self.s.misdiagnosis_loss * (self.base_err - exp_err) + self.s.pair_weight * (self.base_c - exp_c)

    def score_test(self, action: ActionSpec) -> ScoredAction:
        td = self.store.test_def(action.test)
        r = CHANNEL_RELIABILITY[action.type]
        dists = [self.store.test_dist_eff(i, td.id) for i in self.ids]
        outcomes = []
        for o in td.outcomes:
            liks = [r * d[o] + (1 - r) * td.background[o] for d in dists]
            outcomes.append((liks, td.background[o]))
        gain = self._gain(outcomes)
        return ScoredAction(action, gain - self.s.cost_weight * action.cost - action.risk, gain, [td.id])


def target_prob(action: ActionSpec, differential: list[dict[str, Any]], store: CardStore) -> float:
    """Highest posterior among diseases for which some outcome of the test is >= 2x its background."""
    if not action.test:
        return 1.0
    td = store.test_def(action.test)
    best = 0.0
    for d in differential:
        if store.test_dist(d["disease_id"], td.id) is None:
            continue
        dist = store.test_dist_eff(d["disease_id"], td.id)
        if any(dist[o] >= 2 * td.background[o] for o in td.outcomes):
            best = max(best, d["prob"])
    return best


def live_cant_miss(differential: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    return [d for d in differential[1:] if d["cant_miss"] and d["prob"] >= settings.exclusion_threshold]


def threshold_commit_ready(differential: list[dict[str, Any]], settings: Settings) -> bool:
    if not differential or differential[0]["prob"] < settings.commit_threshold:
        return False
    return settings.objective != "clinical" or not live_cant_miss(differential, settings)


def candidate_actions(
    differential: list[dict[str, Any]], store: CardStore, settings: Settings,
    observed: set[str], done_ids: set[str], only_for: str | None = None,
) -> list[tuple[ActionSpec, list[str]]]:
    if only_for:
        ids = [only_for]
    else:
        ids = [d["disease_id"] for d in differential[: settings.top_k_load]]
        ids += [d["disease_id"] for d in differential if d["cant_miss"] and d["prob"] >= settings.cant_miss_floor]
    out = []
    allowed = set(getattr(settings, "allowed_action_types", ()) or ())
    for a in store.actions_for(list(dict.fromkeys(ids)), getattr(settings, "planner_max_codes", 0)):
        if a.id in done_ids or (allowed and a.type not in allowed):
            continue
        if a.test:
            if a.test in observed:
                continue
            gate = MIN_TARGET_PROB.get(a.type)
            if gate is not None and target_prob(a, differential, store) < gate:
                continue
            out.append((a, [a.test]))
        else:
            codes = [c for c in a.reveals if c not in observed]
            if codes:
                out.append((a, codes))
    return out


def score_all(differential, other_prob, store, settings, cands) -> list[ScoredAction]:
    if not differential:
        return []
    model = _Model(differential, other_prob, store, settings)
    scored = [model.score_test(a) if a.test else model.score_findings(a, codes) for a, codes in cands]
    return sorted(scored, key=lambda s: s.voi, reverse=True)


def _conditional_batch(model: "_Model", cands: list[ScoredAction], limit: int, settings: Settings) -> tuple[list[ScoredAction], float]:
    """D1: greedy batch; each added action must be worth its cost GIVEN the actions already chosen."""
    cands = [c for c in cands if c.voi > 0][:10]
    if not cands:
        return [], 0.0
    batch = [cands[0]]
    gain = cands[0].gain
    cost = settings.cost_weight * cands[0].action.cost + cands[0].action.risk
    seen = set(cands[0].reveals)
    while len(batch) < limit:
        best, best_m, best_g = None, 0.0, gain
        for c in cands:
            if c in batch or set(c.reveals) & seen:
                continue
            g = model.joint_gain(batch + [c])
            m = g - gain - (settings.cost_weight * c.action.cost + c.action.risk)
            if m > best_m:
                best, best_m, best_g = c, m, g
        if best is None or best_m <= settings.voi_epsilon:
            break
        batch.append(best)
        seen |= set(best.reveals)
        cost += settings.cost_weight * best.action.cost + best.action.risk
        gain = best_g
    return batch, gain - cost


def build_options(scored: list[ScoredAction], settings: Settings, model: "_Model | None" = None) -> list[Option]:
    options: list[Option] = []
    groups = {"history": ({"history"}, settings.max_batch_history), "exam": ({"exam"}, settings.max_batch_history),
              "labs": ({"poc", "lab"}, settings.max_batch_tests)}
    for kind, (types, limit) in groups.items():
        if model is not None and getattr(settings, "conditional_batches", False):
            batch, value = _conditional_batch(model, [s for s in scored if s.action.type in types], limit, settings)
            if batch:
                options.append(Option(kind, batch, value - settings.turn_cost))
            continue
        batch, seen = [], set()
        for s in scored:
            if s.action.type in types and s.voi > 0 and not (set(s.reveals) & seen):
                batch.append(s)
                seen |= set(s.reveals)
                if len(batch) >= limit:
                    break
        if batch:
            options.append(Option(kind, batch, sum(s.voi for s in batch) - settings.turn_cost))
    for s in scored:
        if s.action.type in ("imaging", "advanced_imaging", "procedure"):
            options.append(Option("single", [s], s.voi - settings.turn_cost))
    if not options and scored:
        options.append(Option("single", [scored[0]], scored[0].voi - settings.turn_cost))
    return sorted(options, key=lambda o: o.value, reverse=True)


def decide(
    differential: list[dict[str, Any]], other_prob: float, store: CardStore, settings: Settings, turn: int,
    observed: set[str], done_ids: set[str], critic_block: bool = False, stall_count: int = 0,
    veto: dict[str, Any] | None = None,
) -> Decision:
    """`critic_block` is the v0.2 veto (a bare flag). `veto` is the v0.3 structured veto (D2):
    {"alternative": disease_id}. It blocks a threshold commit only if some action separates the leader from
    the named alternative with positive value, scored with the ordinary pair weights; otherwise the commit goes
    ahead and the objection is reported as unresolved."""
    if turn >= settings.max_turns - 1:
        return Decision(True, f"turn budget reached ({turn}/{settings.max_turns})")

    cands = candidate_actions(differential, store, settings, observed, done_ids)
    model = _Model(differential, other_prob, store, settings) if differential else None
    scored = sorted([model.score_test(a) if a.test else model.score_findings(a, codes) for a, codes in cands],
                    key=lambda x: x.voi, reverse=True) if model else []
    options = build_options(scored, settings, model)
    best = options[0].value if options else 0.0
    leader = differential[0]["prob"] if differential else 0.0
    can_commit = turn >= settings.min_turns_before_commit

    def objection() -> Decision | None:
        """A named objection (critic veto or report dissent) holds the commit only while an action that separates
        the leader from the named alternative has positive value."""
        lead_id, alt = differential[0]["disease_id"], veto["alternative"]
        alt_d = next((d for d in differential if d["disease_id"] == alt), None)
        if alt_d is None or alt == lead_id:
            return None
        pair = [a for a in candidate_actions(differential, store, settings, observed, done_ids, only_for=lead_id)]
        pair += [a for a in candidate_actions(differential, store, settings, observed, done_ids, only_for=alt)
                 if a[0].id not in {x[0].id for x in pair}]
        focus = score_all(differential, other_prob, store, settings, pair)
        if focus and focus[0].voi > settings.voi_epsilon:
            return Decision(False, f"objection: separating {differential[0]['name']} from {alt_d['name']}", [focus[0]], best, scored)
        return None

    has_veto = bool(veto and veto.get("alternative") and differential)
    # A report dissent applies to every commit except the turn budget; a critic veto (v0.3 D2, as tested) only
    # to threshold commits.
    dissent = has_veto and veto.get("source") == "report_dissent"

    if can_commit and threshold_commit_ready(differential, settings):
        if has_veto:
            held = objection()
            if held:
                return held
            reason = f"leader {leader:.2f} >= {settings.commit_threshold:.2f}; unresolved objection: {veto['alternative']}"
            return Decision(True, reason, best_voi=best, scored=scored)
        if not critic_block:
            return Decision(True, f"leader {leader:.2f} >= {settings.commit_threshold:.2f}", best_voi=best, scored=scored)
        # v0.2 critic veto: separate the leader from its strongest challenger instead of committing.
        for d in differential[1:3]:
            focus = score_all(differential, other_prob, store, settings,
                              candidate_actions(differential, store, settings, observed, done_ids, only_for=d["disease_id"]))
            if focus:
                return Decision(False, f"critic veto: separating from {d['name']}", [focus[0]], best, scored)

    if can_commit and best <= settings.voi_epsilon and (leader >= settings.min_commit_prob or
                                                        (stall_count + 1 >= settings.stall_turns and differential)):
        if dissent:
            held = objection()
            if held:
                return held
        why = (f"no option worth its cost (best {best:.3f}), leader {leader:.2f}" if leader >= settings.min_commit_prob
               else f"stalled: no worthwhile option for {settings.stall_turns} turns, leader {leader:.2f}")
        if dissent:
            why += f"; unresolved objection: {veto['alternative']}"
        return Decision(True, why, best_voi=best, scored=scored)
    if not options:
        return Decision(False, "no scored action; free question", [], best, scored)
    return Decision(False, f"best turn option: {options[0].kind} (value {best:.3f})", options[0].actions, best, scored)


def is_test(action_type: str) -> bool:
    return action_type in TEST_TYPES
