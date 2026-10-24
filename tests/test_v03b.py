"""v0.3 second pass: C1, C4, D1, D2, E1, F1, G2, G4. Each checks the mechanism and that its flag restores v0.2."""

from dxagent.graph.nodes import Nodes
from dxagent.llm import ScriptedLLM
from dxagent.reasoning.belief import compute_posterior
from dxagent.reasoning.planner import _Model, build_options, candidate_actions, decide
from dxagent.runtime import Runtime
from dxagent.schemas import CriticOutput, HypothesisOutput, ProposedHypothesis, ReportOutput

from .conftest import finding, hyp, result


def nodes(settings, store, handlers):
    return Nodes(Runtime(settings=settings, doctor=ScriptedLLM(handlers), store=store))


def ph(did, bucket="common"):
    return ProposedHypothesis(name=did, disease_id=did, prior_bucket=bucket, cant_miss=False, rationale="")


def diff_of(store, settings, hyps, f=(), t=()):
    return compute_posterior(hyps, list(f), store, settings, list(t))


# ------------------------------------------------------------------ C1 open-world guard


def test_c1_reprompts_once_when_nothing_new(store, settings):
    settings.open_world_enforce = True  # off by default since the v0.3 record; the mechanism is still tested
    calls = []

    def gen(system, user):
        calls.append(system)
        return HypothesisOutput(hypotheses=[ph("pneumonia")] if len(calls) == 1 else [ph("pericarditis")])

    n = nodes(settings, store, {HypothesisOutput: gen})
    state = {"turn": 4, "hypotheses": [hyp(store, "pneumonia", 0.08)], "findings": [], "differential": []}
    out = n.hypothesize(state)
    assert len(calls) == 2 and "organ system" in calls[1]
    assert "pericarditis" in [h["disease_id"] for h in out["hypotheses"]] and out["open_world_reprompts"] == 1
    calls.clear()
    out = n.hypothesize({**state, "open_world_reprompts": settings.open_world_max})
    assert len(calls) == 1  # capped per case
    settings.open_world_enforce = False
    calls.clear()
    n.hypothesize(state)
    assert len(calls) == 1


# ------------------------------------------------------------------ D1 conditional batches


def test_d1_same_cluster_questions_are_not_batched_at_full_value(store, settings):
    hyps = [hyp(store, "heart_failure_exacerbation", 0.08), hyp(store, "pneumonia", 0.08), hyp(store, "copd_exacerbation", 0.08)]
    diff = diff_of(store, settings, hyps)
    m = _Model(diff["differential"], diff["other_prob"], store, settings)
    cands = candidate_actions(diff["differential"], store, settings, set(), set())
    scored = {a.id: (m.score_test(a) if a.test else m.score_findings(a, c)) for a, c in cands}
    orth = next(v for k, v in scored.items() if v.reveals == ["orthopnea"])
    pnd = next(v for k, v in scored.items() if v.reveals == ["paroxysmal_nocturnal_dyspnea"])
    joint = m.joint_gain([orth, pnd])
    # Same cluster: the pair is worth much less than the sum of its parts.
    assert joint < 0.8 * (orth.gain + pnd.gain)
    assert abs(m.joint_gain([orth]) - orth.gain) < 1e-9


def test_d1_flag_restores_additive_batches(store, settings):
    hyps = [hyp(store, "heart_failure_exacerbation", 0.08), hyp(store, "pneumonia", 0.08)]
    diff = diff_of(store, settings, hyps)
    m = _Model(diff["differential"], diff["other_prob"], store, settings)
    cands = candidate_actions(diff["differential"], store, settings, set(), set())
    scored = sorted([m.score_test(a) if a.test else m.score_findings(a, c) for a, c in cands], key=lambda s: -s.voi)
    on = build_options(scored, settings, m)
    settings.conditional_batches = False
    off = build_options(scored, settings, m)
    h_on = next(o for o in on if o.kind == "history")
    h_off = next(o for o in off if o.kind == "history")
    assert h_on.value <= h_off.value + 1e-9  # joint value never exceeds the naive sum


# ------------------------------------------------------------------ D2 structured veto


def _ready(store, settings):
    hyps = [hyp(store, "pneumonia", 0.25), hyp(store, "pulmonary_embolism", 0.02)]
    f = [finding("fever"), finding("productive_cough"), finding("crackles")]
    t = [result("img_chest_xray", "consolidation")]
    return hyps, f, t, diff_of(store, settings, hyps, f, t)


def test_d2_veto_blocks_only_with_a_positive_value_separating_action(store, settings):
    hyps, f, t, res = _ready(store, settings)
    d = res["differential"]
    assert d[0]["disease_id"] == "pneumonia" and d[0]["prob"] >= settings.commit_threshold
    obs = {x["code"] for x in f} | {"img_chest_xray"}
    dec = decide(d, res["other_prob"], store, settings, 6, obs, set(), veto={"alternative": "pulmonary_embolism"})
    assert not dec.commit and dec.actions and "objection" in dec.reason
    # When nothing can separate them any more, the commit goes ahead with the objection recorded.
    settings.voi_epsilon = 1e9
    dec = decide(d, res["other_prob"], store, settings, 6, obs, set(), veto={"alternative": "pulmonary_embolism"})
    assert dec.commit and "unresolved objection" in dec.reason


def test_d2_critic_veto_not_repeated_without_new_evidence_and_capped(store, settings):
    hyps, f, t, res = _ready(store, settings)
    crit = lambda s, u: CriticOutput(veto_alternative="pulmonary_embolism", veto_reason="cant_miss_not_excluded")  # noqa: E731
    n = nodes(settings, store, {CriticOutput: crit})
    base = {"hypotheses": hyps, "findings": f, "test_results": t, "differential": res["differential"],
            "entropy": 0.2, "turn": 6}
    o1 = n.critic(base)
    assert o1["veto"]["alternative"] == "pulmonary_embolism" and o1["vetoes"][-1]["honored"]
    o2 = n.critic({**base, **o1, "turn": 7, "critic_turn": 6})  # same evidence
    assert o2["veto"] == {} and o2["vetoes"][-1]["why"] == "repeat without new evidence"
    f2 = f + [finding("tachycardia", turn=7)]
    o3 = n.critic({**base, **o2, "findings": f2, "turn": 8, "critic_turn": 7})  # new evidence: allowed again
    assert o3["vetoes"][-1]["honored"]
    settings.veto_ceiling = 2
    f3 = f2 + [finding("hemoptysis", present=False, turn=8)]
    o4 = n.critic({**base, **o3, "findings": f3, "turn": 9, "critic_turn": 8})
    assert o4["vetoes"][-1]["why"] == "ceiling reached"


def test_d2_veto_without_alternative_does_not_block_and_legacy_flag(store, settings):
    hyps, f, t, res = _ready(store, settings)
    n = nodes(settings, store, {CriticOutput: lambda s, u: CriticOutput(block_commit=True)})
    base = {"hypotheses": hyps, "findings": f, "test_results": t, "differential": res["differential"], "entropy": 0.2, "turn": 6}
    out = n.critic(base)
    assert out["critic_block"] is False and not out.get("veto")
    settings.structured_veto = False
    assert n.critic(base)["critic_block"] is True


# ------------------------------------------------------------------ E1 deterministic final diagnosis


def test_e1_final_is_top_and_dissent_triggers_one_reevaluation(store, settings):
    hyps, f, t, res = _ready(store, settings)
    rep = lambda s, u: ReportOutput(final_diagnosis="Pericarditis", rationale="x", agrees_with_top=False,  # noqa: E731
                                     dissent_diagnosis="Pericarditis", dissent_reason="rub")
    n = nodes(settings, store, {ReportOutput: rep})
    st = {"hypotheses": hyps, "findings": f, "test_results": t, "differential": res["differential"], "turn": 6}
    o1 = n.report(st)
    assert o1["phase"] == "gathering" and "pericarditis" in [h["disease_id"] for h in o1["hypotheses"]]
    assert Nodes.route_after_report(o1) == "load_knowledge"
    o2 = n.report({**st, **o1})
    assert o2["phase"] == "committed" and o2["final_report"]["final_diagnosis"] == "Pneumonia"
    assert o2["final_report"]["llm_final_diagnosis"] == "Pericarditis"
    settings.deterministic_final = False  # v0.2: the LLM's name is the answer
    o3 = n.report(st)
    assert o3["final_report"]["final_diagnosis"] == "Pericarditis"


def test_e1_dissent_naming_a_listed_disease_becomes_an_objection(store, settings):
    """Revised E1: the common case (the LLM prefers a disease already on the list) is no longer only logged."""
    hyps, f, t, res = _ready(store, settings)
    hyps = hyps + [hyp(store, "pulmonary_embolism", 0.02)] if "pulmonary_embolism" not in [h["disease_id"] for h in hyps] else hyps
    rep = lambda s, u: ReportOutput(final_diagnosis="Pulmonary embolism", rationale="x", agrees_with_top=False,  # noqa: E731
                                     dissent_diagnosis="Pulmonary embolism", dissent_reason="calf swelling")
    n = nodes(settings, store, {ReportOutput: rep})
    st = {"hypotheses": hyps, "findings": f, "test_results": t, "differential": res["differential"], "turn": 6}
    o1 = n.report(st)
    assert o1["phase"] == "gathering" and o1["veto"] == {"alternative": "pulmonary_embolism", "turn": 6, "source": "report_dissent"}
    assert o1["report_dissent"][-1]["outcome"] == "objection"
    # The objection also holds a "no option worth its cost" commit while a separating action has value.
    d = res["differential"]
    obs = {x["code"] for x in f} | {"img_chest_xray"}
    settings.commit_threshold = 1.01  # force the no-option branch
    settings.turn_cost = 1e6
    dec = decide(d, res["other_prob"], store, settings, 6, obs, set(), veto=o1["veto"])
    assert not dec.commit and "objection" in dec.reason
    settings.dissent_as_objection = False  # previous E1: a listed disease is only logged
    o2 = n.report(st)
    assert o2["phase"] == "committed" and o2["report_dissent"][-1]["outcome"] == "already on the differential"


# ------------------------------------------------------------------ F1 quorum (kept for contrast cards)


def test_f1_quorum_for_contrast_samples():
    from dxagent.knowledge.contrast import _quorum

    assert _quorum(2, True) == 2 and _quorum(3, True) == 2 and _quorum(2, False) == 1


# ------------------------------------------------------------------ G2 / G4 evaluation


def test_g2_leak_terms_match_synonyms_by_channel():
    from dxagent.env.agentclinic import Case, SimulatedEncounter
    from dxagent.schemas import LeakTermsOutput

    case = Case(index=0, objective="o", patient={}, exam={}, tests={}, diagnosis="Myasthenia gravis")
    terms = ScriptedLLM({LeakTermsOutput: lambda s, u: LeakTermsOutput(terms=["MG", "myasthenia"])})
    pat = ScriptedLLM({"text": lambda s, u: "My neurologist said it might be myasthenia."})
    enc = SimulatedEncounter(case, pat, pat, terms_llm=terms)
    enc.respond("What brings you in?", "history")
    enc.respond("Order AChR antibodies", "lab")
    assert enc.leak_flags == 1 and enc.leaks["patient"] == ["myasthenia"] and len(enc.leaks["records"]) == 1


def test_g4_full_brier():
    from dxagent.bench.runner import brier_full

    r = {"differential_probs": [{"prob": 0.7}, {"prob": 0.2}], "other_prob": 0.1, "match_rank": 1}
    assert abs(brier_full(r) - (0.09 + 0.04 + 0.01)) < 1e-9
    r["match_rank"] = None
    assert abs(brier_full(r) - (0.49 + 0.04 + 0.81)) < 1e-9
