import math

from dxagent.knowledge.cards import CardStore
from dxagent.reasoning.belief import apply_numeric_rules, compute_posterior, merge_findings
from tests.conftest import finding, hyp, result


def probs(res):
    return {d["disease_id"]: d["prob"] for d in res["differential"]}


def two(store):
    return [hyp(store, "pulmonary_embolism", 0.05), hyp(store, "pneumonia", 0.05)]


def test_aliases_resolve(store):
    assert store.get("PE").id == "pulmonary_embolism"
    assert store.get("myocardial infarction").id == "acute_coronary_syndrome"


def test_runtime_ratio_is_between_the_two_diseases(store, settings):
    """After one observation, odds(PE:pneumonia) change by exactly P(obs|PE)/P(obs|pneumonia)."""
    before = probs(compute_posterior(two(store), [], store, settings))
    after = probs(compute_posterior(two(store), [finding("fever", conf=1.0)], store, settings))
    ratio = (after["pulmonary_embolism"] / after["pneumonia"]) / (before["pulmonary_embolism"] / before["pneumonia"])
    expected = store.finding_p_eff("pulmonary_embolism", "fever") / store.finding_p_eff("pneumonia", "fever")
    assert math.isclose(ratio, expected, rel_tol=1e-6)


def test_categorical_test_outcome(store, settings):
    res = probs(compute_posterior(two(store), [], store, settings, [result("img_chest_xray", "consolidation")]))
    assert res["pneumonia"] > 0.5 > res["pulmonary_embolism"]
    res2 = probs(compute_posterior(two(store), [], store, settings, [result("img_ctpa", "filling_defect")]))
    assert res2["pulmonary_embolism"] > 0.5


def test_negative_d_dimer_lowers_pe(store, settings):
    res = probs(compute_posterior(two(store), [], store, settings, [result("lab_d_dimer", "normal")]))
    assert res["pulmonary_embolism"] < res["pneumonia"]


def test_confidence_shrinks_toward_background(store):
    p_card = store.finding_p("pulmonary_embolism", "sudden_onset")[0]
    p_eff = store.finding_p_eff("pulmonary_embolism", "sudden_onset")
    bg = store.background("sudden_onset")
    assert min(p_card, bg) < p_eff < max(p_card, bg)


def test_unlisted_finding_is_uninformative(store, settings):
    base = probs(compute_posterior(two(store), [], store, settings))
    res = probs(compute_posterior(two(store), [finding("wheeze", source="exam")], store, settings))
    assert math.isclose(base["pulmonary_embolism"], res["pulmonary_embolism"])


def test_correlated_findings_not_double_counted(store, settings):
    hyps = [hyp(store, "heart_failure_exacerbation", 0.05), hyp(store, "pneumonia", 0.05)]
    one = probs(compute_posterior(hyps, [finding("orthopnea")], store, settings))
    both = probs(compute_posterior(hyps, [finding("orthopnea"), finding("paroxysmal_nocturnal_dyspnea")], store, settings))
    assert math.isclose(one["heart_failure_exacerbation"] / one["pneumonia"], both["heart_failure_exacerbation"] / both["pneumonia"], rel_tol=1e-6) or \
        both["heart_failure_exacerbation"] / both["pneumonia"] < 2 * one["heart_failure_exacerbation"] / one["pneumonia"]


def test_observation_confidence_tempers_update(store, settings):
    strong = probs(compute_posterior(two(store), [finding("hemoptysis", conf=0.95)], store, settings))
    weak = probs(compute_posterior(two(store), [finding("hemoptysis", conf=0.3)], store, settings))
    assert strong["pulmonary_embolism"] > weak["pulmonary_embolism"]


def test_chief_complaint_not_recounted(store, settings):
    base = probs(compute_posterior(two(store), [], store, settings))
    cc = probs(compute_posterior(two(store), [finding("hemoptysis", cc=True, turn=0)], store, settings))
    assert math.isclose(base["pulmonary_embolism"], cc["pulmonary_embolism"])


def test_merge_keeps_chief_complaint_flag():
    merged = merge_findings([finding("fever", cc=True, turn=0)], [finding("fever", turn=1)])
    assert merged[0]["chief_complaint"] is True and merged[0]["turn"] == 1


def test_numeric_rules_map_to_test_outcomes(store):
    out = apply_numeric_rules([{"analyte": "d_dimer", "value": 1800, "unit": "ng/mL"}, {"analyte": "troponin", "value": 0.01, "unit": "ng/mL"}], store, 5)
    got = {t["test_id"]: t["outcome"] for t in out}
    assert got == {"lab_d_dimer": "elevated", "lab_troponin": "normal"}


def test_numeric_rules_refuse_unit_mismatch(store):
    assert apply_numeric_rules([{"analyte": "d_dimer", "value": 1.8, "unit": "mg/L"}], store, 5) == []


def test_perturbation_is_deterministic_and_bounded(settings):
    a = CardStore(settings.registry_dir, None, perturb=2.0, perturb_seed=1, card_dirs=[settings.fixture_dir])
    b = CardStore(settings.registry_dir, None, perturb=2.0, perturb_seed=1, card_dirs=[settings.fixture_dir])
    base = CardStore(settings.registry_dir, None, card_dirs=[settings.fixture_dir])
    p0 = base.finding_p("pneumonia", "fever")[0]
    pa, pb = a.finding_p("pneumonia", "fever")[0], b.finding_p("pneumonia", "fever")[0]
    assert pa == pb
    odds = lambda p: p / (1 - p)
    assert 0.5 - 1e-9 <= odds(pa) / odds(p0) <= 2.0 + 1e-9


# ---------------------------------------------------------------- reverse index

from dxagent.reasoning.belief import reverse_index_candidates  # noqa: E402


def test_reverse_index_recovers_missed_disease(store, settings):
    """The LLM proposed only PE and ACS; a consolidation on X-ray plus fever must bring in pneumonia."""
    hyps = [hyp(store, "pulmonary_embolism", 0.02), hyp(store, "acute_coronary_syndrome", 0.02)]
    f = [finding("fever"), finding("productive_cough")]
    t = [result("img_chest_xray", "consolidation")]
    added = reverse_index_candidates(hyps, f, store, settings, t, complaint="chest_pain")
    assert "pneumonia" in [c["disease_id"] for c in added]
    # v0.3 (C3) / v0.3.1 (P1): the candidate enters with its card's prevalence bucket for the complaint.
    pna = next(c for c in added if c["disease_id"] == "pneumonia")
    assert pna["prior"] == 0.08 and pna["prior_source"] == "reverse_index:common"
    settings.reverse_prevalence_prior = settings.prior_context = False  # v0.2 rule still available
    added = reverse_index_candidates(hyps, f, store, settings, t)
    assert "pneumonia" in [c["disease_id"] for c in added]
    assert all(c["prior"] == settings.reverse_prior and c["prior_source"] == "reverse_index" for c in added)


def test_reverse_index_respects_cap_and_threshold(store, settings):
    hyps = [hyp(store, "pneumonia", 0.08)]
    added = reverse_index_candidates(hyps, [finding("fever")], store, settings, [])
    assert len(added) <= settings.reverse_max_per_turn
    settings.reverse_evidence_rule, settings.reverse_bf = True, 1e6
    assert reverse_index_candidates(hyps, [finding("fever")], store, settings, []) == []
    settings.reverse_evidence_rule = False
    settings.reverse_min_prob = 0.99
    assert reverse_index_candidates(hyps, [finding("fever")], store, settings, []) == []


def test_reverse_index_ignores_evidence_free_cards(store, settings):
    """With no observation any card speaks to, nothing is added, however weak the differential."""
    hyps = [hyp(store, "pneumonia", 0.001)]
    assert reverse_index_candidates(hyps, [finding("unmapped_code_xyz")], store, settings, []) == []


def test_reverse_index_can_be_disabled_in_graph(settings):
    from dxagent.graph.nodes import Nodes
    from dxagent.knowledge.cards import CardStore
    from dxagent.offline import make_offline_llm
    from dxagent.runtime import Runtime

    st = CardStore(settings.registry_dir, None, card_dirs=[settings.fixture_dir])
    state = {"hypotheses": [hyp(st, "pulmonary_embolism", 0.02)], "findings": [finding("fever")],
             "test_results": [result("img_chest_xray", "consolidation")], "turn": 5}
    on = Nodes(Runtime.create(settings, doctor=make_offline_llm(st), store=st)).update_belief(state)
    settings.reverse_index = False
    off = Nodes(Runtime.create(settings, doctor=make_offline_llm(st), store=st)).update_belief(state)
    assert "pneumonia" in [h["disease_id"] for h in on["hypotheses"]]
    assert "pneumonia" not in [h["disease_id"] for h in off["hypotheses"]]
    assert on["hypothesis_trace"][-1]["sources"]["Pneumonia"].startswith("reverse_index")
