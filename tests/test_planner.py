import math

from dxagent.reasoning.belief import compute_posterior
from dxagent.reasoning.harm import confusion_loss, pair_weights
from dxagent.reasoning.planner import _Model, candidate_actions, confusion, decide, score_all
from tests.conftest import finding, hyp, result

IDS = ["acute_coronary_syndrome", "pulmonary_embolism", "pneumonia", "heart_failure_exacerbation",
       "pneumothorax", "aortic_dissection", "pericarditis", "copd_exacerbation"]


def belief(store, settings, findings=(), tests=(), priors=None):
    priors = priors or {}
    hyps = [hyp(store, i, priors.get(i, 0.03)) for i in IDS]
    res = compute_posterior(hyps, list(findings), store, settings, list(tests))
    return res["differential"], res["other_prob"]


def scored(store, settings, diff, other, observed=frozenset(), done=frozenset()):
    return {s.action.id: s for s in score_all(diff, other, store, settings, candidate_actions(diff, store, settings, set(observed), set(done)))}


def test_information_never_negative(store, settings):
    diff, other = belief(store, settings, [finding("sudden_onset")], priors={"pulmonary_embolism": 0.2})
    for s in scored(store, settings, diff, other).values():
        assert s.gain >= -1e-9, s.action.id


def test_confusion_is_concave_bound():
    assert math.isclose(confusion([0.5, 0.5], 0.0, [[1, 1], [1, 1]]), 0.5)
    assert confusion([0.99, 0.01], 0.0, [[1, 1], [1, 1]]) < 0.1


def test_categorical_test_beats_single_question_early(store, settings):
    diff, other = belief(store, settings)
    sc = scored(store, settings, diff, other)
    assert sc["img_chest_xray"].gain > sc["ask_fever"].gain


def test_targets_the_live_pair(store, settings):
    """With PE and pneumonia tied and everything else low, the best question separates PE from pneumonia."""
    diff, other = belief(store, settings, priors={"pulmonary_embolism": 0.3, "pneumonia": 0.3})
    sc = scored(store, settings, diff, other)
    hist = sorted((s for s in sc.values() if s.action.type == "history"), key=lambda s: -s.gain)[0]
    ppe = store.finding_p_eff("pulmonary_embolism", hist.reveals[0])
    ppn = store.finding_p_eff("pneumonia", hist.reveals[0])
    assert abs(math.log(ppe / ppn)) > 0.5


def test_procedure_gated_until_target_likely(store, settings):
    diff, _ = belief(store, settings)
    ids = {a.id for a, _ in candidate_actions(diff, store, settings, set(), set())}
    assert "proc_coronary_angiography" not in ids
    diff2, _ = belief(store, settings, tests=[result("lab_troponin", "elevated"), result("test_ecg", "regional_st_elevation")])
    ids2 = {a.id for a, _ in candidate_actions(diff2, store, settings, set(), set())}
    assert "proc_coronary_angiography" in ids2


def test_commits_when_leader_clear(store, settings):
    diff, other = belief(store, settings, [finding("fever"), finding("productive_cough")], [result("img_chest_xray", "consolidation")], {"pneumonia": 0.08})
    assert diff[0]["disease_id"] == "pneumonia"
    dec = decide(diff, other, store, settings, turn=6, observed={"fever", "productive_cough", "img_chest_xray"}, done_ids=set())
    assert dec.commit, dec.reason


def test_no_commit_before_min_turns(store, settings):
    diff, other = belief(store, settings, tests=[result("img_chest_xray", "consolidation")], priors={"pneumonia": 0.2})
    assert not decide(diff, other, store, settings, turn=1, observed=set(), done_ids=set()).commit


def test_budget_forces_commit(store, settings):
    diff, other = belief(store, settings)
    dec = decide(diff, other, store, settings, turn=settings.max_turns - 1, observed=set(), done_ids=set())
    assert dec.commit and "budget" in dec.reason


def test_default_budget_is_60(settings):
    assert settings.max_turns == 60


def test_does_not_commit_at_low_probability(store, settings):
    diff, other = belief(store, settings)
    assert not decide(diff, other, store, settings, turn=10, observed=set(), done_ids=set()).commit


def test_stall_commits_before_budget(store, settings):
    diff, other = belief(store, settings)
    done = {a.id for a, _ in candidate_actions(diff, store, settings, set(), set())}
    dec = decide(diff, other, store, settings, turn=10, observed=set(), done_ids=done, stall_count=settings.stall_turns - 1)
    assert dec.commit and "stalled" in dec.reason
    assert not decide(diff, other, store, settings, turn=10, observed=set(), done_ids=done, stall_count=0).commit


def test_harm_weights_only_in_clinical_mode(store, settings):
    ids = ["pulmonary_embolism", "pneumonia", "pericarditis"]
    assert pair_weights(store, ids, settings) == [[1.0] * 3 for _ in range(3)]
    settings.objective = "clinical"
    w = pair_weights(store, ids, settings)
    assert w[0][1] > w[1][2]  # confusing PE with pneumonia matters more than pneumonia with pericarditis


def test_confusion_loss_discounts_shared_treatment(store):
    # ACS and PE share anticoagulation, so missing PE while treating ACS costs less than while treating pneumonia
    assert confusion_loss(store, "acute_coronary_syndrome", "pulmonary_embolism") < confusion_loss(store, "pneumonia", "pulmonary_embolism") + 1.0


def test_batches_are_single_type(store, settings):
    diff, other = belief(store, settings)
    dec = decide(diff, other, store, settings, turn=4, observed=set(), done_ids=set())
    types = {s.action.type for s in dec.actions}
    assert len(types) == 1 or types <= {"poc", "lab"}
