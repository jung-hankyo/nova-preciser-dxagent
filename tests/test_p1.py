"""v0.3.1 prior context (P1): the prior conditions on S alone, by construction, and the diagnostics around it."""


from dxagent.config import PRIOR_BUCKET
from dxagent.graph.nodes import Nodes
from dxagent.knowledge.cards import CardStore
from dxagent.knowledge.complaints import ComplaintRegistry
from dxagent.llm import ScriptedLLM
from dxagent.priors import median_bucket, reconcile, select_complaint
from dxagent.reasoning.belief import context_coverage, prior_sensitivity, reverse_index_candidates
from dxagent.runtime import Runtime
from dxagent.schemas import (ComplaintMatchOutput, ExtractedComplaint, ExtractedFinding, ExtractionOutput,
                             HypothesisOutput, PriorAssessment, PriorItem, ProposedHypothesis)

from .conftest import finding, hyp

OPENING = "I came in because of chest pain. I also have a fever and a cough with sputum."


def ef(code, present=True, evidence=""):
    return ExtractedFinding(code=code, label=code.replace("_", " "), present=present, confidence=0.9,
                            source="history", evidence=evidence)


def opening_extraction(age=None):
    def h(system, user):
        return ExtractionOutput(
            findings=[ef("chest_pain_pleuritic", evidence="chest pain"), ef("fever", evidence="a fever"),
                      ef("productive_cough", evidence="a cough with sputum")],
            complaints=[ExtractedComplaint(label="fever", quote="a fever", stated_reason=False, finding_code="fever"),
                        ExtractedComplaint(label="chest discomfort", quote="chest pain", stated_reason=True,
                                           finding_code="chest_pain_pleuritic")],
            age=age, problem_representation="x")
    return h


def generator(system, user):  # deliberately "leaky": calls pneumonia very common because it saw fever and cough
    return HypothesisOutput(hypotheses=[
        ProposedHypothesis(name="Pneumonia", disease_id="pneumonia", prior_bucket="very_common", cant_miss=False, rationale=""),
        ProposedHypothesis(name="Pulmonary embolism", disease_id="pulmonary_embolism", prior_bucket="rare", cant_miss=True,
                           rationale=""),
    ])


def make_nodes(settings, store, prompts, extraction=None):
    def assess(system, user):
        prompts.append(user)
        return PriorAssessment(items=[PriorItem(disease_id="pneumonia", prior_bucket="common"),
                                      PriorItem(disease_id="pulmonary_embolism", prior_bucket="uncommon")])

    llm = ScriptedLLM({ExtractionOutput: extraction or opening_extraction(), HypothesisOutput: generator,
                       PriorAssessment: assess, ComplaintMatchOutput: lambda s, u: ComplaintMatchOutput(acuity=3)})
    return Nodes(Runtime(settings=settings, doctor=llm, store=store))


def run_turn(n, state):
    for node in ("perceive", "hypothesize", "load_knowledge"):
        state.update(getattr(n, node)(state) or {})
    return state


# ------------------------------------------------------------------ complaint registry and rule


def test_registry_alias_auto_confirm_new(tmp_path, settings):
    calls = []

    def match(system, user):
        calls.append(user)
        return ComplaintMatchOutput(same_as="chest_pain" if "exertion" in user else "", acuity=2)

    reg = ComplaintRegistry(settings.registry_dir, tmp_path, llm=ScriptedLLM({ComplaintMatchOutput: match}))
    assert reg.resolve("Chest discomfort") == (reg.get("chest_pain"), "alias")
    assert reg.resolve("chest pains")[1] == "auto"  # concept Dice >= 0.85, no LLM call
    assert calls == []
    c, how = reg.resolve("chest pain on exertion")  # Dice 0.8: one LLM confirmation
    assert (c.key, how) == ("chest_pain", "confirmed") and len(calls) == 1
    c, how = reg.resolve("double vision")  # nothing similar: new entry, acuity from the LLM, persisted
    assert (c.key, how, c.acuity) == ("double_vision", "new", 2)
    again = ComplaintRegistry(settings.registry_dir, tmp_path)
    assert again.resolve("double vision")[0].acuity == 2
    assert again.resolve("chest pain on exertion")[1] == "alias"


def test_select_complaint_rule():
    c = lambda stated, acuity, pos: {"stated_reason": stated, "acuity": acuity, "pos": pos}  # noqa: E731
    assert select_complaint([c(False, 1, 0), c(True, 4, 9)]) == (1, "stated")  # stated reason beats severity
    assert select_complaint([c(True, 3, 0), c(True, 2, 9)]) == (1, "acuity")  # among stated, the most urgent
    assert select_complaint([c(True, 2, 5), c(True, 2, 1)]) == (1, "earliest")
    assert select_complaint([c(False, 3, 0)]) == (0, "only")
    assert select_complaint([]) == (-1, "none")


def test_perceive_flags_exactly_the_selected_complaint(settings, store):
    n = make_nodes(settings, store, [])
    out = n.perceive({"last_reply": OPENING, "turn": 0})
    assert [f["code"] for f in out["findings"] if f["chief_complaint"]] == ["chest_pain_pleuritic"]
    S = out["prior_context"]
    assert (S["complaint"], S["rule"], S["finding_match"]) == ("chest_pain", "stated", "code")


# ------------------------------------------------------------------ the prior assessor sees S only


def test_assessor_sees_only_S_and_overrides_the_generator(settings, store):
    prompts = []
    n = make_nodes(settings, store, prompts)
    st = run_turn(n, {"last_reply": OPENING, "presentation": OPENING, "turn": 0})
    assert len(prompts) == settings.prior_samples
    for p in prompts:  # information hiding: the complaint is there, the other findings are not
        assert "Presenting complaint: chest pain" in p
        assert "fever" not in p and "cough" not in p and "sputum" not in p
    by = {h["disease_id"]: h for h in st["hypotheses"]}
    assert by["pneumonia"]["prior"] == PRIOR_BUCKET["common"]  # not the generator's very_common
    assert by["pneumonia"]["proposed_bucket"] == "very_common"
    assert {e["route"] for e in st["prior_log"]} == {"assessor"}  # seed card buckets agree within one step


def test_buckets_are_reassigned_when_S_changes(settings, store):
    prompts = []
    n = make_nodes(settings, store, prompts)
    st = run_turn(n, {"last_reply": OPENING, "presentation": OPENING, "turn": 0})
    n0 = len(prompts)
    st.update({"last_reply": "nothing new", "turn": 1})
    run_turn(n, st)
    assert len(prompts) == n0  # S unchanged: no new assessment
    n.llm.handlers[ExtractionOutput] = lambda s, u: ExtractionOutput(age=71, problem_representation="x")
    st.update({"last_reply": "I am 71.", "turn": 2})
    run_turn(n, st)
    assert st["prior_context"]["age"] == 71 and len(prompts) == n0 + settings.prior_samples
    assert "Age: 71" in prompts[-1]
    assert any(e["reassigned"] for e in st["prior_log"])


def test_flag_off_keeps_the_generator_bucket(settings, store):
    settings.prior_context = False
    prompts = []
    n = make_nodes(settings, store, prompts)
    st = run_turn(n, {"last_reply": OPENING, "presentation": OPENING, "turn": 0})
    assert prompts == [] and "prior_context" not in st
    assert {h["disease_id"]: h["prior"] for h in st["hypotheses"]}["pneumonia"] == PRIOR_BUCKET["very_common"]


def test_reconcile_and_median():
    assert reconcile("common", "rare", 2) == ("uncommon", "reconciled")  # two steps apart: meet in the middle
    assert reconcile("very_common", "uncommon", 2) == ("common", "reconciled")
    assert reconcile("very_common", "rare", 2) == ("uncommon", "reconciled")  # 1.5 rounds toward the middle
    assert reconcile("common", "uncommon", 2) == ("uncommon", "assessor")  # close: the assessor (sees age, sex)
    assert reconcile(None, "rare", 2) == ("rare", "assessor") and reconcile("rare", None, 2) == ("rare", "card")
    assert median_bucket(["common", "rare", "uncommon"]) == "uncommon"
    assert median_bucket(["very_common", "uncommon"]) == "common"


# ------------------------------------------------------------------ complaint-keyed prevalence


def test_reverse_index_uses_complaint_keyed_prevalence(settings, store):
    hyps = [hyp(store, "pulmonary_embolism", 0.02)]
    f = [finding("fever"), finding("productive_cough"), finding("crackles"), finding("temp_gt_38")]
    pna = lambda cands: next(c for c in cands if c["disease_id"] == "pneumonia")  # noqa: E731
    assert pna(reverse_index_candidates(hyps, f, store, settings, [], complaint="chest_pain"))["prior_source"] == "reverse_index:common"
    other = pna(reverse_index_candidates(hyps, f, store, settings, [], complaint="headache"))
    assert other["prior"] == settings.reverse_prior and other["prior_source"] == "reverse_index:default"


# ------------------------------------------------------------------ diagnostics


def test_prior_sensitivity_detects_a_prior_decided_leader(settings, store):
    hyps = [hyp(store, "pneumonia", 0.25), hyp(store, "pulmonary_embolism", 0.005)]
    weak = [finding("sudden_onset")]  # mildly favours PE, not enough against a 50x prior ratio
    s = prior_sensitivity(hyps, weak, store, settings, [], alpha=0.5)
    assert s["leader"] == "pneumonia" and s["changed_flat"]
    strong = [finding("fever"), finding("productive_cough"), finding("crackles")]
    assert not prior_sensitivity(hyps, strong, store, settings, [])["changed_flat"]


def test_context_coverage_counts_findings_no_top_card_explains(store):
    diff = [{"disease_id": "pneumonia"}, {"disease_id": "heart_failure_exacerbation"}]
    f = [finding("productive_cough", turn=0, cc=True), finding("fever", turn=1), finding("prior_vte", turn=2),
         finding("mining_work_exposure", turn=1), finding("unilateral_leg_swelling", turn=5)]
    cov = context_coverage(f, diff, store)
    assert cov["opening_intake"] == [3, 2]  # fever is covered; prior VTE and the exposure moved nothing
    assert cov["later"] == [1, 1] and set(cov["uncovered"]) == {"prior_vte", "mining_work_exposure", "unilateral_leg_swelling"}


# ------------------------------------------------------------------ step 5: context reaches the update as evidence


def test_vital_signs_map_to_findings_deterministically(store):
    from dxagent.reasoning.belief import apply_numeric_rules, apply_vital_rules

    vals = [{"analyte": "heart_rate", "value": 118, "unit": "/min"}, {"analyte": "temperature_f", "value": 99.0, "unit": "F"},
            {"analyte": "resp_rate", "value": 30, "unit": "/min"}, {"analyte": "sbp", "value": 85, "unit": "mmHg"}]
    got = {f["code"]: f["present"] for f in apply_vital_rules(vals, store, turn=2)}
    assert got == {"tachycardia": True, "temp_gt_38": False, "tachypnea": True, "hypotension": True}
    ox = apply_numeric_rules([{"analyte": "spo2", "value": 88, "unit": "%"}], store, 2)
    assert [(t["test_id"], t["outcome"]) for t in ox] == [("test_pulse_oximetry", "hypoxia")]
    assert apply_numeric_rules([{"analyte": "spo2", "value": 0.88, "unit": "fraction"}], store, 2) == []  # no conversion


def test_perceive_lets_the_vital_rule_override_the_llm(settings, store):
    from dxagent.schemas import LabValue

    def h(system, user):
        return ExtractionOutput(findings=[ExtractedFinding(code="tachycardia", label="fast pulse", present=False, confidence=0.6,
                                                           source="exam", evidence="")],
                                lab_values=[LabValue(analyte="heart_rate", value=124, unit="/min")], problem_representation="x")

    n = make_nodes(settings, store, [], extraction=h)
    out = n.perceive({"last_reply": "HR 124", "turn": 3, "prior_context": {"complaint": "chest_pain"}})
    tc = next(f for f in out["findings"] if f["code"] == "tachycardia")
    assert tc["present"] and tc["confidence"] == 0.97


def test_registry_categories_and_coverage_by_category(store):
    assert store.finding_def("prior_vte").category == "risk_factor" and store.finding_def("tachycardia").category == "vital"
    diff = [{"disease_id": "pneumonia"}]
    cov = context_coverage([finding("fever", turn=1), finding("prior_vte", turn=1), finding("mining_work_exposure", turn=1)], diff, store)
    assert cov["by_category"] == {"symptom": [1, 0], "risk_factor": [1, 1], "unregistered": [1, 1]}


def test_extraction_prompt_still_asks_for_context():
    from dxagent.prompts import EXTRACT_SYSTEM

    assert "temperature_f" in EXTRACT_SYSTEM


def test_runner_summarizes_the_p1_diagnostics():
    from dxagent.bench.runner import p1_summary

    recs = [{"prior_sensitivity": {"changed_flat": True, "changed_tempered": False},
             "prior_log": [{"proposed": "very_common", "assessor": "uncommon", "route": "assessor", "votes": ["uncommon", "rare", "uncommon"]},
                           {"proposed": "rare", "assessor": "rare", "route": "assessor", "votes": ["rare"] * 3}],
             "prior_context": {"rule": "stated", "finding_match": "code"},
             "context_coverage": {"opening_intake": [4, 1], "by_category": {"risk_factor": [2, 1]}}},
            {"prior_sensitivity": {"changed_flat": False, "changed_tempered": False}, "prior_log": [], "context_coverage": {}}]
    s = p1_summary(recs)
    assert s["prior_decides_flat"] == 0.5 and s["prior_leak_gap"] == {"mean_buckets": 1.0, "share_ge2": 0.5, "n": 2}
    assert s["prior_vote_disagreement"] == 0.5 and s["complaint_rule"] == {"stated": 1}
    assert s["context_uncovered_opening_intake"] == 0.25 and s["context_uncovered_by_category"]["risk_factor"]["uncovered"] == 0.5
