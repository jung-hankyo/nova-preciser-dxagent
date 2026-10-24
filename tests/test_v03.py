"""v0.3 fixes: one block per fix, each checking the mechanism and that its flag restores v0.2 behavior."""

from dxagent.config import PRIOR_BUCKET
from dxagent.graph.nodes import Nodes
from dxagent.knowledge.cards import CardStore, FindingDef
from dxagent.llm import ScriptedLLM
from dxagent.reasoning.belief import compute_posterior, merge_findings, reverse_index_candidates
from dxagent.reasoning.planner import decide
from dxagent.runtime import Runtime
from dxagent.schemas import (ExtractedFinding,
                             ExtractionOutput, HypothesisOutput, ProposedHypothesis)

from .conftest import finding, hyp, result


def nodes(settings, store, handlers):
    return Nodes(Runtime(settings=settings, doctor=ScriptedLLM(handlers), store=store))


def extraction(*fs, prompts=None):
    def h(system, user):
        if prompts is not None:
            prompts.append(user)
        return ExtractionOutput(findings=list(fs), problem_representation="x")
    return h


def ef(code, present=True, primary=False, conf=0.9):
    return ExtractedFinding(code=code, label=code.replace("_", " "), present=present, confidence=conf,
                            source="history", evidence="", primary_complaint=primary)


# ------------------------------------------------------------------ A1 primary complaint


def test_a1_only_primary_complaint_is_excluded(store, settings):
    n = nodes(settings, store, {ExtractionOutput: extraction(ef("fever", primary=True), ef("productive_cough"), ef("chest_pain_pleuritic"))})
    out = n.perceive({"last_reply": "fever, cough with sputum, pain on breathing in", "turn": 0})
    assert [f["code"] for f in out["findings"] if f["chief_complaint"]] == ["fever"]
    settings.primary_complaint_only = settings.prior_context = False  # P1 requires A1
    out = n.perceive({"last_reply": "fever, cough with sputum, pain on breathing in", "turn": 0})
    assert all(f["chief_complaint"] for f in out["findings"])


def test_a1_first_present_finding_is_primary_when_none_flagged(store, settings):
    n = nodes(settings, store, {ExtractionOutput: extraction(ef("fever", present=False), ef("productive_cough"), ef("hemoptysis"))})
    out = n.perceive({"last_reply": "no fever, cough, some blood", "turn": 0})
    assert [f["code"] for f in out["findings"] if f["chief_complaint"]] == ["productive_cough"]


def test_a1_presenting_findings_reach_the_reverse_index(store, settings):
    hyps = [hyp(store, "pulmonary_embolism", 0.02)]
    pres = [finding("productive_cough", cc=True), finding("fever", turn=0), finding("crackles", turn=0), finding("temp_gt_38", turn=0)]
    added = reverse_index_candidates(hyps, pres, store, settings, [])
    assert "pneumonia" in [c["disease_id"] for c in added]
    v02 = [{**f, "chief_complaint": True} for f in pres]  # v0.2: the whole presentation was excluded
    assert reverse_index_candidates(hyps, v02, store, settings, []) == []


# ------------------------------------------------------------------ A2 vocabulary by reply text


def test_a2_reply_text_offers_codes_outside_the_hypotheses(store, settings):
    prompts = []
    n = nodes(settings, store, {ExtractionOutput: extraction(prompts=prompts)})
    state = {"last_reply": "I coughed up some blood this morning", "turn": 4,
             "hypotheses": [hyp(store, "pericarditis", 0.08)], "last_actions": []}
    assert "hemoptysis" not in store.codes_for("pericarditis")
    n.perceive(state)
    assert "- hemoptysis:" in prompts[-1]
    settings.lexicon_vocab = False
    n.perceive(state)
    assert "- hemoptysis:" not in prompts[-1]


# ------------------------------------------------------------------ A3 canonical codes


def test_a3_alias_rewrites_cards_and_vocabulary(tmp_path, settings):
    store = CardStore(settings.registry_dir, tmp_path, card_dirs=[settings.fixture_dir])
    store.add_registry([FindingDef(code="pyrexia", label="pyrexia", background=0.2)])
    card = store.get("pneumonia")
    card.findings.append(card.findings[0].model_copy(update={"code": "pyrexia", "conf": 0.1}))
    store.add_alias("pyrexia", "fever")
    assert store.canonical("pyrexia") == "fever"
    assert [f.code for f in store.get("pneumonia").findings].count("fever") == 1  # higher-confidence entry kept
    assert "pyrexia" not in store.vocabulary()
    assert CardStore(settings.registry_dir, tmp_path, card_dirs=[settings.fixture_dir]).canonical("pyrexia") == "fever"  # persisted


def test_a3_extracted_synonym_resolves_to_canonical(store, settings):
    store.add_alias("coughing_blood", "hemoptysis", persist=False)
    n = nodes(settings, store, {ExtractionOutput: extraction(ef("coughing_blood"))})
    out = n.perceive({"last_reply": "I coughed blood", "turn": 3, "hypotheses": [hyp(store, "pneumonia", 0.08)]})
    assert out["findings"][0]["code"] == "hemoptysis"


# ------------------------------------------------------------------ B1 diagnostic


def test_b1_list_effect_is_logged_when_the_list_changes(store, settings):
    settings.reverse_index = False
    n = nodes(settings, store, {})
    f = [finding("crackles"), finding("bilateral_leg_edema"), finding("orthopnea")]
    base = [hyp(store, "heart_failure_exacerbation", 0.25), hyp(store, "pneumonia", 0.25)]
    first = n.update_belief({"hypotheses": base, "findings": f, "turn": 5})
    st = {"hypotheses": base + [hyp(store, "pericarditis", 0.25), hyp(store, "pneumothorax", 0.25)], "findings": f, "turn": 6,
          "differential": first["differential"]}
    log = n.update_belief(st)["list_effect_log"][-1]
    assert log["added"] and log["p_new_list"] < log["p_same_evidence_old_list"]


# ------------------------------------------------------------------ B2 gap filling


# ------------------------------------------------------------------ B3 background pass


# ------------------------------------------------------------------ C2 out-of-list trigger


# ------------------------------------------------------------------ C3 reverse index and discovery


def test_c3_prevalence_prior_and_bayes_factor_rule(store, settings):
    settings.prior_context = False  # legacy single-bucket prevalence (v0.3 C3)
    settings.reverse_evidence_rule = True  # rejected alternative, kept behind a flag
    hyps = [hyp(store, "pulmonary_embolism", 0.02), hyp(store, "acute_coronary_syndrome", 0.08)]
    f = [finding("fever"), finding("productive_cough")]
    t = [result("img_chest_xray", "consolidation")]
    pna = next(c for c in reverse_index_candidates(hyps, f, store, settings, t) if c["disease_id"] == "pneumonia")
    assert pna["prior"] == PRIOR_BUCKET["common"]
    # Evidence the current list already explains: nothing is added even though cards speak to it.
    hyps2 = [hyp(store, "pneumonia", 0.08)]
    assert "pneumonia" not in [c["disease_id"] for c in reverse_index_candidates(hyps2, f, store, settings, t)]
    assert all(c["disease_id"] != "copd_exacerbation" for c in reverse_index_candidates(hyps2, f, store, settings, t))


def test_c3_discovery_after_two_unexplained_turns_and_capped(store, settings):
    settings.cardless_discovery = True  # off by default since v0.3 (lowered top-1)
    calls = []

    def disc(system, user):
        calls.append(system)
        return HypothesisOutput(hypotheses=[ProposedHypothesis(name="Sarcoidosis", disease_id="sarcoidosis",
                                                               prior_bucket="rare", cant_miss=False, rationale="x")])
    n = nodes(settings, store, {HypothesisOutput: disc})
    f = [{**finding("erythema_nodosum", conf=0.9), "label": "erythema nodosum"}]
    base = {"turn": 9, "hypotheses": [hyp(store, "pneumonia", 0.08)] * 3, "findings": f}
    assert n._discovery_target({**base, "unexplained_streak": {"erythema_nodosum": 1}}) is None
    out = n.hypothesize({**base, "unexplained_streak": {"erythema_nodosum": 2}})
    assert any("cause THIS finding" in s for s in calls) and out["cardless_queries"] == 1
    assert any(h["disease_id"] == "sarcoidosis" and h.get("prior_source", "").startswith("discovery") for h in out["hypotheses"])
    assert n._discovery_target({**base, "unexplained_streak": {"erythema_nodosum": 5},
                                "cardless_queries": settings.cardless_max_queries}) is None


def test_c3_prevalence_prior_with_v02_admission(store, settings):
    """Shipped C3: the card's bucket as prior, admitted by the v0.2 posterior test (evidence rule off)."""
    settings.prior_context = False  # legacy single-bucket prevalence (v0.3 C3)
    settings.reverse_evidence_rule, settings.reverse_prevalence_prior = False, True
    hyps = [hyp(store, "pulmonary_embolism", 0.02), hyp(store, "acute_coronary_syndrome", 0.02)]
    f = [finding("fever"), finding("productive_cough")]
    t = [result("img_chest_xray", "consolidation")]
    pna = next(c for c in reverse_index_candidates(hyps, f, store, settings, t) if c["disease_id"] == "pneumonia")
    assert pna["prior"] == PRIOR_BUCKET["common"] and pna["prior_source"] == "reverse_index:common"
    settings.reverse_prevalence_prior = False  # v0.2: fixed prior
    pna = next(c for c in reverse_index_candidates(hyps, f, store, settings, t) if c["disease_id"] == "pneumonia")
    assert pna["prior"] == settings.reverse_prior and pna["prior_source"] == "reverse_index"
