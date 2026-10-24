"""Contrast cards (v0.4): LLM-elicited, restricted to existing codes and tests, capped confidence, cached."""

from dxagent.knowledge.cards import CardStore
from dxagent.knowledge.contrast import ContrastService
from dxagent.llm import ScriptedLLM
from dxagent.schemas import ContrastDraft, ContrastDraftFinding, ContrastDraftTest, OutcomeLevel


def draft(extra_code: str = "invented_sign", test_id: str = "img_chest_xray") -> ContrastDraft:
    return ContrastDraft(
        findings=[ContrastDraftFinding(code="unilateral_leg_swelling", label="one-sided leg swelling", source="exam",
                                       freq_in_a="occasional", freq_in_b="rare", background_freq="uncommon", cluster="dvt"),
                  ContrastDraftFinding(code=extra_code, label="a sign nobody has coded", source="exam",
                                       freq_in_a="very_common", freq_in_b="rare", background_freq="rare", cluster="x")],
        tests=[ContrastDraftTest(test_id=test_id, name="Chest X-ray", type="imaging",
                                 outcomes_a=[OutcomeLevel(outcome="normal", level="common")],
                                 outcomes_b=[OutcomeLevel(outcome="consolidation", level="very_common")])],
        management_differs=True,
    )


def service(tmp_path, settings, d):
    store = CardStore(settings.registry_dir, tmp_path, card_dirs=[settings.fixture_dir])
    return ContrastService(ScriptedLLM({ContrastDraft: lambda s, u: d}), store, settings), store


def test_contrast_uses_existing_codes_only_and_caps_confidence(tmp_path, settings):
    svc, store = service(tmp_path, settings, draft())
    cc = svc.draft_contrast("pulmonary_embolism", "pneumonia", "dyspnea", samples=2)
    assert cc is not None and store.has_contrast("pneumonia", "pulmonary_embolism")
    assert {f.code for f in cc.findings} == {"unilateral_leg_swelling"}  # the invented code is dropped
    assert store.finding_def("invented_sign") is None  # and never registered
    assert all(f.conf <= settings.contrast_conf_cap for f in cc.findings)
    assert all(t.conf <= settings.contrast_conf_cap for t in cc.tests.values())
    assert cc.provenance.status == "llm_contrast"
    assert (tmp_path / "contrasts").exists()


def test_contrast_drops_unknown_tests_and_fills_only_gaps(tmp_path, settings):
    svc, store = service(tmp_path, settings, draft(test_id="img_invented_scan"))
    cc = svc.draft_contrast("pulmonary_embolism", "pneumonia", "dyspnea", samples=2)
    assert cc is not None and cc.tests == {}
    # a disease card entry always wins over the contrast entry
    own = store.get("pulmonary_embolism").finding_map.get("unilateral_leg_swelling")
    if own is not None:
        assert store.finding_p("pulmonary_embolism", "unilateral_leg_swelling")[0] == own.p


def test_contrast_cards_never_enter_the_reverse_index(tmp_path, settings):
    svc, store = service(tmp_path, settings, draft())
    svc.draft_contrast("pulmonary_embolism", "pneumonia", "dyspnea", samples=2)
    assert all(store.index_eligible(store.get(c)) for c in store.cards_speaking_to({"unilateral_leg_swelling"}, []))
