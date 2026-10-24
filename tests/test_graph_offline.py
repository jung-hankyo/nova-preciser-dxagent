"""End-to-end: the LangGraph loop runs intake, planning, categorical tests, interrupts, and commit offline."""

from dxagent.knowledge.cards import CardStore
from dxagent.offline import make_offline_llm
from dxagent.runtime import Runtime
from dxagent.session import DxSession

TRUTH = {"fever": True, "productive_cough": True, "crackles": True, "temp_gt_38": True, "tachycardia": True, "chest_pain_pleuritic": True}
TESTS = {"img_chest_xray": "consolidation", "lab_cbc": "leukocytosis", "lab_d_dimer": "elevated", "test_pulse_oximetry": "hypoxia"}


def make_session(settings):
    store = CardStore(settings.registry_dir, None, card_dirs=[settings.fixture_dir])
    return DxSession(Runtime.create(settings, doctor=make_offline_llm(store), store=store)), store


def answer(res, store):
    vocab = store.vocabulary()
    parts = []
    for a in res.actions:
        if a.get("test"):
            td = store.test_def(a["test"])
            parts.append(f"{td.name}: {TESTS.get(td.id, max(td.background, key=td.background.get)).replace('_', ' ')}")
        for c in a.get("reveals", []):
            parts.append(vocab[c]["label"] if TRUTH.get(c) else "no " + vocab[c]["label"])
    return ". ".join(parts) or "nothing else to add"


def test_full_encounter_reaches_diagnosis(settings):
    sess, store = make_session(settings)
    res = sess.start("I have chest pain and shortness of breath.")
    first = []
    while not res.done:
        first.append([a["id"] for a in res.actions])
        res = sess.reply(answer(res, store))
    assert first[:3] == [["open_hpi"], ["pmh_meds_social"], ["vitals_general_exam"]]
    assert res.report["final_diagnosis"] == "Pneumonia"
    st = sess.state
    assert any(t["test_id"] == "img_chest_xray" for t in st["test_results"])
    assert st["action_log"]


def test_budget_commit(settings):
    settings.max_turns = 5
    sess, _ = make_session(settings)
    res = sess.start("I feel unwell.")
    n = 0
    while not res.done and n < 10:
        res = sess.reply("nothing else to add")
        n += 1
    assert res.done
