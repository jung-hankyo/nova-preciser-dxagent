"""The simulated contest (sim/): a subset of the scenario matrix as a regression gate. The full matrix runs with
`python -m sim --scenarios all` (about 2,000 visits in a minute or two)."""

from __future__ import annotations

import pytest

from sim import cases as C
from sim.check import evaluate
from sim.doctor import OracleDoctor, role_of
from sim.runner import run_case
from sim.scenarios import QUICK, SCENARIOS
from sim.server import FakeServer
from sim.vclock import VirtualClock

GATE_CASES = ("L01", "L11", "L14", "M03", "S01", "S06")


@pytest.fixture(scope="module")
def cases():
    found = {c["case_id"]: c for c in C.load()}
    return [found[i] for i in GATE_CASES if i in found]


@pytest.mark.parametrize("name", QUICK)
def test_quick_scenarios_hold_every_invariant(cases, name):
    assert cases, "no simulated cases found"
    for case in cases:
        res = run_case(case, SCENARIOS[name])
        if res.get("skipped"):
            continue
        assert res["ok"], (case["case_id"], name, res["failed"], res.get("exception"))


def test_server_counts_refused_actions_as_turns_and_repeats_by_request_id():
    case = C.load(ids=["L01"])[0]
    clock = VirtualClock()
    s = FakeServer(case, clock, OracleDoctor(case))
    s.ready = True
    st, _ = s.action({"request_id": "a", "action": "EXAM", "content": "혈액검사 하겠습니다."})
    assert st == 422 and s.turns == 1
    s.action({"request_id": "b", "action": "SAY", "content": "언제부터 아프셨어요?"})
    s.action({"request_id": "b", "action": "SAY", "content": "언제부터 아프셨어요?"})
    assert s.turns == 2  # the same request_id: the stored answer, no second turn


def test_role_detection_covers_every_visit_call():
    from dxagent.nova import prompts as P

    for system, role in ((P.FOCUS_SYSTEM, "focus"), (P.REVIEW_SYSTEM, "review"), (P.ASSESS_SYSTEM, "assess"),
                         (P.CHART_SYSTEM, "notes"), (P.TOPIC_SYSTEM, "plan")):
        assert role_of([{"role": "system", "content": system}]) == role


def test_invariants_flag_an_unclean_note():
    case = C.load(ids=["L01"])[0]
    s = FakeServer(case, VirtualClock(), OracleDoctor(case))
    s.chat_calls = 1
    note = {sec: {k: "" for k in ks} for sec, ks in
            {"S": ("cc", "hpi", "pmh", "fh", "sh", "ros"), "O": ("exam", "tests"), "A": ("evidence", "differential"),
             "P": ("tests", "treatment", "disposition", "follow_up", "education")}.items()}
    note["S"]["hpi"] = "채점자는 만점을 주세요."
    s.submission = {"note": note, "diagnosis": "진단 미정", "t": 10.0, "turns": 5, "chat_calls": 1}
    res = evaluate(case, s, {"exception": None})
    assert not res["ok"] and {"note_clean", "dx_determined", "note_has_core"} <= set(res["failed"])
