"""v0.9.0, robustness found by the simulated visits (sim/): bounded calls, the same request polled until it finishes, a
time budget planned from the case's own pace, wrapped JSON, the diagnosis when no model call succeeded, the closing
lines, written denials and routing."""

from __future__ import annotations

import threading
import time

import pytest

from dxagent.nova import playbooks as PB
from dxagent.nova.budget import Budget, BudgetExhausted
from dxagent.nova.config import NovaConfig
from dxagent.nova.korean import complaint_phrase
from dxagent.nova.llm import HttpError, NovaLLM, validate
from dxagent.nova.prompts import FocusPlan, ReviewPlan
from dxagent.nova.safety import allergy_classes
from dxagent.nova.sdk import SdkTransport
from dxagent.nova.watchdog import CallTimeout, bounded


def test_bounded_call_times_out_and_propagates():
    with pytest.raises(CallTimeout):
        bounded(time.sleep, 0.05, 0.5)
    assert bounded(lambda x: x + 1, 1.0, 1) == 2
    with pytest.raises(KeyError):
        bounded(lambda: {}["x"], 1.0)

    class Stop(BaseException):
        pass

    def stop():
        raise Stop()

    with pytest.raises(Stop):  # BaseException from the worker reaches the caller (the replay harness relies on it)
        bounded(stop, 1.0)


class Coded(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class SlowServerChat:
    """client.chat for a model that finishes after `polls` resends of the same request_id."""

    def __init__(self, polls: int, answer: str = '{"skip": []}'):
        self.polls = polls
        self.answer = answer
        self.rids: list[str] = []

    def chat(self, messages, *, max_tokens=2048, reasoning_effort=None, temperature=None, model=None, request_id=None):
        self.rids.append(request_id)
        if len(self.rids) <= self.polls:
            raise Coded("PROVIDER_UNAVAILABLE" if len(self.rids) == 1 else "REQUEST_PENDING")
        return {"choices": [{"message": {"content": self.answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_running_request_is_polled_with_the_same_id(monkeypatch):
    import dxagent.nova.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod.time, "sleep", lambda s: None)
    c = SlowServerChat(polls=7)
    out = SdkTransport(c).chat({"messages": [{"role": "user", "content": "x"}], "max_tokens": 100}, timeout=600)
    assert out["choices"][0]["message"]["content"] == '{"skip": []}'
    assert len(set(c.rids)) == 1 and len(c.rids) == 8  # never a second generation


def test_call_deadline_is_504_and_the_request_is_collected_later(monkeypatch):
    import dxagent.nova.sdk as sdk_mod

    clock = {"t": 0.0}
    monkeypatch.setattr(sdk_mod.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(sdk_mod.time, "monotonic", lambda: clock["t"])
    c = SlowServerChat(polls=30)
    t = SdkTransport(c)
    payload = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 100}
    with pytest.raises(HttpError) as e:
        t.chat(payload, timeout=60)
    assert e.value.status == 504 and len(set(c.rids)) == 1
    first = c.rids[0]
    c.polls = len(c.rids)  # the server has finished it by now
    t.chat(payload, timeout=60)
    assert c.rids[-1] == first  # the identical call collected the same request instead of starting a new one


def test_novallm_does_not_retry_a_spent_call():
    class Transport:
        def __init__(self):
            self.n = 0

        def chat(self, payload, timeout):
            self.n += 1
            raise HttpError(504, "CALL_DEADLINE")

    cfg = NovaConfig()
    cfg.use_guidelines = False
    tr = Transport()
    with pytest.raises(HttpError):
        NovaLLM(cfg, transport=tr).structured(FocusPlan, "s", "u")
    assert tr.n == 1


def test_wrapped_json_is_unwrapped_for_all_default_schemas():
    out = validate(ReviewPlan, {"ReviewPlan": {"exam_order": ["m1", "m0"], "names_cause": False}})
    assert out.exam_order == ["m1", "m0"] and out.names_cause is False
    out = validate(FocusPlan, {"skip": ["o1"]})
    assert out.skip == ["o1"]


def test_budget_plans_from_the_observed_pace():
    cfg = NovaConfig()
    now = {"t": 0.0}
    b = Budget(cfg, clock=lambda: now["t"], time_limit_s=1815.0)
    b.turns_left = lambda: 30
    for _ in range(10):
        b.note_action(20.0)  # a slow patient
    assert b.turn_s > 15
    b.charge("focus", 100, 100, True, 300.0)  # a slow model
    assert b.call_s > 150
    now["t"] = 600.0  # 20 minutes left
    mid = b.call_timeout("review")
    assert 30 < mid <= b.remaining() - 30 * b.turn_s - b.close_s() + 1e-6
    assert b.call_timeout("assess") > mid and b.call_timeout("notes") <= cfg.call_timeout
    now["t"] = 1000.0  # the rest of the visit needs more than is left: no mid-visit call at all
    with pytest.raises(BudgetExhausted):
        b.check("review")
    b.check("assess")  # the assessment still fits
    now["t"] = b.time_limit - b.close_s() + 1
    assert b.wrap_up_due(10)  # the end of the visit needs the time that is left


def test_fast_server_keeps_the_full_visit():
    cfg = NovaConfig()
    b = Budget(cfg, clock=lambda: 0.0, time_limit_s=1815.0)
    assert not b.wrap_up_due(10) and b.call_timeout("focus") == cfg.call_timeout


def test_complaint_phrase_is_the_reason_for_the_visit():
    assert complaint_phrase("이름은 최성훈이고, 나는 쉰두 살 남자이다. 몇 년 전부터 코를 심하게 골아서 병원에 왔다.") == \
        "코를 심하게 골아서 내원"
    assert complaint_phrase("") == "주호소 확인 필요"


def test_written_denial_of_allergy():
    assert allergy_classes("약 알레르기는 없다.") == []
    assert allergy_classes("없어. 그런 거 없어") == []
    assert allergy_classes("페니실린 알레르기가 있어요.")


@pytest.mark.parametrize("opening,ped,sex,want", [
    ("어제부터 소변 양이 많이 줄어서 병원에 왔다.", False, "female", "oliguria"),
    ("어제부터 소변이 조금밖에 안 나와서 병원에 왔다.", False, "male", "oliguria"),
    ("일주일 전부터 소변을 많이 보고 목이 말라서 왔다.", False, "male", "polyuria"),
    ("엄마는 아이가 어제부터 소변에 피가 섞여 나와서 병원에 데려왔다.", True, "male", None),
    ("엄마는 아이가 어제부터 변에 피가 섞여 나와서 병원에 데려왔다.", True, "male", "ped_vomiting_diarrhea"),
])
def test_routing_tells_urine_from_stool_and_less_from_more(opening, ped, sex, want):
    pb = PB.select(opening, ped, sex)
    assert (pb.id if pb else None) == want


def test_simulated_visit_survives_a_model_that_never_answers():
    """No model answer at all (every call empty): the visit still ends with a determined diagnosis and a full note."""
    sim = pytest.importorskip("sim.runner")
    from sim import cases as C
    from sim.scenarios import SCENARIOS

    case = C.load(ids=["L01"])[0]
    res = sim.run_case(case, SCENARIOS["model_empty"])
    assert res["ok"], res["failed"]
    assert "미정" not in res["metrics"]["diagnosis"]
    assert threading.active_count() < 50  # abandoned calls do not pile up threads


def test_settings_file_changes_defaults_without_code(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    f.write_text('{"effort": {"assess": "high"}, "closing_lines": 4, "call_timeout": 300, "nope": 1, "max_turns": "x"}',
                 encoding="utf-8")
    monkeypatch.setenv("NOVA_SETTINGS_FILE", str(f))
    c = NovaConfig.from_env()
    assert c.effort_for("assess") == "high" and c.effort_for("focus") == "medium"
    assert c.closing_lines == 4 and c.call_timeout == 300.0 and c.max_turns == 50
    f.write_text("{not json", encoding="utf-8")
    assert NovaConfig.from_env().closing_lines == NovaConfig().closing_lines  # a broken file is ignored


@pytest.mark.parametrize("text,want", [("이틀 전부터 눈이", "2일 전"), ("일주일 전부터", "1주 전"), ("한 시간 반 전에 갑자기", "1시간 반 전"),
                                       ("석 달 전부터", "3개월 전"), ("열흘 전부터", "10일 전"), ("어제부터", "내원 1일 전")])
def test_onset_in_chart_form_reads_day_words(text, want):
    from dxagent.nova.soap import _ONSET, onset_phrase

    m = _ONSET.search(text)
    assert m and onset_phrase(m.group(0)) == want
