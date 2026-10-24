"""Friendly talk within the visit's turns, records of what was done, the follow-up said and written, and no question
or maneuver repeated (v0.9.1). All offline."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from dxagent.nova import checklist as CL
from dxagent.nova import playbooks as PB
from dxagent.nova.agent import (NovaAgent, _smoked_without_years, exam_key, make_store, patient_question,
                                reported_symptom, reported_topic, split_line, symptom_keys)
from dxagent.nova.config import NovaConfig
from dxagent.nova.env import LocalSimEnv
from dxagent.nova.ledger import Entry
from dxagent.nova.offline import OfflineNovaLLM
from dxagent.nova.prompts import ChartOutput, PlanItem, ReviewPlan
from dxagent.nova.protocol import check_say
from dxagent.nova.soap import _turn_tags, complete_signs

CASES = Path(__file__).resolve().parents[1] / "dxagent" / "nova" / "cases"

CHEST = {
    "id": "t_chest", "title": {"ko": "가슴 답답함", "en": "Chest tightness"}, "population": "any", "pain": False,
    "match": {"any": ["가슴이 답답"]},
    "history": [
        {"id": "onset", "q": "언제부터 그러셨어요?", "item": "Onset", "p": 1},
        {"id": "dyspnea", "q": "숨이 차세요?", "item": "Dyspnea", "p": 1},
        {"id": "sweat", "q": "아플 때 식은땀이 나나요?", "item": "Diaphoresis", "p": 1},
        {"id": "night_sweat", "q": "밤에 식은땀이 나세요?", "item": "Night sweats", "p": 1},
        {"id": "fhx_heart", "q": "가족 중 심장병 있는 분 있나요?", "item": "FHx: heart disease", "sec": "fhx", "p": 1},
    ],
    "exams": [{"name": "심음 청진", "item": "Heart", "p": 1}, {"name": "폐 청진", "item": "Chest", "p": 1}],
    "differential": [{"en": "Stable angina", "ko": "안정형 협심증"},
                     {"en": "Acute coronary syndrome", "ko": "급성 관상동맥 증후군", "cant_miss": True}],
    "tests": [{"name": "12-lead ECG", "for": ["Stable angina", "Acute coronary syndrome"], "why": "허혈 확인"}],
    "plan": {"treatment": ["항협심증 약물 검토"], "disposition": "외래 추적. 근거 : 활력징후 안정",
             "follow_up": ["1주 후 재진 : 심전도·혈액검사 결과 확인"], "red_flags": ["쉬어도 낫지 않는 가슴 통증", "실신"]},
    "education": ["검사로 심장을 확인할게요."],
}


def _cfg(**kw) -> NovaConfig:
    c = NovaConfig()
    c.use_guidelines = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


@pytest.fixture(scope="module")
def store():
    return make_store(_cfg())


def _case(qa: list[tuple[str, str]], opening: str = "가슴이 답답해서 왔어요.", age: int = 58, sex: str = "M") -> dict:
    case = json.loads((CASES / "sample_pe.json").read_text(encoding="utf-8"))
    case["opening"] = {"text": opening, "name": "", "age": age, "sex": sex}
    case["qa"] = [{"keys": [k], "answer": a} for k, a in qa]
    case["exam"] = [{"keys": ["심음"], "result": "S1·S2 규칙적, 심잡음(-)"}, {"keys": ["폐 청진"], "result": "수포음(-)"}]
    return case


def _play(case: dict, store, cfg=None, pb=None, **kw):
    cfg = cfg or _cfg()
    pb = pb or PB.parse(copy.deepcopy(CHEST))
    orig = PB.select
    PB.select = lambda *a, **k: pb
    try:
        agent = NovaAgent(cfg, store=store, offline=True, **kw)
        env = LocalSimEnv(case, cfg)
        return agent.run(env), env
    finally:
        PB.select = orig


def _texts(env, kind=None) -> list[str]:
    return [a["text"] for a in env.actions if kind is None or a["kind"] == kind]


TOLD = [("언제부터", "어제부터요. 가슴이 답답하고 숨이 약간 차고 식은땀이 났어요. 이거 심각한 건가요?"),
        ("가족 중에 앓는", "아버지가 심근경색으로 돌아가셨어요."), ("밤에", "밤에는 괜찮아요.")]


# -- what counts as already answered --------------------------------------------------------------------------------


def _e(turn: int, reply: str, kind: str = "SAY", section: str = "hpi") -> Entry:
    return Entry(id=f"e{turn}", turn=turn, kind=kind, section=section, label="", text="", reply=reply)


def test_presence_questions_answered_by_an_earlier_reply():
    told = [_e(0, "머리가 아파서 왔어요.", "opening"), _e(3, "가슴이 답답할 때 숨이 약간 차고 식은땀이 날 때도 있어요."),
            _e(5, "멈춰서 쉬면 삼에서 오 분 안에 좋아져요."), _e(7, "열은 없어요."), _e(8, "계단을 오르면 숨이 차요.")]
    assert reported_symptom("아플 때 식은땀이 나나요?", told)[:1] == (3,)
    assert reported_symptom("숨이 차지는 않으세요?", told)[2] == [True]
    assert reported_symptom("열이 나세요?", told)[2] == [False]  # denied counts as answered
    assert reported_symptom("쉬면 좋아지나요?", told) is not None
    assert reported_symptom("계단 오를 때 숨이 차세요?", told)[0] == 8  # the condition must be in that reply
    assert reported_symptom("머리가 아프세요?", told)[0] == 0  # the opening counts
    # not presence questions, or a condition no reply named: still asked
    for q in ("두통이 갑자기 시작됐나요?", "누우면 숨이 더 차세요?", "밤에 식은땀이 나세요?", "열이나 오한이 있었나요?",
              "얼마나 자주 숨이 차세요?", "죽고 싶다는 생각이 드세요?"):
        assert reported_symptom(q, told) is None, q
    assert symptom_keys("손이 떨리지는 않으세요?") == []


def test_background_topic_needs_a_named_history_not_a_worry():
    fhx = [_e(9, "아버지가 예순 살에 심근경색으로 돌아가셨어요.", section="fhx")]
    assert reported_topic("가족 중 심장병 있는 분 있나요?", "fhx", fhx) == (9, fhx[0].reply)
    worry = [_e(4, "혹시 심장에 큰 문제가 있는 건 아닌지 걱정돼요.", section="pmh")]
    assert reported_topic("심장병을 진단받은 적 있나요?", "pmh", worry) is None
    one = [_e(5, "고혈압약을 먹고 있어요.", section="pmh")]
    assert reported_topic("고혈압이나 당뇨가 있으세요?", "pmh", one) is None  # both topics must be named
    assert reported_topic("가족 중 심장병 있는 분 있나요?", "fhx", [_e(9, "아니요, 없어요.", section="fhx")]) is None


def test_one_maneuver_under_two_wordings():
    assert exam_key("심장 청진을 하겠습니다.") == exam_key("심음 청진하겠습니다.")
    assert exam_key("폐 청진을 하겠습니다.") == exam_key("호흡음 청진하겠습니다.")
    assert exam_key("복부 촉진을 하겠습니다.") == exam_key("배를 촉진하겠습니다.")
    assert exam_key("우상복부 촉진하겠습니다.") != exam_key("복부 촉진을 하겠습니다.")  # a different maneuver
    assert exam_key("무릎 반사를 보겠습니다.") == exam_key("슬개건 반사 확인하겠습니다.")


def test_smoking_years_in_sino_korean_words_are_years():
    assert not _smoked_without_years("담배는 하루 한 갑씩 삼십 년째 피우고 있어요.")
    assert not _smoked_without_years("스무 년 정도 피웠어요.")
    assert _smoked_without_years("네, 하루 반 갑 피워요.")


def test_patient_questions_and_long_lines():
    assert patient_question("이거 심각한 건가요?") and patient_question("혹시 암은 아닐까요?")
    assert not patient_question("아니요, 없어요.") and not patient_question("네?")
    assert split_line("검사 결과를 보고 치료를 정할게요. 일주일 뒤에 다시 뵐게요.", 30) == [
        "검사 결과를 보고 치료를 정할게요.", "일주일 뒤에 다시 뵐게요."]


# -- the visit --------------------------------------------------------------------------------------------------------


def test_what_the_patient_already_told_is_recorded_not_asked_again(store):
    rec, env = _play(_case(TOLD), store)
    said = " ".join(_texts(env, "SAY"))
    assert "숨이 차세요?" not in said and "식은땀이 나나요?" not in said
    assert "밤에 식은땀이 나세요?" in said  # a condition nobody mentioned
    assert "가족 중 심장병 있는 분 있나요?" not in said  # the general family question's answer named it
    told = {e["ask"]: e for e in rec["ledger"] if e["source"] == "volunteered"}
    assert told["숨이 차세요?"]["note"] == "Dyspnea (+)" and told["아플 때 식은땀이 나나요?"]["note"] == "Diaphoresis (+)"
    assert all(e["turn"] is not None and e["turn"] > 0 for e in told.values())
    s = env.submission["note"]["S"]
    assert "Dyspnea (+)" in s["hpi"] + s["ros"] and "Diaphoresis (+)" in s["hpi"] + s["ros"]
    assert "심근경색" in s["fh"] and s["fh"].count("심근경색") == 1  # written once
    assert env.violations == [] and len(_texts(env)) == len(set(_texts(env)))


def test_skipping_can_be_switched_off(store):
    rec, env = _play(_case(TOLD), store, _cfg(skip_answered=False))
    said = " ".join(_texts(env, "SAY"))
    assert "숨이 차세요?" in said and not any(e["source"] == "volunteered" for e in rec["ledger"])


def test_a_question_from_the_patient_is_acknowledged_then_answered_at_the_end(store):
    rec, env = _play(_case(TOLD), store)
    says = _texts(env, "SAY")
    i = next(k for k, t in enumerate(says) if t.endswith(CHEST["history"][0]["q"]))
    assert says[i + 1].startswith(CL.ASKED_BACK)  # "끝나고 자세히 설명드릴게요. ..."
    assert all(check_say(t)[0] for t in says)


def test_the_examination_is_announced_once_and_recorded(store):
    rec, env = _play(_case(TOLD), store)
    acts = env.actions
    first_exam = next(k for k, a in enumerate(acts) if a["kind"] == "EXAM")
    intro = acts[first_exam - 1]
    assert intro["kind"] == "SAY" and intro["text"] == CL.EXAM_INTRO
    assert sum(1 for a in acts if a["text"] == CL.EXAM_INTRO) == 1
    edu = env.submission["note"]["P"]["education"]
    assert "진찰 전 설명함" in edu and CL.EXAM_INTRO in edu
    rec2, env2 = _play(_case(TOLD), store, _cfg(exam_intro=False))
    assert CL.EXAM_INTRO not in _texts(env2) and rec2["turns"] == rec["turns"] - 1


def test_the_time_out_words_the_announcement(store):
    def factory(budget):
        llm = OfflineNovaLLM(store, budget)
        llm.handlers[ReviewPlan] = lambda s, u: ReviewPlan(exam_intro_ko="이제 가슴 소리를 들어볼게요.")
        return llm

    rec, env = _play(_case(TOLD), store, llm_factory=factory)
    assert "이제 가슴 소리를 들어볼게요." in _texts(env, "SAY")

    def bad(budget):  # too long, a question: the fixed line instead
        llm = OfflineNovaLLM(store, budget)
        llm.handlers[ReviewPlan] = lambda s, u: ReviewPlan(exam_intro_ko="이제 가슴과 배와 다리를 차례로 진찰할 건데 괜찮으세요?")
        return llm

    rec, env = _play(_case(TOLD), store, llm_factory=bad)
    assert CL.EXAM_INTRO in _texts(env, "SAY")


def test_six_closing_lines_end_with_the_follow_up(store):
    rec, env = _play(_case(TOLD), store)
    lines = [line for line, _ in rec["closing_said"]]
    assert len(lines) == 6 and rec["turns"] <= 50
    assert any(re.search(r"(다시|뒤에|재진)", ln) for ln in lines)
    p = env.submission["note"]["P"]
    assert "환자에게 안내함" in p["follow_up"] and "진료 끝에 설명함" in p["education"]
    assert re.search(r"\d+\s*(일|주)", p["follow_up"])  # when to come back


def test_gathering_end_and_the_spare_turn():
    assert _cfg().gather_end() == 44 and _cfg(exam_intro=False).gather_end() == 43
    assert _cfg(closing_lines=5, exam_intro=False).gather_end() == 44


def test_plain_acknowledgments_stay_rare(store):
    qa = [("언제부터", "어제부터 계속 답답했어요."), ("숨이", "네, 계단을 오르면 숨이 많이 차요."),
          ("식은땀", "네, 아플 때 식은땀이 났어요."), ("밤에", "네, 밤에도 식은땀이 나요."),
          ("가족", "아버지가 고혈압이 있으세요."), ("약", "혈압약을 먹고 있어요.")]
    rec, env = _play(_case(qa), store, _cfg(skip_answered=False))
    says = _texts(env, "SAY")
    acks = [k for k, t in enumerate(says) if t.startswith(CL.LIGHT_ACKS)]
    assert 0 < len(acks) <= _cfg().light_acks_max
    assert all(b - a >= 3 for a, b in zip(acks, acks[1:]))


# -- the note ---------------------------------------------------------------------------------------------------------


def test_items_the_chart_left_out_are_added_back():
    e1 = Entry(id="e1", turn=4, kind="SAY", section="hpi", label="식은땀", text="식은땀이 나셨나요?", reply="네, 났어요.",
               item="Diaphoresis", ask="식은땀이 나셨나요?")
    e2 = Entry(id="e2", turn=6, kind="SAY", section="ros", label="발열", text="열이 나세요?", reply="아니요.", item="Fever",
               ask="열이 나세요?")
    e3 = Entry(id="e3", turn=7, kind="SAY", section="ros", label="기침", text="기침을 하세요?", reply="아니요.", item="Cough",
               ask="기침을 하세요?")
    res = {("S", "hpi"): "58세 남자. 어제부터 가슴 답답함.", ("S", "ros"): "Fever/Chill (-/ )\nOthers : 없음"}
    added = complete_signs(res, {("S", "hpi"): [e1], ("S", "ros"): [e2, e3]})
    assert added == ["Diaphoresis (+)", "Cough (-)"]  # Fever was charted
    assert res[("S", "hpi")].endswith("Diaphoresis (+)") and res[("S", "ros")].endswith("Others : Cough (-)")


def test_evidence_turn_tags_must_name_turns_that_happened():
    assert _turn_tags("머피 징후(+) : 담낭염 시사 (T31, T32)", 40).endswith("(T31, T32)")
    assert _turn_tags("머피 징후(+) : 담낭염 시사 (T31, T52)", 40) == "머피 징후(+) : 담낭염 시사"


def test_chart_prompt_marks_facts_told_before_asked(store):
    seen = []

    def factory(budget):
        llm = OfflineNovaLLM(store, budget)
        default = llm.handlers.get(ChartOutput)

        def chart(system, user):
            seen.append(user)
            return default(system, user) if default else ChartOutput()

        llm.handlers[ChartOutput] = chart
        return llm

    _play(_case(TOLD), store, llm_factory=factory)
    assert seen and "자발 진술" in seen[0] and "Q: 언제부터 그러셨어요?" in seen[0]
    assert "안녕하세요" not in seen[0].split("[S.cc + S.hpi]")[1]  # the bare question, not the greeting


# -- after the stand-in visits ------------------------------------------------------------------------------------------


def test_the_same_question_in_other_words():
    from dxagent.nova.agent import same_question

    assert same_question("최근 성관계를 하셨나요?", "성관계를 하신 적 있나요?")
    assert same_question("소변 보실 때 불편하세요?", "소변볼 때 아프세요?")
    assert not same_question("담배를 피우세요?", "담배는 몇 년 피우셨어요?")  # a question word makes it another
    assert not same_question("누우면 숨이 더 차세요?", "숨이 차세요?")
    assert not same_question("가슴이 두근거리나요?", "가슴이 아프세요?")


def test_vomiting_and_sudden_onset_told_before_asked():
    told = [_e(0, "한 시간 전에 갑자기 가슴이 아파서 왔어요.", "opening"), _e(9, "밤새 메스꺼웠고 오늘 아침에 한 번 토했어요.")]
    assert reported_symptom("토하신 적 있나요?", told)[0] == 9
    assert reported_symptom("갑자기 아프기 시작했나요?", told)[0] == 0
    assert reported_symptom("두통이 갑자기 시작됐나요?", told) is None  # its other words are not in that reply


def test_the_note_guard_leaves_open_questions_to_the_chart():
    e = Entry(id="e1", turn=4, kind="SAY", section="hpi", label="양상", text="어떤 느낌으로 아프세요?", reply="아니요, 그런 건 없어요.",
              item="Character", ask="어떤 느낌으로 아프세요?")
    res = {("S", "hpi"): "22세 여자. 어제 저녁부터 배꼽 주위 통증."}
    assert complete_signs(res, {("S", "hpi"): [e]}) == []
    d = Entry(id="e2", turn=5, kind="SAY", section="hpi", label="호흡곤란", text="숨이 차세요?", reply="아니요.", item="Dyspnea",
              ask="숨이 차세요?")
    res = {("S", "hpi"): "호흡곤란 (-)."}
    assert complete_signs(res, {("S", "hpi"): [d]}) == []  # recorded in Korean already


def test_particles_after_test_names():
    from dxagent.nova.soap import _ro

    assert [w + _ro(w) for w in ("트로포닌", "소변배양", "CT", "흉부 X선", "심전도")] == [
        "트로포닌으로", "소변배양으로", "CT로", "흉부 X선으로", "심전도로"]


def test_education_separates_what_was_said(store):
    rec, env = _play(_case(TOLD), store)
    edu = env.submission["note"]["P"]["education"].splitlines()
    assert edu[0].startswith("- 진찰 전 설명함") and edu[1].startswith("- 진료 끝에 설명함")
    assert all(not ln.startswith("- 교육") or "내용" in ln for ln in edu)


def test_the_time_out_sees_the_queued_questions():
    from dxagent.nova.prompts import review_user

    u = review_user("배가 아파요.", "", 22, "female", "", "", "", [("m0", "복부 촉진", 1)], (30, 44),
                    ["성관계를 하신 적 있나요?"])
    assert "still to be asked" in u and "성관계를 하신 적 있나요?" in u
