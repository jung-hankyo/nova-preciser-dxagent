"""v0.8.0, the evidence review: numbers spoken in Korean in the note check, one-entity diagnoses, follow-up questions
after a positive answer, pelvic exams only for sexually active patients, the diagnostic time-out, tests linked to what
they settle and the plan's named warning signs."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from dxagent.nova import checklist as CL
from dxagent.nova import playbooks as PB
from dxagent.nova.agent import NovaAgent, make_store
from dxagent.nova.config import NovaConfig
from dxagent.nova.env import LocalSimEnv
from dxagent.nova.korean import one_diagnosis, positive_answer
from dxagent.nova.offline import OfflineNovaLLM
from dxagent.nova.prompts import DX_RULE_CHOOSE, PlanItem, ReviewPlan
from dxagent.nova.soap import ground_lines, korean_numbers, numbers_grounded

CASES = Path(__file__).resolve().parents[1] / "dxagent" / "nova" / "cases"

HEADACHE = {
    "id": "t_headache", "title": {"ko": "두통", "en": "Headache"}, "population": "any", "pain": True,
    "match": {"any": ["머리가 아파"]},
    "history": [
        {"id": "onset", "q": "언제부터 아프셨어요?", "item": "Onset", "p": 1},
        {"id": "painkiller", "q": "진통제를 드시나요?", "item": "Analgesic use", "sec": "meds", "p": 1,
         "then": [{"id": "days", "q": "한 달에 며칠 드세요?", "item": "Analgesic days", "known": "며칠|일|열흘|보름"},
                  {"id": "since", "q": "언제부터 그렇게 드셨어요?", "item": "Analgesic duration"}]},
        {"id": "tbc", "q": "결핵이나 간염을 앓은 적 있나요?", "item": "Tbc / hepatitis", "sec": "pmh", "p": 1},
        {"id": "sexual", "q": "성관계를 하신 적 있나요?", "item": "Sexual activity", "sec": "ob", "p": 1,
         "only": "adult", "gate": "sexual"},
        {"id": "nausea", "q": "속이 메스껍나요?", "item": "Nausea", "p": 2},
    ],
    "exams": [{"name": "안저 검사", "item": "H/E/E/N/T", "p": 1},
              {"name": "질경 검사", "item": "GU", "p": 1, "only": ["female", "sexually_active"]},
              {"name": "경부 강직", "item": "Neck", "p": 2},
              {"name": "측두동맥 촉진", "item": "H/E/E/N/T", "p": 2}],
    "differential": [{"en": "Medication-overuse headache", "ko": "약물과용두통"},
                     {"en": "Migraine", "ko": "편두통"},
                     {"en": "Subarachnoid hemorrhage", "ko": "거미막밑출혈", "cant_miss": True}],
    "tests": [{"name": "Brain CT", "for": ["Subarachnoid hemorrhage"], "why": "출혈 배제"},
              {"name": "두통 일지", "for": ["Medication-overuse headache", "Migraine"], "why": "진통제 복용일 확인"},
              {"name": "소변 임신 반응 검사", "for": ["safety"], "why": "약물·영상 전 임신 배제"}],
    "plan": {"treatment": ["원인 약물 중단 계획", "예방 약물 검토"], "disposition": "외래 추적. 근거 : 신경학적 이상 없음",
             "follow_up": ["2주 후 재진 : 두통 일지 확인"], "red_flags": ["갑자기 생긴 극심한 두통", "팔다리 마비"]},
    "education": ["진통제는 줄여 나가야 해요.", "두통 일지를 써 주세요."],
}


def _pb(**changes):
    d = copy.deepcopy(HEADACHE)
    d.update(changes)
    return PB.parse(d)


def _cfg(**kw) -> NovaConfig:
    c = NovaConfig()
    c.use_guidelines = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


@pytest.fixture(scope="module")
def store():
    return make_store(_cfg())


def _case(opening: str, qa: list[tuple[str, str]], age: int = 41, sex: str = "F") -> dict:
    case = json.loads((CASES / "sample_pe.json").read_text(encoding="utf-8"))
    case["opening"] = {"text": opening, "name": "", "age": age, "sex": sex}
    case["qa"] = [{"keys": [k], "answer": a} for k, a in qa]
    case["exam"] = [{"keys": ["안저"], "result": "유두부종 없습니다."}, {"keys": ["질경"], "result": "특이 소견 없습니다."}]
    return case


def _play(case: dict, store, pb, cfg=None, **kw):
    cfg = cfg or _cfg()
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


# -- the note's number check ----------------------------------------------------------------------------------------


def test_numbers_spoken_in_korean_are_read_as_digits():
    assert korean_numbers("서른두 살이에요") == ["32"]
    assert korean_numbers("백육십사 센티미터") == ["164"]
    assert korean_numbers("예순두 살") == ["62"]
    assert korean_numbers("삼 점 이 키로") == ["3.2"]
    assert "3" in korean_numbers("석 달 전부터") and "6" in korean_numbers("반 년 전")
    assert korean_numbers("열이 나요") == [] and korean_numbers("목이 쉰 소리") == []  # words, not numbers


def test_chart_lines_are_checked_one_by_one():
    src = "마흔다섯 살 남자. 원래 칠십이 키로였는데 지금 육십육 키로예요. 두 달 전부터"
    assert numbers_grounded("45세 남자.\n평소 72 kg → 현재 66 kg (6 kg 감소), 2개월 전부터", src)  # 72-66 on the line
    assert not numbers_grounded("평소 72 kg → 현재 65 kg", src)
    kept, dropped = ground_lines("45세 남자.\n키 175 cm\n2개월 전부터 체중 감소", src)
    assert kept == "45세 남자.\n2개월 전부터 체중 감소" and dropped == ["키 175 cm"]
    assert numbers_grounded("이번 주부터 악화", "이번 주")  # "이번" in the chart's own text is not a number


# -- diagnosis, answers ----------------------------------------------------------------------------------------------


def test_one_diagnosis_entity():
    assert one_diagnosis("r/o Acute cholecystitis") == "Acute cholecystitis"
    assert one_diagnosis("급성 담낭염 의증") == "급성 담낭염"
    assert one_diagnosis("GERD 및 기능성 소화불량") == "GERD"
    assert one_diagnosis("Sedative, hypnotic, or anxiolytic use disorder") == \
        "Sedative, hypnotic, or anxiolytic use disorder"


def test_prompts_do_not_name_the_public_samples_answers():
    from dxagent.nova import prompts

    text = " ".join(str(getattr(prompts, n)) for n in dir(prompts) if n.isupper())
    for leak in ("Graves", "그레이브스", "constitutional delay", "sedative-hypnotic", "수면제 처방", "56 kg"):
        assert leak not in text
    assert "symptom without its cause" in DX_RULE_CHOOSE


def test_positive_answers():
    assert positive_answer("네, 타이레놀 먹어요", "진통제를 드시나요?")
    assert positive_answer("거의 매일 먹어요")
    assert not positive_answer("아니요")
    assert not positive_answer("예전엔 먹었는데 지금은 안 먹어요")


# -- playbook format -------------------------------------------------------------------------------------------------


def test_new_playbook_fields_validate():
    assert PB.validate(copy.deepcopy(HEADACHE)) == []
    bad = copy.deepcopy(HEADACHE)
    bad["tests"].append({"name": "MRI", "for": ["Tension headache"]})
    bad["plan"]["red_flags"] = ["하나"]
    bad["history"][1]["then"].append({"id": "x", "q": "약 이름과 용량은요?", "item": "x"})
    bad["history"][1]["then"].append({"id": "y", "q": "더 있나요?", "item": "y"})
    errs = " ".join(PB.validate(bad))
    assert "not a differential name" in errs and "red_flags" in errs and "at most 3" in errs


def test_playbook_item_raises_a_standard_item():
    plan = CL.build_plan([], "", "male", 41, True, False, 8, playbook=_pb())
    tbc = [s for s in plan.background if s.item == "Tbc / hepatitis"]
    assert len(tbc) == 1 and tbc[0].id == "pmh_tbc_hep" and tbc[0].tier == 1  # asked once, at priority 1


def test_linked_tests():
    pb = _pb()
    assert pb.tests_for(["Subarachnoid hemorrhage"]) == ["Brain CT", "소변 임신 반응 검사"]
    assert pb.test_line("Brain CT") == "Brain CT : 출혈 배제"


# -- the visit -------------------------------------------------------------------------------------------------------


def test_follow_ups_after_a_positive_answer(store):
    rec, env = _play(_case("머리가 아파서 왔어요.", [("진통제", "네, 타이레놀을 자주 먹어요."), ("며칠", "보름 정도요.")]),
                     store, _pb())
    says = " ".join(_texts(env, "SAY"))
    assert "한 달에 며칠 드세요?" in says and "언제부터 그렇게 드셨어요?" in says
    rec, env = _play(_case("머리가 아파서 왔어요.", [("진통제", "네, 한 달에 열흘 정도 먹어요.")]), store, _pb())
    says = " ".join(_texts(env, "SAY"))
    assert "한 달에 며칠 드세요?" not in says and "언제부터 그렇게 드셨어요?" in says  # `known` already answered
    rec, env = _play(_case("머리가 아파서 왔어요.", [("진통제", "아니요, 안 먹어요.")]), store, _pb())
    assert "한 달에 며칠" not in " ".join(_texts(env, "SAY"))


@pytest.mark.parametrize("age,answer,done", [(41, "네, 있어요.", True), (41, "아니요, 없어요.", False),
                                             (16, "아니요.", False), (16, "네.", True)])
def test_pelvic_exam_only_when_sexually_active(store, age, answer, done):
    rec, env = _play(_case("머리가 아파서 왔어요.", [("성관계", answer)], age=age), store, _pb())
    assert any("질경" in t for t in _texts(env, "EXAM")) == done


def test_no_pelvic_exam_for_a_child(store):
    rec, env = _play(_case("아이가 머리가 아프대요.", [], age=8), store, _pb())
    assert not any("질경" in t for t in _texts(env, "EXAM"))
    assert not any("성관계" in t for t in _texts(env, "SAY"))


def test_time_out_asks_reorders_and_skips(store):
    def factory(budget):
        llm = OfflineNovaLLM(store, budget)

        def review(system, user):
            ids = [ln[1:ln.index("]")] for ln in user.splitlines() if ln.startswith("[m")]
            opt = [ln[1:ln.index("]")] for ln in user.splitlines() if ln.startswith("[m") and ln.endswith("(p2)")]
            return ReviewPlan(names_cause=False, cant_miss_open=["Subarachnoid hemorrhage"],
                              questions=[PlanItem(text_ko="두통이 갑자기 시작됐나요?", item="Sudden onset")],
                              exams=[PlanItem(text_ko="대면 시야 검사하겠습니다.", item="H/E/E/N/T")],
                              exam_order=list(reversed(ids)), skip_exams=opt[:1])
        llm.handlers[ReviewPlan] = review
        return llm

    rec, env = _play(_case("머리가 아파서 왔어요.", [("성관계", "아니요.")]), store, _pb(), llm_factory=factory)
    acts = _texts(env)
    i_q = next(i for i, t in enumerate(acts) if t.endswith("두통이 갑자기 시작됐나요?"))
    exams = _texts(env, "EXAM")
    assert all(acts.index(e) > i_q for e in exams)  # the time-out's question before the exam
    assert "대면 시야 검사하겠습니다." in exams
    ev = next(e for e in rec["events"] if e["kind"] == "review")
    assert ev["names_cause"] is False and ev["skipped"] == 1
    assert "Subarachnoid hemorrhage" in json.dumps(rec["events"], ensure_ascii=False)


def test_visit_ends_with_six_explanations_within_the_cap(store):
    rec, env = _play(_case("머리가 아파서 왔어요.", [("진통제", "네, 거의 매일 먹어요.")]), store, _pb())
    assert rec["turns"] <= 49 and env.violations == []
    assert len(rec["closing_said"]) == 6
    assert any(re.search(r"다시 (뵙|뵐)|재진|뒤에", line) for line, _ in rec["closing_said"])  # the follow-up is said


def test_plan_names_linked_tests_and_warning_signs(store):
    rec, env = _play(_case("머리가 아파서 왔어요.", [("진통제", "네, 거의 매일 먹어요.")]), store, _pb())
    p = env.submission["note"]["P"]
    assert "즉시 내원" in p["follow_up"] and "팔다리 마비" in p["follow_up"]
    assert p["treatment"] and "표준 초기 치료" not in p["treatment"]
    a = env.submission["note"]["A"]
    assert "앞선 감별진단" not in a["differential"] and "턴" not in a["evidence"]


@pytest.mark.parametrize("answer,asked", [("네, 하루 한 갑 피워요.", True), ("네, 20년 피웠어요.", False),
                                          ("아니요, 안 피워요.", False)])
def test_smoking_years_follow_up(store, answer, asked):
    """Pack-years: a smoker is asked for how many years when the answer does not say (v0.8.0)."""
    rec, env = _play(_case("머리가 아파서 왔어요.", [("담배", answer)]), store, _pb())
    assert ("담배는 몇 년 피우셨어요?" in _texts(env, "SAY")) == asked


def test_safety_follow_up_is_not_capped(store):
    d = copy.deepcopy(HEADACHE)
    d["history"].insert(1, {"id": "mood", "q": "요즘 기분이 많이 가라앉나요?", "item": "Depressed mood", "p": 1,
                            "then": [{"id": "suicidal", "q": "죽고 싶다는 생각이 드시나요?", "item": "Suicidal ideation",
                                      "safety": True}]})
    assert PB.validate(copy.deepcopy(d)) == []
    rec, env = _play(_case("머리가 아파서 왔어요.", [("가라앉", "네, 많이 우울해요."), ("진통제", "네, 매일 먹어요.")]),
                     store, PB.parse(d), cfg=_cfg(followup_max=0))
    says = _texts(env, "SAY")
    assert "죽고 싶다는 생각이 드시나요?" in says and "한 달에 며칠 드세요?" not in says


def test_plan_cap_is_not_the_end_of_gathering(store):
    """The cap plans the visit; gathering runs to turn 44, and turns left after the time-out go back to the
    priority-2 items cut only for turns (0.8.0 integration: gathering had stopped at the cap)."""
    case = json.loads((CASES / "sample_pe.json").read_text(encoding="utf-8"))
    agent = NovaAgent(_cfg(soft_turn_cap=24), store=store, offline=True)
    env = LocalSimEnv(case, _cfg(soft_turn_cap=24))
    rec = agent.run(env)
    assert any(e["kind"] == "backfill" for e in rec["events"])
    assert rec["turns"] > 24 + 5 and rec["turns"] <= 49 and env.violations == []
