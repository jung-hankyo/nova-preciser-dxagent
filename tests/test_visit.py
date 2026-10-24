"""The lean visit (core off, the production default): model calls, the focus call and the selection, the turn
target, communication, and the note's layout. All offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dxagent.nova import checklist as CL
from dxagent.nova.agent import NovaAgent, make_store
from dxagent.nova.config import NovaConfig
from dxagent.nova.env import LocalSimEnv
from dxagent.nova.korean import acknowledgment, address, empathy_line, parse_name
from dxagent.nova.offline import OfflineNovaLLM
from dxagent.nova.prompts import FocusPlan, LeadDx, PlanItem
from dxagent.nova.protocol import check_say
from dxagent.nova.soap import exam_lines, group_tests, onset_phrase, region

CASES = Path(__file__).resolve().parents[1] / "dxagent" / "nova" / "cases"


def _case(name: str) -> dict:
    return json.loads((CASES / f"{name}.json").read_text(encoding="utf-8"))


def _cfg(**kw) -> NovaConfig:
    c = NovaConfig()
    c.use_guidelines = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


@pytest.fixture(scope="module")
def store():
    return make_store(_cfg())


def _play(case: dict, cfg: NovaConfig, store, **kw):
    agent = NovaAgent(cfg, store=store, offline=True, **kw)
    env = LocalSimEnv(case, cfg)
    return agent.run(env), env


def _says(env) -> list[str]:
    return [a["text"] for a in env.actions if a["kind"] == "SAY"]


# -- model calls and the core switch ------------------------------------------------------------------------------


def test_lean_visit_makes_four_calls(store):
    """Focus, diagnostic time-out, assessment, chart (v0.8.0: the time-out is the one call added to the lean visit)."""
    rec, env = _play(_case("sample_pe"), _cfg(), store)
    roles = rec["budget"]["by_role"]
    assert set(roles) == {"focus", "review", "assess", "notes"} and rec["budget"]["calls"] == 4
    assert rec["playbook"] == "dyspnea" and env.violations == [] and env.submission["diagnosis"]


def test_core_mode_restores_the_core(store):
    rec, _ = _play(_case("sample_pe"), _cfg(core=True, use_fixtures=True), make_store(_cfg(use_fixtures=True)))
    roles = rec["budget"]["by_role"]
    assert {"perceive", "hypothesize"} <= set(roles) and "focus" not in roles


def test_no_playbook_visit_plans_the_topic_first(store):
    rec, _ = _play(_case("sample_pe"), _cfg(playbooks=False), store)
    assert set(rec["budget"]["by_role"]) == {"plan", "focus", "review", "assess", "notes"}


# -- the focus call and the selection -----------------------------------------------------------------------------


def _focus_llm(store, skip_text: str, add_text: str, question: str):
    """A focus call that skips the optional item asking `skip_text`, adds the extra item asking `add_text` and writes
    one new question."""

    class Focus(OfflineNovaLLM):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)

            def plan(system: str, user: str) -> FocusPlan:
                ids = {}
                for line in user.splitlines():
                    if line.startswith("[") and "] " in line:
                        i, t = line[1:].split("] ", 1)
                        ids[t] = i
                return FocusPlan(topic_ko="급성 호흡곤란", leading=[LeadDx(name_en="Pulmonary embolism", name_ko="폐색전증",
                                                                          why="수술 후 종아리 부종")],
                                 skip=[ids[skip_text]], add=[ids[add_text]],
                                 questions=[PlanItem(text_ko=question, target="calf pain", item="Calf pain")],
                                 exams=[PlanItem(text_ko="심전도 검사하겠습니다.", target="ECG", item="ECG", section="exam"),
                                        PlanItem(text_ko="종아리 압통 확인하겠습니다.", target="calf tenderness",
                                                 item="Calf", section="exam")])

            self.handlers[FocusPlan] = plan

    return lambda b: Focus(store, b)


def test_focus_call_skips_adds_and_asks(store):
    from dxagent.nova import playbooks as PB

    pb = PB.load_all()["dyspnea"]
    optional = next(s.ask for s in pb.history if s.tier == 2)
    extra = next(s.ask for s in pb.history if s.tier == 3)
    rec, env = _play(_case("sample_pe"), _cfg(), store,
                     llm_factory=_focus_llm(store, optional, extra, "종아리를 누르면 아프세요?"))
    asked = _says(env)
    assert all(optional not in t for t in asked)  # skipped
    assert any(extra in t for t in asked)  # added
    assert any("종아리를 누르면 아프세요?" in t for t in asked)  # the focus call's own question
    assert rec["leading"][0]["en"] == "Pulmonary embolism" and optional and extra
    exams = [a["text"] for a in env.actions if a["kind"] == "EXAM"]
    assert "종아리 압통 확인하겠습니다." in exams  # its own maneuver
    assert not any("심전도" in t for t in exams)  # a test is not an exam (EXAM_NOT_RECOGNIZED would cost a turn)


def test_failed_focus_call_keeps_the_deterministic_visit(store):
    rec, env = _play(_case("sample_pe"), _cfg(), store,
                     llm_factory=lambda b: OfflineNovaLLM(store, b, fail_roles=("focus",)))
    assert any(e["kind"] == "focus_failed" for e in rec["events"]) and env.submission["diagnosis"]
    assert env.violations == [] and rec["turns"] >= 25


def test_turn_target_is_an_aim_not_a_cut(store):
    """The turn target is the usual length the focus call trims toward; nothing is cut only to reach it (50 is the
    cap; priority-2 items are cut only past the plan cap or when they reach the turns kept for later steps)."""
    from dxagent.nova import playbooks as PB

    runs = {t: _play(_case("sample_pe"), _cfg(turn_target=t), store) for t in (30, 40)}
    (r30, e30), (r40, e40) = runs[30], runs[40]
    assert _says(e30) == _says(e40) and r30["turns"] == r40["turns"] <= 49  # the offline focus call skips nothing
    pb = PB.load_all()["dyspnea"]
    asked = " ".join(_says(e40))
    p2 = [s for s in pb.history if s.tier == 2 and pb._keep(s.id, "male", False)]
    assert all(s.ask in asked for s in p2 if s.id not in r40["dropped"])  # every priority-2 item asked unless cut
    p3 = [s.ask for s in pb.history if s.tier == 3]
    assert not any(q in asked for q in p3)  # priority 3 only when the focus call adds it
    assert e40.violations == [] and all(a["accepted"] for a in e40.actions)


def test_focus_call_is_told_the_turn_budget():
    from dxagent.nova.prompts import focus_user

    u = focus_user("숨이 차요.", "", 58, "male", "", "", ["a"], [("o0", "b")], [], turns=(14, 44, 40, 50))
    assert "14 used" in u and "turn 44" in u and "about 40" in u and "50 is the hard limit" in u
    assert "do not skip a needed item only to save turns" in u


def test_soft_cap_cuts_priority_two_but_never_priority_one(store):
    from dxagent.nova import playbooks as PB

    rec, env = _play(_case("sample_pe"), _cfg(soft_turn_cap=26), store)
    pb = PB.load_all()["dyspnea"]
    must = [s.ask for s in pb.history if s.tier == 1 and pb._keep(s.id, "male", False)]
    asked = " ".join(_says(env))
    told = {e["ask"] for e in rec["ledger"] if e["source"] == "volunteered"}  # answered before it was asked
    assert all(q in asked or q in told for q in must if q != "언제부터 숨이 차셨어요?")  # every priority-1 item
    # the cut items may come back through the backfill when turns are left, never past the uncapped visit
    assert rec["dropped"] and rec["turns"] <= _play(_case("sample_pe"), _cfg(), store)[0]["turns"]


def test_visit_ends_when_the_plan_is_done(store):
    """No filler questions: the visit ends when the plan does, under the 50-turn limit."""
    rec, env = _play(_case("sample_pe"), _cfg(), store)
    asked = _says(env)
    assert len(asked) == len(set(asked)) and rec["turns"] <= 49
    focus = next(e for e in rec["events"] if e["kind"] == "select")
    assert focus["planned"] <= focus["target"]  # the plan fit: nothing was cut for turns


# -- communication --------------------------------------------------------------------------------------------------


def test_naming_and_titles():
    assert parse_name("저는 이수진이고, 쉰다섯 살 여자이다.") == "이수진"
    assert parse_name("이름은 정하준이고, 일곱 살 남자아이이다.") == "정하준"
    assert parse_name("안녕하세요, 박정호입니다. 58세 남자예요.") == "박정호"
    assert parse_name("엄마예요. 아이가 열이 나요.") == "" and parse_name("가슴이 아파요.") == ""
    assert address("저는 이수진이고", False) == "이수진 님"
    assert address("이름은 정하준이고, 일곱 살 남자아이이다. 엄마 정미영이 보호자로 함께 왔다.", True) == "어머님"
    assert address("아빠가 데리고 왔다. 아이가 기침을 한다.", True) == "아버님"
    assert address("기침이 나요", False) == "" and address("기침이 나요", False, name="이수진") == "이수진 님"
    assert empathy_line("가슴이 아파요", False, True) == "많이 아프셨겠어요."
    assert empathy_line("살이 빠져서 걱정이 되었다", False, False) == "걱정 많으셨겠어요."
    assert empathy_line("아이가 열이 나요", True, False) == "걱정 많으셨겠어요."


def test_acknowledgments_follow_the_patient_not_denials():
    assert acknowledgment("암일까 봐 걱정돼요.") == "걱정되셨겠어요."
    assert acknowledgment("너무 아파서 잠을 못 자요.") == "많이 아프셨겠어요."
    assert acknowledgment("요즘 너무 힘들어요.") == "많이 힘드셨겠어요."
    for r in ("걱정은 없어요.", "우울하지는 않아요.", "아니요, 그런 건 없어요.", "네, 좀 두근거려요."):
        assert acknowledgment(r) == ""


def test_visit_talks_to_the_patient(store):
    case = _case("sample_pe")
    case["qa"].append({"keys": ["걱정"], "answer": "심장마비일까 봐 너무 걱정돼요."})
    rec, env = _play(case, _cfg(), store)
    says = _says(env)
    assert says[0].startswith("안녕하세요, 박정호 님.")  # greeting by name, folded into the first question
    assert any(t.startswith("많이 아프셨겠어요.") for t in says[1:3])  # the opening's pain acknowledged early
    assert CL.CONCERN.ask in " ".join(says)
    i = next(k for k, t in enumerate(says) if CL.CONCERN.ask in t)
    assert says[i + 1].startswith("걱정되셨겠어요.")  # the voiced worry acknowledged on the next question
    assert any(t.startswith(CL.SIGNPOSTS["background"]) for t in says)  # signpost before the background
    closing = ([a["text"] for a in env.actions if a.get("meta", {}).get("source") == "closing"]
               or says[-_cfg().closing_lines:])
    assert closing[0].startswith("박정호 님,")
    assert all(check_say(t)[0] for t in says) and env.violations == []
    assert len({t for t in says if t.startswith(("많이", "걱정"))}) <= 3  # acknowledgments stay rare


def test_guardian_is_greeted_by_title(store):
    case = _case("sample_pneumonia")
    case["opening"] = {"text": "이름은 이준호이고, 다섯 살 남자아이이다. 엄마가 보호자로 함께 왔다. 엄마는 아이 키가 또래보다 작아서 걱정이다."}
    rec, env = _play(case, _cfg(), store)
    says = _says(env)
    assert says[0].startswith("안녕하세요, 어머님.") and says[1].startswith("걱정 많으셨겠어요.")
    assert not any(w in " ".join(says) for w in ("담배", "술은", "생리"))


# -- the note -----------------------------------------------------------------------------------------------------


def test_exam_is_written_by_region_with_turns():
    assert region("Heart") == "Chest" and region("CVA") == "Back" and region("Neuro") == "Extremities / Neuro"
    assert region("Mental status") == "General" and region("whatever") == "Others"
    text = exam_lines([("Neck", "갑상선 촉진 미만성 비대(+)", 21), ("Chest", "폐 청진 clear", 23),
                       ("Neck", "경부 림프절 촉지(-)", 22)], "BP 120/80 mmHg (내원 시)")
    assert text.splitlines() == ["V/S : BP 120/80 mmHg (내원 시)",
                                 "Neck : 갑상선 촉진 미만성 비대(+) (턴 21); 경부 림프절 촉지(-) (턴 22)",
                                 "Chest : 폐 청진 clear (턴 23)"]
    assert region("General", "신체 계측하겠습니다.") == "Growth" and region("V/S", "기립성 혈압 측정하겠습니다.") == "V/S"
    text = exam_lines([("V/S", "기립성 혈압 수축기 강하 6 mmHg", 30), ("Growth", "키 105cm (3백분위수 미만)", 31)],
                      "BP 100/60 mmHg (내원 시)")
    assert text.splitlines() == ["V/S : BP 100/60 mmHg (내원 시); 기립성 혈압 수축기 강하 6 mmHg (턴 30)",
                                 "Growth : 키 105cm (3백분위수 미만) (턴 31)"]  # one V/S line, then Growth


def test_tests_one_line_per_purpose_by_kind():
    """P.tests: one line per purpose, ordered by kind (each test tied to what it settles)."""
    lines = group_tests(["혈액검사 : CBC, CRP : 염증 확인", "복부 초음파 : 담석 확인", "12-lead ECG : 허혈 배제",
                         "소변 임신 반응 검사 : 방사선 영상 검사 전 확인", "혈액검사 : lipase : 급성 췌장염 감별",
                         "Upper endoscopy", "Stool culture : 세균성 장염 감별"])
    assert lines == ["- 혈액검사 : CBC, CRP : 염증 확인", "- 혈액검사 : lipase : 급성 췌장염 감별",
                     "- 소변검사 : 소변 임신 반응 검사 : 방사선 영상 검사 전 확인",
                     "- 영상검사 : 복부 초음파 : 담석 확인", "- 심전도 : 12-lead ECG : 허혈 배제",
                     "- 내시경 : Upper endoscopy", "- 기타 : Stool culture : 세균성 장염 감별"]


def test_note_follows_the_guide(store):
    rec, env = _play(_case("sample_pe"), _cfg(), store)
    note = env.submission["note"]
    s, o, a, p = note["S"], note["O"], note["A"], note["P"]
    assert s["cc"].startswith("C.C : 호흡곤란") and "Onset :" in s["cc"]
    assert s["hpi"].startswith("58세 남자.")
    assert s["ros"].splitlines()[0].startswith("General weakness (") and "Others :" in s["ros"]
    assert s["pmh"].splitlines()[0].startswith("DM / HTN") and "Allergy" in s["pmh"]
    asked_tbc = any("결핵" in t for t in _says(env))  # the dyspnea playbook asks it at priority 1 (v0.8.0)
    assert ("Tbc / hepatitis ( / )" in s["pmh"]) != asked_tbc  # a template item not asked: "( )", never "(-)"
    lines = o["exam"].splitlines()
    assert lines[0].startswith("V/S :") and all(" : " in ln for ln in lines)
    heads = [ln.split(" : ")[0] for ln in lines]
    assert len(heads) == len(set(heads))  # one line per region
    for e in rec["ledger"]:
        if e["kind"] == "EXAM" and e["accepted"]:
            assert f"(턴 {e['turn']})" in o["exam"]
    assert a["evidence"].startswith("#1. ") and "\n- 근거 : " in a["evidence"]
    assert all(ln.startswith("- ") for ln in p["tests"].splitlines())
    assert p["disposition"] and "진료 끝에 설명함" in p["education"] and "진찰 전 설명함" in p["education"]


def test_model_written_chart_is_grouped_and_guarded(store):
    from dxagent.nova.prompts import Assessment, ChartLine, ChartOutput

    class Chart(OfflineNovaLLM):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.handlers[ChartOutput] = self._chart
            base = self.handlers[Assessment]
            self.handlers[Assessment] = lambda s, u: base(s, u).model_copy(update={"evidence_ko": [
                "T10 오른쪽 종아리 부종 : 심부정맥혈전증 동반 가능", "T12 D-dimer 5,000 상승"]})  # 2nd: never measured

        @staticmethod
        def _chart(system: str, user: str) -> ChartOutput:
            ids = [ln.split("]")[0][1:] for ln in user.splitlines() if ln.startswith("[e")]
            exam = [i for i in ids if "EXAM" not in i]
            return ChartOutput(cc="C.C : 급성 호흡곤란\nOnset : 오늘 아침", hpi="58세 남자. 오늘 아침부터 갑자기 숨이 참.",
                               exam=[ChartLine(id=i, text=f"Chest : 청진 소견 {k}") for k, i in enumerate(exam)])

    rec, env = _play(_case("sample_pe"), _cfg(), store, llm_factory=lambda b: Chart(store, b))
    note = env.submission["note"]
    assert note["S"]["cc"].startswith("C.C : 급성 호흡곤란") and note["S"]["hpi"].startswith("58세 남자.")
    heads = [ln.split(" : ")[0] for ln in note["O"]["exam"].splitlines()]
    assert len(heads) == len(set(heads)) and heads[0] == "V/S"
    assert "종아리 부종" in note["A"]["evidence"] and "5,000" not in note["A"]["evidence"]
    assert any(e["kind"] == "evidence_rejected" for e in rec["events"])


def test_onset_in_chart_form():
    """As a CPX chart writes it: "Onset : 3개월 전", "Onset : 내원 1일 전"."""
    assert onset_phrase("석 달 동안") == onset_phrase("3개월째") == "3개월 전"
    assert onset_phrase("두 달 전부터") == "2개월 전" and onset_phrase("7년 전부터") == "7년 전"
    assert onset_phrase("어제부터") == "내원 1일 전" and onset_phrase("오늘 아침부터") == "내원 당일"
    assert onset_phrase("며칠 전부터") == "며칠 전부터"  # left as said
