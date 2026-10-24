"""Presentation playbooks (dxagent/nova/playbooks): file validity, routing from the opening statement, plan building,
and a playbook-led visit. All offline; needs only the pinned runtime wheels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dxagent.nova import checklist as CL
from dxagent.nova import playbooks as PB
from dxagent.nova.agent import NovaAgent, make_store
from dxagent.nova.config import NovaConfig
from dxagent.nova.env import LocalSimEnv
from dxagent.nova.guidelines import GuidelineLibrary
from dxagent.nova.korean import is_pediatric, parse_demographics
from dxagent.nova.protocol import char_len

ROOT = Path(__file__).resolve().parents[1]

# One plain opening per playbook (our own wording), and harder variants.
PROBES = {
    "chest_pain": "쉰 살 남자이다. 어제부터 가슴이 아파서 왔다.",
    "palpitation": "마흔 살 여자이다. 가슴이 두근거려서 왔다.",
    "dyspnea": "예순 살 남자이다. 숨이 차서 왔다.",
    "cough": "서른 살 남자이다. 기침이 2주째 나서 왔다.",
    "hemoptysis": "쉰다섯 살 남자이다. 기침할 때 피가 나와서 왔다.",
    "syncope": "스무 살 여자이다. 아침에 쓰러져서 왔다.",
    "edema": "예순 살 여자이다. 다리가 부어서 왔다.",
    "high_bp": "쉰 살 남자이다. 건강검진에서 혈압이 높다고 해서 왔다.",
    "abdominal_pain": "서른 살 여자이다. 배가 아파서 왔다.",
    "dyspepsia": "마흔 살 남자이다. 속이 쓰리고 소화가 안 돼서 왔다.",
    "vomiting": "스물다섯 살 여자이다. 어제부터 토해서 왔다.",
    "diarrhea": "서른 살 남자이다. 설사를 해서 왔다.",
    "constipation": "일흔 살 여자이다. 변비가 심해서 왔다.",
    "upper_gi_bleeding": "쉰 살 남자이다. 피를 토해서 왔다.",
    "hematochezia": "예순 살 남자이다. 대변에 피가 섞여 나와서 왔다.",
    "jaundice": "쉰 살 남자이다. 눈이 노랗게 변해서 왔다.",
    "abdominal_distension": "예순 살 남자이다. 배가 불러 와서 왔다.",
    "dysphagia": "예순다섯 살 남자이다. 음식이 잘 안 넘어가서 왔다.",
    "fever": "서른 살 여자이다. 열이 나서 왔다.",
    "fatigue": "마흔 살 여자이다. 너무 피곤해서 왔다.",
    "weight_loss": "쉰 살 남자이다. 살이 빠져서 왔다.",
    "weight_gain": "서른다섯 살 여자이다. 살이 쪄서 왔다.",
    "polyuria": "마흔다섯 살 남자이다. 목이 자주 마르고 소변을 많이 봐서 왔다.",
    "dyslipidemia": "쉰 살 남자이다. 건강검진에서 콜레스테롤이 높다고 해서 왔다.",
    "oliguria": "예순 살 남자이다. 소변 양이 줄어서 왔다.",
    "easy_bruising": "마흔 살 여자이다. 몸에 멍이 자주 생겨서 왔다.",
    "neck_pain": "마흔 살 여자이다. 뒷목이 아파서 왔다.",
    "nasal_symptoms": "서른 살 남자이다. 콧물이 계속 나서 왔다.",
    "neck_mass": "서른 살 여자이다. 목에 혹이 만져져서 왔다.",
    "headache": "서른 살 여자이다. 머리가 아파서 왔다.",
    "dizziness": "예순 살 여자이다. 어지러워서 왔다.",
    "seizure": "스물다섯 살 남자이다. 어제 경련을 해서 왔다.",
    "weakness_numbness": "일흔 살 남자이다. 오른쪽 팔에 힘이 빠져서 왔다.",
    "memory_loss": "일흔다섯 살 여자이다. 자꾸 깜빡해서 왔다.",
    "tremor": "예순 살 남자이다. 손이 떨려서 왔다.",
    "altered_consciousness": "여든 살 남자이다. 아들이 보호자로 함께 왔다. 어제부터 정신이 오락가락해서 왔다.",
    "joint_pain": "쉰 살 여자이다. 무릎이 아파서 왔다.",
    "back_pain": "마흔 살 남자이다. 허리가 아파서 왔다.",
    "dysuria": "스물다섯 살 여자이다. 소변볼 때 아파서 왔다.",
    "hematuria": "예순 살 남자이다. 소변에 피가 나와서 왔다.",
    "voiding_difficulty": "일흔 살 남자이다. 소변 줄기가 약해서 왔다.",
    "menstrual_irregularity": "스물두 살 여자이다. 석 달째 생리를 안 해서 왔다.",
    "dysmenorrhea": "스무 살 여자이다. 생리통이 심해서 왔다.",
    "vaginal_discharge": "서른 살 여자이다. 질 분비물이 많아져서 왔다.",
    "abnormal_uterine_bleeding": "쉰다섯 살 여자이다. 폐경 후에 피가 비쳐서 왔다.",
    "breast_mass": "마흔다섯 살 여자이다. 유방에 혹이 만져져서 왔다.",
    "insomnia": "예순 살 여자이다. 잠을 잘 못 자서 왔다.",
    "depressed_mood": "서른 살 남자이다. 우울해서 왔다.",
    "anxiety": "스물여덟 살 여자이다. 불안해서 왔다.",
    "rash": "서른 살 남자이다. 몸에 두드러기가 나서 왔다.",
    "ped_fever": "네 살 남자아이이다. 엄마가 보호자로 함께 왔다. 아이가 열이 나서 왔다.",
    "short_stature": "여덟 살 여자아이이다. 엄마가 보호자로 함께 왔다. 키가 작아서 왔다.",
    "developmental_delay": "두 살 남자아이이다. 엄마가 보호자로 함께 왔다. 아직 말을 못 해서 왔다.",
    "ped_vomiting_diarrhea": "한 살 여자아이이다. 엄마가 보호자로 함께 왔다. 아이가 토하고 설사를 해서 왔다.",
    "ped_cough": "세 살 남자아이이다. 엄마가 보호자로 함께 왔다. 아이가 기침을 해서 왔다.",
    "sore_throat": "스무 살 여자이다. 목이 아파서 왔다.",
    "ear_symptoms": "서른 살 남자이다. 귀가 아파서 왔다.",
    "ped_rash": "다섯 살 여자아이이다. 엄마가 보호자로 함께 왔다. 아이 몸에 발진이 생겨서 왔다.",
}
VARIANTS = [
    ("headache", "서른네 살 여자이다. 머리가 자주 아파서 병원에 왔다."),
    ("depressed_mood", "서른한 살 남자이다. 두 달 전부터 기분이 계속 가라앉아서 왔다."),
    ("chest_pain", "가슴이 조이는 것처럼 아파서 왔어요."),
    ("abdominal_pain", "오른쪽 아랫배가 아파서 왔다."),
    ("dizziness", "머리가 핑 돌아서 왔어요."),
    ("weakness_numbness", "입이 돌아가서 왔다."),
    ("insomnia", "수면제를 처방받으러 왔다."),
    ("upper_gi_bleeding", "짜장 같은 변을 봐서 왔어요."),
    ("edema", "발목이 붓고 아파서 왔다."),  # "목이 붓고" (sore throat) must not match inside "발목이"
    ("sore_throat", "목이 붓고 아파서 왔다."),
    # found by the clinical review
    ("edema", "양쪽 발목이 부어서 왔어요."),  # an excluded word ("목이 부") must not match inside "발목이"
    ("polyuria", "물을 많이 마시고 소변을 많이 봐요."),
    ("polyuria", "소변을 많이 봐요."),  # match words keep their adverbs: volume, not frequency
    ("dysuria", "소변을 자주 봐요."),
    ("insomnia", "자다가 자주 깨요."),
    ("weakness_numbness", "갑자기 오른쪽 팔에 힘이 없어서 왔어요."),  # not fatigue
    ("weakness_numbness", "입이 돌아갔어요."),
    ("seizure", "어제 갑자기 쓰러져서 온몸을 떨었어요."),  # not syncope
    ("syncope", "잠깐 의식이 없었어요."),
    ("dizziness", "쓰러질 것 같이 어지러워요."),
    ("hemoptysis", "기침을 하는데 피가 나와요."),
    ("upper_gi_bleeding", "토했는데 피가 섞여 나왔어요."),
    ("hematochezia", "변 볼 때 피가 나와요."),
    ("weight_gain", "살이 많이 쪘어요."),
    ("ear_symptoms", "한쪽 귀가 갑자기 안 들려요."),
    ("joint_pain", "무릎이 붓고 아파요."),
    ("tremor", "양손이 떨려요."),
    ("fever", "몸이 떨리고 열이 나요."),
    ("altered_consciousness", "할머니가 이상한 말을 하세요."),
    # urology and women's health review
    ("hematuria", "소변이 빨개요."),
    ("hematuria", "소변 볼 때 피가 나와요."),
    ("hematuria", "피 섞인 소변이 나와요."),
    ("voiding_difficulty", "예순 살 남자이다. 소변이 시원하게 안 나와요."),
    ("voiding_difficulty", "소변을 잘 못 봐요."),
    ("voiding_difficulty", "쉰 살 여자이다. 기침할 때 소변이 새요."),  # not cough
    ("voiding_difficulty", "밤에 소변 때문에 자주 깨요."),
    ("dysuria", "스무 살 여자이다. 소변을 볼 때 아파요."),
    ("menstrual_irregularity", "스물두 살 여자이다. 생리를 3개월째 안 해요."),  # durations are removed like adverbs
    ("menstrual_irregularity", "스물두 살 여자이다. 생리가 3달째 없어요."),
    ("dysmenorrhea", "열아홉 살 여자이다. 생리 기간에 배가 아파요."),  # not abdominal pain
    ("vaginal_discharge", "서른 살 여자이다. 밑이 가려워요."),  # not rash
    ("vaginal_discharge", "서른 살 여자이다. 분비물에서 냄새가 나요."),
    ("abnormal_uterine_bleeding", "쉰다섯 살 여자이다. 폐경했는데 피가 나와요."),
    ("abnormal_uterine_bleeding", "마흔 살 여자이다. 생리를 너무 많이 해요."),
    ("hematochezia", "마흔 살 여자이다. 변기에 피가 비쳐요."),  # not uterine bleeding
    ("insomnia", "쉰다섯 살 여자이다. 폐경했는데 잠을 못 자요."),
    ("breast_mass", "마흔 살 여자이다. 가슴에 몽우리가 만져져요."),
    ("cough", "기침이 2주째 안 멈춰요."),
    ("headache", "머리가 이틀째 아파요."),
    ("hemoptysis", "침에 피가 섞여 나와요."),  # "침" must start a word: not inside "아침"
]
NOT_ROUTED = ["걸을 때 몸이 불안정해요.", "콧물이 노랗게 변했어요.", "배에 물집이 생겼어요.",
              "마흔 살 여자이다. 잇몸에서 피가 비쳐요."]  # a generic "피가 비쳐" must not mean uterine bleeding


def _route(opening: str):
    age, sex = parse_demographics(opening)
    pb = PB.select(opening, is_pediatric(opening, age), sex)
    return pb.id if pb else None


def test_every_playbook_is_valid_and_probed():
    assert PB.validate_all() == []
    books = PB.load_all()
    assert len(books) >= 50 and set(PROBES) == set(books)
    for pb in books.values():
        assert pb.differential and pb.exams and pb.education
        assert all(char_len(s.ask) <= 30 for s in pb.history + pb.exams)


@pytest.mark.parametrize("want,opening", sorted(PROBES.items()) + VARIANTS)
def test_routing_from_the_opening(want, opening):
    assert _route(opening) == want


@pytest.mark.parametrize("opening", NOT_ROUTED)
def test_words_that_must_not_misroute(opening):
    assert _route(opening) not in ("anxiety", "jaundice", "abdominal_distension")


def test_population_filters():
    assert _route("서른 살 여자이다. 열이 나서 왔다.") == "fever"
    assert _route("네 살 남자아이이다. 엄마가 보호자로 함께 왔다. 아이가 열이 나서 왔다.") == "ped_fever"
    assert PB.select("생리통이 심해서 왔다.", sex="male") is None
    menu = dict(PB.menu(pediatric=True))
    assert "ped_fever" in menu and "chest_pain" not in menu


def test_plan_from_a_playbook():
    pb = PB.load_all()["chest_pain"]
    plan = CL.build_plan(["chest_pain"], "어제부터 가슴이 아파서 왔다.", "male", 58, True, True, 5, playbook=pb)
    asks = [s.ask for s in plan.hpi]
    assert "언제부터 아프셨어요?" not in asks  # onset already stated
    assert asks[0] == "갑자기 아프기 시작했나요?" and CL.HPI_PRIOR in plan.hpi
    bg = [s.ask for s in plan.background]
    assert "콜레스테롤이 높다고 들으셨어요?" in bg and "고혈압이나 당뇨가 있으세요?" in bg
    assert [s.ask for s in plan.exams][:2] == ["심음 청진하겠습니다.", "폐 청진하겠습니다."]
    assert all(pb.priority[s.id] == 3 for s in plan.ros if s.section != "ros")  # priority 3 waits for the end


def test_sex_limited_items_and_exams():
    pb = PB.load_all()["voiding_difficulty"]
    male = CL.build_plan([], "소변 줄기가 약해서 왔다.", "male", 70, False, False, 8, playbook=pb)
    female = CL.build_plan([], "소변이 새서 왔다.", "female", 60, False, False, 8, playbook=pb)
    m_ex, f_ex = {s.label for s in male.exams}, {s.label for s in female.exams}
    assert "전립선 촉진" in m_ex and "양손 골반 내진" not in m_ex
    assert "양손 골반 내진" in f_ex and "전립선 촉진" not in f_ex
    asks = {s.ask for s in male.hpi + male.background + male.ros}
    assert "자연분만을 몇 번 하셨나요?" not in asks


def test_playbook_led_visit_fits_the_turn_limit():
    cfg = NovaConfig()
    cfg.use_guidelines = False
    agent = NovaAgent(cfg, store=make_store(cfg), offline=True)
    case = json.loads((ROOT / "dxagent" / "nova" / "cases" / "sample_pe.json").read_text(encoding="utf-8"))
    env = LocalSimEnv(case, cfg)
    rec = agent.run(env)
    starts = [e for e in rec["events"] if e.get("kind") == "start"]
    assert starts and starts[0].get("playbook") == "dyspnea"  # "숨이 차고 가슴이 아파서": the first complaint leads
    exams = [a for a in env.actions if a["kind"] == "EXAM" and a["accepted"]]
    assert env.submission["action"] == "DIAGNOSE" and rec["turns"] <= cfg.max_turns and len(exams) >= 4
    assert env.violations == []


def test_guideline_paths(tmp_path):
    (tmp_path / "US_StatPearls").mkdir()
    (tmp_path / "US_StatPearls" / "a.md").write_text("# Chest Pain\n", encoding="utf-8")
    (tmp_path / "catalog.csv").write_text("파일명,제목\nUS_StatPearls/a.md,Chest Pain\n", encoding="utf-8")
    lib = GuidelineLibrary(tmp_path)
    docs = lib.get(["US_StatPearls/a.md", "US_StatPearls/missing.md"])
    assert [d.title for d in docs] == ["Chest Pain"]


def test_every_maneuver_is_one_specific_action():
    """The examiner refuses a bundled, whole-region or unspecified request (EXAM_NOT_SPECIFIC) and a request that is
    not an examination (EXAM_NOT_RECOGNIZED), each at a turn's cost. Every
    playbook maneuver is a standard vocabulary name or one specific action."""
    from dxagent.nova.protocol import check_exam

    for name in PB.EXAM_NAMES:
        req = PB.exam_request(name)
        assert check_exam(req)[0] and PB.specific_exam(req), name
        if name not in PB.SAMPLE_EXAMS:  # standard names may bundle ("간·비장 촉진"); our own names do not
            assert "·" not in name and not name.endswith("진찰"), name
    # refused requests: each is caught by one of the two checks
    for refused in ("팔 떨림 확인하겠습니다", "혈압과 맥박을 측정하겠습니다", "허리·골반 긴장 확인하겠습니다",
                    "머리, 목, 가슴을 관찰하겠습니다", "진찰 전부 해 주세요", "식사 기록을 확인하겠습니다",
                    "잘 때 숨소리 및 심장 소리 청진하겠습니다"):
        assert not (check_exam(refused)[0] and PB.specific_exam(refused)), refused
    for accepted in ("두 팔을 뻗게 해 떨림을 보겠습니다", "발목 반사를 보겠습니다", "복부 부위 관찰하겠습니다",
                     "심장 청진을 하겠습니다", "폐 청진하겠습니다"):
        assert check_exam(accepted)[0] and PB.specific_exam(accepted), accepted
