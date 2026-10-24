"""Documentation checklist: what a grader expects to see asked and recorded, whatever the diagnosis.

The core planner values a question only by how much it separates the diagnoses on the list, so it never asks about
allergies or family history. The contest scores "necessary history and exam actually done AND recorded" (25 points),
and the official note template (development guide, "초진 의무기록 작성 안내") has fixed fields:

    S.cc   C.C, Onset, 통증평가 (NRS, 부위, 양상, 기간)
    S.hpi  onset, course, character, aggravating/relieving factors, associated symptoms, prior care, visit
    S.pmh  DM / HTN, previous stroke, Tbc / hepatitis, Op. Hx, Medicine history, Allergy, each (+)/(-)
    S.fh   family history;  S.sh  Smoking (p/d, yrs), Alcohol, occupation;  S.ros  (+)/(-) items
    O.exam body part : finding (턴 N)

Each slot is ONE Korean question of at most 30 code points, or one exam maneuver, with:
- `item`: the chart item name it fills (e.g. "DM / HTN", "Fever/Chill"), so the note can say "(-)" only for what was
  actually asked, and "( )" for what was not;
- `codes`: registry finding codes it may reveal (the extractor checks them explicitly);
- `alt`: single-topic (question, item) pairs asked one by one instead if the patient answers "한 번에 하나씩 물어봐
  주세요." (e.g. "고혈압이나 당뇨가 있으세요?" -> "고혈압이 있으세요?" (HTN), "당뇨가 있으세요?" (DM));
- `tier`: 1 always asked, 2 asked unless the mid-visit focus call drops it or the plan would run past the soft cap
  (46 turns; 50 is the limit), 3 asked only when the focus call selects it (playbook items carry their priority p as the tier).

Ledger sections: hpi, pmh, meds, allergy, fhx, shx, ob, ros (S); exam (O). The note maps pmh + meds + allergy + ob to
S.pmh, fhx to S.fh, shx to S.sh.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

GREETING = "안녕하세요."  # prefixed to the first question when it fits in 30 characters (with the name if it fits)
EMPATHY = "많이 불편하셨겠어요."  # prefixed to one early question when it fits (korean.empathy_line picks the wording)


@dataclass(frozen=True)
class Slot:
    id: str
    section: str  # hpi | pmh | meds | allergy | fhx | shx | ob | ros | exam
    label: str  # Korean label (notes, logs)
    ask: str  # one question (<= 30 code points) or one exam maneuver
    codes: tuple[str, ...] = ()  # registry finding codes this may reveal
    kind: str = "SAY"  # SAY | EXAM
    item: str = ""  # chart item name in the note ("Allergy", "Fever/Chill", "Chest")
    alt: tuple[tuple[str, str], ...] = ()  # single-topic fallbacks: (question, chart item)
    tier: int = 1  # 1 always, 2 unless dropped (focus call, soft cap), 3 only if the focus call selects it


# -- History of present illness -------------------------------------------------------------------------------------

CC_ASK = Slot("cc", "hpi", "주호소", "어디가 불편해서 오셨어요?", item="C.C")  # only if the start information has none

HPI_ONSET = Slot("hpi_onset", "hpi", "발병 시점", "언제부터 그러셨어요?", ("sudden_onset",), item="Onset")
HPI_PAIN = [
    Slot("hpi_character", "hpi", "양상", "어떻게 아프세요?", item="양상"),
    Slot("hpi_location", "hpi", "위치·방사", "아픈 곳이 다른 데로 퍼지나요?",
         ("chest_pain_radiating_arm", "chest_pain_tearing_back"), item="방사"),
    Slot("hpi_severity", "hpi", "강도", "통증은 10점 중 몇 점인가요?", item="NRS"),
    Slot("hpi_modifiers", "hpi", "악화·완화 요인", "언제 더 심해지세요?",
         ("chest_pain_pleuritic", "chest_pain_exertional", "chest_pain_positional_relief"), item="악화 요인"),
]
HPI_GENERAL = [
    Slot("hpi_course", "hpi", "경과", "증상이 점점 심해지고 있나요?", item="경과"),
    Slot("hpi_severity_gen", "hpi", "중증도", "일상생활이 힘들 정도인가요?", item="중증도"),
    Slot("hpi_modifiers", "hpi", "악화·완화 요인", "언제 더 심해지세요?", ("orthopnea",), item="악화 요인", tier=2),
]
HPI_PRIOR = Slot("hpi_prior", "hpi", "이전 유사 증상", "전에도 이런 적이 있으셨어요?", item="이전 유사 증상", tier=2)
HPI_CARE = Slot("hpi_care", "hpi", "이전 진료", "이 증상으로 병원에 가 보셨어요?", item="이전 진료", tier=2)
# The patient's own concern (ideas, concerns, expectations): asked once, answered in the closing explanation.
CONCERN = Slot("hpi_concern", "hpi", "환자의 걱정", "가장 걱정되는 점이 무엇인가요?", item="환자의 걱정")

# -- Background -----------------------------------------------------------------------------------------------------

BACKGROUND = [
    Slot("pmh_dm_htn", "pmh", "고혈압·당뇨", "고혈압이나 당뇨가 있으세요?", ("hypertension_history",),
         item="DM / HTN", alt=(("고혈압이 있으세요?", "HTN"), ("당뇨가 있으세요?", "DM"))),
    Slot("pmh_other", "pmh", "기타 질환", "그 밖에 앓는 병이 있으세요?", ("copd_history", "prior_vte"),
         item="기타 질환", tier=2),
    Slot("pmh_tbc_hep", "pmh", "결핵·간염", "결핵이나 간염을 앓은 적 있나요?", item="Tbc / hepatitis",
         alt=(("결핵을 앓은 적 있나요?", "Tbc"), ("간염을 앓은 적 있나요?", "Hepatitis")), tier=3),
    Slot("surgery", "pmh", "수술·입원력", "수술이나 입원한 적 있으세요?", ("recent_immobilization_or_surgery",),
         item="Op. Hx", alt=(("수술받은 적 있으세요?", "Op. Hx"), ("입원한 적 있으세요?", "입원력")), tier=2),
    Slot("meds", "meds", "복용 약물", "드시는 약이 있으세요?", item="Medicine history"),
    Slot("allergy", "allergy", "약물 알레르기", "약 알레르기가 있으세요?", item="Allergy"),
    Slot("fhx", "fhx", "가족력", "가족 중에 앓는 병이 있으세요?", item="FHx"),
    Slot("smoking", "shx", "흡연", "담배를 피우세요?", ("smoking_history",), item="Smoking"),
    Slot("alcohol", "shx", "음주", "술은 얼마나 드세요?", item="Alcohol"),
    Slot("job", "shx", "직업", "어떤 일을 하세요?", item="직업", tier=3),
]
SMOKING_AMOUNT = Slot("smoking_amount", "shx", "흡연량", "하루에 몇 개비 피우세요?", item="Smoking p/d")
SMOKING_YEARS = Slot("smoking_years", "shx", "흡연 기간", "담배는 몇 년 피우셨어요?", item="Smoking years")  # pack-years
OB_LMP = Slot("ob_lmp", "ob", "마지막 월경·임신 가능성", "마지막 생리는 언제였나요?", item="LMP")
OB_CYCLE = Slot("ob_cycle", "ob", "월경 주기 변화", "생리 주기는 규칙적인가요?", item="Menstruation", tier=2)

# Children (under 13, or a guardian speaking): asked of the guardian; no smoking, alcohol, job or LMP.
PEDIATRIC = [
    Slot("ped_birth", "pmh", "출생력", "아이는 임신 몇 주에 태어났나요?", item="출생력"),
    Slot("ped_bw", "pmh", "출생 체중", "태어날 때 몸무게는 얼마였나요?", item="출생 체중", tier=2),
    Slot("ped_dev", "pmh", "발달력", "걷고 말하는 건 또래와 비슷했나요?", item="발달력"),
    Slot("ped_vax", "pmh", "예방접종", "예방접종은 다 맞았나요?", item="예방접종"),
    Slot("ped_pmh", "pmh", "과거 질환", "아이가 크게 앓은 병이 있나요?", item="기타 질환"),
    Slot("ped_meds", "meds", "복용 약물", "아이가 먹는 약이 있나요?", item="Medicine history"),
    Slot("ped_allergy", "allergy", "약물 알레르기", "아이가 약 알레르기가 있나요?", item="Allergy"),
    Slot("ped_fhx", "fhx", "가족력", "가족 중에 앓는 병이 있나요?", item="FHx"),
    Slot("ped_diet", "shx", "식습관", "아이가 밥은 잘 먹나요?", item="식습관", tier=2),
]
PED_PARENT_HT = Slot("ped_parent_ht", "fhx", "부모 키", "부모님 키는 어떻게 되세요?", item="부모 키")

# -- Review of systems and standard maneuvers, per complaint group --------------------------------------------------

FEVER = Slot("ros_fever", "ros", "발열·오한", "열이나 오한이 있었나요?", ("fever",), item="Fever/Chill",
             alt=(("열이 나셨나요?", "Fever"), ("오한이 있었나요?", "Chill")))
COUGH = Slot("ros_cough", "ros", "기침·가래", "기침이나 가래가 있나요?", ("productive_cough",), item="Cough/Sputum",
             alt=(("기침을 하세요?", "Cough"), ("가래가 나오나요?", "Sputum")))
DYSPNEA = Slot("ros_dyspnea", "ros", "호흡곤란", "숨이 차지는 않으세요?", item="Dyspnea")
WEAKNESS = Slot("ros_weakness", "ros", "국소 신경학적 증상", "팔다리에 힘이 빠지거나 저리나요?", item="Weakness/Numbness",
                alt=(("팔다리에 힘이 빠지나요?", "Weakness"), ("팔다리가 저리나요?", "Numbness")))
URINE = Slot("ros_urine", "ros", "배뇨 증상", "소변볼 때 불편하지는 않으세요?", item="Dysuria")
WEIGHT = Slot("ros_weight", "ros", "체중 감소", "최근에 살이 빠지셨나요?", item="Weight loss")
APPETITE = Slot("ros_appetite", "ros", "식욕", "입맛은 어떠세요?", item="Appetite")
HEAT = Slot("ros_heat", "ros", "더위 못 참음", "더위를 많이 타세요?", item="Heat intolerance")
COLD = Slot("ros_cold", "ros", "추위 못 참음", "추위를 많이 타세요?", item="Cold intolerance")
SWEAT = Slot("ros_sweat", "ros", "발한", "땀이 많이 나세요?", item="Sweating")
NIGHT_SWEAT = Slot("ros_night_sweat", "ros", "야간 발한", "밤에 식은땀이 나세요?", item="Night sweats")
PALP = Slot("ros_palpitation", "ros", "두근거림", "가슴이 두근거리나요?", item="Palpitation")
BOWEL = Slot("ros_bowel", "ros", "배변 변화", "대변 보는 게 달라졌나요?", item="Bowel habit change")
THIRST = Slot("ros_thirst", "ros", "갈증·다음", "목이 자주 마르세요?", item="Polydipsia")
MOOD = Slot("ros_mood", "ros", "우울감", "기분이 우울하지는 않으세요?", item="Depressed mood")
SLEEP = Slot("ros_sleep", "ros", "수면", "잠은 잘 주무세요?", item="Insomnia")
MELENA = Slot("ros_melena", "ros", "흑색변", "검은 변을 보신 적 있나요?", item="Melena")

ROS: dict[str, list[Slot]] = {
    "chest_pain": [
        DYSPNEA,
        Slot("ros_diaphoresis", "ros", "식은땀", "식은땀이 나셨나요?", ("diaphoresis",), item="Diaphoresis"),
        Slot("ros_palpitation", "ros", "두근거림", "가슴이 두근거리나요?", item="Palpitation"),
        FEVER,
    ],
    "dyspnea": [
        COUGH,
        Slot("ros_chest_pain", "ros", "흉통", "가슴이 아프지는 않으세요?", ("chest_pain_pleuritic",), item="Chest pain"),
        Slot("ros_orthopnea", "ros", "기좌호흡", "누우면 숨이 더 차세요?", ("orthopnea",), item="Orthopnea"),
        Slot("ros_leg", "ros", "다리 부종", "다리가 붓지는 않으세요?", ("unilateral_leg_swelling", "bilateral_leg_edema"),
             item="Leg swelling"),
        FEVER,
    ],
    "cough": [
        FEVER,
        Slot("ros_sputum", "ros", "가래 양상", "가래는 무슨 색인가요?", ("productive_cough",), item="Sputum"),
        Slot("ros_hemoptysis", "ros", "객혈", "가래에 피가 섞여 나오나요?", ("hemoptysis",), item="Hemoptysis"),
        DYSPNEA,
    ],
    "abdominal": [
        Slot("ros_nv", "ros", "구역·구토", "구역질이나 구토가 있나요?", item="N/V",
         alt=(("속이 메스꺼우세요?", "Nausea"), ("토하신 적 있나요?", "Vomiting"))),
        Slot("ros_bowel", "ros", "배변 변화", "대변 보는 게 달라졌나요?", item="Bowel habit change"),
        URINE, FEVER, APPETITE,
    ],
    "headache": [
        WEAKNESS,
        Slot("ros_vision", "ros", "시야 이상", "물체가 겹쳐 보이나요?", item="Diplopia"),
        Slot("ros_vomit", "ros", "구토", "토하신 적이 있나요?", item="Vomiting"),
        FEVER,
    ],
    "fever": [
        COUGH, URINE,
        Slot("ros_diarrhea", "ros", "설사", "설사를 하시나요?", item="Diarrhea"),
        Slot("ros_headache", "ros", "두통", "머리가 아프지는 않으세요?", item="Headache"),
        Slot("ros_rash", "ros", "발진", "피부에 발진이 생겼나요?", item="Rash"),
    ],
    "dizziness": [
        Slot("ros_vertigo", "ros", "회전성 어지럼", "주변이 빙빙 도는 느낌인가요?", item="Vertigo"),
        Slot("ros_hearing", "ros", "청력 변화", "귀가 잘 안 들리지는 않으세요?", item="Hearing loss"),
        WEAKNESS,
        Slot("ros_syncope", "ros", "실신", "정신을 잃은 적이 있나요?", item="Syncope"),
    ],
    "back_pain": [
        Slot("ros_leg_neuro", "ros", "하지 신경 증상", "다리가 저리거나 힘이 빠지나요?", item="Leg numbness/weakness",
             alt=(("다리가 저리세요?", "Leg numbness"), ("다리에 힘이 빠지세요?", "Leg weakness"))),
        Slot("ros_sphincter", "ros", "배뇨·배변 장애", "소변보기가 힘드세요?", item="Urinary retention"),
        FEVER, WEIGHT,
    ],
    "weight": [APPETITE, HEAT, PALP, BOWEL, THIRST, NIGHT_SWEAT, MOOD],
    "sleep": [
        Slot("ros_sleep_onset", "ros", "입면 장애", "잠들기가 힘드세요?", item="입면 장애"),
        Slot("ros_sleep_maint", "ros", "수면 유지 장애", "자다가 자주 깨세요?", item="수면 유지 장애"),
        MOOD,
        Slot("ros_snoring", "ros", "코골이", "코를 심하게 고세요?", item="Snoring"),
        Slot("ros_daytime", "ros", "주간 졸림", "낮에 많이 졸리세요?", item="Daytime sleepiness"),
        Slot("ros_caffeine", "shx", "카페인", "커피는 하루 몇 잔 드세요?", item="Caffeine"),
        Slot("ros_nocturia", "ros", "야간뇨", "밤에 소변 보러 깨세요?", item="Nocturia"),
    ],
    "growth": [
        APPETITE,
        Slot("ros_ped_ill", "ros", "잦은 질병", "아이가 자주 아픈 편인가요?", item="잦은 질병"),
        Slot("ros_puberty", "ros", "사춘기 징후", "몸에 털이 나기 시작했나요?", item="사춘기 징후"),
        Slot("ros_ped_bowel", "ros", "배변", "아이가 설사를 자주 하나요?", item="Diarrhea"),
        Slot("ros_ped_headache", "ros", "두통", "아이가 머리 아프다고 하나요?", item="Headache"),
    ],
    "fatigue": [SLEEP, WEIGHT, MOOD, COLD, MELENA, FEVER],
    "palpitation": [
        Slot("ros_palp_onset", "ros", "발작성 여부", "두근거림이 갑자기 시작되나요?", item="Paroxysmal"),
        DYSPNEA,
        Slot("ros_chest_pain_p", "ros", "흉통", "가슴이 아프지는 않으세요?", item="Chest pain"),
        Slot("ros_syncope_p", "ros", "실신", "정신을 잃은 적이 있나요?", item="Syncope"),
        HEAT,
    ],
    "general": [FEVER, WEIGHT, APPETITE],
}


def _ex(id_: str, label: str, ask: str, item: str, codes: tuple[str, ...] = ()) -> Slot:
    return Slot(id_, "exam", label, ask, codes, "EXAM", item=item)


LUNG = _ex("ex_lung", "폐 청진", "폐 청진을 하겠습니다.", "Chest", ("crackles", "wheeze", "unilateral_reduced_breath_sounds"))
HEART = _ex("ex_heart", "심음 청진", "심음 청진하겠습니다.", "Heart", ("s3_gallop", "pericardial_friction_rub"))
LEGS = _ex("ex_legs", "하지 말초 진찰", "하지 말초 진찰하겠습니다.", "Extremities",
           ("unilateral_leg_swelling", "bilateral_leg_edema"))
JVP = _ex("ex_jvp", "경정맥", "목 정맥 확장을 확인하겠습니다.", "Neck", ("jvd",))
ABD_PALP = _ex("ex_abd_palp", "복부 촉진", "복부 촉진을 하겠습니다.", "Abdomen")
ABD_REBOUND = _ex("ex_abd_rebound", "반발 압통", "복부 반발 압통을 확인하겠습니다.", "Abdomen")
ABD_BOWEL = _ex("ex_bowel_sounds", "장음", "장음 청진하겠습니다.", "Abdomen")
CVA = _ex("ex_cva", "늑골척추각 압통", "늑골척추각 압통을 확인하겠습니다.", "CVA")
THROAT = _ex("ex_throat", "인두", "인두를 보겠습니다.", "H/E/E/N/T")
SKIN = _ex("ex_skin", "피부", "피부 발진을 확인하겠습니다.", "Skin")
GENERAL = _ex("ex_general", "전반적 외관", "전반적 외관을 시진하겠습니다.", "General")
MENTAL = _ex("ex_mental", "의식·지남력", "지남력을 평가하겠습니다.", "General")
CONJ = _ex("ex_conj", "결막·공막", "결막 시진하겠습니다.", "H/E/E/N/T")
THYROID = _ex("ex_thyroid", "갑상선 촉진", "갑상선을 촉진하겠습니다.", "Neck")
NECK_LN = _ex("ex_neck_ln", "경부 림프절", "목 림프절을 촉진하겠습니다.", "Neck")
TREMOR = _ex("ex_tremor", "자세성 떨림", "자세성 떨림을 확인하겠습니다.", "Neuro")
DTR = _ex("ex_dtr", "심부건반사", "심부건반사를 확인하겠습니다.", "Neuro")
SKIN_TEMP = _ex("ex_skin_temp", "피부 온도·발한", "피부 온도·발한 촉진하겠습니다.", "Skin")
EYE_SIGNS = _ex("ex_eye_signs", "갑상선 안징후", "갑상선 안징후를 확인하겠습니다.", "H/E/E/N/T")
ANTHRO = _ex("ex_anthro", "신체 계측", "신체 계측을 하겠습니다.", "General")
GROWTH_CHART = _ex("ex_growth_chart", "성장도표", "성장도표 백분위수를 확인하겠습니다.", "General")
BODY_PROP = _ex("ex_body_prop", "신체 비율", "신체 비율을 평가하겠습니다.", "General")
TANNER = _ex("ex_tanner", "사춘기 성숙도", "사춘기 성숙도를 평가하겠습니다.", "GU")
DYSMORPH = _ex("ex_dysmorph", "안면 기형", "얼굴 기형 소견을 보겠습니다.", "H/E/E/N/T")
LIVER_SPLEEN = _ex("ex_liver_spleen", "간·비장 촉진", "간·비장 촉진하겠습니다.", "Abdomen")
ORTHO_BP = _ex("ex_ortho", "기립성 혈압", "기립성 혈압을 측정하겠습니다.", "V/S")
RADIAL = _ex("ex_radial", "요골동맥 맥박", "요골동맥 맥박을 촉진하겠습니다.", "V/S")

EXAMS: dict[str, list[Slot]] = {
    "chest_pain": [HEART, LUNG, LEGS],
    "dyspnea": [LUNG, HEART, LEGS, JVP],
    "cough": [LUNG, THROAT, HEART],
    "abdominal": [ABD_PALP, ABD_REBOUND, ABD_BOWEL, CVA],
    "headache": [_ex("ex_neck", "경부 강직", "목 경직을 확인하겠습니다.", "Neck"),
                 _ex("ex_pupil", "동공 반사", "동공 반사를 확인하겠습니다.", "Neuro"),
                 _ex("ex_motor", "사지 근력", "사지 근력 검사하겠습니다.", "Neuro")],
    "fever": [LUNG, THROAT, ABD_PALP, SKIN],
    "dizziness": [_ex("ex_nystagmus", "안구진탕", "안구진탕을 확인하겠습니다.", "Neuro"),
                  _ex("ex_finger_nose", "소뇌 협조운동", "소뇌 협조운동 검사하겠습니다.", "Neuro"),
                  _ex("ex_romberg", "롬베르그 검사", "롬베르그 검사를 하겠습니다.", "Neuro")],
    "back_pain": [_ex("ex_spine", "척추 압통", "척추 압통을 확인하겠습니다.", "Back"),
                  _ex("ex_slr", "하지 직거상 검사", "하지 직거상 검사를 하겠습니다.", "Neuro"),
                  _ex("ex_leg_motor", "사지 근력", "사지 근력 검사하겠습니다.", "Neuro")],
    "weight": [GENERAL, THYROID, NECK_LN, HEART, TREMOR, ABD_PALP],
    "sleep": [GENERAL, MENTAL, THYROID, HEART],
    "growth": [ANTHRO, GROWTH_CHART, BODY_PROP, THYROID, TANNER, DYSMORPH],
    "fatigue": [GENERAL, CONJ, THYROID, NECK_LN, HEART, LIVER_SPLEEN],
    "palpitation": [RADIAL, HEART, THYROID, TREMOR, LUNG],
    "general": [GENERAL, LUNG, HEART, ABD_PALP],
}

SECTION_ORDER = ("hpi", "pmh", "meds", "allergy", "fhx", "shx", "ob", "ros")
SECTION_KO = {"cc": "주호소", "hpi": "현병력", "pmh": "과거력", "meds": "약물", "allergy": "약물 알레르기", "fhx": "가족력",
              "shx": "사회력", "ob": "산과력", "ros": "계통 문진", "exam": "진찰 소견", "test": "검사 결과",
              "vitals": "활력징후", "education": "환자 설명"}
# Ledger section -> official note field
NOTE_FIELD = {"hpi": ("S", "hpi"), "pmh": ("S", "pmh"), "meds": ("S", "pmh"), "allergy": ("S", "pmh"),
              "ob": ("S", "ros"), "fhx": ("S", "fh"), "shx": ("S", "sh"), "ros": ("S", "ros"),
              "exam": ("O", "exam"), "test": ("O", "tests")}


@dataclass
class EncounterPlan:
    hpi: list[Slot] = field(default_factory=list)
    background: list[Slot] = field(default_factory=list)
    ros: list[Slot] = field(default_factory=list)
    exams: list[Slot] = field(default_factory=list)


def _dedupe(slots: list[Slot]) -> list[Slot]:
    seen, out = set(), []
    for s in slots:
        if s.id not in seen and s.ask not in seen:
            seen |= {s.id, s.ask}
            out.append(s)
    return out


def _tiered(slots: list[Slot], tier: int, first: int = 0) -> list[Slot]:
    """The same slots with `tier` from position `first` on (the earlier ones keep theirs)."""
    return [x if i < first else replace(x, tier=max(x.tier, tier)) for i, x in enumerate(slots)]


def build_plan(groups: list[str], opening: str, sex: Optional[str], age: Optional[int], pain: bool,
               onset_known: bool, exam_max: int, ask_alcohol: bool = True,
               ask_job: bool = False, ask_tbc_hep: bool = True, ask_prior_care: bool = True,
               pediatric: bool = False, playbook: Any = None, playbook_exam_max: int = 8,
               concern: bool = True) -> EncounterPlan:
    if playbook is not None:
        return _playbook_plan(playbook, sex, age, onset_known, ask_alcohol, ask_job, ask_tbc_hep, ask_prior_care,
                              pediatric, playbook_exam_max, concern)
    groups = groups or ["general"]
    hpi = [] if onset_known else [HPI_ONSET]
    if pain:
        hpi += HPI_PAIN
    else:  # "when is it worse" fits symptoms, not weight loss, growth or sleep (the topic plan asks the right thing)
        hpi += [s for s in HPI_GENERAL if not (s.id == "hpi_modifiers" and set(groups) & {"weight", "growth", "sleep"})]
    hpi.append(HPI_PRIOR)
    if ask_prior_care:
        hpi.append(HPI_CARE)
    if concern:
        hpi.append(CONCERN)
    bg = _background(sex, age, ask_alcohol, ask_job, ask_tbc_hep, pediatric)
    if pediatric and "growth" in groups:
        bg.insert(-1, PED_PARENT_HT)
    if not pediatric and sex == "female" and (age is None or 12 <= age <= 55) and \
            any(g in groups for g in ("weight", "fatigue", "sleep")):
        bg.append(OB_CYCLE)
    ros: list[Slot] = []
    exams: list[Slot] = []
    for g in groups:
        ros += ROS.get(g, ROS["general"])
        exams += EXAMS.get(g, EXAMS["general"])
    # a complaint group is coarser than a playbook: its review items are optional, its first 3 maneuvers standard
    return EncounterPlan(hpi=_dedupe(hpi), background=_dedupe(bg), ros=_tiered(_dedupe(ros), 2),
                         exams=_tiered(_dedupe(exams)[:exam_max], 2, first=3))


def _background(sex: Optional[str], age: Optional[int], ask_alcohol: bool, ask_job: bool, ask_tbc_hep: bool,
                pediatric: bool) -> list[Slot]:
    if pediatric:
        return list(PEDIATRIC)
    skip = {"alcohol": not ask_alcohol, "job": not ask_job, "pmh_tbc_hep": not ask_tbc_hep}
    bg = [s for s in BACKGROUND if not skip.get(s.id, False)]
    if sex == "female" and (age is None or 12 <= age <= 55):
        bg.append(OB_LMP)
    return bg


def _playbook_plan(pb: Any, sex: Optional[str], age: Optional[int], onset_known: bool, ask_alcohol: bool,
                   ask_job: bool, ask_tbc_hep: bool, ask_prior_care: bool, pediatric: bool,
                   exam_max: int, concern: bool = True) -> EncounterPlan:
    """The visit from a presentation playbook: its complaint history, prior episodes and care, the patient's concern;
    the standard background plus the playbook's complaint-specific background; its review items; its maneuvers.
    Every item carries its priority as the tier; priority-3 items are candidates the focus call may select."""
    kw = {"sex": sex, "pediatric": pediatric}
    hpi = [s for s in pb.section("hpi", **kw) if not (onset_known and s.id == f"pb_{pb.id}_onset")]
    own = {s.id[len(f"pb_{pb.id}_"):] for s in pb.history}
    hpi += [HPI_PRIOR] if "prior" not in own else []
    hpi += [HPI_CARE] if ask_prior_care and "care" not in own else []
    hpi += [CONCERN] if concern and "concern" not in own else []
    bg = _background(sex, age, ask_alcohol, ask_job, ask_tbc_hep, pediatric)
    own_bg = pb.section("pmh", "meds", "fhx", "shx", "ob", **kw)
    # a playbook item naming a standard chart item sets that item's priority (e.g. Tbc / hepatitis at p1 for cough)
    raise_to = {s.item: s.tier for s in own_bg}
    bg = [replace(s, tier=min(s.tier, raise_to[s.item])) if s.item in raise_to else s for s in bg]
    std_items = {s.item for s in bg}
    extra = [s for s in own_bg if s.item not in std_items]
    order = {k: i for i, k in enumerate(SECTION_ORDER)}
    bg = sorted(bg + extra, key=lambda s: order.get(s.section, 99))  # stable: standard items first in each section
    ros = pb.section("ros", **kw)
    return EncounterPlan(hpi=_dedupe(hpi), background=_dedupe(bg), ros=_dedupe(ros), exams=pb.exam_slots(sex, pediatric)[:exam_max])


def all_slots() -> list[Slot]:
    out = ([CC_ASK, HPI_ONSET, HPI_PRIOR, HPI_CARE, CONCERN, OB_LMP, OB_CYCLE, SMOKING_AMOUNT, SMOKING_YEARS,
            PED_PARENT_HT] + HPI_PAIN
           + HPI_GENERAL + BACKGROUND + PEDIATRIC)
    for v in ROS.values():
        out += v
    for v in EXAMS.values():
        out += v
    return _dedupe(out)


ITEM_BY_ASK: dict[str, str] = {s.ask: s.item for s in all_slots() if s.item}  # same text asked by the core planner
ALT_BY_ASK: dict[str, tuple[tuple[str, str], ...]] = {s.ask: s.alt for s in all_slots() if s.alt}
