"""Presentation playbooks: one reviewed YAML file per CPX chief complaint (dxagent/nova/playbooks/*.yaml).

Contest cases are drawn from a known list of outpatient presentations, and what is expected in the interview depends on
the presentation more than on the hidden disease: the characteristics of the complaint, the questions that separate
its usual causes, its red flags, the background and the maneuvers that examine it. A playbook holds exactly that, as
single-topic Korean questions of at most 30 characters and exam requests named as in the samples' exam tables.
The selected playbook is the visit's deterministic floor, independent of the model; one mid-visit focus call trims
its optional items and adds case-specific ones (agent.py).

File format (see chest_pain.yaml; `validate()` enforces it):

    id: chest_pain                       # = file name
    title: {ko: 흉통, en: Chest pain}
    status: draft                        # draft | reviewed (set by the clinical reviewer)
    reviewer: ""                         # who reviewed, and when
    population: any                      # any | adult | child | female
    pain: true                           # the complaint is pain (NRS asked, pain wording)
    match: {any: [가슴이 아프, 흉통], not: []}   # substrings of the opening statement
    history:                             # in asking order within each section and priority
      - {id: location, q: "어디가 아프세요?", item: Location, sec: hpi, p: 1}
      - {id: ocp, q: "피임약을 드시나요?", item: OCP, sec: meds, p: 3, only: female}
      - id: hypnotic                     # `then`: asked right after a positive answer (not a plain denial)
        q: "수면제를 드시나요?"
        item: Hypnotic use
        sec: meds
        p: 1
        then:
          - {id: hyp_dose, q: "하룻밤에 몇 알 드세요?", item: Hypnotic dose, known: "알|정|mg|밀리"}
      - {id: sexual, q: "성관계를 하신 적 있나요?", item: Sexual activity, sec: ob, p: 1, only: adult, gate: sexual}
    exams:
      - {name: 심음 청진, item: Heart, p: 1}           # request text: q, or derived from the name
      - {name: 질경 검사, item: GU, p: 1, only: [female, sexually_active]}
    differential:
      - {en: Acute coronary syndrome, ko: 급성 관상동맥증후군, cant_miss: true}
    tests:                               # the work-up; `for` = differential `en` names it settles, or "safety"
      - {name: "12-lead ECG", for: [Acute coronary syndrome], why: "ST 변화·부정맥 확인"}
      - "Chest X-ray"                    # a bare name is accepted (no link)
    plan:                                # the presentation's plan, for the note's fallback and the assessment
      treatment: ["..."]                 # first-line steps, conditional on the work-up ("<검사> 확인 후 ~")
      disposition: "..."                 # the usual decision and its reason
      follow_up: ["..."]                 # interval and what is re-checked
      red_flags: ["..."]                 # 2-4 named warning symptoms that need an immediate visit
    education: ["...", ...]              # closing explanations (<= 30 characters each)
    guidelines: [US_StatPearls/....md]   # guideline library paths, read on the server
    notes: "free text for the reviewer"

Sections: hpi (the complaint and its associated symptoms), ros (systemic review), pmh, meds, fhx, shx, ob.
Priority p (the slot's tier): 1 always asked; 2 asked unless the mid-visit focus call judges it uninformative for this
patient or the plan would run past its cap; 3 asked only when the focus call or the diagnostic time-out selects it.
`only: female|male|adult|child|sexually_active` (or a list of them, all required) limits an item or a maneuver.
`sexually_active` (pelvic and speculum exams) drops the maneuver for a child, for a patient who denied sexual activity
when asked the item marked `gate: sexual`, and for an adolescent (13-17) unless that item was answered yes. A history
item's `then` lists up to 3 follow-up questions asked right after a positive answer (quantity, duration, escalation),
each skipped when its `known` pattern already appears in the answer. At most 8 maneuvers (config.playbook_exam_max);
2 to 4 closing lines. The item `id: onset` is skipped when the opening already states the onset. The standard
background (DM/HTN, Tbc/hepatitis, operations, medicines, allergy, family history, smoking, alcohol, LMP) is added by
checklist.build_plan; a playbook item with the same chart item as a standard one sets that item's priority instead of
being asked twice (e.g. `{item: Tbc / hepatitis, p: 1}` asks the standard TB question in a cough visit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from dxagent.nova.checklist import Slot
from dxagent.nova.protocol import char_len, check_exam, check_say, single_topic

ROOT = Path(__file__).resolve().parent / "playbooks"
SECTIONS = ("hpi", "ros", "pmh", "meds", "fhx", "shx", "ob")
POPULATIONS = ("any", "adult", "child", "female")
ONLY = ("female", "male", "adult", "child", "sexually_active")
SEXUAL = "sexually_active"
MAX_THEN = 3
TEST_FOR_SAFETY = "safety"
MAX_EXAMS = 8  # config.playbook_exam_max

# Exam names. SAMPLE_EXAMS is the standard maneuver vocabulary of CPX exam tables (the names the examiner maps
# requests onto). EXTRA_EXAMS are standard maneuvers a presentation needs beyond it. A real maneuver missing from a
# case's table is answered with a finding written for it, and a request that is not a physical examination is refused
# at a turn's cost; every name here is a bedside maneuver, and the standard names are preferred.
SAMPLE_EXAMS = {
    "전반적 외관 시진": "General appearance", "의식·지남력 평가": "Mental status", "활력징후 측정": "Vital signs",
    "기립성 혈압 측정": "Orthostatic BP", "요골동맥 맥박 촉진": "Radial pulse palpation",
    "피부 온도·발한 촉진": "Skin temperature and diaphoresis", "결막·공막 시진": "Conjunctiva and sclera inspection",
    "외안근 운동": "Extraocular movement", "안구진탕 확인": "Nystagmus", "갑상선 안징후 검사": "Thyroid eye signs",
    "안면 기형 소견 시진": "Dysmorphic features", "경부 림프절 촉진": "Cervical lymph node palpation",
    "갑상선 촉진": "Thyroid palpation", "갑상선 잡음 청진": "Thyroid bruit", "흉곽 시진": "Chest inspection",
    "폐 청진": "Lung auscultation", "심음 청진": "Heart auscultation", "복부 시진": "Abdominal inspection",
    "장음 청진": "Bowel sound auscultation", "복부 촉진": "Abdominal palpation",
    "간·비장 촉진": "Liver and spleen palpation", "사지 근력 검사": "Limb muscle power",
    "심부건반사": "Deep tendon reflex", "자세성 떨림": "Postural tremor",
    "떨림·서동 관찰": "Resting tremor and bradykinesia", "소뇌 협조운동 검사": "Cerebellar coordination",
    "구음장애 확인": "Dysarthria", "보행 관찰": "Gait", "롬베르그 검사": "Romberg test",
    "인지기능 선별검사": "Cognitive screening", "하지 말초 진찰": "Peripheral pulse and edema",
    "신체 계측": "Anthropometry", "성장도표 백분위수 확인": "Growth chart percentile",
    "신체 비율 평가": "Body proportion", "사춘기 성숙도 평가": "Tanner stage and testicular volume",
    "성장 속도·중간부모키 산출": "Growth velocity and midparental height", "경골앞 점액부종": "Pretibial myxedema",
    "자주색 선조 시진": "Purple striae", "병식 평가": "Insight", "주간 진정·불안 관찰": "Daytime sedation and anxiety",
}
EXTRA_EXAMS = {
    "경정맥압 측정": "Jugular venous pressure", "간경정맥 역류 검사": "Hepatojugular reflux",
    "경동맥 잡음 청진": "Carotid bruit", "흉벽 압통 촉진": "Chest wall tenderness", "흉부 타진": "Chest percussion",
    "촉각진탕": "Tactile fremitus", "양팔 혈압 비교": "Blood pressure in both arms",
    "곤봉지 확인": "Digital clubbing", "인두 시진": "Pharynx inspection", "이경 검사": "Otoscopy",
    "비강 시진": "Nasal inspection", "부비동 압통 확인": "Sinus tenderness", "구강 점막 시진": "Oral mucosa inspection",
    "반발압통": "Rebound tenderness", "머피 징후": "Murphy sign", "로브싱 징후": "Rovsing sign",
    "요근 징후": "Psoas sign", "폐쇄근 징후": "Obturator sign", "복수 이동탁음 확인": "Shifting dullness (ascites)",
    "늑골척추각 압통": "Costovertebral angle tenderness", "서혜부 탈장 확인": "Inguinal hernia examination",
    "직장수지검사": "Digital rectal examination", "방광 촉진": "Bladder palpation",
    "고환 촉진": "Testicular palpation", "양손 골반 내진": "Bimanual pelvic examination", "질경 검사": "Speculum examination",
    "유방 촉진": "Breast palpation", "겨드랑이 림프절 촉진": "Axillary lymph node palpation",
    "척추 압통 촉진": "Spinal tenderness", "요추 운동 범위": "Lumbar range of motion",
    "하지직거상 검사": "Straight leg raise", "촉각 검사": "Light touch sensation", "회음부 감각 검사": "Perineal (saddle) sensation",
    "안면 근력 검사": "Facial muscle strength (CN VII)", "혀 운동 검사": "Tongue movement (CN XII)", "경부 강직": "Neck stiffness", "동공 반사": "Pupillary reflex",
    "안저 검사": "Fundoscopy", "측두동맥 촉진": "Temporal artery palpation",
    "딕스-홀파이크 검사": "Dix-Hallpike test", "두부 충동 검사": "Head impulse test",
    "속삭임 청력 검사": "Whispered voice test", "소리 반응 관찰": "Response to sound (child)", "손 관절 촉진": "Hand joint palpation",
    "손목 관절 촉진": "Wrist joint palpation", "무릎 관절 촉진": "Knee joint palpation",
    "어깨 관절 운동 범위 검사": "Shoulder range of motion", "발가락 관절 촉진": "Toe joint palpation",
    "피부 시진": "Skin inspection", "정동 평가": "Affect", "안검 부종 시진": "Eyelid edema",
    "다모증 평가": "Hirsutism", "유즙 분비 확인": "Galactorrhea", "흑색가시세포증 시진": "Acanthosis nigricans",
    "단섬유 감각 검사": "Monofilament test", "모세혈관 재충전 시간": "Capillary refill time",
    "피부 긴장도 확인": "Skin turgor", "대천문 촉진": "Anterior fontanelle", "근긴장도 평가": "Muscle tone",
    "모로 반사 확인": "Moro reflex", "발달 선별검사": "Developmental screening",
    "바빈스키 징후": "Babinski sign", "회내근 표류 검사": "Pronator drift", "케르니그 징후": "Kernig sign",
    "전립선 촉진": "Prostate palpation", "복부 잡음 청진": "Abdominal bruit", "복부 타진": "Abdominal percussion",
    "퍼덕떨림 확인": "Asterixis", "유양돌기 압통 확인": "Mastoid tenderness", "이주 압통 확인": "Tragal tenderness",
    "턱관절 촉진": "Temporomandibular joint palpation", "니콜스키 징후": "Nikolsky sign",
    "피부묘기증 확인": "Dermographism", "유리압진 검사": "Diascopy (blanching)", "가워스 징후": "Gowers sign",
    "팔렌 검사": "Phalen test", "티넬 징후": "Tinel sign", "스펄링 검사": "Spurling test",
    "고관절 운동 범위 검사": "Hip range of motion", "시력 검사": "Visual acuity", "침샘 촉진": "Salivary gland palpation",
    "앙와위 회전 검사": "Supine roll test", "후방 당김 검사": "Pull test (postural instability)",
    "항문 괄약근 긴장도 확인": "Anal sphincter tone", "대면 시야 검사": "Visual fields by confrontation",
    "웨버 검사": "Weber test", "린네 검사": "Rinne test",
    # added with the evidence review (v0.8.0): single discriminating maneuvers the audits asked for
    "맥버니점 압통": "McBurney point tenderness", "담낭 촉진": "Gallbladder palpation", "항문 시진": "Anal inspection",
    "기침 스트레스 검사": "Cough stress test", "발목 관절 촉진": "Ankle joint palpation", "주의력 검사": "Attention test",
    "외음부 시진": "Vulvar inspection", "유방 시진": "Breast inspection",
    "쇄골상 림프절 촉진": "Supraclavicular lymph node palpation", "서혜부 림프절 촉진": "Inguinal lymph node palpation",
    "자궁경부 운동 압통 확인": "Cervical motion tenderness", "가피 확인": "Eschar search",
    "하지 함요부종 확인": "Pitting edema of the legs", "아킬레스건 반사": "Achilles reflex",
    "허리둘레 측정": "Waist circumference", "종아리 둘레 측정": "Calf circumference", "체중 측정": "Body weight",
    "낙하산 반사 확인": "Parachute reflex", "혀 시진": "Tongue inspection", "구역 반사": "Gag reflex",
    "경부 운동 범위 검사": "Neck range of motion", "점상출혈 확인": "Petechiae and purpura",
    "황색종 시진": "Xanthoma inspection", "각막환 확인": "Corneal arcus", "손가락-코 검사": "Finger-to-nose test",
}
EXAM_NAMES = {**SAMPLE_EXAMS, **EXTRA_EXAMS}
_ACTION_TAIL = re.compile(r"(시진|촉진|청진|타진|검사|평가|측정|확인|관찰|진찰|산출|계측)$")


_LABELS: set[str] = set()


def specific_exam(text: str) -> bool:
    """A model-written maneuver the examiner will take as ONE maneuver: a name from the vocabulary, or a free request
    without "·" and without a whole-region "진찰"/"검진" (the examiner refuses those as too broad, at a turn's cost)."""
    from dxagent.nova.korean import exam_label

    if not _LABELS:
        _LABELS.update(exam_label(exam_request(n)) for n in EXAM_NAMES)
    label = exam_label(text)
    if label in _LABELS:
        return True
    if re.fullmatch(r"(손|팔|다리|몸|전신)?\s*(떨림|반사|감각|근력|신경)", label):
        return False  # a symptom or a system without its method ("손 떨림 확인": postural or resting?)
    return "·" not in label and not re.search(r"(진찰|검진|신경학적\s*검사)$", label)


def exam_request(name: str) -> str:
    """'심음 청진' -> '심음 청진하겠습니다.'; '심부건반사' -> '심부건반사 확인하겠습니다.'"""
    return f"{name}하겠습니다." if _ACTION_TAIL.search(name) else f"{name} 확인하겠습니다."


@dataclass(frozen=True)
class Playbook:
    id: str
    title_ko: str
    title_en: str
    status: str
    population: str
    pain: bool
    match_any: tuple[str, ...]
    match_not: tuple[str, ...]
    history: tuple[Slot, ...]
    priority: dict = field(default_factory=dict, hash=False, compare=False)  # slot id -> 1 | 2 | 3
    sex_only: dict = field(default_factory=dict, hash=False, compare=False)  # slot id -> tuple of ONLY values
    exams: tuple[Slot, ...] = ()
    differential: tuple[dict, ...] = ()
    tests: tuple[str, ...] = ()
    education: tuple[str, ...] = ()
    guidelines: tuple[str, ...] = ()
    # parent slot id -> ((follow-up slot, `known` pattern, safety), ...): asked right after a positive answer
    followups: dict = field(default_factory=dict, hash=False, compare=False)
    gate_id: str = ""  # slot id of the sexual-activity item (`gate: sexual`), "" if none
    test_links: dict = field(default_factory=dict, hash=False, compare=False)  # test name -> {"for": (...), "why": str}
    plan: dict = field(default_factory=dict, hash=False, compare=False)  # treatment, disposition, follow_up, red_flags

    def _keep(self, slot_id: str, sex: Optional[str], pediatric: Optional[bool]) -> bool:
        """An item limited by `only` is dropped when the patient is known not to fit (female/male, adult/child; a
        `sexually_active` maneuver is never planned for a child, and is checked again at the visit: agent.py)."""
        for only in self.sex_only.get(slot_id, ()):
            if only in ("female", "male") and sex and sex != only:
                return False
            if only in ("adult", "child") and pediatric is not None and (only == "child") != pediatric:
                return False
            if only == SEXUAL and pediatric:
                return False
        return True

    def followups_for(self, slot: Slot) -> tuple:
        """The follow-ups of a planned slot: the playbook item's own, or, for a standard item a playbook item raised
        (same chart item, e.g. Alcohol), that playbook item's."""
        own = self.followups.get(slot.id)
        if own or slot.id.startswith(f"pb_{self.id}_"):
            return own or ()
        for h in self.history:
            if h.item and h.item == slot.item and h.id in self.followups:
                return self.followups[h.id]
        return ()

    def needs_sexual(self, slot_id: str) -> bool:
        return SEXUAL in self.sex_only.get(slot_id, ())

    def tests_for(self, names: list[str]) -> list[str]:
        """The work-up linked to these diagnoses (English names, case-insensitive), then the safety tests."""
        want = {n.strip().lower() for n in names if n}
        hit = [t for t in self.tests if want & {x.lower() for x in self.test_links.get(t, {}).get("for", ())}]
        safety = [t for t in self.tests if TEST_FOR_SAFETY in self.test_links.get(t, {}).get("for", ()) and t not in hit]
        return hit + safety

    def test_line(self, name: str) -> str:
        """'<test> : <purpose>' for the note's plan ("" purpose: the name alone)."""
        why = self.test_links.get(name, {}).get("why", "")
        return f"{name} : {why}" if why else name

    def section(self, *names: str, max_p: int = 3, sex: Optional[str] = None,
                pediatric: Optional[bool] = None) -> list[Slot]:
        """History slots of these sections, priority 1 before 2 before 3 (file order within a priority), without the
        items limited to another sex or age group."""
        out = [s for s in self.history if s.section in names and self.priority.get(s.id, 2) <= max_p
               and self._keep(s.id, sex, pediatric)]
        return sorted(out, key=lambda s: self.priority.get(s.id, 2))

    def exam_slots(self, sex: Optional[str] = None, pediatric: Optional[bool] = None) -> list[Slot]:
        """Maneuvers in priority order, without those limited to another sex or age group."""
        return [e for e in self.exams if self._keep(e.id, sex, pediatric)]

    def differential_en(self, n: int = 6) -> list[str]:
        return [d["en"] for d in self.differential[:n]]

    def cant_miss(self) -> list[str]:
        return [d["en"] for d in self.differential if d.get("cant_miss")]


def _slot(pid: str, h: dict) -> Slot:
    return Slot(f"pb_{pid}_{h['id']}", h.get("sec", "hpi"), h.get("label") or h.get("item") or h["id"], h["q"].strip(),
                (), "SAY", item=h.get("item", ""), tier=int(h.get("p", 2)))


def _then(pid: str, h: dict) -> tuple:
    """A history item's follow-up questions: ((slot, known pattern, safety), ...), in the parent's section."""
    return tuple((Slot(f"pb_{pid}_{h['id']}_{c['id']}", h.get("sec", "hpi"), c.get("item") or c["id"], c["q"].strip(),
                       (), "SAY", item=c.get("item", ""), tier=1), str(c.get("known") or ""), bool(c.get("safety")))
                 for c in h.get("then") or [])


def _only(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _test_name(t: Any) -> str:
    return t if isinstance(t, str) else str((t or {}).get("name", ""))


def _exam(pid: str, e: dict) -> Slot:
    name = e["name"]
    return Slot(f"pb_{pid}_ex_{EXAM_NAMES.get(name, name)}", "exam", name, (e.get("q") or exam_request(name)).strip(), (),
                "EXAM", item=e.get("item") or EXAM_NAMES.get(name, name), tier=int(e.get("p", 2)))


def parse(data: dict[str, Any]) -> Playbook:
    pid = data["id"]
    hist = data.get("history") or []
    return Playbook(
        id=pid, title_ko=data["title"]["ko"], title_en=data["title"]["en"], status=data.get("status", "draft"),
        population=data.get("population", "any"), pain=bool(data.get("pain", False)),
        match_any=tuple((data.get("match") or {}).get("any") or ()), match_not=tuple((data.get("match") or {}).get("not") or ()),
        history=tuple(_slot(pid, h) for h in hist), priority={f"pb_{pid}_{h['id']}": int(h.get("p", 2)) for h in hist},
        sex_only={**{f"pb_{pid}_{h['id']}": _only(h["only"]) for h in hist if h.get("only")},
                  **{_exam(pid, e).id: _only(e["only"]) for e in data.get("exams") or [] if e.get("only")}},
        exams=tuple(_exam(pid, e) for e in sorted(data.get("exams") or [], key=lambda e: int(e.get("p", 2)))),
        differential=tuple(data.get("differential") or ()),
        tests=tuple(_test_name(t) for t in data.get("tests") or () if _test_name(t)),
        education=tuple(data.get("education") or ()), guidelines=tuple(data.get("guidelines") or ()),
        followups={f"pb_{pid}_{h['id']}": _then(pid, h) for h in hist if h.get("then")},
        gate_id=next((f"pb_{pid}_{h['id']}" for h in hist if h.get("gate") == "sexual"), ""),
        test_links={t["name"]: {"for": tuple(t.get("for") or ()), "why": str(t.get("why") or "").strip()}
                    for t in data.get("tests") or () if isinstance(t, dict) and t.get("name")},
        plan=dict(data.get("plan") or {}))


@lru_cache(maxsize=4)
def load_all(root: str = str(ROOT)) -> dict[str, Playbook]:
    out: dict[str, Playbook] = {}
    for f in sorted(Path(root).glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("id"):
            out[data["id"]] = parse(data)
    return out


def _fits(pb: Playbook, pediatric: bool, sex: Optional[str]) -> bool:
    if pb.population == "child":
        return pediatric
    if pb.population == "adult":
        return not pediatric
    if pb.population == "female":
        return sex != "male"
    return True


_ADVERBS = re.compile(r"(?:(?<=\s)|^)(자주|계속|많이|너무|좀|조금|갑자기|자꾸|심하게|점점|요즘|가끔|매우|아주|살짝|약간|자꾸만|계속해서)\s+")
# Durations placed between subject and verb ("생리를 3개월째 안 해요", "입맛이 두 달째 없어요", "일주일 넘게").
_DURATIONS = re.compile(r"(?:(?<=\s)|^)(?:(?:\d+|한|두|세|석|네|넉|다섯|여섯|일곱|여덟|아홉|열|한두|두세|서너|몇|여러)\s*"
                        r"(?:개월|달|주|일|년|시간)|(?:이|삼|사|오|육)\s*(?:개월|주)|일주일|며칠|이틀|사흘|나흘|보름|반년)"
                        r"\s*(?:째|간|동안|넘게|이상|정도|쯤|가량|전부터)?\s+")


_BODY = ("목", "배", "눈", "귀", "코", "손", "발", "등", "입", "혀", "침")  # one-syllable nouns that end compounds (발목, 아침)


def _find(text: str, word: str) -> int:
    """First position of `word` in `text`; a word that starts with a one-syllable noun from _BODY must start a word
    itself ("목이 붓고" is not found inside "발목이 붓고", nor "침에 피" inside "아침에 피")."""
    i = text.find(word)
    while i > 0 and word[:1] in _BODY and "가" <= text[i - 1] <= "힣":
        i = text.find(word, i + 1)
    return i


def select(opening: str, pediatric: bool = False, sex: Optional[str] = None,
           books: Optional[dict[str, Playbook]] = None) -> Optional[Playbook]:
    """The playbook whose match words appear earliest in the opening statement (the reason for the visit is usually
    stated first). Among words found at the same position the longest wins; a population-specific playbook (child,
    female) beats a general one on a tie. Each word is tried on the opening as written and with common adverbs
    and durations removed, so "머리가 자주 아파서" matches "머리가 아파" and "생리를 3개월째 안 해요" matches "생리를 안 해", while
    "소변을 자주" still tells frequency from volume."""
    books = load_all() if books is None else books
    raw = (opening or "").lower()
    texts = (raw, _DURATIONS.sub("", _ADVERBS.sub("", raw)))

    def pos(word: str) -> int:
        hits = [i for i in (_find(t, word.lower()) for t in texts) if i >= 0]
        return min(hits) if hits else -1

    best: Optional[tuple[int, int, int, str]] = None
    for pb in books.values():
        if not _fits(pb, pediatric, sex) or any(pos(w) >= 0 for w in pb.match_not):
            continue
        for w in pb.match_any:
            i = pos(w)
            if i >= 0:
                key = (i, -len(w), 0 if pb.population != "any" else 1, pb.id)
                if best is None or key < best:
                    best = key
    return books[best[3]] if best else None


def menu(pediatric: bool = False, sex: Optional[str] = None) -> list[tuple[str, str]]:
    """(id, Korean title) of the playbooks that fit the patient: the list the topic plan may choose from."""
    return [(pb.id, pb.title_ko) for pb in load_all().values() if _fits(pb, pediatric, sex)]


# -- validation ---------------------------------------------------------------------------------------------------

def validate(data: dict[str, Any], name: str = "") -> list[str]:
    """Problems with one playbook file (empty when it is valid)."""
    errs: list[str] = []
    where = name or str(data.get("id"))

    def err(msg: str) -> None:
        errs.append(f"{where}: {msg}")

    for k in ("id", "title", "population", "match", "history", "exams", "differential", "education"):
        if k not in data:
            err(f"missing {k}")
    if errs:
        return errs
    if name and Path(name).stem != data["id"]:
        err(f"id {data['id']!r} differs from the file name")
    if data["population"] not in POPULATIONS:
        err(f"population {data['population']!r}")
    if data.get("status", "draft") not in ("draft", "reviewed"):
        err(f"status {data.get('status')!r}")
    if not (data["match"] or {}).get("any"):
        err("match.any is empty")
    def question(hid: str, q: str) -> None:
        ok, why = check_say(q)
        if not ok:
            err(f"{hid}: {why}: {q}")
        if not q.endswith("?"):
            err(f"{hid}: a question ends with '?': {q}")
        if not single_topic(q):
            err(f"{hid}: more than one topic: {q}")

    def no_empty(kind: str, entry: Any) -> None:
        """An unquoted comma inside a YAML flow mapping splits a value into empty keys ({en: A, B} -> en: A, B: null)."""
        if isinstance(entry, dict):
            empty = [k for k, v in entry.items() if v is None]
            if empty:
                err(f"{kind} {entry.get('id') or entry.get('name') or entry.get('en')!r}: empty keys {empty} "
                    f"(quote a value that contains a comma or a colon)")

    for key in ("history", "exams", "differential", "tests"):
        for entry in data.get(key) or []:
            no_empty(key, entry)
            for c in (entry.get("then") or []) if isinstance(entry, dict) else []:
                no_empty("follow-up", c)
    for d in data.get("differential") or []:
        if isinstance(d, dict) and set(d) - {"en", "ko", "cant_miss"}:
            err(f"differential {d.get('en')!r}: keys en, ko, cant_miss only, not {sorted(set(d) - {'en', 'ko', 'cant_miss'})}")
    ids = set()
    gates = 0
    for h in data["history"]:
        hid = h.get("id")
        if not hid or hid in ids:
            err(f"history id {hid!r} missing or repeated")
        ids.add(hid)
        question(hid, (h.get("q") or "").strip())
        if h.get("sec", "hpi") not in SECTIONS:
            err(f"{hid}: section {h.get('sec')!r}")
        if int(h.get("p", 2)) not in (1, 2, 3):
            err(f"{hid}: priority {h.get('p')!r}")
        if not h.get("item"):
            err(f"{hid}: no chart item")
        if any(o not in ONLY for o in _only(h.get("only"))):
            err(f"{hid}: only {h.get('only')!r}")
        if h.get("gate") not in (None, "sexual"):
            err(f"{hid}: gate {h.get('gate')!r} (only 'sexual')")
        gates += h.get("gate") == "sexual"
        then = h.get("then") or []
        if not isinstance(then, list) or len(then) > MAX_THEN:
            err(f"{hid}: then is a list of at most {MAX_THEN} follow-up questions")
            then = []
        for c in then:
            cid = f"{hid}.{c.get('id')}"
            if not c.get("id") or cid in ids:
                err(f"follow-up id {cid!r} missing or repeated")
            ids.add(cid)
            question(cid, (c.get("q") or "").strip())
            if not c.get("item"):
                err(f"{cid}: no chart item")
            if c.get("safety") not in (None, True, False):
                err(f"{cid}: safety is true or false")
            if c.get("known"):
                try:
                    re.compile(str(c["known"]))
                except re.error as exc:
                    err(f"{cid}: known is not a pattern: {exc}")
    if gates > 1:
        err("more than one item marked gate: sexual")
    names = set()
    for e in data["exams"]:
        n = e.get("name", "")
        if n not in EXAM_NAMES:
            err(f"exam {n!r} is not a known maneuver name")
        if n in names:
            err(f"exam {n!r} repeated")
        if any(o not in ONLY for o in _only(e.get("only"))):
            err(f"exam {n!r}: only {e.get('only')!r}")
        names.add(n)
        q = (e.get("q") or exam_request(n)).strip()
        ok, why = check_exam(q)
        if not ok or char_len(q) > 30:
            err(f"exam {n!r}: {why or 'over 30 characters'}: {q}")
    for key in ("education", "guidelines"):
        if not all(isinstance(x, str) for x in data.get(key) or []):
            err(f"{key}: every entry must be text (quote entries containing ': ')")
    dx_names = {str(d.get("en", "")).lower() for d in data["differential"]}
    for t in data.get("tests") or []:
        if isinstance(t, str):
            continue
        if not isinstance(t, dict) or not isinstance(t.get("name"), str) or not t["name"].strip():
            err(f"tests: an entry is a name or {{name, for, why}}: {t!r}")
            continue
        for f in t.get("for") or []:
            if str(f).lower() not in dx_names and f != TEST_FOR_SAFETY:
                err(f"test {t['name']!r}: for {f!r} is not a differential name (en) or {TEST_FOR_SAFETY!r}")
        if t.get("why") is not None and not isinstance(t.get("why"), str):
            err(f"test {t['name']!r}: why must be text")
    plan = data.get("plan")
    if plan is not None:
        if not isinstance(plan, dict) or set(plan) - {"treatment", "disposition", "follow_up", "red_flags"}:
            err("plan: keys treatment, disposition, follow_up, red_flags")
        else:
            for key in ("treatment", "follow_up", "red_flags"):
                v = plan.get(key, [])
                if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
                    err(f"plan.{key}: a list of text lines")
            if not isinstance(plan.get("disposition", ""), str):
                err("plan.disposition: text")
            if not 2 <= len(plan.get("red_flags") or []) <= 4:
                err("plan.red_flags: 2 to 4 warning symptoms")
    if len(data["exams"]) > MAX_EXAMS:
        err(f"{len(data['exams'])} exams: only the first {MAX_EXAMS} are ever requested")
    if not 2 <= len(data["education"]) <= 4:
        err("2 to 4 closing lines")
    for d in data["differential"]:
        if not d.get("en") or not d.get("ko"):
            err(f"differential entry without en/ko: {d}")
    if len(data["differential"]) < 3:
        err("fewer than 3 differential diagnoses")
    for line in data["education"]:
        ok, why = check_say(line)
        if not ok:
            err(f"education: {why}: {line}")
    return errs


def validate_all(root: Path = ROOT) -> list[str]:
    errs: list[str] = []
    seen: dict[str, str] = {}
    for f in sorted(root.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as exc:
            errs.append(f"{f.name}: YAML error: {exc}")
            continue
        errs += validate(data, f.name)
        seen[data.get("id", f.stem)] = f.name
    return errs


def sheet(root: Path = ROOT) -> str:
    """All playbooks as one Markdown review sheet (docs/PLAYBOOKS.md), in file order."""
    sec_ko = {"hpi": "현병력", "ros": "계통 문진", "pmh": "과거력", "meds": "약물", "fhx": "가족력", "shx": "사회력",
              "ob": "산과력"}
    out = ["# Presentation playbooks: review sheet", "",
           "Generated by `python -m dxagent.nova.playbooks --sheet docs/PLAYBOOKS.md` from "
           "`dxagent/nova/playbooks/*.yaml`. Edit the YAML files, not this sheet. A playbook becomes `reviewed` when the "
           "clinical reviewer sets `status: reviewed` and `reviewer` in its file.", "",
           "Priority: 1 always asked, 2 unless the focus call skips it or the plan would run past the soft cap (46 of the 50 "
           "turns), 3 only if the "
           "focus call adds it. ★ = can't-miss diagnosis. "
           "Exam names in italics are not in the standard vocabulary (for a maneuver a case's table lacks, the "
           "examiner writes a finding consistent with the case, usually normal).", ""]
    files = sorted(root.glob("*.yaml"))
    out += ["| Playbook | Population | Status | Items | Exams |", "|---|---|---|---|---|"]
    datas = [yaml.safe_load(f.read_text(encoding="utf-8")) for f in files]
    for d in datas:
        out.append(f"| [{d['title']['ko']} ({d['id']})](#{d['id']}) | {d['population']} | {d.get('status', 'draft')} | "
                   f"{len(d['history'])} | {len(d['exams'])} |")
    for d in datas:
        out += ["", f"## {d['id']}", "", f"**{d['title']['ko']}** ({d['title']['en']}) · population `{d['population']}` · "
                f"status `{d.get('status', 'draft')}`" + (f" · reviewer {d['reviewer']}" if d.get("reviewer") else ""), "",
                "Opening words: " + ", ".join(f"`{w}`" for w in d["match"].get("any", []))
                + (" · not: " + ", ".join(f"`{w}`" for w in d["match"].get("not", [])) if d["match"].get("not") else ""),
                "", "| p | Section | Question | Chart item |", "|---|---|---|---|"]
        for h in sorted(d["history"], key=lambda h: (int(h.get("p", 2)), SECTIONS.index(h.get("sec", "hpi")))):
            only = f" ({'+'.join(_only(h['only']))} only)" if h.get("only") else ""
            gate = " (gate)" if h.get("gate") else ""
            out.append(f"| {h.get('p', 2)} | {sec_ko.get(h.get('sec', 'hpi'))} | {h['q']}{only}{gate} | {h['item']} |")
            for c in h.get("then") or []:
                out.append(f"| ↳ | if yes | {c['q']} | {c['item']} |")
        out += ["", "Exams: " + ", ".join(
            (f"{e['name']}" if e["name"] in SAMPLE_EXAMS else f"*{e['name']}*") + f" (p{e.get('p', 2)})"
            + (f" ({'+'.join(_only(e['only']))} only)" if e.get("only") else "")
            for e in sorted(d["exams"], key=lambda e: int(e.get("p", 2))))]
        out += ["", "Differential: " + "; ".join(("★ " if x.get("cant_miss") else "") + f"{x['ko']} ({x['en']})"
                                                for x in d["differential"])]
        tests = []
        for t in d.get("tests") or []:
            if isinstance(t, str):
                tests.append(t)
            else:
                tests.append(t["name"] + (f" ({t['why']})" if t.get("why") else "")
                             + (f" → {', '.join(t.get('for') or [])}" if t.get("for") else ""))
        out += ["", "Tests: " + "; ".join(tests)]
        plan = d.get("plan") or {}
        if plan:
            out += ["", "Plan: " + " · ".join(
                f"{k}: " + ("; ".join(plan[k]) if isinstance(plan[k], list) else str(plan[k]))
                for k in ("treatment", "disposition", "follow_up", "red_flags") if plan.get(k))]
        out += ["", "Closing lines: " + " / ".join(f"\"{x}\"" for x in d["education"])]
        out += ["", "References: " + ", ".join(f"`{g}`" for g in d.get("guidelines") or [])]
        if d.get("notes"):
            out += ["", "Notes: " + " ".join(str(d["notes"]).split())]
    return "\n".join(out) + "\n"


def budget(pb: Playbook, sex: Optional[str], age: Optional[int]) -> tuple[int, int, int]:
    """Planned gathering turns for one patient: (priority 1 only, priorities 1-2, follow-ups available). The standard
    background, prior episodes, care and the concern are included; closing explanations and the diagnostic time-out
    are not (config: closing_lines, review_*)."""
    from dxagent.nova import checklist as CL

    pediatric = age is not None and age < 13
    plan = CL.build_plan([], "", sex, age, pb.pain, False, 8, pediatric=pediatric, playbook=pb)
    slots = plan.hpi + plan.background + plan.ros + plan.exams
    p1 = sum(1 for x in slots if x.tier <= 1)
    p2 = sum(1 for x in slots if x.tier <= 2)
    then = sum(len(pb.followups.get(x.id, ())) for x in slots if x.tier <= 2)
    return p1, p2, then


def budget_table() -> str:
    rows = ["playbook  patient  p1  p1+p2  follow-ups"]
    for pb in load_all().values():
        patients = {"child": [("male", 6)], "adult": [("male", 45), ("female", 45)], "female": [("female", 30)]}.get(
            pb.population, [("male", 45), ("female", 45), ("female", 6)])
        for sex, age in patients:
            p1, p2, then = budget(pb, sex, age)
            rows.append(f"{pb.id}  {sex[0]}{age}  {p1}  {p2}  {then}")
    return "\n".join(rows)


if __name__ == "__main__":  # python -m dxagent.nova.playbooks [--sheet PATH] [--budget]: validate (sheet, turn table)
    import sys

    if "--budget" in sys.argv:
        print(budget_table())
    problems = validate_all()
    print("\n".join(problems) or f"{len(load_all())} playbooks valid")
    if "--sheet" in sys.argv and not problems:
        target = Path(sys.argv[sys.argv.index("--sheet") + 1])
        target.write_text(sheet(), encoding="utf-8")
        print(f"wrote {target}")
    sys.exit(1 if problems else 0)
