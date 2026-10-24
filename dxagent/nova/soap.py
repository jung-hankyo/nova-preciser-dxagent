"""The first-visit note in the official format (development guide, "초진 의무기록 작성 안내").

    note = {"S": {"cc", "hpi", "pmh", "fh", "sh", "ros"},
            "O": {"exam", "tests"},
            "A": {"evidence", "differential"},
            "P": {"tests", "treatment", "disposition", "follow_up", "education"}}

(nova_sdk 1.0 RECORD_FIELDS; the primary diagnosis is the separate `diagnosis` argument of client.diagnose and opens
A.evidence as "#1. <diagnosis>".) Every field is a string ("" when there is nothing to write). A missing or extra
field is refused (INVALID_RECORD_FIELDS; the SDK raises ValueError before sending), so the note is built from the
SDK's `empty_record()` when it is importable, else from FIELDS. Limits: the rendered note <= 100,000 UTF-8 bytes,
the diagnosis one line <= 500 bytes.

S and O come only from the ledger: what was asked or examined, with exam and test results tagged "(턴 N)". One LLM
call writes them in Korean chart style (C.C / P.I / DM / HTN (-/-) / Fever/Chill (-/-) / Chest : clear). Two
deterministic guards check it against what was said:
  numbers  every number in a line must appear in its source: the answers, the age and the vital signs, with numbers
           spoken in Korean words read as digits (서른두 살 = 32, 오십육 키로 = 56, 석 달 = 3, 사흘 = 3), and a
           difference of two grounded numbers on the same line allowed (56 kg → 49 kg, 7 kg 감소); a line that fails
           is dropped (the rest of the field is kept), and a field with no line left is rewritten from the raw answers;
  signs    an item marked (+) or (-) whose keywords never appear in the history (e.g. "Stroke (-)" when stroke was
           never discussed) is reset to "( )", the guide's mark for "not confirmed".
A and P come from the assessment call and the computed differential, then pass safety.py. Without any LLM the note
is still complete (raw answers, evidence from the core, a template plan).
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, Optional

from dxagent.knowledge.cards import slug
from dxagent.nova.checklist import NOTE_FIELD
from dxagent.nova.korean import complaint_sentence, exam_label, yes_no
from dxagent.nova.playbooks import TEST_FOR_SAFETY as TEST_SAFETY
from dxagent.nova.prompts import CHART_SYSTEM, ChartOutput, chart_user
from dxagent.nova.protocol import MAX_DIAGNOSIS_BYTES, MAX_NOTE_BYTES, injection_free, sanitize_note
from dxagent.nova.safety import lint_plan
from dxagent.reasoning.belief import evidence_breakdown

if TYPE_CHECKING:  # pragma: no cover
    from dxagent.nova.agent import Encounter

FIELDS: dict[str, tuple[str, ...]] = {
    "S": ("cc", "hpi", "pmh", "fh", "sh", "ros"),
    "O": ("exam", "tests"),
    "A": ("evidence", "differential"),
    "P": ("tests", "treatment", "disposition", "follow_up", "education"),
}


def note_shape_ok(note: object, fields: dict) -> bool:
    """True when `note` has exactly the sections and fields of `fields`, every value a string."""
    if not isinstance(note, dict) or note.keys() != fields.keys():
        return False
    for section, names in fields.items():
        part = note[section]
        if not isinstance(part, dict) or set(part) != set(names) or any(not isinstance(v, str) for v in part.values()):
            return False
    return True


FIELD_KO = {("S", "cc"): "주호소", ("S", "hpi"): "현병력", ("S", "pmh"): "과거력·약물·알레르기", ("S", "fh"): "가족력",
            ("S", "sh"): "사회력", ("S", "ros"): "계통 문진", ("O", "exam"): "진찰 소견", ("O", "tests"): "검사 결과",
            ("A", "evidence"): "근거", ("A", "differential"): "감별진단",
            ("P", "tests"): "추가 검사", ("P", "treatment"): "치료", ("P", "disposition"): "처치 결정",
            ("P", "follow_up"): "추적", ("P", "education"): "환자 교육"}

# Chart item -> Korean words that must appear in the history for a (+) or (-) to be grounded.
ITEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dm": ("당뇨", "혈당"), "htn": ("고혈압", "혈압"), "tbc": ("결핵",), "hepatitis": ("간염",),
    "stroke": ("뇌졸중", "중풍", "뇌경색", "뇌출혈"), "previous stroke": ("뇌졸중", "중풍", "뇌경색", "뇌출혈"),
    "op. hx": ("수술",), "op hx": ("수술",), "op": ("수술",), "medicine history": ("약",), "allergy": ("알레르기",),
    "smoking": ("담배", "흡연"), "alcohol": ("술", "음주"), "fever": ("열",), "chill": ("오한", "춥", "으슬"),
    "cough": ("기침",), "sputum": ("가래",), "dyspnea": ("숨",), "chest pain": ("가슴", "흉통"),
    "headache": ("두통", "머리"), "nausea": ("구역", "메스", "울렁"), "vomiting": ("구토", "토"), "n": ("구역", "메스", "울렁"),
    "v": ("구토", "토"), "diarrhea": ("설사",), "constipation": ("변비", "대변"), "dysuria": ("소변",),
    "hematuria": ("소변", "피"), "abdominal pain": ("배", "복통"), "palpitation": ("두근",), "syncope": ("정신", "기절", "실신"),
    "diaphoresis": ("땀",), "orthopnea": ("누우", "누워"), "pnd": ("자다", "밤에"), "weight loss": ("살", "체중"),
    "appetite": ("입맛", "식욕"), "anorexia": ("입맛", "식욕"), "rash": ("발진",), "vertigo": ("빙빙", "어지"),
    "dizziness": ("어지",), "hearing loss": ("귀", "청력", "들리"), "diplopia": ("겹쳐", "두 개"), "weakness": ("힘",),
    "numbness": ("저리", "감각"), "leg swelling": ("다리", "붓", "부어"), "edema": ("붓", "부종", "부어"),
    "hemoptysis": ("피",), "sore throat": ("목", "인후"), "myalgia": ("근육", "몸살"), "fatigue": ("피곤", "피로", "기운"),
    "jaundice": ("노랗", "황달"), "melena": ("변", "검"), "hematochezia": ("변", "피"), "bowel habit change": ("대변", "변"),
    "urinary retention": ("소변",), "fhx": ("가족",), "lmp": ("생리", "월경"), "leg numbness": ("저리",),
    "leg weakness": ("힘",), "dm / htn": ("당뇨", "혈당", "고혈압", "혈압"),
}

_NUM = re.compile(r"\d+(?:\.\d+)?")
_KO_NUM = {"하루": "1", "어제": "1", "어젯밤": "1", "이틀": "2", "그제": "2", "그저께": "2", "사흘": "3", "나흘": "4",
           "닷새": "5", "엿새": "6", "일주일": "7", "한 달": "1", "두 달": "2", "한 시간": "1", "두 시간": "2",
           "한 번": "1", "두 번": "2", "세 번": "3", "한 갑": "1", "반 갑": "0.5", "반갑": "0.5", "한 병": "1", "두 병": "2",
           "한 잔": "1", "두 잔": "2", "세 잔": "3", "1주": "7"}
_SIGNED = re.compile(r"(?P<name>[A-Za-z][A-Za-z.]*(?:\s+[A-Za-z][A-Za-z.]*)*(?:\s*/\s*[A-Za-z][A-Za-z.]*(?:\s+[A-Za-z][A-Za-z.]*)*)*)"
                     r"\s*\(\s*(?P<signs>[+\-]?\s*(?:/\s*[+\-]?\s*)*)\)")
_TURN_WORD = "턴"

# O.exam is written by body region, in this order (the guide's parts; Heart and Breast under Chest, as in the model
# records). A maneuver's chart item names its region.
REGIONS = ("V/S", "Growth", "General", "Skin", "H/E/E/N/T", "Neck", "Chest", "Abdomen", "Back", "GU",
           "Extremities / Neuro", "Others")  # the reference chart's order (Growth: a child's measurements and percentiles)
_REGION_OF = {"v/s": "V/S", "vital signs": "V/S", "general": "General", "mental status": "General",
              "development": "General", "h/e/e/n/t": "H/E/E/N/T", "heent": "H/E/E/N/T", "neck": "Neck",
              "thyroid": "Neck", "chest": "Chest", "heart": "Chest", "lung": "Chest", "breast": "Chest",
              "abdomen": "Abdomen", "back": "Back", "cva": "Back", "spine": "Back", "gu": "GU", "rectal": "GU",
              "extremities": "Extremities / Neuro", "neuro": "Extremities / Neuro",
              "extremities / neuro": "Extremities / Neuro", "skin": "Skin", "growth": "Growth",
              "anthropometry": "Growth", "lymph nodes": "Neck", "genitalia": "GU", "genitalia / tanner": "GU",
              "pulse": "V/S"}
_GROWTH = re.compile(r"(신체\s?계측|성장\s?도표|성장\s?속도|중간부모키|키와?\s?체중)")

# S.ros: the reference chart's fixed block, then "Others".
ROS_BLOCK = (("General weakness", "Fatigue"), ("Headache", "Dizziness"), ("Chest pain", "Palpitation"),
             ("Fever/Chill/Cough/Sputum/Rhinorrhea",), ("Abdominal pain", "Melena", "Hematochezia", "Hematemesis"),
             ("Anorexia/Nausea/Vomiting/Diarrhea/Constipation",))
_ROS_ALIAS = {"weakness": "General weakness", "chills": "Chill", "nausea": "Nausea", "vomiting": "Vomiting",
              "anorexia": "Anorexia", "appetite": "Anorexia", "dizziness": "Dizziness", "vertigo": "Dizziness",
              "palpitation": "Palpitation", "palpitations": "Palpitation", "fever": "Fever", "chill": "Chill",
              "cough": "Cough", "sputum": "Sputum", "rhinorrhea": "Rhinorrhea", "headache": "Headache",
              "fatigue": "Fatigue", "chest pain": "Chest pain", "abdominal pain": "Abdominal pain", "melena": "Melena",
              "hematochezia": "Hematochezia", "hematemesis": "Hematemesis", "diarrhea": "Diarrhea",
              "constipation": "Constipation",
              "general weakness": "General weakness"}

# P.tests is grouped by kind, in this order (as in the reference chart).
TEST_KINDS = ("혈액검사", "소변검사", "영상검사", "심전도", "내시경", "기타")
_TEST_KIND = (("기타", re.compile(r"(stool|대변|분변|객담|sputum|swab|도말|피부|biopsy|생검|청력|폐기능|spirometr)", re.I)),
              ("심전도", re.compile(r"(ECG|EKG|심전도|Holter|홀터)", re.I)),
              ("내시경", re.compile(r"(내시경|scopy\b|colonoscop|endoscop|bronchoscop|cystoscop|방광경|기관지경)", re.I)),
              ("영상검사", re.compile(r"(X선|X-ray|엑스레이|\bCT\b|CT |MRI|초음파|ultrasound|sonograph|촬영|영상|스캔|scan|"
                                  r"mammograph|조영|echocardiograph|심초음파|DEXA|골밀도|angiograph)", re.I)),
              ("소변검사", re.compile(r"(소변|urine|urinalysis|요검사|요배양)", re.I)),
              ("혈액검사", re.compile(r"(혈액|혈청|혈당|피검사|CBC|TSH|T4|BUN|creatinine|Cr\b|glucose|HbA1c|LFT|AST|ALT|"
                                  r"electrolyte|전해질|troponin|트로포닌|CRP|ESR|D-dimer|lipid|지질|ferritin|철|호르몬|"
                                  r"hCG|BNP|culture|배양|serolog|항체|antibod|level|coagulation|응고|PT|aPTT|vitamin|"
                                  r"비타민|calcium|칼슘|prolactin|cortisol|testosterone|FSH|LH|PSA|amylase|lipase|"
                                  r"ammonia|gas|ABG|lactate)", re.I)))
_ONSET = re.compile(r"(?:(?:오늘|어제|그제|그저께|지난|며칠|몇)\s?\S{0,6}\s?부터|"
                    r"(?:\d+|한|두|세|석|네|넉|다섯|여섯|일곱|여덟|아홉|열|반|이|삼|사|오|육)\s?(?:시간|일|주|개월|달|년)\s?"
                    r"(?:전\s?부터|전|째|동안|사이|부터))")

_NUM_WORD = {"한": 1, "두": 2, "세": 3, "석": 3, "네": 4, "넉": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
             "열": 10, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6}


def onset_phrase(found: str) -> str:
    """Chart form of an onset phrase: "석 달 전" / "3개월째" / "두 달 동안" -> "3개월 전" / "3개월 전" / "2개월 전"."""
    s = re.sub(r"\s?(동안|사이)$", " 전", found.strip())
    day = next((v for k, v in (("오늘", "내원 당일"), ("어제", "내원 1일 전"), ("그저께", "내원 2일 전"),
                               ("그제", "내원 2일 전")) if s.startswith(k)), "")
    if day:
        return day
    if re.match(r"반\s?년", s):
        return "6개월 전"
    m = re.match(r"(\d+|다섯|여섯|일곱|여덟|아홉|한|두|세|석|네|넉|열|이|삼|사|오|육)\s?(시간|일|주|개월|달|년)\s?(.*)$", s)
    if not m:
        return s
    n = m.group(1) if m.group(1).isdigit() else str(_NUM_WORD[m.group(1)])
    unit = "개월" if m.group(2) == "달" else m.group(2)
    return f"{n}{unit} 전"


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def sdk_empty_record() -> Optional[dict]:
    """The SDK's own template (nova_sdk.empty_record at the ZIP root), if importable."""
    try:
        import nova_sdk  # type: ignore

        rec = nova_sdk.empty_record()
        return rec if isinstance(rec, dict) else None
    except Exception:
        return None


def empty_note(template: Optional[dict] = None) -> dict[str, dict[str, str]]:
    if isinstance(template, dict) and all(isinstance(v, dict) for v in template.values()):
        return {sec: {k: "" for k in fields} for sec, fields in copy.deepcopy(template).items()}
    return {sec: {k: "" for k in fields} for sec, fields in FIELDS.items()}


def put(note: dict, sec: str, key: str, value: str) -> None:
    """Write a field only if the template has it (never add a field: INVALID_RECORD_FIELDS)."""
    if sec in note and key in note[sec]:
        note[sec][key] = sanitize_note((value or "").strip())


def flatten(note: dict) -> dict[str, str]:
    out = {}
    for sec, fields in note.items():
        lines = [f"- {FIELD_KO.get((sec, k), k)}: {v}" if "\n" not in v else f"- {FIELD_KO.get((sec, k), k)}:\n{v}"
                 for k, v in fields.items() if v]
        out[sec] = "\n".join(lines)
    out["text"] = "\n\n".join(f"[{s}]\n{out[s]}" for s in note)
    return out


def enforce_limits(note: dict, diagnosis: str) -> tuple[dict, str]:
    while len(str(note).encode("utf-8")) > MAX_NOTE_BYTES * 0.95:  # trim the longest field
        sec, key = max(((s, k) for s in note for k in note[s]), key=lambda x: len(note[x[0]][x[1]]))
        note[sec][key] = note[sec][key][: max(200, len(note[sec][key]) // 2)]
    d = diagnosis.encode("utf-8")[:MAX_DIAGNOSIS_BYTES].decode("utf-8", "ignore")
    return note, d


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


# Numbers spoken in Korean words (the simulated patients and the start information spell them: "서른두 살",
# "오십육 키로그람", "석 달"). Native numerals (서른두) and Sino-Korean ones (오십육, 백육십사, 삼 점 이) become digits; a
# single-syllable numeral counts only before a counter (두 번, 삼 일), since 이/사/오/네/세 are also ordinary words.
_NATIVE_UNIT = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "석": 3, "서": 3, "넷": 4, "네": 4, "넉": 4,
                "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
_NATIVE_TEN = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80,
               "아흔": 90}
_SINO = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
_COUNTER = (r"(?=\s?(?:살|세|키로|킬로|kg|센티|cm|미터|개월|달|년|주|일|시간|분|번|회|알|정|잔|병|갑|개비|개|점|도|"
            r"밀리|미리|mg|ml|층|장|프로|퍼센트|%|배|명|째|차례|시|그램|그람|씩|cc|리터|칸|군데|가지))")
_NATIVE = re.compile(r"(?<![가-힣])(?P<ten>스물|스무|서른|마흔|예순|일흔|여든|아흔|열|쉰)?"
                     r"(?P<unit>하나|다섯|여섯|일곱|여덟|아홉|한|둘|두|셋|세|석|서|넷|네|넉)?"
                     r"(?P<cnt>" + _COUNTER + r")?")
_SINO_RX = re.compile(r"(?<![가-힣])(?P<n>[영공일이삼사오육륙칠팔구십백천]+)(?:\s?점\s?(?P<d>[영공일이삼사오육륙칠팔구]+))?"
                      r"(?P<cnt>" + _COUNTER + r")?")
_DAYS = {"하루": "1", "이틀": "2", "사흘": "3", "나흘": "4", "닷새": "5", "엿새": "6", "이레": "7", "열흘": "10",
         "보름": "15", "일주일": "7", "반년": "6", "반 년": "6", "반 알": "0.5", "반알": "0.5", "한두": "1 2",
         "두세": "2 3", "서너": "3 4", "두어": "2"}
_TAGS = re.compile(r"(?:\bT\d+\b|\(?턴\s?\d+\)?|\[e\d+\])")


def _sino_value(word: str) -> Optional[int]:
    total, cur = 0, 0
    for ch in word:
        if ch in _SINO:
            cur = _SINO[ch]
        else:
            mult = {"십": 10, "백": 100, "천": 1000}[ch]
            total += (cur or 1) * mult
            cur = 0
    return total + cur


def korean_numbers(text: str) -> list[str]:
    """Digits for the numbers spoken in Korean words in `text` ("서른두 살" -> ["32"], "백육십사 센티" -> ["164"])."""
    out: list[str] = []
    t = text or ""
    for m in _NATIVE.finditer(t):
        ten, unit, cnt = m.group("ten"), m.group("unit"), m.group("cnt") is not None
        if not (ten or unit):
            continue
        if not cnt and (not ten or ten in ("열", "쉰") or (unit and not ten)):
            continue  # "열이 나요", "목이 쉰", "세 사람" without a counter: not a number
        out.append(str(_NATIVE_TEN.get(ten, 0) + _NATIVE_UNIT.get(unit, 0)))
    for m in _SINO_RX.finditer(t):
        word, dec, cnt = m.group("n"), m.group("d"), m.group("cnt") is not None
        if word[0] in "십백천" and len(word) == 1 and not cnt:
            continue
        if not cnt and not (len(word) >= 2 and re.search("[십백천]", word)):
            continue  # a single syllable (이, 사, 오, 일) is a number only before a counter
        if re.search(r"[십백천][십백천]", word) or word[-1] in "십백천" and len(word) > 1 and word[-2] in "십백천":
            continue
        val = _sino_value(word)
        if val is None:
            continue
        out.append(f"{val}.{''.join(str(_SINO[c]) for c in dec)}" if dec else str(val))
    for word, d in {**_KO_NUM, **_DAYS}.items():
        if word in t:
            out += d.split()
    return out


def _norm_num(n: str) -> str:
    return n.rstrip("0").rstrip(".") if "." in n else n


def _numbers(text: str, spoken: bool = False) -> set[str]:
    """The numbers in `text`: digits and fixed words (사흘, 한 달); with `spoken`, every number spoken in Korean words
    (for the source only: in the chart's own text a word such as 이번 must not count as 2)."""
    t = _TAGS.sub(" ", text or "")
    nums = set(_NUM.findall(t))
    if spoken:
        nums |= set(korean_numbers(t))
    else:
        nums |= {d for word, d in _KO_NUM.items() if word in t}
    return {_norm_num(n) for n in nums}


def _line_grounded(line: str, src: set[str]) -> bool:
    """Every number of the line is in the source, or is the difference of two grounded numbers on the same line
    ("평소 56 kg → 현재 49 kg (7 kg 감소)")."""
    nums = _numbers(line)
    missing = nums - src
    if not missing:
        return True
    have = [float(n) for n in nums & src]
    diffs = {round(abs(a - b), 2) for a in have for b in have if a != b}
    return all(float(m) in diffs for m in missing)


def numbers_grounded(field_text: str, source: str) -> bool:
    src = _numbers(source, spoken=True)
    return all(_line_grounded(ln, src) for ln in (field_text or "").splitlines() or [""])


def ground_lines(field_text: str, source: str) -> tuple[str, list[str]]:
    """The field without its lines whose numbers the source does not support. Returns (kept text, dropped lines)."""
    src = _numbers(source, spoken=True)
    kept, dropped = [], []
    for ln in (field_text or "").splitlines():
        (kept if _line_grounded(ln, src) else dropped).append(ln)
    return "\n".join(kept).strip(), dropped


def ground_signs(field_text: str, source: str) -> tuple[str, list[str]]:
    """Reset (+)/(-) to ( ) for known items never discussed in `source`. Returns (text, reset item names)."""
    reset: list[str] = []
    src = source or ""

    def fix(m: re.Match) -> str:
        names = [n.strip() for n in m.group("name").split("/")]
        signs = [x.strip() for x in m.group("signs").split("/")] if m.group("signs").strip() else [""]
        if len(signs) != len(names):
            return m.group(0)
        out_signs = []
        for name, sign in zip(names, signs):
            kws = ITEM_KEYWORDS.get(name.lower().strip(" ."))
            if sign and kws is not None and not any(k in src for k in kws):
                reset.append(name)
                sign = ""
            out_signs.append(sign or " ")
        return f"{m.group('name')} ({'/'.join(out_signs)})"

    return _SIGNED.sub(fix, field_text or ""), reset


# ---------------------------------------------------------------------------
# S and O
# ---------------------------------------------------------------------------


def _clip(text: str, n: int = 120) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t if len(t) <= n else t[: n - 1] + "…"


def _field_entries(enc: "Encounter") -> dict[tuple[str, str], list]:
    out: dict[tuple[str, str], list] = {}
    for e in enc.ledger.accepted():
        if e.kind in ("SAY", "EXAM", "TEST") and e.section in NOTE_FIELD:
            out.setdefault(NOTE_FIELD[e.section], []).append(e)
    return out


def _row(e) -> str:
    return f"[{e.id}] ({_TURN_WORD} {e.turn}) Q: {e.text} / A: {_clip(e.reply, 300)}"


# Items named after a function, signed as the symptom when the question asks about its loss ("입맛이 없으세요?" ->
# "아니요": Anorexia (-), not "Appetite (-)", which would read as no appetite).
_SIGN_NAME = {"appetite": "Anorexia", "sleep": "Insomnia", "weight": "Weight change", "urine": "Urinary symptom"}


def _item_line(e) -> str:
    """One raw S line: a chart sign when the answer is a plain yes/no to a named item, else 'item : answer' (the
    answer clipped: the raw fallback is a chart, not a transcript)."""
    yn = yes_no(e.reply, e.text)
    if e.item and yn is not None:
        n = len(e.item.split("/"))
        name = e.item
        if e.item.strip().lower() in _SIGN_NAME and re.search(r"(없|떨어|줄었|못\s|안\s)", e.text or ""):
            name = _SIGN_NAME[e.item.strip().lower()]  # the question asks about the loss: sign the symptom
        if not yn[0]:
            return f"{name} ({'/'.join(['-'] * n)})"
        if n == 1:
            return f"{name} (+)"
    head = e.item or e.label
    if not head or head == e.text:
        return f"{e.text} → {_clip(e.reply, 60)}"
    return f"{head} : {_clip(e.reply, 60)}"


def _who(enc: "Encounter") -> str:
    """"32세 여자" (age and sex as the chart writes them), or ""."""
    age, sex = enc.ctx.get("age"), enc.ctx.get("sex")
    ko = {"female": "여자", "male": "남자"}.get(sex or "", "")
    return f"{age}세 {ko}".strip() if age is not None else ko


def _vitals_chart(line: str) -> str:
    rep = (("혈압", "BP"), ("맥박", "HR"), ("호흡수", "RR"), ("체온", "BT"), ("산소포화도", "SpO2"), (" 회/분", "/min"),
           ("회/분", "/min"))
    for a, b in rep:
        line = line.replace(a, b)
    return line


def region(item: str, text: str = "") -> str:
    """The note's part for an exam entry: by its chart item, except a child's measurements (Growth)."""
    if text and _GROWTH.search(text):
        return "Growth"
    return _REGION_OF.get((item or "").strip().lower(), "Others")


def exam_lines(rows: list[tuple[str, str, Optional[int]]], vitals: str = "") -> str:
    """O.exam by region: rows of (region, finding, turn) -> "Neck : 갑상선 촉진 미만성 비대 (턴 21); ..." in REGIONS
    order, the vital signs first."""
    by: dict[str, list[str]] = {}
    for reg, finding, turn in rows:
        by.setdefault(reg if reg in REGIONS else "Others", []).append(
            f"{finding} ({_TURN_WORD} {turn})" if turn is not None else finding)
    if vitals:  # the vital signs given at the start, then any measured during the visit (orthostatic, pulse)
        by["V/S"] = [vitals] + by.get("V/S", [])
    return "\n".join(f"{reg} : " + "; ".join(by[reg]) for reg in REGIONS if by.get(reg))


def _finding(e) -> str:
    """'갑상선 촉진 양측 미만성 비대(+)': the maneuver, then the result as reported (the maneuver is not repeated when
    the result already names it)."""
    label, reply = exam_label(e.text), _clip(e.reply, 300)
    return reply if reply.replace(" ", "").startswith(label.replace(" ", "")) else f"{label} {reply}"


def _sign(e) -> str:
    yn = yes_no(e.reply, e.text)
    return "" if yn is None else ("+" if yn[0] else "-")


def ros_block(entries: list) -> str:
    """S.ros without the model: the fixed block with (+)/(-) for the items asked and answered yes or no, ( ) for the
    rest, then "Others" for every other item asked."""
    signs: dict[str, str] = {}
    others: list[str] = []
    for e in entries:
        name = _ROS_ALIAS.get((e.item or "").strip().lower())
        sign = _sign(e)
        if name and sign and name not in signs:
            signs[name] = sign
        elif e.item:
            others.append(f"{e.item} ({sign})" if sign else f"{e.item} : {_clip(e.reply, 60)}")
        else:
            others.append(f"{e.text} → {_clip(e.reply, 60)}")
    lines = []
    for group in ROS_BLOCK:
        parts = []
        for g in group:
            names = g.split("/")
            parts.append(f"{g} ({'/'.join(signs.get(n, ' ') for n in names)})")
        lines.append("  ".join(parts))
    lines.append("Others : " + (", ".join(others) if others else "없음"))
    return "\n".join(lines)


_TEMPLATE = {"pmh": ("DM / HTN", "Tbc / hepatitis", "Op. Hx", "Medicine history", "Allergy"),
             "fh": ("DM / HTN",), "sh": ("Smoking", "Alcohol")}
_TEMPLATE_KEYS = {"dm": "DM / HTN", "htn": "DM / HTN", "tbc": "Tbc / hepatitis", "hepatitis": "Tbc / hepatitis",
                  "op": "Op. Hx", "medicine": "Medicine history", "allergy": "Allergy", "smoking": "Smoking",
                  "alcohol": "Alcohol"}


def _template_of(item: str) -> str:
    head = re.split(r"[\s/.:(]", (item or "").strip().lower(), maxsplit=1)[0]
    return _TEMPLATE_KEYS.get(head, "")


def _with_template(key: str, rows: list) -> list[str]:
    """Item lines in the template's order ("( )" for a template item never asked: the guide's "not confirmed"),
    then the other items asked. For the family history, free answers go to "Others"."""
    lines: dict[str, list[str]] = {}
    others: list[str] = []
    for e in rows:
        t = _template_of(e.item)
        if t and t in _TEMPLATE.get(key, ()):
            lines.setdefault(t, []).append(_item_line(e))
        elif key == "fh" and _sign(e) == "" :
            others.append(_clip(e.reply, 160))
        else:
            others.append(_item_line(e))
    out = []
    for item in _TEMPLATE.get(key, ()):
        if item in lines:
            out += lines[item]
        else:
            n = len(item.split("/"))
            out.append(f"{item} ({'/'.join([' '] * n)})")
    if key == "fh":
        return out + ([f"Others : {'; '.join(others)}"] if others else [])
    return out + others


def raw_s_o(enc: "Encounter") -> dict[tuple[str, str], str]:
    """S and O from the ledger with no LLM: faithful, plain."""
    fe = _field_entries(enc)
    opening = enc.ctx.get("opening", "")
    said = complaint_sentence(opening, 160)
    out: dict[tuple[str, str], str] = {}
    hpi = fe.get(("S", "hpi"), [])
    pb = getattr(enc, "playbook", None)
    topic = ((enc.assessment.chief_complaint_ko if getattr(enc, "assessment", None) is not None else "")
             or getattr(getattr(enc, "focus", None), "topic_ko", "") or getattr(getattr(enc, "topic", None), "topic_ko", "")
             or (pb.title_ko if pb is not None else ""))
    cc = [f"C.C : {topic.strip() or said}"]
    onset = next((e for e in hpi if e.item == "Onset"), None)
    m = _ONSET.search(onset.reply if onset else "") or _ONSET.search(opening)
    if m:  # chart form ("3개월 전"), from the answer to the onset question or else the opening
        cc.append("Onset : " + onset_phrase(m.group(0)))
    elif onset:
        cc.append(f"Onset : {_clip(onset.reply, 60)}")
    nrs = next((e for e in hpi if e.item == "NRS"), None)
    if nrs:
        cc.append(f"통증평가 : NRS {_clip(nrs.reply, 40)}")
    elif not enc.ctx.get("pain") and not (pb is not None and pb.pain):
        cc.append("통증평가 : 해당 없음")
    out[("S", "cc")] = "\n".join(cc)
    who = _who(enc)
    out[("S", "hpi")] = (f"{who}. " if who else "") + "내원 시 \"" + said + "\"" + "".join(
        f"\n- {_item_line(e)}" for e in hpi)
    for key in ("pmh", "fh", "sh"):  # one item per line, as in the reference chart
        out[("S", key)] = "\n".join(_with_template(key, fe.get(("S", key), [])))
    out[("S", "ros")] = ros_block(fe.get(("S", "ros"), []) + [e for e in hpi if _ROS_ALIAS.get((e.item or "").lower())])
    vit = f"{_vitals_chart(enc.ctx['vitals_line'])} (내원 시)" if enc.ctx.get("vitals_line") else ""
    out[("O", "exam")] = exam_lines([(region(e.item, e.text), _finding(e), e.turn) for e in fe.get(("O", "exam"), [])], vit)
    out[("O", "tests")] = "\n".join(f"{e.item or exam_label(e.text)} : {_clip(e.reply)} ({_TURN_WORD} {e.turn})"
                                    for e in fe.get(("O", "tests"), []))
    return out


def _known_numbers(enc: "Encounter") -> str:
    """The numbers the chart may use besides the answers: the age and the vital signs given at the start."""
    age = enc.ctx.get("age")
    return f"{age if age is not None else ''} {enc.ctx.get('vitals_line') or ''}".strip()


def chart_s_o(enc: "Encounter") -> dict[tuple[str, str], str]:
    """S and O written by the LLM in chart style, field by field checked against the ledger; raw where it fails."""
    raw = raw_s_o(enc)
    cfg = enc.cfg
    if not cfg.notes_call or enc.budget.remaining() < cfg.final_margin_s * 1.5:
        return raw
    fe = _field_entries(enc)
    opening = enc.ctx.get("opening", "")
    fields = {"S.cc + S.hpi": [_row(e) for e in fe.get(("S", "hpi"), [])],
              "S.pmh": [_row(e) for e in fe.get(("S", "pmh"), [])], "S.fh": [_row(e) for e in fe.get(("S", "fh"), [])],
              "S.sh": [_row(e) for e in fe.get(("S", "sh"), [])], "S.ros": [_row(e) for e in fe.get(("S", "ros"), [])],
              "O.exam": [_row(e) for e in fe.get(("O", "exam"), [])], "O.tests": [_row(e) for e in fe.get(("O", "tests"), [])]}
    try:
        out: ChartOutput = enc.llm.structured(ChartOutput, CHART_SYSTEM, chart_user(opening, enc.ctx.get("vitals_line", ""), fields))
    except Exception as exc:
        enc._event("chart_failed", error=str(exc)[:160])
        return raw
    known = _known_numbers(enc)
    history = opening + " " + known + " " + " ".join(f"{e.text} {e.reply}" for k, es in fe.items() if k[0] == "S"
                                                    for e in es)
    res = dict(raw)
    for key in ("cc", "hpi", "pmh", "fh", "sh", "ros"):
        text = (getattr(out, key) or "").strip()
        if not text:
            continue
        src = history if key in ("cc", "hpi", "ros") else opening + " " + known + " " + " ".join(
            f"{e.text} {e.reply}" for e in fe.get(("S", key), []))
        if not injection_free(text):
            enc._event("chart_field_rejected", field=key, text=_clip(text, 160))
            continue
        text, dropped = ground_lines(text, src)
        if dropped:
            enc._event("chart_lines_dropped", field=key, lines=[_clip(x, 100) for x in dropped])
        if not text:
            enc._event("chart_field_rejected", field=key, text="")
            continue
        text, reset = ground_signs(text, history)
        if reset:
            enc._event("chart_signs_reset", field=key, items=reset)
        res[("S", key)] = text
    for okey in ("exam", "tests"):
        entries = {e.id: e for e in fe.get(("O", okey), [])}
        by_id = {x.id: (x.text or "").strip() for x in getattr(out, okey)}
        rows: list[tuple[str, str, Optional[int]]] = []
        for eid, e in entries.items():
            t = by_id.get(eid, "")
            if t and injection_free(t) and numbers_grounded(t, e.text + " " + e.reply) and _TURN_WORD not in t:
                head, _, body = t.partition(" : ")
                reg = _REGION_OF.get(head.strip().lower()) if body else None
                rows.append((reg or region(e.item, e.text), body.strip() if reg else t, e.turn))
            else:
                rows.append((region(e.item, e.text), _finding(e), e.turn))
        if okey == "exam":
            vit = f"{_vitals_chart(enc.ctx['vitals_line'])} (내원 시)" if enc.ctx.get("vitals_line") else ""
            text = exam_lines(rows, vit)
        else:
            text = "\n".join(f"{f} ({_TURN_WORD} {t})" for _, f, t in rows)
        if text:
            res[("O", okey)] = text
    return res


# ---------------------------------------------------------------------------
# A and P
# ---------------------------------------------------------------------------


def _unstable(labs: list[dict[str, Any]]) -> str:
    v = {x["analyte"]: x["value"] for x in labs}
    why = []
    if v.get("sbp", 999) < 90:
        why.append("수축기 혈압 90 mmHg 미만")
    if v.get("spo2", 100) < 90:
        why.append("산소포화도 90% 미만")
    if v.get("heart_rate", 0) > 130:
        why.append("맥박 130회/분 초과")
    if v.get("resp_rate", 0) > 30:
        why.append("호흡수 30회/분 초과")
    return ", ".join(why)


_NORMAL = re.compile(r"(특이\s*소견\s*없|정상|이상\s*없|없음|음성|\(-\))")


def _ledger_evidence(enc: "Encounter", limit: int = 4) -> list[str]:
    """A.evidence without the model, in the reference chart's shape (history, examination, decisive negatives):
    "문진 : 호흡곤란 (+), 기좌호흡 (+)" / "진찰 : 폐 청진 양측 하부 수포음" / "V/S : HR 112/min" / "음성 : 흉통 (-), 객혈 (-)"."""
    pos, neg, exam = [], [], []
    for e in enc.ledger.accepted():
        if e.kind == "SAY" and e.item and e.section in ("hpi", "ros") and e.section != "bundled":
            sign = _sign(e)
            if sign == "+":
                pos.append(f"{e.item} (+)")
            elif sign == "-":
                neg.append(f"{e.item} (-)")
        elif e.kind == "EXAM" and e.reply and not _NORMAL.search(e.reply):
            exam.append(_finding(e) if len(e.reply) <= 60 else f"{exam_label(e.text)} {_clip(e.reply, 60)}")
    out = []
    if pos:
        out.append("문진 : " + ", ".join(dict.fromkeys(pos[:6])))
    if exam:
        out.append("진찰 : " + "; ".join(exam[:3]))
    unstable = _unstable(enc.ctx.get("labs", []))
    if unstable:
        out.append(f"V/S : {unstable}")
    if neg:
        out.append("음성 소견 : " + ", ".join(dict.fromkeys(neg[:6])))
    return out[:limit]


def _fallback_evidence(enc: "Encounter") -> list[str]:
    st = enc.state
    diff = st.get("differential") or []
    findings = [f for f in st.get("findings", []) if not f.get("chief_complaint")]
    by_code = {f["code"]: f for f in findings}
    parts = []
    if not enc.cfg.core:  # the lean visit: the ledger's facts (the core's findings are only the vital-sign rules)
        return _ledger_evidence(enc) or ["문진과 진찰 소견을 종합한 판단"]
    if diff and enc.final.get("source") in ("core", "fallback"):
        for row in evidence_breakdown(diff[0]["disease_id"], findings, st.get("test_results", []), enc.store, limit=5):
            f = by_code.get(row["obs"][1:])
            if f is not None and f.get("evidence"):
                parts.append(f"{'+' if f['present'] else '-'} '{_clip(f['evidence'], 40)}' ({_TURN_WORD} {f.get('turn', 0)})")
    if not parts:
        parts = [f"'{_clip(f['evidence'], 40)}' ({_TURN_WORD} {f.get('turn', 0)})"
                 for f in findings if f["present"] and f.get("evidence")][:5]
    if not parts:
        parts = _ledger_evidence(enc)
    return parts or ["문진과 진찰 소견을 종합한 판단"]


def group_tests(lines: list[str]) -> list[str]:
    """P.tests one line per purpose, ordered by kind as in the reference chart: "- 혈액검사 : CBC, CRP : 염증 확인". A
    line that names its kind ("영상검사 : 복부 초음파 : 담석 확인") keeps it; others are classified by the test's name
    (else by the whole line, else "기타"). A line repeated is written once."""
    by: dict[str, list[str]] = {}
    for ln in lines:
        t = re.sub(r"^[-•*\s]+", "", str(ln)).strip()
        if not t:
            continue
        head, sep, rest = t.partition(":")
        kind = head.strip() if sep and head.strip() in TEST_KINDS else ""
        body = rest.strip() if kind else t
        if not kind:
            name = re.split(r"\s:\s|\(", body, maxsplit=1)[0]  # the test's name, not its reason
            kind = next((k for k, rx in _TEST_KIND if rx.search(name)), "")
            kind = kind or next((k for k, rx in _TEST_KIND if rx.search(body)), "기타")
        if body not in by.setdefault(kind, []):
            by[kind].append(body)
    return [f"- {k} : {b}" for k in TEST_KINDS for b in by.get(k, [])]


def assessment_and_plan(enc: "Encounter") -> dict[tuple[str, str], str]:
    a = enc.assessment
    en, ko = enc.final.get("en", ""), enc.final.get("ko", "")
    primary = f"{ko} ({en})" if ko and en and ko != en else (ko or en or "진단 미정")
    evidence = [x.strip() for x in (a.evidence_ko if a is not None else []) if x and x.strip()]
    if not evidence and a is not None and a.rationale_ko.strip():
        evidence = [a.rationale_ko.strip()]
    # every number in an evidence bullet must come from the visit (turn tags removed, the age and vital signs added)
    source = _TAGS.sub(" ", enc.ledger.transcript(10 ** 6)) + " " + _known_numbers(enc)
    kept = [x for x in evidence if injection_free(x) and numbers_grounded(x, source)]
    if len(kept) < len(evidence):
        enc._event("evidence_rejected", items=[_clip(x, 80) for x in evidence if x not in kept])
    evidence = kept
    if not evidence:
        evidence = _fallback_evidence(enc)
    out: dict[tuple[str, str], str] = {("A", "diagnosis"): primary,
                                       ("A", "evidence"): f"#1. {primary}\n- 근거 : {evidence[0]}"
                                       + "".join(f"\n- {x}" for x in evidence[1:5])}
    pb = getattr(enc, "playbook", None)

    def settle(name_en: str, name_ko: str = "") -> str:
        """The playbook's test linked to a diagnosis ("" if none)."""
        if pb is None:
            return ""
        hits = [t for t in pb.tests_for([name_en, name_ko]) if TEST_SAFETY not in pb.test_links.get(t, {}).get("for", ())]
        return hits[0] if hits else ""

    def r_o(name_ko: str, name_en: str, reason: str, test: str) -> str:
        name = f"{name_ko} ({name_en})" if name_ko and name_en and name_en != name_ko else (name_ko or name_en)
        reason = reason.strip().rstrip(".") or "이 증상에서 감별이 필요한 질환"
        test = test.strip()
        return f"r/o {name} : {reason}." + (f" {test}로 감별" if test else "")

    diffs: list[str] = []
    if a is not None:
        for d in a.differentials[:5]:
            diffs.append(r_o(d.name_ko, d.name_en, d.reason_ko, d.test_ko or settle(d.name_en, d.name_ko)))
    named = " ".join(diffs).lower()
    for d in enc.state.get("differential") or []:
        if len(diffs) >= 5:
            break
        if d["name"].lower() in (en or "").lower() or d["name"].lower() in named:
            continue
        if d["prob"] >= 0.03 or d.get("cant_miss"):
            diffs.append(r_o("", d["name"], f"계산된 감별 확률 {d['prob']:.0%}", settle(d["name"])))
    for d in enc.dangerous:  # every dangerous alternative still possible is named
        if d["name"].lower() not in " ".join(diffs).lower():
            diffs.append(r_o("", d["name"], "놓치면 위험한 질환, 배제 필요", (d.get("tests") or [""])[0] or settle(d["name"])))
    lead = [d for d in getattr(enc, "leading", []) if d["en"].lower() != (en or "").lower()]
    if a is None:  # no assessment: the time-out's (or focus call's) other leading diagnoses, then the can't-miss causes
        for d in lead[:3]:
            if d["en"].lower() not in " ".join(diffs).lower():
                why = "; ".join(x for x in (d.get("why"), d.get("against") and f"반대 : {d['against']}") if x)
                diffs.append(r_o(d.get("ko", ""), d["en"], why or "문진 소견상 감별 필요", settle(d["en"], d.get("ko", ""))))
    if pb is not None and len(diffs) < 3:
        for d in pb.differential:
            if len(diffs) >= 5:
                break
            if d.get("cant_miss") and d["en"].lower() not in " ".join(diffs).lower() and d["en"].lower() != (en or "").lower():
                diffs.append(r_o(d["ko"], d["en"], "이 증상에서 배제가 필요한 질환", settle(d["en"], d["ko"])))
    out[("A", "differential")] = "\n".join(diffs)

    plan = a.plan.model_dump() if a is not None else {}
    plan_ref = pb.plan if pb is not None else {}
    if not plan.get("tests"):
        plan["tests"] = [f"{name} : 감별진단 확인" for _, name, _ in enc.tests_ranked[:3]]
    if not plan.get("tests") and pb is not None:  # the work-up linked to the diagnosis and its alternatives
        names = [en, ko] + [d["en"] for d in getattr(enc, "leading", [])] + pb.cant_miss()
        if a is not None:
            names += [d.name_en for d in a.differentials]
        linked = pb.tests_for(names)
        plan["tests"] = [pb.test_line(t) for t in (linked or list(pb.tests))[:8]]
    if not plan.get("treatment"):
        card = enc.store.get(slug(en)) or enc.store.get(en)
        if card is not None and card.harm.treatments:
            plan["treatment"] = [f"{t} (표준 치료 원칙에 따름)" for t in card.harm.treatments]
        else:
            plan["treatment"] = list(plan_ref.get("treatment") or [])
    if not plan.get("disposition") and plan_ref.get("disposition"):
        plan["disposition"] = str(plan_ref["disposition"])
    if not plan.get("follow_up"):
        plan["follow_up"] = list(plan_ref.get("follow_up") or ["1~2주 후 재진 : 검사 결과 확인 후 치료 계획 결정"])
    flags = [f for f in plan_ref.get("red_flags") or [] if f]
    if flags and not any("즉시" in x or "바로" in x for x in plan["follow_up"]):
        plan["follow_up"].append(f"{'·'.join(flags)} 시 예정일 전이라도 즉시 내원")
    if not plan.get("education"):
        plan["education"] = list(pb.education) if pb is not None else []
    card = enc.store.get(slug(en)) or enc.store.get(en)
    ctx = {"allergy_text": enc._section_text("allergy"), "sex": enc.ctx.get("sex"), "age": enc.ctx.get("age"),
           "unstable": _unstable(enc.ctx.get("labs", [])),
           "lethal_primary": bool(card is not None and card.harm.untreated_course == "lethal" and card.harm.urgency == "hours"),
           "dangerous": enc.dangerous, "confirm": enc.confirm_test}
    plan, safety_notes = lint_plan(plan, ctx)
    enc.plan_out, enc.safety_notes = plan, safety_notes
    out[("P", "tests")] = "\n".join(group_tests(plan["tests"]))
    out[("P", "treatment")] = "\n".join(f"- {t}" for t in plan["treatment"] + [f"주의: {n}" for n in safety_notes if "알레르기" in n])
    out[("P", "disposition")] = plan["disposition"]
    out[("P", "follow_up")] = "\n".join(f"- {t}" for t in plan["follow_up"])
    edu = [f"- {t}" for t in plan["education"]]
    said = [f"\"{line}\" ({_TURN_WORD} {turn})" for line, turn in enc.closing_said]
    if said:
        edu.append("- 진료 중 설명함: " + ", ".join(said))
    out[("P", "education")] = "\n".join(edu)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_soap(enc: "Encounter", allow_llm: bool = True) -> dict[str, Any]:
    """{"note": official structure, "S"/"O"/"A"/"P"/"text": flattened, "diagnosis": str}."""
    note = empty_note(getattr(enc, "note_template", None) or sdk_empty_record())
    so = chart_s_o(enc) if allow_llm else raw_s_o(enc)
    ap = assessment_and_plan(enc)
    for (sec, key), val in {**so, **ap}.items():
        put(note, sec, key, val)
    note, diagnosis = enforce_limits(note, ap[("A", "diagnosis")])
    return {"note": note, "diagnosis": diagnosis, **flatten(note)}


def minimal_soap(enc: "Encounter") -> dict[str, Any]:
    """The same note without any LLM call (deadline, budget, or an error in build_soap)."""
    return build_soap(enc, allow_llm=False)


def emergency_soap(enc: "Encounter", diagnosis: str) -> dict[str, Any]:
    """Last resort when even the raw note fails: a valid record with the opening, the answers and the diagnosis."""
    note = empty_note(getattr(enc, "note_template", None) or sdk_empty_record())
    put(note, "S", "cc", f"C.C : {complaint_sentence(enc.ctx.get('opening', ''), 160)}")
    for kind, sec, key in (("SAY", "S", "hpi"), ("EXAM", "O", "exam")):
        rows = [f"{e.text} → {_clip(e.reply)} ({_TURN_WORD} {e.turn})" for e in enc.ledger.accepted()
                if e.kind == kind and e.section not in ("education", "bundled")]
        put(note, sec, key, "\n".join(rows))
    put(note, "A", "diagnosis", diagnosis)
    put(note, "A", "evidence", f"#1. {diagnosis}")
    note, diagnosis = enforce_limits(note, diagnosis or "진단 미정")
    return {"note": note, "diagnosis": diagnosis, **flatten(note)}
