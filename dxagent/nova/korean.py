"""Deterministic Korean helpers: yes/no answers, vital signs, demographics, complaint groups, onset mentions.

These run with no LLM call. The yes/no fast path is deliberately conservative: only short, unhedged answers are
read here; everything else goes to the extractor.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# yes / no
# ---------------------------------------------------------------------------

_YES = re.compile(
    r"^(네|예|응|맞아요|맞습니다|맞아|그래요|그렇습니다|그랬어요|있어요|있습니다|있었어요|있었습니다|있죠|했어요|했습니다|"
    r"피워요|피웁니다|마셔요|마십니다|나요|났어요|나왔어요|아파요|아팠어요|차요|찼어요|부었어요|그런 것 같아요)(?=$|[\s,.!~])"
)
_NO = re.compile(
    r"^(아니요|아뇨|아니오|아니에요|아닙니다|아니|없어요|없습니다|없었어요|없었습니다|없죠|전혀|안\s?(그래요|해요|피워요|"
    r"마셔요|났어요|나요|아파요|아팠어요|차요|부었어요|그랬어요)|그런\s?(건|것은|적은|적)?\s?없|별로\s?없|딱히\s?없)(?=$|[\s,.!~요어습])"
)
_HEDGE = re.compile(r"(모르|글쎄|가끔|조금|약간|살짝|간혹|때때로|애매|기억이 안|잘 안|확실하지)")


_NEG_QUESTION = re.compile(r"지\s*(는|도)?\s*않")  # "열이 나지는 않나요?": a bare "네" means "no fever"
_BARE = re.compile(r"^[\s,.!~]*(요|그래요|맞아요|그렇습니다)?[\s,.!~]*$")


def yes_no(reply: str, question: str = "") -> Optional[tuple[bool, float]]:
    """(present, confidence) for a short unambiguous Korean yes/no answer; None otherwise. After a negatively phrased
    question ("~지는 않나요?") a bare "네" / "아니요" is ambiguous (Korean answers agree with the negation), so only an
    answer that names the state ("네, 안 나요", "아니요, 나요") counts."""
    t = re.sub(r"\s+", " ", (reply or "").strip())
    if not t or len(t) > 30 or _HEDGE.search(t):
        return None
    t2 = re.sub(r"^[\"'“”‘’(\[]+", "", t)
    y_m, n_m = _YES.match(t2), _NO.match(t2)
    if question and _NEG_QUESTION.search(question) and (y_m or n_m):
        rest = t2[(y_m or n_m).end():]
        if (y_m or n_m).group(0) in ("네", "예", "응", "아니요", "아뇨", "아니오", "아니", "그래요", "맞아요", "맞습니다",
                                     "그렇습니다"):
            if re.search(r"(없|않|안\s)", rest):  # the answer names the absence: "네, 안 나요" / "아니요, 없어요"
                return (False, 0.85)
            return None  # bare, or the content decides ("아니요, 열이 나요"): left to the model
    y, n = bool(y_m), bool(n_m)
    if y and not n:
        if re.search(r"(없|아니|않|안 )", t2[2:]):  # "네, 없어요" / "네 그런 적은 없어요" are denials
            return (False, 0.85)
        return (True, 0.85)
    if n and not y:
        if re.search(r"(있어요|있습니다|있었어요)$", t2) and not re.search(r"없", t2):
            return None
        return (False, 0.85)
    return None


_DENIAL_START = re.compile(r"^(아니|아뇨|없|안\s|전혀)")
_DENIAL_END = re.compile(r"(없어요|없습니다|없었어요|없고요|없는데요|않아요|않습니다|않았어요|안 해요|안 먹어요|안 피워요|"
                         r"안 마셔요|안 했어요|아니요|아뇨|아니에요|아닙니다|모르겠어요|몰라요)[.!~]*$")


def positive_answer(reply: str, question: str = "") -> bool:
    """The answer reports the asked item (a "yes", or any content that is not a plain denial): a follow-up on it is
    worth a turn. "네, 졸피뎀 먹어요" and "가끔 먹어요" are positive; "아니요", "그런 적 없어요", "예전엔 먹었는데
    지금은 안 먹어요" are not."""
    yn = yes_no(reply, question)
    if yn is not None:
        return yn[0]
    t = re.sub(r"\s+", " ", (reply or "").strip())
    if not t:
        return False
    return not (_DENIAL_START.match(t) or _DENIAL_END.search(t))


_DX_PREFIX = re.compile(r"^(?:r/o|rule[ -]out|probable|possible|suspected|의증|추정)\s*[:：]?\s*", re.I)
_DX_SUFFIX = re.compile(r"\s*(?:의증|추정|의심)$")


def one_diagnosis(text: str) -> str:
    """The primary diagnosis as ONE entity (the 5-point grading lowers a list or an unrelated second diagnosis): no
    "r/o", "의증" or "probable", and only the first of diagnoses joined by " 및 ", " 또는 ", ";", " + " or a line break."""
    t = re.sub(r"\s+", " ", (text or "").strip().splitlines()[0] if (text or "").strip() else "")
    t = _DX_SUFFIX.sub("", _DX_PREFIX.sub("", t))
    for sep in (" 및 ", " 또는 ", ";", " + "):
        if sep in t:
            t = t.split(sep)[0]
    return t.strip(" ,.;")


# ---------------------------------------------------------------------------
# Vital signs
# ---------------------------------------------------------------------------

_NUM = r"(\d{1,3}(?:\.\d+)?)"
_VITAL_TEXT = [
    ("bp", re.compile(r"(?:혈압|BP|B/P|blood pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})", re.I)),
    ("heart_rate", re.compile(r"(?:맥박|심박수|심박|HR|PR|pulse|heart rate)\s*[:=]?\s*" + _NUM, re.I)),
    ("resp_rate", re.compile(r"(?:호흡수|호흡|RR|resp(?:iratory)? rate)\s*[:=]?\s*" + _NUM, re.I)),
    ("temperature", re.compile(r"(?:체온|BT|temp(?:erature)?)\s*[:=]?\s*" + _NUM, re.I)),
    ("spo2", re.compile(r"(?:산소\s*포화도|SpO2|SpO₂|SaO2|O2\s*sat(?:uration)?)\s*[:=]?\s*" + _NUM, re.I)),
]
_KEYMAP = {
    "bp": "bp", "혈압": "bp", "blood_pressure": "bp", "bloodpressure": "bp", "b/p": "bp",
    "sbp": "sbp", "systolic": "sbp", "수축기혈압": "sbp", "dbp": "dbp", "diastolic": "dbp", "이완기혈압": "dbp",
    "hr": "heart_rate", "pr": "heart_rate", "pulse": "heart_rate", "맥박": "heart_rate", "심박수": "heart_rate",
    "heart_rate": "heart_rate", "heartrate": "heart_rate",
    "rr": "resp_rate", "호흡수": "resp_rate", "호흡": "resp_rate", "resp_rate": "resp_rate", "respiratory_rate": "resp_rate",
    "bt": "temperature", "temp": "temperature", "체온": "temperature", "temperature": "temperature",
    "temperature_c": "temperature", "body_temperature": "temperature",
    "spo2": "spo2", "산소포화도": "spo2", "sao2": "spo2", "o2sat": "spo2", "oxygen_saturation": "spo2",
}
_LABEL_KO = {"sbp": ("혈압", "mmHg"), "heart_rate": ("맥박", "회/분"), "resp_rate": ("호흡수", "회/분"),
             "temperature_c": ("체온", "℃"), "temperature_f": ("체온", "℉"), "spo2": ("산소포화도", "%")}


def _first_number(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else None


def parse_vitals(vitals: Any) -> tuple[list[dict[str, Any]], str]:
    """(lab_values for the core's vital-sign rules, a Korean display line)."""
    vals: dict[str, float] = {}
    dbp: Optional[float] = None
    if isinstance(vitals, dict):
        for k, v in vitals.items():
            key = _KEYMAP.get(re.sub(r"[\s\-]", "_", str(k).strip().lower()), _KEYMAP.get(str(k).strip()))
            if key == "bp":
                m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", str(v))
                if m:
                    vals["sbp"], dbp = float(m.group(1)), float(m.group(2))
            elif key == "sbp":
                x = _first_number(v)
                if x is not None:
                    vals["sbp"] = x
            elif key == "dbp":
                dbp = _first_number(v)
            elif key:
                x = _first_number(v)
                if x is not None:
                    vals[key] = x
    elif vitals:
        text = str(vitals)
        for key, rx in _VITAL_TEXT:
            m = rx.search(text)
            if not m:
                continue
            if key == "bp":
                vals["sbp"], dbp = float(m.group(1)), float(m.group(2))
            else:
                vals[key] = float(m.group(1))
    if "temperature" in vals:
        t = vals.pop("temperature")
        vals["temperature_f" if t > 50 else "temperature_c"] = t
    labs, shown = [], []
    for key in ("sbp", "heart_rate", "resp_rate", "temperature_c", "temperature_f", "spo2"):
        if key in vals:
            label, unit = _LABEL_KO[key]
            labs.append({"analyte": key, "value": vals[key], "unit": unit})
            v = f"{vals[key]:g}"
            shown.append(f"{label} {v}/{dbp:g} {unit}" if key == "sbp" and dbp is not None else f"{label} {v}{unit}" if unit in ("%", "℃", "℉") else f"{label} {v} {unit}")
    if not shown and vitals:
        shown.append(str(vitals) if not isinstance(vitals, dict) else ", ".join(f"{k} {v}" for k, v in vitals.items()))
    return labs, ", ".join(shown)


# ---------------------------------------------------------------------------
# Demographics, complaint groups, onset
# ---------------------------------------------------------------------------


_KO_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80, "아흔": 90}
_KO_UNITS = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7,
             "여덟": 8, "아홉": 9}
_KO_AGE = re.compile(r"(?<![가-힣])(열|스물|스무|서른|마흔|쉰|예순|일흔|여든|아흔)?\s?(한|하나|두|둘|세|셋|네|넷|다섯|여섯|일곱|여덟|아홉)?"
                     r"\s?(?:살|세)(?![가-힣])")
_SINO = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}


def korean_number(text: str) -> Optional[int]:
    """Native Korean (서른두) or Sino-Korean (삼십이, 백오) number words up to 999; None if not a number."""
    t = (text or "").replace(" ", "")
    if not t:
        return None
    if t.isdigit():
        return int(t)
    m = re.fullmatch(r"(열|스물|스무|서른|마흔|쉰|예순|일흔|여든|아흔)?(한|하나|두|둘|세|셋|네|넷|다섯|여섯|일곱|여덟|아홉)?", t)
    if m and (m.group(1) or m.group(2)):
        return _KO_TENS.get(m.group(1) or "", 0) + _KO_UNITS.get(m.group(2) or "", 0)
    m = re.fullmatch(r"(?:([일이삼사오육칠팔구])?(백))?(?:([일이삼사오육칠팔구])?(십))?([일이삼사오육칠팔구])?", t)
    if m and any(m.groups()):
        n = (_SINO.get(m.group(1), 1) * 100 if m.group(2) else 0) + (_SINO.get(m.group(3), 1) * 10 if m.group(4) else 0)
        return n + (_SINO.get(m.group(5), 0) if m.group(5) else 0)
    return None


def parse_demographics(text: str) -> tuple[Optional[int], Optional[str]]:
    """Age and sex of the patient: the first age mentioned (the patient introduces themself, or the guardian
    introduces the child first), in digits ("62세") or words ("서른두 살", "여섯 살"); "생후 n개월" gives 0."""
    t = text or ""
    age = None
    found: list[tuple[int, int]] = []
    m = re.search(r"(\d{1,3})\s*(?:세|살|years?\s*old|y/?o)", t, flags=re.I)
    if m:
        found.append((m.start(), int(m.group(1))))
    for m in _KO_AGE.finditer(t):
        if m.group(1) or m.group(2):
            found.append((m.start(), _KO_TENS.get(m.group(1) or "", 0) + _KO_UNITS.get(m.group(2) or "", 0)))
    m = re.search(r"생후\s*(\d+|[가-힣]{1,3})\s*(개월|주|일)", t)
    if m:
        found.append((m.start(), 0))
    if found:
        age = min(found)[1]
    sex = None
    head = t[: (min(found)[0] + 40) if found else len(t)]  # the patient's own description comes first
    for scope in (head, t):
        if re.search(r"(남성|남자|남아|\(남\)|\b남/|\bmale\b|\bman\b|\bM/\d|아들)", scope, flags=re.I):
            sex = "male"
        elif re.search(r"(여성|여자|여아|\(여\)|\b여/|\bfemale\b|\bwoman\b|\bF/\d|딸)", scope, flags=re.I):
            sex = "female"
        if sex:
            break
    return age, sex


_GUARDIAN = re.compile(r"(보호자|엄마|아빠|어머니|아버지|할머니|할아버지)[^.]{0,30}(함께|같이|데리고|대신)|아이|아기|남아|여아")


def is_pediatric(text: str, age: Optional[int]) -> bool:
    """A child (under 13), or a visit where a guardian speaks for the patient."""
    if age is not None:
        return age < 13
    return bool(_GUARDIAN.search(text or ""))


# -- addressing the patient ----------------------------------------------------------------------------------------

_NAME = re.compile(r"(?:이름은|저는|제\s*이름은)\s*([가-힣]{2,4}?)(?:이고|이며|이에요|예요|입니다|이다|라고)"
                   r"|^\s*(?:안녕하세요[,.!]?\s*)?([가-힣]{2,4}?)(?:입니다|이에요|예요)[,.!\s]")
_NOT_NAMES = {"엄마", "아빠", "어머니", "아버지", "할머니", "할아버지", "보호자", "남편", "아내", "딸", "아들", "환자", "학생",
              "직장인", "주부", "회사원"}
_GUARDIAN_TITLE = (("할아버지", "할아버님"), ("할머니", "할머님"), ("어머니", "어머님"), ("엄마", "어머님"),
                   ("아버지", "아버님"), ("아빠", "아버님"))


def parse_name(text: str) -> str:
    """The patient's name from a self-introduction ("저는 이수진이고", "이름은 정하준이고"), else ""."""
    m = _NAME.search(text or "")
    name = (m.group(1) or m.group(2)) if m else ""
    return "" if name in _NOT_NAMES else name


def address(text: str, pediatric: bool, name: str = "") -> str:
    """How the doctor addresses the speaker: the guardian of a child by relation ("어머님"), an adult by name with 님
    ("이수진 님"; `name` from the start information when it gives one), else "" (no name given)."""
    t = text or ""
    if pediatric:
        hits = [(t.find(k), title) for k, title in _GUARDIAN_TITLE if k in t]
        return min(hits)[1] if hits else "보호자님"
    name = name or parse_name(t)
    return f"{name} 님" if name else ""


def empathy_line(text: str, pediatric: bool, pain: bool) -> str:
    """One acknowledgment for the start of the history, chosen from the opening (worry, pain, a child)."""
    if pediatric or "걱정" in (text or ""):
        return "걱정 많으셨겠어요."
    return "많이 아프셨겠어요." if pain else "많이 불편하셨겠어요."


_CUE = (("걱정되셨겠어요.", re.compile(r"(걱정|무서|불안|겁이|겁나)")),
        ("많이 힘드셨겠어요.", re.compile(r"(힘들|괴로|죽겠|우울|속상|눈물|미치겠)")),
        ("많이 아프셨겠어요.", re.compile(r"(너무 아프|너무 아파|못 참|참을 수가 없|참기 힘들)")))
_CUE_DENIED = re.compile(r"^.{0,7}?(없|않|아니|괜찮)")


def acknowledgment(reply: str) -> str:
    """A short acknowledgment when the patient voices worry, distress or severe pain; "" otherwise, or when the cue
    is denied right after it ("걱정은 없어요", "우울하지는 않다"; "불안해서 잠이 안 와요" still counts)."""
    r = reply or ""
    if re.match(r"^\s*(아니요|아뇨)", r):
        return ""
    for line, rx in _CUE:
        for m in rx.finditer(r):
            if not _CUE_DENIED.match(r[m.end():]):
                return line
    return ""


GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "chest_pain": ("가슴", "흉통", "흉부", "chest"),
    "dyspnea": ("숨이", "숨쉬", "숨 쉬", "숨차", "숨이 차", "호흡곤란", "호흡이", "breath", "dyspnea"),
    "cough": ("기침", "가래", "cough", "sputum"),
    "abdominal": ("배가", "배를", "배 ", "배는", "복통", "복부", "명치", "윗배", "아랫배", "옆구리", "속이", "구토", "설사", "abdominal", "stomach"),
    "headache": ("머리", "두통", "headache"),
    "fever": ("열이", "열나", "고열", "발열", "오한", "fever"),
    "dizziness": ("어지러", "어지럼", "현기증", "빙빙", "dizz", "vertigo"),
    "back_pain": ("허리", "요통", "back pain"),
    # common outpatient complaints (CPX-style cases)
    "weight": ("살이 빠", "살이 많이 빠", "체중", "몸무게", "살이 쪄", "살이 찌"),
    "sleep": ("잠을", "잠이", "불면", "수면", "잠들"),
    "growth": ("키가", "키는 또래", "성장", "작은 것 같", "발달"),
    "fatigue": ("피곤", "피로", "기운이 없", "기력", "무기력"),
    "palpitation": ("두근", "심장이 빨리", "가슴이 뛰"),
}


def detect_groups(text: str, limit: int = 2) -> list[str]:
    """Complaint groups named in the opening statement, in order of first mention (at most `limit`)."""
    t = (text or "").lower()
    pos = []
    for g, kws in GROUP_KEYWORDS.items():
        idx = [t.find(k.lower()) for k in kws if t.find(k.lower()) >= 0]
        if idx:
            pos.append((min(idx), g))
    pos.sort()
    return [g for _, g in pos][:limit]


_ONSET = re.compile(r"(부터|전부터|어제|그제|그저께|오늘|아침|저녁|새벽|지난주|지난달|며칠|몇 시간|\d+\s*(시간|일|주|개월|달|년)\s*(전|째|정도)|"
                    r"(한|두|세|석|네|넉|다섯|여섯|일곱|여덟|아홉|열|몇|수|일|이|삼|사|오|육)\s*(시간|주|달|개월|해|년)\s*(전|동안|째|사이))")
_PAIN = re.compile(r"(아프|아파|아픈|통증|쑤시|저리|따끔|뻐근|결리|조이|pain|ache)")


def onset_known(text: str) -> bool:
    return bool(_ONSET.search(text or ""))


def is_pain(text: str) -> bool:
    return bool(_PAIN.search(text or ""))


def first_sentence(text: str, limit: int = 120) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:limit]


_VISIT = re.compile(r"(왔|오게 되|오게 됐|내원|방문|진료를 받|병원에)")
_INTRO = re.compile(r"(이름은|입니다\.?$|살\s?(여자|남자|남자아이|여자아이)|보호자로|키는|몸무게는|체중은)")


def complaint_sentence(text: str, limit: int = 120) -> str:
    """The sentence of the opening statement that states the reason for the visit: a sentence saying why the patient
    came, preferring complaint words, never the self-introduction or a bare measurement."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "")) if x.strip()]
    if not sents:
        return ""
    best, score = sents[0], -99
    for i, x in enumerate(sents):
        sc = 3 * bool(_VISIT.search(x)) + 2 * bool(detect_groups(x) or is_pain(x)) + bool("걱정" in x)
        sc -= 3 * bool(_INTRO.search(x) and not detect_groups(x))
        sc -= 0.01 * i
        if sc > score:
            best, score = x, sc
    return best[:limit]


# a particle only before a space: "평가하겠습니다" keeps its 가 ("신체 비율 평가"), "부기를 비교하겠습니다" drops 를
_EXAM_TAIL = re.compile(r"(?:\s*(?:을|를|이|가)\s+|\s*)(확인|비교|측정|시행)?\s*(하겠습니다|해 보겠습니다|보겠습니다|찍겠습니다|합니다)\.?$")


def exam_label(text: str) -> str:
    """'폐 청진을 하겠습니다.' -> '폐 청진'; '양쪽 다리 부기를 비교하겠습니다.' -> '양쪽 다리 부기'."""
    return _EXAM_TAIL.sub("", (text or "").strip()).strip() or (text or "").strip()
