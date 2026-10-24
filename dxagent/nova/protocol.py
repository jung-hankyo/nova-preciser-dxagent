"""Contest protocol: actions, observations, and the checks every outgoing message passes.

- SAY: a question, explanation or empathy, one question at a time. EXAM: ONE maneuver as a sentence ("복부
  청진하겠습니다"); a bundled request or one that is not a physical examination is refused and costs a turn. TEST:
  final round only.
- SAY, EXAM and TEST: at most 30 code points after NFC normalization; longer content is refused, never truncated.
- DIAGNOSE: the S/O/A/P note with fixed fields plus one primary diagnosis; no turn.
- A question asking for several things at once is answered with a request to ask one at a time (BUNDLE_REPLY).
- Every outgoing text is filtered for anything that reads like an instruction to the virtual patient or the grader.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

KINDS = ("SAY", "EXAM", "TEST", "DIAGNOSE")
MAX_CONTENT = 30  # code points after NFC, for SAY, EXAM and TEST
MAX_NOTE_BYTES = 100_000
MAX_DIAGNOSIS_BYTES = 500

# Error codes, as the SDK reports them.
# NovaError.code carries them; a ValueError comes from the SDK's own checks before anything is sent (no turn).
REFUSAL_CODES = ("CONTENT_TOO_LONG", "EMPTY_CONTENT", "EXAM_NOT_SPECIFIC", "EXAM_NOT_RECOGNIZED", "TEST_NOT_SPECIFIC",
                 "TEST_NOT_OFFERED", "ACTION_NOT_AVAILABLE", "INVALID_ACTION_FIELDS")
# Refused by the server, no answer, but one turn used (a wrongly sent request). The other refusals cost no turn.
TURN_USING_CODES = ("EMPTY_CONTENT", "CONTENT_TOO_LONG", "EXAM_NOT_SPECIFIC", "EXAM_NOT_RECOGNIZED")
STOP_CODES = ("TURN_LIMIT_REACHED", "SESSION_EXPIRED", "SESSION_CLOSED", "RUN_NOT_ACTIVE")  # only DIAGNOSE (or nothing)
# A model of the server failed to give a usable answer (the Doctor's empty answer, the virtual patient or the
# examiner off format): no turn, no charge; send a NEW request.
RETRY_NEW_CODES = ("PROVIDER_RESPONSE_INVALID", "PATIENT_RESPONSE_INVALID", "EXAMINER_RESPONSE_INVALID")
# The request may still be running or may not have arrived: resend the SAME request_id, so it is not executed twice
# (PROVIDER_UNAVAILABLE, REQUEST_PENDING: the call has not finished; its answer comes back to the same request_id).
RETRY_SAME_CODES = ("API_UNAVAILABLE", "INVALID_API_RESPONSE", "PROVIDER_UNAVAILABLE", "REQUEST_PENDING",
                    "LOCAL_GATEWAY_UNAVAILABLE", "MODEL_NOT_READY")  # MODEL_NOT_READY: 409 before client.ready()
RETRY_CODES = RETRY_NEW_CODES + RETRY_SAME_CODES
MODEL_LIMIT_CODES = ("MODEL_CALL_LIMIT", "MODEL_TOKEN_BUDGET")  # the per-case call and token limits
RECORD_CODES = ("INVALID_RECORD_FIELDS", "RECORD_TOO_LARGE", "SINGLE_DIAGNOSIS_REQUIRED", "FINAL_RECORD_REQUIRED")
ERROR_CODES = (REFUSAL_CODES + STOP_CODES + RETRY_CODES + MODEL_LIMIT_CODES + RECORD_CODES +
               ("INVALID_MESSAGES", "INVALID_COMPLETION_FIELDS", "MODEL_NOT_ALLOWED", "RESPONSE_TOO_LARGE",
                "API_ERROR", "NOT_FOUND"))
BUNDLE_REPLY = re.compile(r"한\s*번에\s*하나씩")  # the patient's answer to a bundled question


@dataclass
class Opening:
    text: str  # the patient's first words (name, age, sex, reason for the visit)
    vitals: Any = None  # dict or free text, as the environment gives them
    age: Optional[int] = None
    sex: Optional[str] = None  # "male" | "female"
    name: str = ""
    case_id: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class Action:
    kind: str  # SAY | EXAM | TEST | DIAGNOSE
    text: str = ""
    diagnosis: str = ""
    note: Optional[dict] = None  # DIAGNOSE: {"S": {...}, "O": {...}, "A": {...}, "P": {...}} (fixed fields)
    soap: Optional[dict] = None  # DIAGNOSE: the same note flattened to text (logs, local simulator)
    meta: dict = field(default_factory=dict)  # internal: slot id, finding codes, source (never sent)

    def to_json(self) -> dict[str, Any]:
        """The HTTP body of POST /v1/session/actions (request_id is added by the transport)."""
        if self.kind == "DIAGNOSE":
            return {"action": "DIAGNOSE", "note": self.note or {}, "diagnosis": self.diagnosis}
        return {"action": self.kind, "content": self.text}


@dataclass
class Observation:
    text: str = ""
    turn: Optional[int] = None  # turns used so far, if the environment reports it
    accepted: bool = True  # False: the action was refused (bundled exam, not an exam, ...); info["turn_used"] when
    # the refusal still cost a turn (TURN_USING_CODES)
    done: bool = False  # the environment will accept no more SAY/EXAM/TEST (turn or time limit)
    code: str = ""  # error code when refused (EXAM_NOT_SPECIFIC, CONTENT_TOO_LONG, ...)
    info: dict = field(default_factory=dict)


def as_dict(x: Any) -> dict:
    return asdict(x)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

# Unambiguously interrogative Korean endings. ("-어요/-세요" are also declarative, so they never count by themselves.)
_Q_END = re.compile(r"(나요|까요|니까|인가요|던가요|는지요|ㄴ가요|을까요|ㄹ까요)$")
_INJECTION = re.compile(
    r"(채점|만점|점수를\s*(주|매)|평가자|프롬프트|시스템\s*(메시지|프롬프트)|지시(를|사항을)?\s*무시|이전\s*지시|역할을\s*바꾸|"
    r"ignore\s+(all|any|previous|prior)|system\s+prompt|grader|you\s+are\s+now|as\s+an\s+ai|jailbreak)",
    re.I,
)
_BUNDLE = re.compile(r"(,|，|/|\s및\s|그리고|와\s|과\s|랑\s|하고\s|또한|및)")
_NOT_SPECIFIC = re.compile(r"(다\s*해|전부|모두|전체|다\s*봐|이상\s*소견만|결과만|알아서|필요한\s*진찰|신체\s*검진)")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def char_len(text: str) -> int:
    """The contest's count: Unicode code points after NFC, spaces, punctuation and line breaks included."""
    return len(nfc(text))


def question_count(text: str) -> int:
    """Clauses ending in "?" plus clauses ending in an unambiguous question ending without one
    ("숨이 차나요, 가슴이 아프세요?" counts 2; "많이 힘드셨겠어요. 언제부터요?" counts 1)."""
    n = 0
    for clause, mark in re.findall(r"([^?？.!,，]+)([?？.!,，]?)", text or ""):
        c = clause.strip()
        if not c:
            continue
        if mark in ("?", "？") or _Q_END.search(c):
            n += 1
    return n


def injection_free(text: str) -> bool:
    return not _INJECTION.search(text or "")


def check_say(text: str, max_chars: int = 30) -> tuple[bool, str]:
    t = (text or "").strip()
    if not t:
        return False, "empty"
    if char_len(t) > max_chars:
        return False, f"{char_len(t)} characters > {max_chars}"
    if question_count(t) > 1:
        return False, "more than one question"
    if not injection_free(t):
        return False, "meta-instruction"
    if "\n" in t:
        return False, "line break"
    return True, ""


_CHAIN = re.compile(r"(?<=[가-힣])(고|거나|며|든지)\s")


def single_topic(text: str) -> bool:
    """For model-written questions: one topic. The virtual patient answers only the first of several questions, so a
    chain such as "땀이 나고 두근거리고 손도 떨리나요?" would leave the rest unanswered (and recorded wrongly).
    Rejects two or more chained clauses, or a list with two commas."""
    t = text or ""
    return len(_CHAIN.findall(t)) < 2 and t.count(",") + t.count("，") < 2


# Not a physical examination: history, laboratory or imaging tests, treatment, records or explanations. The examiner
# refuses these with EXAM_NOT_RECOGNIZED, which costs a turn, so an EXAM never names one.
_NOT_EXAM = re.compile(r"(혈액|채혈|피\s?검사|소변\s?검사|소변을\s?받|객담|배양|X\s?-?선|엑스\s?레이|\b(?:CT|MRI|ECG|EKG)\b|"
                       r"초음파|심전도|내시경|조직\s?검사|검사지|혈당|처방|투여|주사|수액|기록|일지|설명|물어|여쭤)", re.I)


def _check_action(t: str, max_chars: int) -> tuple[bool, str]:
    if not t:
        return False, "empty"
    if char_len(t) > max_chars:
        return False, f"{char_len(t)} characters > {max_chars}"
    if _BUNDLE.search(t):
        return False, "looks like more than one maneuver"
    if _NOT_SPECIFIC.search(t):
        return False, "not one specific maneuver"
    if not injection_free(t) or "\n" in t:
        return False, "meta-instruction or line break"
    return True, ""


def check_exam(text: str, max_chars: int = MAX_CONTENT) -> tuple[bool, str]:
    t = (text or "").strip()
    ok, why = _check_action(t, max_chars)
    if ok and _NOT_EXAM.search(t):
        return False, "not a physical examination (EXAM_NOT_RECOGNIZED costs a turn)"
    return ok, why


def check_test(text: str, max_chars: int = MAX_CONTENT) -> tuple[bool, str]:
    return _check_action((text or "").strip(), max_chars)


def error_code(obj: Any) -> str:
    """The contest error code carried by an exception (NovaError.code), a response dict or a string ("" if none).
    Any upper-case code on `.code` is returned, known or not; a ValueError from the SDK's own checks maps to
    CONTENT_TOO_LONG (action text) or INVALID_RECORD_FIELDS (note)."""
    for attr in ("code", "error_code", "error"):
        v = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
        if isinstance(v, dict):
            v = v.get("code") or v.get("error_code")
        if isinstance(v, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", v.strip()):
            return v.strip()
    text = str(obj)
    for code in ERROR_CODES:
        if code in text:
            return code
    if isinstance(obj, ValueError):
        if "NFC" in text or "30" in text:
            return "CONTENT_TOO_LONG"
        if "note" in text or "S/O/A/P" in text or "empty_record" in text:
            return "INVALID_RECORD_FIELDS"
        if "diagnosis" in text:
            return "FINAL_RECORD_REQUIRED"
    return ""


def sanitize_note(text: str) -> str:
    """Drop any line of a note that reads like an instruction to the grader or the patient simulator."""
    return "\n".join(line for line in (text or "").split("\n") if injection_free(line))
