"""NovaAgent: one contest encounter.

The encounter is a generator: it yields one Action (SAY / EXAM / TEST, then DIAGNOSE) and receives one Observation per
turn. `NovaAgent.run(env)` drives it against a pull-style environment; `StepAgent` exposes `reset(opening)` and
`step(observation)` for a push-style harness. Either way the generator always ends with a DIAGNOSE: any exception, a
spent budget, the turn limit or the closing deadline moves it to the closing steps, and the SOAP note falls back to
templates if the model cannot be called.

The visit (each step one action per turn; model calls in brackets):

    turn 0   opening + vital signs (no turn); presentation playbook from the opening words
             [topic plan, only when no playbook matched: may pick one from a menu, else writes the questions]
    HPI      the complaint's priority-1 items, greeting and one acknowledgment folded into the questions
             [focus: leading diagnoses; optional items to skip; priority-3 items to add; up to 3 questions, 2 maneuvers]
             selection: priority 1 always; 2 unless skipped; 3 if added; priority-2 items cut only past the soft cap
    HPI      the selected priority-2 items, prior episodes and care, the patient's concern
    history  past history, medications, allergy, family, smoking, alcohol, LMP, plus the playbook's background
    ROS      the selected review items
    exam     the playbook's maneuvers, then the focus call's
    close    [assessment and plan], 3 explanation lines to the patient
    DIAGNOSE [chart: S and O in chart style] + A and P, checked by soap.py's guards and safety.py

With a playbook that is 3 model calls per case (4 without). `NOVA_CORE=1` restores the Bayesian core around the same
visit: finding extraction after each block, hypotheses, the critic, value-of-information questions and the question
bank (about 13 calls); it only adds information when knowledge cards cover the case.
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from dxagent.config import DEFAULT_ACTION_COST
from dxagent.graph.nodes import Nodes
from dxagent.knowledge.cards import CardStore, slug
from dxagent.knowledge.complaints import ComplaintRegistry
from dxagent.nova import checklist as CL
from dxagent.nova import playbooks as PB
from dxagent.nova.budget import Budget, BudgetExhausted
from dxagent.nova.config import NovaConfig
from dxagent.nova.korean import (acknowledgment, address, complaint_phrase, detect_groups, empathy_line, exam_label,
                                 is_pain, is_pediatric, one_diagnosis, onset_known, parse_demographics, parse_vitals,
                                 positive_answer, yes_no)
from dxagent.nova.ledger import Entry, Ledger
from dxagent.nova.llm import NovaLLM
from dxagent.nova.prompts import (ASSESS_SYSTEM, BANK_SYSTEM, DX_RULE_CHOOSE, DX_RULE_FIXED, EXAM_NAMES, FOCUS_SYSTEM,
                                  GUARDIAN_RULE, PHRASE_SYSTEM, REVIEW_SYSTEM, TESTS_RULE_FINAL, TESTS_RULE_PRELIM,
                                  TOPIC_SYSTEM, Assessment, FocusPlan, PhraseOutput, QuestionBank, ReviewPlan, TopicPlan,
                                  assess_user, bank_user, focus_user, phrase_user, review_user, topic_user)
from dxagent.nova.protocol import (BUNDLE_REPLY, REFUSAL_CODES, Action, Observation, Opening, check_exam, check_say,
                                   check_test, injection_free, single_topic)
from dxagent.nova.safety import lint_lines
from dxagent.reasoning import planner as PL
from dxagent.reasoning.belief import apply_vital_rules, merge_findings
from dxagent.runtime import Runtime

log = logging.getLogger(__name__)

TEST_TYPES = ("poc", "lab", "imaging", "advanced_imaging", "procedure")


def make_store(cfg: NovaConfig) -> CardStore:
    """Registry + compiled kb/ (if present) + provisional card directories (+ the illustrative fixtures)."""
    s = cfg.core_settings()
    dirs = [Path(d) for d in cfg.cards_dirs]
    if cfg.use_fixtures:
        dirs.append(s.fixture_dir)
    kb = cfg.kb_dir if cfg.kb_dir and Path(cfg.kb_dir).exists() else None
    return CardStore(s.registry_dir, None, 1.0, 0, card_dirs=dirs, kb_dir=kb,
                     inherit_depth=s.inherit_depth, inherit_decay=s.inherit_decay)


_UNDETERMINED = re.compile(r"^\s*$|(undetermined|unknown|not determined|uncertain|미정|미상|불명|알\s*수\s*없)", re.I)


def _prefix(prefix: str, text: str, limit: int) -> str:
    joined = f"{prefix} {text}"
    return joined if check_say(joined, limit)[0] else text


_ADULT_ONLY = re.compile(r"(담배|흡연|술은|술을|음주|생리|월경|임신 가능|성관계|직업|무슨 일을|어떤 일을)")


def _norm(text: str) -> str:
    return re.sub(r"[\s?.!,~]", "", text or "")


def _smoker_without_amount(reply: str) -> bool:
    """A current smoker whose answer gives no amount: worth one "how many a day" question (S.sh wants p/d)."""
    t = reply or ""
    if re.search(r"(끊|안\s*피|금연|피우지\s*않|안 해|없어요|아니)", t):
        return False
    if re.search(r"(\d|갑|개비|한두|두세|서너)", t):
        return False
    yn = yes_no(t)
    return (yn is not None and yn[0]) or bool(re.search(r"피(워|웁|움|우고|웠)", t))


def _smoked_without_years(reply: str) -> bool:
    """A current or former smoker whose answer gives no duration: one "how many years" question (pack-years).
    Years in digits, native words (스무 년) or Sino-Korean words (삼십 년째) count as given."""
    t = reply or ""
    if re.search(r"(안\s*피|피우지\s*않|피운\s*적\s*(이\s*)?없|안 해요|없어요|^아니)", t):
        return False
    return not re.search(r"(\d+|[한두세네다여일아열스서마쉰예]\S{0,3}|[일이삼사오육칠팔구십]+)\s*년", t)


# A presence question ("식은땀이 나셨나요?") whose symptom an earlier reply or the opening already reported or denied
# ("가슴이 답답할 때 숨이 차고 식은땀이 날 때도 있어요.") is not asked again: the reply is recorded for the note instead.
# (question, reply, chart) patterns per symptom. The question pattern names the symptom with a presence predicate, so
# "두통이 갑자기 시작됐나요?" (an attribute of the symptom) or "누우면 숨이 더 차세요?" is not a presence question. Every
# symptom a question names must have been answered; a question asking how often, how long or which, or one whose
# words before the symptom (a time, a place, a trigger: "밤에 식은땀이 ...", "계단 오를 때 ...") the earlier reply does
# not contain, is still asked.
_SYMPTOM_WORDS = (  # (question, earlier reply, chart) patterns per symptom
    (r"식은땀이?\s?(나|났|흘|있)", r"식은땀", r"(식은땀|발한|diaphoresis)"),
    (r"두근(거리|거림이 ?있|거린)", r"두근", r"(두근|심계항진|palpitation)"),
    (r"(메스꺼우|메스껍|메슥거리|구역질?이? ?(나|있|하))", r"(메스꺼|메스껍|메슥|구역질)", r"(구역|오심|nausea)"),
    (r"(토하(세|셨|시|신|나)|토한 적|구토(가|를)? ?(하|있|나))", r"(토하|토했|토를|구토|토할|토한)", r"(구토|vomiting)"),
    (r"설사(를|가)? ?(하|있|나)", r"(설사|묽은 변|물똥)", r"(설사|diarrhea)"),
    (r"변비(가)? ?(있|생기)", r"변비", r"(변비|constipation)"),
    (r"기침(을|이)? ?(하|나|있)", r"기침", r"(기침|cough)"), (r"가래(가|는)? ?(나|있|끓)", r"가래", r"(가래|객담|sputum)"),
    (r"콧물(이)? ?(나|흐르|있)", r"콧물", r"(콧물|비루|rhinorrhea)"),
    (r"(오한이 ?(있|나|들)|으슬으슬)", r"(오한|으슬)", r"(오한|chill)"),
    (r"열이\s?(나|있|났)", r"(열이 (나|났|있|올라|오르)|열은 |열도 |발열|고열|미열)", r"(발열|fever)"),
    (r"숨이 (차|가빠)", r"(숨이 (약간 |좀 |많이 |조금 )?(차|찬|가빠)|숨찬|숨이 막|호흡곤란)", r"(호흡곤란|숨참|dyspnea)"),
    (r"(정신을 잃|기절하|실신하)", r"(정신을 잃|기절|실신|쓰러)", r"(실신|의식 소실|syncope)"),
    (r"어지러(우|워|운|움이 ?있)", r"(어지러|어지럼|빙빙)", r"(어지럼|dizziness|vertigo)"),
    (r"(두통이 ?(있|나|생기)|머리가 아프|머리도 아프)", r"(두통|머리가 아프|머리가 아파|머리도 아)", r"(두통|headache)"),
    (r"입맛(이|은)? ?(없|떨어|있)", r"(입맛|식욕)", r"(식욕|입맛|anorexia|appetite)"),
    (r"(저리|저려)", r"(저리|저려)", r"(저림|저리|감각 저하|numbness)"),
    (r"(다른 (곳|데)(으)?로|어디로).{0,6}(퍼지|뻗치|번지)", r"(퍼져|퍼지|뻗쳐|뻗치|번져|번지)", r"(방사|퍼짐|radiation)"),
    (r"^쉬면 (좋아|나아|괜찮|덜)", r"쉬(면|니까|고 나면)[^.,]{0,20}(좋아|나아|괜찮|덜|없어|가라앉)", r"(휴식|안정 시|쉬면)"),
    (r"(피가 섞|피를 토|객혈)", r"(피가 섞|피를 토|객혈|각혈)", r"(객혈|혈담|hemoptysis)"),
    (r"(검은 변|검은색 변|흑변)", r"(검은 변|검은색 변|흑변|짜장)", r"(흑색변|흑변|melena)"),
    (r"(살이 빠지|살이 빠졌|체중이 줄)", r"(살이 (많이 |좀 |조금 )?빠|체중이 (줄|감소)|몸무게가 줄)", r"(체중 감소|weight loss)"),
    (r"갑자기\s?(심하게\s?)?(아프기|생기|시작)", r"갑자기", r"(갑자기|급성|sudden|abrupt)"),
)
_PRESENCE = re.compile(r"(나요|있나요|있으세요|세요|하나요|하시나요|었나요|았나요|적 있나요|드시나요|습니까)\?\s*$")
_QUALIFIER = re.compile(r"(얼마나|언제|몇|어떤|어디서|어떻게|자주|오래|무슨|왜|무엇|뭐|처음|점점|더 )")
_DENIED_AFTER = re.compile(r"^[^.?!,]{0,10}(없|않|아니|안 )")
_NEVER_SKIP = re.compile(r"(죽고|자살|해치|극단|성관계|임신)")  # safety and the pelvic-exam gate are always asked
_CONDITION_FREE = {"아플", "아프실", "아프면", "그럴", "증상이", "그때", "통증이", "혹시", "요즘", "평소", "지금", "다른",
                   "곳으로", "데로", "아픈", "어디로"}  # "아플 때 식은땀이 ...": the complaint itself


def symptom_keys(question: str) -> list[int]:
    """The symptoms (indexes of _SYMPTOM_WORDS) a presence question asks about; [] for any other question."""
    q = question or ""
    if not _PRESENCE.search(q) or _QUALIFIER.search(q) or _NEVER_SKIP.search(q):
        return []
    return [i for i, (q_rx, _, _) in enumerate(_SYMPTOM_WORDS) if re.search(q_rx, q)]


def _condition_words(question: str, key: int) -> list[str]:
    """The stems of the words before the symptom ("계단 오를 때 숨이 차세요?" -> 계단, 오를; "밤에 식은땀이 ..." -> 밤에),
    one of which the earlier reply must contain; [] when the question starts with the symptom."""
    m = re.search(_SYMPTOM_WORDS[key][0], question)
    head = question[:m.start()] if m else ""
    return [w[:2] if len(w) <= 3 else w[:3] for w in re.findall(r"[가-힣]{2,}", head) if w not in _CONDITION_FREE]


def reported_symptom(question: str, entries: list) -> Optional[tuple[int, str, list[bool]]]:
    """(turn, reply, [reported per symptom]) when earlier replies (or the opening, turn 0) already mentioned every
    symptom the presence question asks about, reported or denied, and any condition it names; None otherwise. The
    turn and reply are those of the last reply needed."""
    keys = symptom_keys(question)
    if not keys:
        return None
    found: list[tuple[int, str, bool]] = []
    for i in keys:
        a_rx = _SYMPTOM_WORDS[i][1]
        cond = _condition_words(question, i)
        hit = None
        for e in entries:
            if e.kind not in ("SAY", "opening") or e.source == "volunteered" or e.turn is None \
                    or e.section == "bundled" or not e.accepted:
                continue
            r = e.reply or ""
            m = re.search(a_rx, r)
            if m and (not cond or any(w in r for w in cond)):  # "계단 오를 때": 계단 or 오르 in the reply
                hit = (e.turn, r, not _DENIED_AFTER.search(r[m.end():]))
                break
        if hit is None:
            return None
        found.append(hit)
    turn, reply, _ = max(found, key=lambda x: x[0])
    return turn, reply, [x[2] for x in found]


# A specific background question whose topic an earlier answer in the same block already named ("가족 중에 앓는 병이
# 있으세요?" -> "아버지가 심근경색으로 돌아가셨어요." makes "가족 중 심장병 있는 분 있나요?" a repeat). A general "no"
# does not answer a specific question: only a named topic does.
_TOPIC_WORDS = (
    (r"심장", r"(심장|심근경색|협심증|부정맥|심부전|스텐트)"), (r"(뇌졸중|중풍)", r"(뇌졸중|중풍|뇌경색|뇌출혈)"),
    (r"당뇨", r"당뇨"), (r"고혈압", r"고혈압"), (r"(콜레스테롤|고지혈)", r"(콜레스테롤|고지혈|이상지질)"),
    (r"암(?=[이을은에으과와도 치,.]|$)", r"암(?=[이을은에으과와도 ,.]|$)"),  # 위암으로, 암 치료; not 암로디핀
    (r"결핵", r"결핵"), (r"간염", r"간염"), (r"천식", r"천식"), (r"갑상선", r"갑상선"), (r"치매", r"치매"),
    (r"(콩팥|신장)", r"(콩팥|신장|투석)"), (r"(우울|정신)", r"(우울|정신|조현|조울|공황)"),
    (r"(갑자기 사망|돌연사|갑자기 돌아)", r"(갑자기 (사망|돌아)|돌연사|급사)"),
)
_TOPIC_BLOCK = {"fhx": ("fhx",), "pmh": ("pmh",), "meds": ("meds", "pmh")}
_WORRY = re.compile(r"(걱정|아닌지|아닐까|혹시|무서|겁나|불안)")


def reported_topic(question: str, section: str, entries: list) -> Optional[tuple[int, str]]:
    """(turn, reply) of an earlier answer in the same block (family history, past history, medications) that names
    the topic the question asks about and does not deny it; None otherwise."""
    block = _TOPIC_BLOCK.get(section)
    if not block or _NEVER_SKIP.search(question or ""):
        return None
    hits: list[tuple[int, str]] = []
    for q_rx, a_rx in _TOPIC_WORDS:
        if not re.search(q_rx, question or ""):
            continue
        hit = None
        for e in entries:
            r = e.reply or ""
            m = re.search(a_rx, r)
            if (e.kind == "SAY" and e.accepted and e.section in block and e.source != "volunteered" and e.turn and m
                    and not re.search(r"^[^.?!]{0,12}(없|않|아니)", r[m.end():]) and not re.match(r"^\s*(아니요|아뇨)", r)
                    and not _WORRY.search(r)):  # "심장에 문제가 있는 건 아닌지 걱정돼요": a worry, not a history
                hit = (e.turn, r)
                break
        if hit is None:
            return None  # "고혈압이나 당뇨가 있으세요?": both topics must have been named
        hits.append(hit)
    return max(hits) if hits else None


# Two questions worded differently that ask the same thing ("최근 성관계를 하셨나요?" / "성관계를 하신 적 있나요?",
# "소변 보실 때 불편하세요?" / "소변볼 때 아프세요?"): the same content stems (first two syllables of each word, without
# time words, auxiliaries and endings; 불편/통증 read as 아프). Equal sets only, so "누우면 숨이 더 차세요?" stays apart
# from "숨이 차세요?" and "가슴이 두근거리나요?" from "가슴이 아프세요?".
_STEM_STOP = {"최근", "요즘", "혹시", "지금", "평소", "하셨", "하신", "하세", "하시", "있나", "있으", "있었", "있어", "드시",
              "드세", "보실", "보신", "보셨", "볼때", "적이", "그런", "어떠", "되세", "되나", "나요", "세요", "많이", "자주",
              "조금", "계속", "주로", "아이", "아이가"}
_STEM_SAME = {"불편": "아프", "통증": "아프", "아파": "아프", "아픈": "아프", "아플": "아프", "소변": "소변", "피가": "피"}


def question_stems(question: str) -> frozenset:
    words = re.findall(r"[가-힣]{2,}", question or "")
    stems = {_STEM_SAME.get(w[:2], w[:2]) for w in words}
    return frozenset(x for x in stems if x not in _STEM_STOP)


def same_question(a: str, b: str) -> bool:
    """Same content stems and the same question words (몇, 얼마나, 언제, ...): "담배를 피우세요?" is not "담배는 몇 년
    피우셨어요?"."""
    sa, sb = question_stems(a), question_stems(b)
    return bool(sa) and sa == sb and set(_QUALIFIER.findall(a or "")) == set(_QUALIFIER.findall(b or ""))


# One maneuver under two wordings ("심장 청진" / "심음 청진하겠습니다.") is one exam: model-added maneuvers are matched
# to the planned and done ones by this key (the whole label, anchored, so "우상복부 촉진" is not "복부 촉진").
_EXAM_SAME = ((r"^(심장|심음|심장음|심장 소리)\s?(을\s?|를\s?)?청진", "심음 청진"),
              (r"^(폐|양측 폐|폐음|호흡음|숨소리)\s?(을\s?|를\s?)?청진", "폐 청진"),
              (r"^(배|복부|복부를|배를)\s?(촉진|눌러)", "복부 촉진"), (r"^(장음|장음을|장 소리)\s?(청진)?", "장음 청진"),
              (r"^(목\s?정맥|경정맥)", "경정맥"), (r"^(목|경부)\s?림프절", "경부 림프절 촉진"),
              (r"^(무릎\s?반사|슬개건\s?반사)", "슬개건 반사"), (r"^(목|경부)\s?(경직|강직)", "경부 강직"),
              (r"^갑상(선|샘)\s?(을\s?|를\s?)?촉진", "갑상선 촉진"), (r"^(하지|다리)\s?(말초|부종|함요)", "하지 말초 진찰"),
              (r"^(복부\s?)?반발\s?압통", "반발 압통"), (r"^(늑골척추각|CVA)", "늑골척추각 압통"))


def exam_key(text: str) -> str:
    t = exam_label(text or "").strip()
    for rx, key in _EXAM_SAME:
        if re.search(rx, t):
            return key
    return re.sub(r"[\s·,.]", "", t)


# A question in the patient's reply ("이거 심각한 건가요?"): acknowledged in the next question, answered at the end.
_PATIENT_QUESTION = re.compile(r"(\?|？|(건가요|인가요|될까요|되나요|해야 하나요|할까요|맞나요|괜찮을까요|괜찮은가요|"
                               r"있을까요|없을까요|어떡하죠|어떻게 해요)\s*[.!~]*\s*$)")


def patient_question(reply: str) -> bool:
    r = (reply or "").strip()
    return len(r) >= 6 and bool(_PATIENT_QUESTION.search(r))


# A closing line that tells the patient the follow-up: when to come back, or what happens next.
_FOLLOW_UP = re.compile(r"(다시 (뵙|봬|뵐|오|와|볼|만나|설명)|재진|다음 (주|번|달|진료|외래)|(뒤|후)에?\s?(다시|오|와|뵙|봬|뵐|결과|외래)|"
                        r"응급실|입원|결과(가|를|는)?\s?(나오|보|확인|나오면)|예약)")


def split_line(line: str, limit: int) -> list[str]:
    """An explanation too long for one utterance, as its sentences that fit ("A해요. B해요." -> two lines); a sentence
    that still does not fit is dropped."""
    parts = [x.strip() for x in re.split(r"(?<=[.!])\s+", (line or "").strip()) if x.strip()]
    return [x for x in parts if len(x) >= 6 and check_say(x, limit)[0] and "?" not in x] if len(parts) > 1 else []


class NovaAgent:
    def __init__(self, cfg: Optional[NovaConfig] = None, store: Optional[CardStore] = None, offline: bool = False,
                 llm_factory: Optional[Callable[[Budget], Any]] = None, clock: Optional[Callable[[], float]] = None):
        self.cfg = cfg or NovaConfig.from_env()
        self.offline = offline
        self.clock = clock or time.monotonic  # resolved now, not at import (the simulator's virtual clock)
        self.store = store or make_store(self.cfg)
        self.llm_factory = llm_factory
        self.guidelines: Any = None  # GuidelineLibrary, loaded by prepare() (or on first use)
        self.note_template: Optional[dict] = None  # nova_sdk.empty_record(), when the SDK is importable
        # Nothing learned in one case is carried to the next (contest rule): phrase caches, banks and question history
        # live in the Encounter. A process is expected to serve one case; run() may still be called again safely.

    def prepare(self, time_budget_s: float = 300.0) -> dict[str, Any]:
        """Per-case setup within prepare()'s 10 minutes: the guideline index and the SDK's note template."""
        from dxagent.nova.guidelines import GuidelineLibrary
        from dxagent.nova.soap import sdk_empty_record

        info: dict[str, Any] = {"cards": len(self.store)}
        self.note_template = sdk_empty_record()
        info["note_template"] = "nova_sdk" if self.note_template else "builtin"
        if self.guidelines is None:
            self.guidelines = GuidelineLibrary.from_config(self.cfg)
        t0 = time.monotonic()
        try:
            info["guideline_docs"] = self.guidelines.load(time_budget_s=time_budget_s)
            info["guideline_index"] = self.guidelines.source
        except Exception as exc:  # never fail a case on the library
            info["guideline_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        info["prepare_s"] = round(time.monotonic() - t0, 1)
        return info

    def new_llm(self, budget: Budget) -> Any:
        if self.llm_factory is not None:
            return self.llm_factory(budget)
        if self.offline:
            from dxagent.nova.offline import OfflineNovaLLM

            return OfflineNovaLLM(self.store, budget)
        return NovaLLM(self.cfg, budget=budget)

    def encounter(self, opening: Opening, time_limit_s: Optional[float] = None) -> "Encounter":
        return Encounter(self, opening, time_limit_s)

    def run(self, env: Any) -> dict[str, Any]:
        """Play one case against a pull-style environment (start / act / submit) and return its record."""
        opening = env.start()
        enc = self.encounter(opening, getattr(env, "seconds_left", None))
        available = getattr(env, "available", None)
        if available is not None and "TEST" not in available:
            enc.tests_blocked = True
        gen = enc.run()
        action = next(gen)
        while action.kind != "DIAGNOSE":
            try:
                obs = env.act(action)
            except Exception as exc:  # the environment failed: treat the action as refused, keep going
                log.warning("environment error on %s: %s", action.kind, exc)
                obs = Observation(text="", accepted=False, info={"error": str(exc)[:200]})
            action = gen.send(obs)
        gen.close()
        result = None
        try:
            result = env.submit(action)
        except Exception as exc:
            enc.errors.append({"where": "submit", "error": str(exc)[:300]})
        return enc.record(action, result)


class StepAgent:
    """Push-style adapter: `reset(opening) -> first action`, then `step(observation) -> next action` until DIAGNOSE."""

    def __init__(self, agent: Optional[NovaAgent] = None):
        self.agent = agent or NovaAgent()
        self.enc: Optional[Encounter] = None
        self._gen: Optional[Generator] = None

    def reset(self, opening: Any) -> Action:
        if isinstance(opening, dict):
            opening = opening_from_dict(opening)
        elif isinstance(opening, str):
            opening = Opening(text=opening)
        self.enc = self.agent.encounter(opening)
        self._gen = self.enc.run()
        return next(self._gen)

    def step(self, observation: Any) -> Action:
        if isinstance(observation, dict):
            observation = observation_from_dict(observation)
        elif isinstance(observation, str):
            observation = Observation(text=observation)
        return self._gen.send(observation)


def opening_from_dict(d: dict[str, Any]) -> Opening:
    text = d.get("opening") or d.get("text") or d.get("message") or d.get("patient") or ""
    if isinstance(text, dict):
        text = text.get("text", "")
    sex = d.get("sex") or d.get("gender")
    if isinstance(sex, str):
        s = sex.strip().lower()
        sex = "male" if s in ("m", "male", "남", "남성", "남자") else "female" if s in ("f", "female", "여", "여성", "여자") else None
    age = d.get("age")
    try:
        age = int(age) if age is not None else None
    except (TypeError, ValueError):
        age = None
    return Opening(text=str(text), vitals=d.get("vitals") or d.get("vital_signs"), age=age, sex=sex,
                   name=str(d.get("name", "")), case_id=str(d.get("case_id", d.get("id", ""))), raw=d)


def observation_from_dict(d: dict[str, Any]) -> Observation:
    text = d.get("text") or d.get("reply") or d.get("observation") or d.get("message") or ""
    turn = d.get("turn")
    return Observation(text=str(text), turn=int(turn) if isinstance(turn, (int, float)) else None,
                       accepted=bool(d.get("accepted", True)), done=bool(d.get("done", False)), info=d)


# ---------------------------------------------------------------------------
# One encounter
# ---------------------------------------------------------------------------


class Encounter:
    def __init__(self, agent: NovaAgent, opening: Opening, time_limit_s: Optional[float] = None):
        self.agent = agent
        self.cfg = agent.cfg
        self.store = agent.store
        self.opening = opening
        self.budget = Budget(self.cfg, clock=agent.clock, time_limit_s=time_limit_s)
        self.budget.turns_left = lambda: max(0, self.cfg.gather_end() - self.turns)
        self.llm = agent.new_llm(self.budget)
        self.settings = self.cfg.core_settings()
        self.nodes: Optional[Nodes] = None  # the Bayesian core, only in the core mode
        if self.cfg.core:
            complaints = ComplaintRegistry(self.settings.registry_dir, None, llm=self.llm,
                                           auto=self.settings.dedup_auto, confirm=self.settings.dedup_confirm)
            rt = Runtime(settings=self.settings, doctor=self.llm, store=self.store, contrast=None, fallbacks=[],
                         prior_llm=self.llm, complaints=complaints)
            self.nodes = Nodes(rt)
        self.ledger = Ledger()
        self.turns = 0
        self.env_done = False
        self.state: dict[str, Any] = {}
        self.ctx: dict[str, Any] = {}
        self.pending: list[tuple[dict[str, Any], Entry]] = []
        self.performed: set[str] = set()
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.core_committed = False
        self.commit_reason = ""
        self.hyp_calls = 0
        self.last_hyp_turn = -99
        self.bank: dict[str, list[dict[str, Any]]] = {"question": [], "exam": [], "test": []}
        self.bank_calls = 0
        self.assessment: Optional[Assessment] = None
        self.final: dict[str, str] = {}
        self.closing_said: list[tuple[str, int]] = []
        self.plan_out: dict[str, Any] = {}
        self.safety_notes: list[str] = []
        self.tests_ranked: list[tuple[str, str, float]] = []
        self.dangerous: list[dict[str, Any]] = []
        self.confirm_test: Optional[str] = None
        self.plan = CL.EncounterPlan()
        self.phrase_cache: dict[tuple[str, str], str] = {}  # (kind, code) -> Korean utterance, this case only
        self.note_template = agent.note_template
        self.guideline_refs: list[dict[str, str]] = []
        self.tests_blocked = False
        self.topic: Optional[TopicPlan] = None
        self.playbook: Any = None  # dxagent.nova.playbooks.Playbook selected for this presentation
        self.topic_questions: list[CL.Slot] = []
        self.topic_exams: list[CL.Slot] = []
        # the focus call (mid-visit): leading diagnoses, optional items dropped, priority-3 items added, new items
        self.focus: Optional[FocusPlan] = None
        self.leading: list[dict[str, str]] = []  # [{"en", "ko", "why"}], most likely first
        self.focus_skip: set[str] = set()
        self.focus_add: set[str] = set()
        self.focus_new: list[CL.Slot] = []
        self.dropped: list[str] = []  # slot ids cut by the selection (focus call, or the soft cap)
        self.cut: list[CL.Slot] = []  # history items cut only for turns (the cap, or yielding to reserved turns)
        # the diagnostic time-out (after the history and the review of systems, before the exam)
        self.review: Optional[ReviewPlan] = None
        self.review_new: list[CL.Slot] = []
        self.reviewed = False
        self.cant_miss_open: list[str] = []
        self.followups = 0  # playbook follow-up questions asked after positive answers
        # communication
        self.greeted = False
        self.empathy_done = False
        self.ack_pending = ""
        self.acks = 0
        self.last_ack_turn = -99
        self.signposted: set[str] = set()
        self.signpost_tries: dict[str, int] = {}
        self.light_pending = False  # a plain acknowledgment ("그러셨군요.") before the next question, when it fits
        self.light_acks = 0
        self.asked_back = False  # the patient asked something: "끝에 자세히 설명드릴게요." before the next question
        self.patient_asks: list[tuple[int, str]] = []  # (turn, reply) where the patient asked a question
        self.exam_announced = False
        self.exam_intro_said: list[tuple[str, int]] = []
        self.skipped: list[dict[str, Any]] = []  # questions not asked because a reply had already answered them

    # -- bookkeeping ----------------------------------------------------------

    def _event(self, kind: str, **kw: Any) -> None:
        self.events.append({"turn": self.turns, "t": round(self.budget.elapsed(), 1), "kind": kind, **kw})

    def _error(self, where: str, exc: BaseException) -> None:
        self.errors.append({"turn": self.turns, "where": where, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                            "trace": traceback.format_exc(limit=4)[-1200:]})
        log.warning("%s failed: %s", where, exc)

    def _node(self, name: str, **kw: Any) -> dict[str, Any]:
        """One step of the Bayesian core (core mode only; {} otherwise or on failure)."""
        if self.nodes is None:
            return {}
        try:
            upd = getattr(self.nodes, name)(self.state, **kw) or {}
        except BudgetExhausted as exc:
            self._event("budget", node=name, reason=str(exc))
            return {}
        except Exception as exc:
            self._error(name, exc)
            return {}
        self.state.update(upd)
        return upd

    def _refresh(self) -> None:
        if self.state.get("hypotheses"):
            self._node("load_knowledge")
            self._node("update_belief")

    def _stop(self, reserve: int = 0) -> bool:
        """Stop gathering: environment limit, turns needed for closing (plus `reserve` turns kept for later steps),
        soft cap, or the time margin."""
        return (self.env_done or self.turns >= self.cfg.gather_end() - reserve
                or self.budget.wrap_up_due(self.turns + reserve))

    def _test_reserve(self) -> int:
        """Turns kept for tests (final round, core mode) while history and exams are taken."""
        if not self.cfg.core or not self.cfg.allow_tests or self.tests_blocked:
            return 0
        return self.cfg.test_max

    def _exam_reserve(self) -> int:
        """Turns kept for the planned maneuvers, the sentence that announces them, and final-round tests while the
        history is taken."""
        left = sum(1 for s in self.plan.exams if s.ask not in self.performed)
        return min(self.cfg.playbook_exam_reserve, left) + self._intro_turns() + self._test_reserve()

    def _intro_turns(self) -> int:
        """1 while the examination is still to be announced to the patient (config exam_intro), else 0."""
        pending = any(s.ask not in self.performed for s in self.plan.exams)
        return 1 if self.cfg.exam_intro and not self.exam_announced and pending else 0

    def _review_reserve(self) -> int:
        """Turns kept for the diagnostic time-out's questions and maneuvers until it has run."""
        cfg = self.cfg
        if self.reviewed or cfg.core or not cfg.review_call:
            return 0
        return min(cfg.review_reserve, cfg.review_questions_max + cfg.review_exams_max)

    # -- the encounter --------------------------------------------------------

    def run(self) -> Generator[Action, Observation, None]:
        try:
            self._start()
            yield from self._gather()
        except GeneratorExit:
            raise
        except Exception as exc:
            self._error("gather", exc)
        try:
            yield from self._close()
        except GeneratorExit:
            raise
        except Exception as exc:
            self._error("close", exc)
        yield self._diagnose()

    def _start(self) -> None:
        op = self.opening
        text = (op.text or "").strip()
        labs, vitals_line = parse_vitals(op.vitals)
        if not labs and not op.vitals:  # vital signs written into the opening text instead of a separate field
            found, line = parse_vitals(text)
            if len(found) >= 2:
                labs, vitals_line = found, line
        age, sex = op.age, op.sex
        p_age, p_sex = parse_demographics(text)
        age = age if age is not None else p_age
        sex = sex or p_sex
        groups = detect_groups(text)
        self.ctx = {"age": age, "sex": sex, "groups": groups, "pain": is_pain(text), "vitals_line": vitals_line,
                    "labs": labs, "opening": text}
        self.ledger.add(turn=0, kind="opening", section="cc", label="주호소", text="", reply=text, source="opening")
        if vitals_line:
            self.ledger.add(turn=0, kind="vitals", section="vitals", label="활력징후", text="", reply=vitals_line,
                            source="opening")
        presentation = text + (f"\n활력징후: {vitals_line}" if vitals_line else "")
        self.state = {
            "presentation": presentation, "presentation_source": "patient", "last_reply": presentation,
            "transcript": [{"role": "patient", "type": "history", "text": presentation, "turn": 0}],
            "turn": 0, "phase": "gathering", "findings": apply_vital_rules(labs, self.store, 0), "test_results": [],
            "hypotheses": [], "last_actions": [], "context": {}, "lab_values": list(labs), "asked_codes": [],
            "done_action_ids": [], "cost_spent": 0.0,
        }
        if text:
            self._read_opening()

    def _read_opening(self) -> None:
        """The playbook (or the topic plan) and the visit plan from the opening statement; in the core mode also its
        findings and the first hypotheses."""
        text = self.ctx["opening"]
        if self.cfg.core:
            self._node("perceive")
            c = self.state.get("context") or {}
            self.ctx["age"] = self.ctx["age"] if self.ctx["age"] is not None else c.get("age")
            self.ctx["sex"] = self.ctx["sex"] or c.get("sex")
        self.ctx["pediatric"] = is_pediatric(text, self.ctx["age"])
        self.ctx["address"] = (address(text, self.ctx["pediatric"], (self.opening.name or "").strip())
                               if self.cfg.address_by_name else "")
        if self.cfg.core:
            self._hypothesize(force=True)
            self._refresh()
        if self.cfg.playbooks:
            from dxagent.nova import playbooks as PB

            try:
                self.playbook = PB.select(text, self.ctx["pediatric"], self.ctx["sex"])
            except Exception as exc:  # a broken playbook file must not stop the visit
                self._error("playbook", exc)
        self._build_plan()
        if self.playbook is None:
            self._plan_topic()
        self._event("start", groups=self.ctx["groups"], age=self.ctx["age"], sex=self.ctx["sex"],
                    pediatric=self.ctx["pediatric"], playbook=self.playbook.id if self.playbook else "",
                    hypotheses=[h["name"] for h in self.state.get("hypotheses", [])],
                    topic=self.topic.topic_ko if self.topic else "", topic_questions=len(self.topic_questions),
                    topic_exams=len(self.topic_exams), address=self.ctx["address"])

    def _build_plan(self) -> None:
        text = self.ctx.get("opening", "")
        self.plan = CL.build_plan(self.ctx["groups"], text, self.ctx["sex"], self.ctx["age"], self.ctx["pain"],
                                  onset_known(text), self.cfg.core_exam_max, self.cfg.ask_alcohol,
                                  self.cfg.ask_job, self.cfg.ask_tbc_hep, self.cfg.ask_prior_care, self.ctx["pediatric"],
                                  playbook=self.playbook, playbook_exam_max=self.cfg.playbook_exam_max,
                                  concern=self.cfg.concern_question)

    def _new_items(self, questions: list, exams: list, n_q: int, n_e: int, prefix: str,
                   first_tier: int = 2, first_n: int = 0) -> tuple[list[CL.Slot], list[CL.Slot]]:
        """Model-written questions and maneuvers as slots: validated (30 characters, one topic, no adult-only question
        to a child's guardian), deduplicated against the plan and what was done. The first `first_n` questions get
        `first_tier`, the rest tier 2."""
        planned = [s.ask for s in self.plan.hpi + self.plan.background + self.plan.ros + self.plan.exams + self.cut]
        seen = {_norm(x) for x in planned + sorted(self.performed)}
        # the same symptom or question asked twice in other words, or one maneuver under another name, is a repeat
        sym_seen = {k for x in planned + sorted(self.performed) for k in symptom_keys(x)}
        exam_seen = {exam_key(s.ask) for s in self.plan.exams} | {exam_key(e.text) for e in self.ledger.entries
                                                                  if e.kind == "EXAM"}
        limit = self.cfg.say_max_chars
        qs: list[CL.Slot] = []
        es: list[CL.Slot] = []
        for i, it in enumerate(questions):
            t = (it.text_ko or "").strip()
            if (len(qs) >= n_q or not check_say(t, limit)[0] or _norm(t) in seen or not single_topic(t)
                    or (self.ctx.get("pediatric") and _ADULT_ONLY.search(t))):
                continue
            keys = symptom_keys(t)
            if (keys and (set(keys) <= sym_seen or reported_symptom(t, self.ledger.entries) is not None)) or \
                    any(same_question(t, x) for x in planned + sorted(self.performed) + [q.ask for q in qs]):
                self._event("repeat_dropped", text=t, why="the question is planned, asked or already told")
                continue
            seen.add(_norm(t))
            sym_seen |= set(keys)
            sec = it.section if it.section in CL.SECTION_ORDER else "hpi"
            tier = first_tier if len(qs) < first_n else 2
            qs.append(CL.Slot(f"{prefix}_q{i}", sec, it.target or t, t, (), "SAY", item=it.item, tier=tier))
        for i, it in enumerate(exams):
            t = (it.text_ko or "").strip()
            if len(es) >= n_e or not check_exam(t, limit)[0] or not PB.specific_exam(t) or _norm(t) in seen:
                continue
            if exam_key(t) in exam_seen:
                self._event("repeat_dropped", text=t, why="the same maneuver is planned or done")
                continue
            seen.add(_norm(t))
            exam_seen.add(exam_key(t))
            es.append(CL.Slot(f"{prefix}_e{i}", "exam", it.target or exam_label(t), t, (), "EXAM",
                              item=it.item or exam_label(t), tier=2))
        return qs, es

    def _insert(self, slots: list[CL.Slot]) -> None:
        """Put new slots into the plan by section: the history of the complaint before prior episodes, care and the
        concern; background and review items at the end of their blocks; maneuvers after the planned ones."""
        tail_ids = {CL.HPI_PRIOR.id, CL.HPI_CARE.id, CL.CONCERN.id}
        for s in slots:
            if s.kind == "EXAM":
                self.plan.exams.append(s)
            elif s.section == "hpi":
                i = next((k for k, x in enumerate(self.plan.hpi) if x.id in tail_ids), len(self.plan.hpi))
                self.plan.hpi.insert(i, s)
            elif s.section == "ros":
                self.plan.ros.append(s)
            else:
                self.plan.background.append(s)

    def _plan_topic(self) -> None:
        """No playbook matched the opening's wording (one call): the model may name one from the menu (the plan is
        rebuilt on it), else it writes the complaint-specific questions and maneuvers and the likely causes."""
        cfg = self.cfg
        if not cfg.topic_plan or (cfg.topic_questions_max <= 0 and cfg.topic_exams_max <= 0):
            return
        menu = []
        if cfg.playbooks:
            from dxagent.nova import playbooks as PB

            menu = PB.menu(self.ctx.get("pediatric", False), self.ctx.get("sex"))
        planned = [s.ask for s in self.plan.hpi + self.plan.background + self.plan.ros + self.plan.exams]
        names = [h["name"] for h in (self.state.get("hypotheses") or [])[:5]]
        guide = self._guideline_text(names[:2], ("history", "physical", "symptoms", "examination", "signs"), 1500)
        try:
            out = self.llm.structured(
                TopicPlan, TOPIC_SYSTEM.format(guardian=GUARDIAN_RULE if self.ctx.get("pediatric") else "",
                                               n_q=cfg.topic_questions_max, n_e=cfg.topic_exams_max,
                                               exam_names=EXAM_NAMES),
                topic_user(self.ctx.get("opening", ""), self.ctx.get("vitals_line", ""), self.ctx.get("age"),
                           self.ctx.get("sex"), names, planned, guide, menu))
        except Exception as exc:
            self._event("topic_failed", error=str(exc)[:160])
            return
        self.topic = out
        self.leading = [{"en": n.strip(), "ko": "", "why": ""} for n in out.likely_causes if n and n.strip()][:5]
        if menu and (out.presentation or "").strip() in {i for i, _ in menu}:
            from dxagent.nova import playbooks as PB

            self.playbook = PB.load_all()[out.presentation.strip()]
            self._build_plan()
            self._event("playbook_from_topic", playbook=self.playbook.id)
            return  # the playbook carries the complaint; the focus call adds case-specific items later
        self.topic_questions, self.topic_exams = self._new_items(out.questions, out.exams, cfg.topic_questions_max,
                                                                 cfg.topic_exams_max, "topic", first_tier=1, first_n=4)
        self._insert(self.topic_questions + self.topic_exams)

    def _ask_complaint(self) -> Generator[Action, Observation, None]:
        """No opening statement was found in the start information: ask for the complaint, then plan from it."""
        res = yield from self._do("SAY", _prefix(CL.GREETING, CL.CC_ASK.ask, self.cfg.say_max_chars), "cc", "주호소", [],
                                  "slot", item="C.C")
        self.greeted = True
        if not res or not res[0].accepted:
            return
        text = res[0].text.strip()
        res[1].section = "cc_answer"  # becomes the opening statement, not an S.hpi line
        self.ctx.update(opening=text, groups=detect_groups(text), pain=is_pain(text))
        self.playbook = None
        self.ledger.entries[0].reply = text
        p_age, p_sex = parse_demographics(text)
        self.ctx["age"] = self.ctx["age"] if self.ctx["age"] is not None else p_age
        self.ctx["sex"] = self.ctx["sex"] or p_sex
        self.state["presentation"] = text + (f"\n활력징후: {self.ctx['vitals_line']}" if self.ctx["vitals_line"] else "")
        self.state["last_reply"] = self.state["presentation"]
        self.state["transcript"][0]["text"] = self.state["presentation"]
        self._read_opening()

    def _gather(self) -> Generator[Action, Observation, None]:
        cfg = self.cfg
        if not self.ctx.get("opening"):
            yield from self._ask_complaint()
        first = [s for s in self.plan.hpi if s.tier <= 1 and s.id != CL.CONCERN.id]
        yield from self._ask(first, "hpi")
        self._core_update(force=True)
        self._focus()
        self._select()
        yield from self._ask(self.plan.hpi, "hpi")  # the selected priority-2 items, prior episodes, care, concern
        yield from self._ask(self.plan.background, "background")
        self._core_update()
        yield from self._core_diagnostics("question", cfg.diag_history_max)
        yield from self._ask(self.plan.ros, "ros")
        self._core_update()
        self._review()
        yield from self._ask([s for s in self.review_new if s.kind == "SAY"], "review")
        yield from self._ask(self._backfill(), "backfill")
        yield from self._ask(self.plan.exams, "exam")
        self._core_update()
        yield from self._core_diagnostics("exam", cfg.diag_exam_max)
        if cfg.allow_tests:
            yield from self._core_diagnostics("test", cfg.test_max)

    # -- asking -----------------------------------------------------------------

    def _ask(self, slots: list[CL.Slot], phase: str) -> Generator[Action, Observation, None]:
        """Ask the slots in order (skipping what was done), each with at most one communication prefix. When the turns
        kept for later steps (maneuvers, the time-out) are reached, priority-2 items yield and priority-1 items may
        still use them; nothing uses the closing explanations' turns or the final round's test turns."""
        for slot in list(slots):
            if self._stop(self._test_reserve()):  # the closing explanations' turns and the final round's tests
                break
            if slot.ask in self.performed:
                continue
            if slot.kind == "SAY" and self._repeat(slot.ask):
                self.performed.add(slot.ask)
                self._event("repeat_dropped", text=slot.ask, why="asked before in other words")
                continue
            told = self._already_told(slot) if slot.kind == "SAY" else None  # recorded even when turns run short
            if told is not None:
                if told[1]:  # reported: the playbook's follow-ups (how much, how long) are still asked
                    yield from self._follow_up(slot, told[0], positive=True)
                continue
            reserve = 0 if phase == "exam" else self._exam_reserve() + (self._review_reserve() if phase != "review" else 0)
            if reserve and self._stop(reserve) and slot.tier >= 2:
                if slot.id not in self.dropped:
                    self.dropped.append(slot.id)  # yields to the turns kept for the maneuvers and the time-out
                    if slot.kind == "SAY" and phase != "backfill":
                        self.cut.append(slot)
                continue
            if self._time_short(slot, phase):
                if slot.id not in self.dropped:
                    self.dropped.append(slot.id)
                    self._event("time_cut", item=slot.ask, phase=phase, remaining=round(self.budget.remaining()),
                                turn_s=round(self.budget.turn_s, 1), call_s=round(self.budget.call_s, 1))
                continue
            if slot.kind == "EXAM" and self._sexual_exam(slot) and not self._sexual_ok():
                self._event("exam_skipped", text=slot.ask, why="pelvic or speculum exam: not sexually active or a minor")
                continue
            if self.cfg.core and phase == "ros" and slot.codes:  # the core already knows the answer
                known = set(self.state.get("asked_codes", [])) | {f["code"] for f in self.state.get("findings", [])}
                if all(c in known for c in slot.codes):
                    continue
            if slot.kind == "EXAM" and self.cfg.exam_intro and not self.exam_announced:
                yield from self._exam_intro()
                if self._stop(self._test_reserve()):
                    break
            yield from self._slot(slot, self._compose(slot, phase))

    def _time_short(self, slot: CL.Slot, phase: str) -> bool:
        """At the observed pace, asking this item would leave too little time for what matters more: priority-2
        history yields to every planned maneuver and the end of the visit, priority-1 history and priority-2 maneuvers
        to the priority-1 maneuvers and the end of the visit; a priority-1 maneuver only to the end of the visit."""
        b = self.budget
        exams = [s for s in self.plan.exams if s.ask not in self.performed and s is not slot]
        if phase == "exam" or slot.kind == "EXAM":
            keep = [s for s in exams if s.tier <= 1] if slot.tier >= 2 else []
        else:
            keep = exams if slot.tier >= 2 else [s for s in exams if s.tier <= 1]
        return b.remaining() - b.turn_s < len(keep) * b.turn_s + max(self.cfg.wrap_margin_s, b.close_s())

    def _compose(self, slot: CL.Slot, phase: str) -> str:
        """The utterance for a slot: the question, with at most one prefix that fits in 30 characters, in this order:
        the greeting (by name or guardian title) on the first question; a promise to explain when the patient just
        asked something; an acknowledgment of worry or distress the patient just voiced; one acknowledgment early in
        the history; a signpost at the start of the background or the review (tried on the block's first 3
        questions); a plain acknowledgment of a substantive answer, at most every third turn."""
        text = slot.ask
        if slot.kind != "SAY":
            return text
        cfg, limit = self.cfg, self.cfg.say_max_chars

        def fits(prefix: str) -> bool:
            return bool(prefix) and check_say(f"{prefix} {text}", limit)[0]

        light, self.light_pending = self.light_pending, False
        if not self.greeted:
            self.greeted = True
            name = self.ctx.get("address", "")
            for g in ([f"{CL.GREETING[:-1]}, {name}."] if name else []) + [CL.GREETING]:
                if fits(g):
                    return f"{g} {text}"
            return text
        if self.asked_back:
            self.asked_back = False
            for line in CL.ASKED_BACK:
                if fits(line):
                    self.last_ack_turn = self.turns
                    return f"{line} {text}"
        if self.ack_pending:
            ack, self.ack_pending = self.ack_pending, ""
            if fits(ack):
                self.acks += 1
                self.last_ack_turn = self.turns
                self.empathy_done = True
                return f"{ack} {text}"
        if cfg.empathy and not self.empathy_done and phase == "hpi":
            line = empathy_line(self.ctx.get("opening", ""), self.ctx.get("pediatric", False),
                                bool(self.ctx.get("pain")))
            if fits(line):
                self.empathy_done = True
                self.last_ack_turn = self.turns
                return f"{line} {text}"
        if cfg.signposts and phase in CL.SIGNPOSTS and phase not in self.signposted:
            tries = self.signpost_tries[phase] = self.signpost_tries.get(phase, 0) + 1
            for sign in CL.SIGNPOSTS[phase]:
                if fits(sign):
                    self.signposted.add(phase)
                    return f"{sign} {text}"
            if tries >= 3:
                self.signposted.add(phase)
        if light and self.light_acks < cfg.light_acks_max and self.turns - self.last_ack_turn >= 3:
            line = CL.LIGHT_ACKS[self.light_acks % len(CL.LIGHT_ACKS)]
            if fits(line):
                self.light_acks += 1
                self.last_ack_turn = self.turns
                return f"{line} {text}"
        return text

    _PELVIC = re.compile(r"(질경|골반\s?내진|자궁경부\s?운동)")

    def _sexual_exam(self, slot: CL.Slot) -> bool:
        pb = self.playbook
        return bool((pb is not None and pb.needs_sexual(slot.id)) or self._PELVIC.search(slot.ask))

    def _sexual_ok(self) -> bool:
        """A speculum or bimanual exam only for a patient who is sexually active: never for a child; for an adolescent
        (13-17) only after a "yes" to the sexual-activity question; for an adult unless she denied it."""
        if self.ctx.get("pediatric"):
            return False
        pb = self.playbook
        gate = next((s for s in (pb.history if pb is not None else ()) if s.id == pb.gate_id), None)
        if gate is None:
            gate_entries = [e for e in self.ledger.accepted() if e.kind == "SAY" and "성관계" in (e.text or "")]
        else:
            gate_entries = [e for e in self.ledger.accepted() if e.kind == "SAY" and gate.ask in (e.text or "")]
        gate_entries = [e for e in gate_entries if e.section != "bundled"]
        if gate_entries:
            return positive_answer(gate_entries[-1].reply, gate_entries[-1].text)
        age = self.ctx.get("age")
        return age is None or age >= 18

    def _follow_up(self, slot: CL.Slot, entry: Entry, positive: Optional[bool] = None) -> Generator[Action, Observation, None]:
        """The playbook's follow-ups (`then`) right after a positive answer: how much, how long, escalating, tried to
        stop. A follow-up whose `known` pattern is already in the answer, or in a reply since, is skipped."""
        pb = self.playbook
        kids = pb.followups_for(slot) if pb is not None else None
        if not kids or not (positive if positive is not None else positive_answer(entry.reply, entry.text)):
            return
        asked = []
        for child, known, safety in kids:
            if self._stop() or (not safety and self.followups >= self.cfg.followup_max):
                continue
            since = " ".join(e.reply or "" for e in self.ledger.accepted()
                             if e.kind == "SAY" and e.turn is not None and entry.turn is not None and e.turn >= entry.turn)
            if child.ask in self.performed or (known and re.search(known, f"{entry.reply or ''} {since}")):
                continue
            if not safety and self._already_told(child) is not None:
                continue
            self.followups += 0 if safety else 1  # a safety follow-up (suicide, ...) is never capped
            asked.append(child.ask)
            yield from self._slot(child)
        if asked:
            self._event("follow_up", parent=slot.id, asked=asked)

    def _repeat(self, question: str) -> bool:
        """The question was already asked in other words (an accepted question with the same content stems)."""
        return any(same_question(question, e.ask or e.text) for e in self.ledger.entries
                   if e.kind == "SAY" and e.accepted and e.source not in ("volunteered", "closing", "frame"))

    def _already_told(self, slot: CL.Slot) -> Optional[tuple[Entry, bool]]:
        """A question that a reply (or the opening) has already answered is not asked again: a presence question about
        symptoms already reported or denied, or a specific background question whose topic the general one's answer
        named. The reply is recorded under the question's chart item (source "volunteered", at that reply's turn) and
        the question counts as done. Returns (entry, reported) or None."""
        if not self.cfg.skip_answered or slot.ask in self.performed:
            return None
        entries = self.ledger.entries
        sym = reported_symptom(slot.ask, entries)
        topic = None if sym is not None else reported_topic(slot.ask, slot.section, entries)
        if sym is None and topic is None:
            return None
        turn, reply = (sym[0], sym[1]) if sym is not None else topic
        signs = sym[2] if sym is not None else [True]
        n = len(slot.item.split("/")) if slot.item else 0
        if sym is not None and slot.item and len(set(signs)) == 1 and n in (1, len(signs)):
            note = f"{slot.item} ({'/'.join(['+' if signs[0] else '-'] * n)})"
        else:
            said = re.sub(r"\s+", " ", reply.strip())[:80]
            note = f"{slot.item or slot.label} : {said}"
        self.performed.add(slot.ask)
        e = self.ledger.add(turn=turn, kind="SAY", section=slot.section, label=slot.label, text=slot.ask, reply=reply,
                            source="volunteered", item=slot.item, note=note, ask=slot.ask)
        self.skipped.append({"ask": slot.ask, "turn": turn, "note": note})
        self._event("already_told", text=slot.ask, turn=turn, note=note)
        return e, any(signs)

    def _exam_intro(self) -> Generator[Action, Observation, None]:
        """One sentence before the first maneuver: what will be examined, and to say if it hurts (the time-out's
        wording when it gave a valid one, else a fixed line)."""
        self.exam_announced = True
        text = self._exam_intro_text()
        res = yield from self._do("SAY", text, "education", "진찰 설명", [], "frame", ask=text)
        if res and res[0].accepted:
            self.exam_intro_said.append((text, self.turns))

    def _exam_intro_text(self) -> str:
        limit = self.cfg.say_max_chars
        cand = (getattr(self.review, "exam_intro_ko", "") or "").strip() if self.review is not None else ""
        if cand and check_say(cand, limit)[0] and "?" not in cand and re.search(r"[가-힣]", cand) \
                and not (self.ctx.get("pediatric") and _ADULT_ONLY.search(cand)) and injection_free(cand):
            return cand
        return CL.EXAM_INTRO_CHILD if self.ctx.get("pediatric") else CL.EXAM_INTRO

    def _listen(self, kind: str, entry: Entry) -> None:
        """After a reply: queue an acknowledgment when the patient voices worry, distress or severe pain, a plain one
        ("그러셨군요.") now and then after a substantive answer, and a promise to explain when the patient asks
        something (the question is answered in the closing explanation)."""
        if kind != "SAY" or not entry.accepted or entry.section in ("bundled",):
            return
        if patient_question(entry.reply):
            self.asked_back = True
            self.patient_asks.append((self.turns, (entry.reply or "").strip()))
        if entry.section == "education":
            return
        ack = acknowledgment(entry.reply)
        if ack and self.acks < self.cfg.cue_empathy_max and self.turns - self.last_ack_turn >= 3:
            self.ack_pending = ack
        elif len((entry.reply or "").strip()) >= 12 and positive_answer(entry.reply, entry.text):
            self.light_pending = True

    # -- the focus call and the selection -------------------------------------------

    def _focus(self) -> None:
        """One call after the history of the complaint: the leading diagnoses, the optional items not worth asking
        for this patient, the priority-3 items worth adding, and a few questions and maneuvers no list covers."""
        cfg = self.cfg
        if cfg.core or not cfg.focus_call:
            return
        lists = self.plan.hpi + self.plan.background + self.plan.ros + self.plan.exams
        todo = [s for s in lists if s.ask not in self.performed]
        planned = [s.ask for s in todo if s.tier <= 1]
        optional = {f"o{i}": s for i, s in enumerate(x for x in todo if x.tier == 2)}
        extra = {f"x{i}": s for i, s in enumerate(x for x in todo if x.tier >= 3)}
        try:
            out = self.llm.structured(
                FocusPlan,
                FOCUS_SYSTEM.format(n_q=cfg.focus_questions_max, n_e=cfg.focus_exams_max,
                                    guardian=GUARDIAN_RULE if self.ctx.get("pediatric") else "", exam_names=EXAM_NAMES),
                focus_user(self.ctx.get("opening", ""), self.ctx.get("vitals_line", ""), self.ctx.get("age"),
                           self.ctx.get("sex"), self._presentation_ref(), self.ledger.transcript(5000), planned,
                           [(i, s.ask) for i, s in optional.items()], [(i, s.ask) for i, s in extra.items()],
                           turns=(self.turns, self.turns + len(planned) + len(optional) + cfg.closing_lines
                                  + self._intro_turns(),
                                  cfg.turn_target, cfg.max_turns)))
        except Exception as exc:
            self._event("focus_failed", error=str(exc)[:160])
            return
        self.focus = out
        self.focus_skip = {optional[i].id for i in out.skip if i in optional}
        self.focus_add = {extra[i].id for i in out.add if i in extra}
        leading = [{"en": d.name_en.strip(), "ko": (d.name_ko or "").strip(), "why": (d.why or "").strip()}
                   for d in out.leading if (d.name_en or "").strip()]
        if leading:
            self.leading = leading[:5]
        qs, es = self._new_items(out.questions, out.exams, cfg.focus_questions_max, cfg.focus_exams_max, "focus")
        self.focus_new = qs + es
        self._insert(self.focus_new)
        self._event("focus", leading=[d["en"] for d in self.leading], skip=len(self.focus_skip),
                    add=len(self.focus_add), questions=len(qs), exams=len(es))

    def _select(self) -> None:
        """Keep priority 1; priority 2 unless the focus call skipped it; priority 3 only if it added it. Turn target
        (about 40, closing lines included) is the aim the focus call trims toward; the plan is cut only when it would run
        past the soft cap (gathering must end there so the closing lines fit under the 50-turn limit): priority-2
        items go first from the review, then the background, the history and the maneuvers."""
        cfg = self.cfg
        new = {s.id for s in self.focus_new}

        def keep(s: CL.Slot) -> bool:
            if s.id in new or s.tier <= 1:
                return True
            return s.id not in self.focus_skip if s.tier == 2 else s.id in self.focus_add

        names = ("hpi", "background", "ros", "exams")
        for n in names:
            lst = getattr(self.plan, n)
            self.dropped += [s.id for s in lst if s.ask not in self.performed and not keep(s)]
            setattr(self.plan, n, [s for s in lst if s.ask in self.performed or keep(s)])

        def left() -> int:
            return sum(1 for n in names for s in getattr(self.plan, n) if s.ask not in self.performed)

        avail = min(cfg.soft_turn_cap, cfg.gather_end()) - self.turns
        for n in ("ros", "background", "hpi", "exams"):
            lst = getattr(self.plan, n)
            for i in range(len(lst) - 1, -1, -1):
                if left() <= avail:
                    break
                s = lst[i]
                if s.tier >= 2 and s.id not in new and s.ask not in self.performed:
                    self.dropped.append(s.id)
                    if s.kind == "SAY":
                        self.cut.append(s)
                    del lst[i]
        self._event("select", planned=left(), target=avail, dropped=len(self.dropped))

    def _review(self) -> None:
        """The diagnostic time-out (one call after the history and the review of systems, before the examination): the
        leading diagnoses with findings for and against, the can't-miss causes still open, whether the leader names a
        cause, up to a few questions that would most change the ranking (asked now), and the planned maneuvers ranked
        by what they separate (optional ones that would not change the assessment are skipped, a few new ones added)."""
        cfg = self.cfg
        if self.reviewed or cfg.core or not cfg.review_call or self.env_done:
            return
        self.reviewed = True
        todo = [s for s in self.plan.exams if s.ask not in self.performed]
        ids = {f"m{i}": s for i, s in enumerate(todo)}
        end = cfg.gather_end()
        must = sum(1 for s in todo if s.tier <= 1)  # an upper bound: unused offers leave room for the announcement
        n_q = max(0, min(cfg.review_questions_max, end - self.turns - must))
        n_e = max(0, min(cfg.review_exams_max, end - self.turns - must - n_q))
        hint = "; ".join(d["en"] for d in self.leading)
        try:
            out = self.llm.structured(
                ReviewPlan,
                REVIEW_SYSTEM.format(n_q=n_q, n_e=n_e, guardian=GUARDIAN_RULE if self.ctx.get("pediatric") else "",
                                     exam_names=EXAM_NAMES),
                review_user(self.ctx.get("opening", ""), self.ctx.get("vitals_line", ""), self.ctx.get("age"),
                            self.ctx.get("sex"), self._presentation_ref(), self.ledger.transcript(6000), hint,
                            [(i, s.ask, s.tier) for i, s in ids.items()], (self.turns, end),
                            [s.ask for s in dict.fromkeys(self.cut)
                             if s.ask not in self.performed and s.id not in self.focus_skip]))
        except Exception as exc:
            self._event("review_failed", error=str(exc)[:160])
            return
        self.review = out
        leading = [{"en": d.name_en.strip(), "ko": (d.name_ko or "").strip(), "why": (d.why or "").strip(),
                    "against": (d.against or "").strip()} for d in out.leading if (d.name_en or "").strip()]
        if leading:
            self.leading = leading[:5]
        self.cant_miss_open = [x.strip() for x in out.cant_miss_open if x and x.strip()][:5]
        qs, es = self._new_items(out.questions, out.exams, n_q, n_e, "review")
        qs = [replace(q, tier=1) for q in qs]
        es = [replace(e, tier=1) for e in es]
        self.review_new = qs + es
        # the exam: ranked planned maneuvers first, optional ones judged uninformative dropped, new ones after the
        # priority-1 maneuvers
        order = [ids[i] for i in dict.fromkeys(out.exam_order) if i in ids]
        ranked = order + [s for s in todo if s not in order]
        skip = {ids[i].id for i in out.skip_exams if i in ids and ids[i].tier >= 2}
        kept = [s for s in ranked if s.id not in skip]
        self.dropped += sorted(skip)
        first = [s for s in kept if s.tier <= 1]
        done = [s for s in self.plan.exams if s.ask in self.performed]
        self.plan.exams = done + first + es + [s for s in kept if s.tier >= 2]
        self._event("review", leading=[d["en"] for d in self.leading], cant_miss_open=self.cant_miss_open,
                    names_cause=out.names_cause, questions=[q.ask for q in qs], exams=[e.ask for e in es],
                    skipped=len(skip), order=[s.ask for s in self.plan.exams if s.ask not in self.performed])

    def _backfill(self) -> list[CL.Slot]:
        """Turns still free once the time-out's questions are asked (the follow-ups and the time-out used fewer than
        were kept for them) go back to the priority-2 history items cut only for turns, in plan order, before the
        exam; items the focus call judged uninformative stay out."""
        if self.cfg.core or not self.cut:
            return []
        end = self.cfg.gather_end()
        exams = sum(1 for s in self.plan.exams if s.ask not in self.performed)
        room = end - self.turns - exams - self._intro_turns() - self._test_reserve()
        out = [s for s in dict.fromkeys(self.cut) if s.ask not in self.performed and s.id not in self.focus_skip]
        out = out[:max(0, room)]
        if out:
            self._event("backfill", items=[s.ask for s in out], room=room)
            self.dropped = [d for d in self.dropped if d not in {s.id for s in out}]
        return out

    # -- core mode ------------------------------------------------------------------

    def _core_update(self, force: bool = False) -> None:
        """Core mode: read the answers of the last block (one extraction call), then hypotheses and belief."""
        if not self.cfg.core:
            self.pending = []
            return
        self._flush()
        self._hypothesize(force=force)
        self._refresh()

    def _core_diagnostics(self, kind: str, n_max: int) -> Generator[Action, Observation, None]:
        """Core mode: questions, maneuvers or tests chosen by value of information (else the question bank)."""
        if not self.cfg.core:
            return
        n = 0
        while n < n_max and not self._stop(self._test_reserve() if kind != "test" else 0):
            if kind == "test" and self.tests_blocked:
                break
            act = self._next_diagnostic(kind)
            if act is None:
                break
            yield from self._diagnostic(act)
            n += 1

    # -- one action -----------------------------------------------------------

    def _do(self, kind: str, text: str, section: str, label: str, codes: list[str], source: str, item: str = "",
            ask: str = ""):
        """Validate, send, record. Returns (observation, entry) or None if the action was not sent.

        A refusal (error code, or accepted=False) is recorded as not accepted; it costs a turn only when the environment
        says so (info["turn_used"]: a wrongly sent SAY/EXAM, protocol.TURN_USING_CODES)."""
        text = (text or "").strip()
        limit = self.cfg.say_max_chars
        ok, why = (check_say(text, limit) if kind == "SAY" else check_exam(text, limit) if kind == "EXAM"
                   else check_test(text, limit))
        if not ok:
            self._event("blocked", action=kind, text=text, why=why)
            return None
        if kind == "TEST" and (not self.cfg.allow_tests or self.tests_blocked):
            return None
        if self.env_done or self.turns >= self.cfg.max_turns:
            return None
        item = item or CL.ITEM_BY_ASK.get(text, "")
        self.performed.add(text)
        t_sent = self.budget.clock()
        obs = yield Action(kind=kind, text=text, meta={"section": section, "codes": codes, "source": source})
        self.budget.note_action(self.budget.clock() - t_sent)
        if obs is None:
            obs = Observation(accepted=False)
        elif isinstance(obs, dict):
            obs = observation_from_dict(obs)
        elif isinstance(obs, str):
            obs = Observation(text=obs)
        if obs.code in REFUSAL_CODES:
            obs.accepted = False
        used = obs.accepted or bool((obs.info or {}).get("turn_used"))
        if used:
            self.turns = obs.turn if isinstance(obs.turn, int) and obs.turn > self.turns else self.turns + 1
            self.state["turn"] = self.turns
        if not obs.accepted:
            self._event("refused", action=kind, text=text, code=obs.code, reply=obs.text[:120], turn_used=used)
            if obs.code == "ACTION_NOT_AVAILABLE" and kind == "TEST":
                self.tests_blocked = True  # the server takes no tests in this round: stop trying
        entry = self.ledger.add(turn=self.turns if obs.accepted else None, kind=kind, section=section, label=label,
                                text=text, reply=obs.text, accepted=obs.accepted, codes=list(codes), source=source,
                                item=item, ask=ask or text)
        if obs.accepted and kind == "SAY" and BUNDLE_REPLY.search(obs.text or ""):
            entry.section = "bundled"  # "한 번에 하나씩 물어봐 주세요.": not an answer, never written into the note
            self._event("bundled", text=text)
        self._listen(kind, entry)
        if obs.accepted:
            role = "patient" if kind == "SAY" else "clinical_team"
            self.state["transcript"] = self.state.get("transcript", []) + [
                {"role": "doctor", "type": kind, "text": text, "turn": self.turns},
                {"role": role, "type": kind, "text": obs.text, "turn": self.turns}]
        if obs.done:
            self.env_done = True
        return obs, entry

    def _mark_asked(self, codes: list[str], action_ids: list[str] = ()) -> None:
        st = self.state
        st["asked_codes"] = sorted(set(st.get("asked_codes", [])) | set(codes))
        ids = list(action_ids) + [f"ask_{c}" for c in codes] + [f"exam_{c}" for c in codes]
        st["done_action_ids"] = list(dict.fromkeys(st.get("done_action_ids", []) + ids))

    def _slot(self, slot: CL.Slot, text: Optional[str] = None):
        text = text or slot.ask
        if text in self.performed or slot.ask in self.performed:
            return
        codes = [c for c in slot.codes if self.store.finding_def(c) is not None]
        res = yield from self._do(slot.kind, text, slot.section, slot.label, codes, "slot", item=slot.item, ask=slot.ask)
        if res is None:
            return
        self.performed.add(slot.ask)  # the question itself, whatever prefix it was asked with
        obs, entry = res
        self._mark_asked(codes)
        if not obs.accepted:
            return
        typ = "exam" if slot.kind == "EXAM" else "history"

        def pend(e: Entry) -> None:
            self.pending.append(({"id": f"slot_{slot.id}", "name": slot.label, "type": typ, "reveals": codes,
                                  "cost": DEFAULT_ACTION_COST[typ], "risk": 0.0}, e))

        if entry.section == "bundled":
            for e in (yield from self._split(slot.alt, slot.section, slot.label, codes)):
                pend(e)
            return
        pend(entry)
        if slot.kind == "SAY":
            yield from self._follow_up(slot, entry)
        if slot.id == "smoking" and self.cfg.smoking_followup:
            if _smoker_without_amount(obs.text) and not self._stop():
                yield from self._slot(CL.SMOKING_AMOUNT)
            if _smoked_without_years(obs.text) and positive_answer(obs.text, text) and not self._stop():
                yield from self._slot(CL.SMOKING_YEARS)

    def _split(self, alt: tuple[tuple[str, str], ...], section: str, label: str, codes: list[str]):
        """Ask each single-topic part of a question the patient wanted one at a time. Returns the answered entries."""
        out: list[Entry] = []
        for ask, item in alt:
            if self._stop():
                break
            r = yield from self._do("SAY", ask, section, label, codes, "slot", item=item)
            if r and r[0].accepted and r[1].section != "bundled":
                out.append(r[1])
        return out

    def _flush(self) -> None:
        """Core mode: extract findings from all answers not yet read (one LLM call per channel)."""
        if not self.cfg.core:
            self.pending = []
        if not self.pending:
            return
        for typ in ("history", "exam"):
            items = [(a, e) for a, e in self.pending if a["type"] == typ]
            if not items:
                continue
            if typ == "history":
                reply = "\n".join(f"Q: {e.text}\nA: {e.reply}" for _, e in items)
            else:
                reply = "\n".join(f"Exam: {e.text}\nResult: {e.reply}" for _, e in items)
            self.state["last_actions"] = [a for a, _ in items]
            self.state["last_reply"] = reply
            self._node("perceive")
        self.pending = []

    def _diagnostic(self, act: dict[str, Any]):
        kind = {"history": "SAY", "exam": "EXAM"}.get(act["type"], "TEST")
        section = act.get("section") or ("exam" if kind == "EXAM" else "test" if kind == "TEST" else "hpi")
        label = act["text"] if kind == "SAY" else exam_label(act["text"])  # Korean, for the note
        res = yield from self._do(kind, act["text"], section, label, act.get("reveals", []), act.get("source", "core"))
        st = self.state
        self._mark_asked(act.get("reveals", []) + act.get("same_codes", []), [act["id"]])
        if res is None:
            return
        obs, entry = res
        st["cost_spent"] = st.get("cost_spent", 0.0) + float(act.get("cost", 0.0))
        if not obs.accepted:
            return
        if entry.section == "bundled":
            parts = yield from self._split(CL.ALT_BY_ASK.get(entry.text, ()), section, label, act.get("reveals", []))
            if not parts:
                return
            obs, entry = Observation(text=" / ".join(e.reply for e in parts), accepted=True), parts[-1]
        reveals = act.get("reveals", [])
        if kind == "SAY" and len(reveals) == 1 and self.cfg.yes_no_fast_path:
            yn = yes_no(obs.text, entry.text)
            if yn is not None:
                fd = self.store.finding_def(reveals[0])
                f = {"code": reveals[0], "label": fd.label if fd else act["name"], "present": yn[0],
                     "confidence": yn[1], "source": "history", "evidence": obs.text[:80], "turn": self.turns,
                     "chief_complaint": False}
                st["findings"] = merge_findings(st.get("findings", []), [f])
                self._event("fast_path", code=reveals[0], present=yn[0])
                self._refresh()
                return
        a = {k: act[k] for k in ("id", "name", "type", "reveals", "test", "cost", "risk") if k in act}
        a["reveals"] = reveals + act.get("same_codes", [])
        st["last_actions"] = [a]
        st["last_reply"] = obs.text if kind == "SAY" else f"{entry.text}\n{obs.text}"
        self._node("perceive")
        self._hypothesize()
        self._refresh()

    # -- choosing the next diagnostic action ----------------------------------

    def _leader_trusted(self) -> bool:
        """The core's leader decides only if it is card-backed, at least core_trust, and the model proposed it too
        (a disease the core added on its own is a candidate to ask about, not a diagnosis to impose)."""
        diff = self.state.get("differential") or []
        if not diff:
            return False
        d = diff[0]
        proposed = {h["disease_id"] for h in self.state.get("hypotheses", [])
                    if not str(h.get("prior_source", "")).startswith("reverse_index")}
        return (bool(d.get("has_card")) and d["prob"] >= self.cfg.core_trust and d.get("card_status") not in ("", "none")
                and d["disease_id"] in proposed)

    def _core_action(self, types: tuple[str, ...]):
        """One action from the core planner restricted to `types`: a dict, "commit", or None (nothing scored)."""
        if not self.state.get("differential"):
            return None
        saved = self.settings.allowed_action_types
        self.settings.allowed_action_types = tuple(t for t in types if t in saved) or ("__none__",)
        try:
            upd = self.nodes.plan(dict(self.state))
        except Exception as exc:
            self._error("plan", exc)
            return None
        finally:
            self.settings.allowed_action_types = saved
        for k in ("action_log", "stall_count", "actions_taken", "unresolved_objection"):
            if k in upd:
                self.state[k] = upd[k]
        if upd.get("phase") == "committing":
            self.core_committed = True  # the core would commit now (latest decision wins)
            self.commit_reason = upd.get("commit_reason", "")
            return "commit"
        self.core_committed = False
        acts = upd.get("last_actions") or []
        if not acts or acts[0].get("free"):
            return None
        return acts[0]

    def _phrase(self, a: dict[str, Any]) -> Optional[str]:
        limit = self.cfg.say_max_chars
        if a.get("test"):
            td = self.store.test_def(a["test"])
            text = (td.ask_ko or f"{td.ko or td.name} 검사를 하겠습니다.") if td else ""
            return text if text and check_test(text, limit)[0] else None
        code = (a.get("reveals") or [""])[0]
        fd = self.store.finding_def(code)
        kind = "exam" if a["type"] == "exam" else "history"
        text = fd.ask_ko if fd and fd.ask_ko else self.phrase_cache.get((kind, code), "")
        if not text:
            try:
                out = self.llm.structured(PhraseOutput, PHRASE_SYSTEM, phrase_user(kind, fd.label if fd else a["name"]))
                text = out.text_ko.strip()
            except Exception as exc:
                self._event("phrase_failed", code=code, error=str(exc)[:120])
                return None
        ok = check_say(text, limit)[0] if kind == "history" else check_exam(text, limit)[0]
        if not ok:
            return None
        self.phrase_cache[(kind, code)] = text
        return text

    def _next_diagnostic(self, kind: str) -> Optional[dict[str, Any]]:
        types = {"question": ("history",), "exam": ("exam",), "test": TEST_TYPES}[kind]
        for _ in range(8):
            a = self._core_action(types)
            if not isinstance(a, dict):
                break
            text = self._phrase(a)
            if text is not None and self.ctx.get("pediatric") and _ADULT_ONLY.search(text):
                text = None  # a child's guardian is not asked about smoking, alcohol, periods or work
            if text is None or text in self.performed:
                # the same maneuver was already done (one auscultation reveals several codes) or unphrasable
                self._mark_asked(a.get("reveals", []), [a["id"]])
                continue
            same = [c for c in self._codes_with_text(text) if c not in a.get("reveals", [])]
            return {**a, "text": text, "label": a["name"], "source": "core", "same_codes": same}
        if self.core_committed and self._leader_trusted():
            return None
        return self._bank_pop(kind)

    def _codes_with_text(self, text: str) -> list[str]:
        """Registry codes whose Korean utterance is this exact text (e.g. lung auscultation: crackles, wheeze, ...)."""
        return [c for c, fd in self.store._findings.items() if fd.ask_ko == text]

    def _bank_pop(self, kind: str) -> Optional[dict[str, Any]]:
        for attempt in range(2):
            q = self.bank[kind]
            while q:
                item = q.pop(0)
                if item["text"] not in self.performed:
                    return item
            if attempt == 0 and self.bank_calls < self.cfg.bank_max_calls:
                self._fill_bank()
        return None

    def _fill_bank(self) -> None:
        self.bank_calls += 1
        if self.pending:
            self._flush()
        st = self.state
        vocab = {}
        for c, v in self.store.vocabulary([d["disease_id"] for d in st.get("differential", [])[:6]] or None).items():
            vocab[c] = v
        try:
            out = self.llm.structured(
                QuestionBank,
                BANK_SYSTEM.format(tests_rule=TESTS_RULE_FINAL if self.cfg.allow_tests else TESTS_RULE_PRELIM),
                bank_user(st.get("problem_rep", ""), st.get("differential", []), self.ledger.transcript(6000),
                          sorted(self.performed), vocab, 4, 3, 3 if self.cfg.allow_tests else 0))
        except Exception as exc:
            self._event("bank_failed", error=str(exc)[:160])
            return
        n = 0
        for it in out.items:
            text = (it.text_ko or "").strip()
            kind = it.kind if it.kind in ("question", "exam", "test") else "question"
            if kind == "test" and not self.cfg.allow_tests:
                continue
            ok = (check_say(text, self.cfg.say_max_chars)[0] if kind == "question"
                  else check_exam(text, self.cfg.say_max_chars)[0])
            if (self.ctx.get("pediatric") and _ADULT_ONLY.search(text)) or (kind == "question" and not single_topic(text)):
                ok = False
            if not ok or text in self.performed:
                continue
            code = slug(it.code or it.target)
            typ = {"question": "history", "exam": "exam", "test": "lab"}[kind]
            n += 1
            self.bank[kind].append({
                "id": f"bank_{kind}_{self.bank_calls}_{n}", "name": it.target or text, "type": typ,
                "reveals": [code] if code and kind != "test" else [], "test": None,
                "cost": DEFAULT_ACTION_COST[typ], "risk": 0.0, "text": text, "label": it.target or text,
                "source": "bank", "section": it.section if kind == "question" else ("exam" if kind == "exam" else "test")})
        self._event("bank", items=n)

    def _hypothesize(self, force: bool = False) -> None:
        if self.hyp_calls >= self.cfg.hypothesize_max:
            return
        if not force and self.turns - self.last_hyp_turn < self.cfg.hypothesize_every:
            return
        before = self.budget.calls
        self._node("hypothesize")
        if self.budget.calls > before:
            self.hyp_calls += 1
            self.last_hyp_turn = self.turns

    # -- closing ----------------------------------------------------------------

    def _close(self) -> Generator[Action, Observation, None]:
        self._flush()
        self._refresh()
        if self.cfg.core and self.cfg.critic and self.state.get("differential"):
            self._node("critic", force=True)
            if self.state.get("critic_added"):
                self._refresh()
            veto = self.state.get("veto") or {}
            if veto.get("turn") == self.state.get("turn") and not self.env_done and \
                    self.turns < self.cfg.max_turns - self.cfg.closing_lines - 1 and \
                    self.budget.remaining() > self.cfg.wrap_margin_s / 2:
                types = ("history", "exam") + (TEST_TYPES if self.cfg.allow_tests else ())
                a = self._core_action(types)
                if isinstance(a, dict):
                    text = self._phrase(a)
                    if text and text not in self.performed:
                        self._event("veto_action", alternative=veto.get("alternative"), text=text)
                        yield from self._diagnostic({**a, "text": text, "label": a["name"], "source": "critic"})
        self._assess()
        for line in self._closing_lines():
            if self.env_done or self.turns >= self.cfg.max_turns or self.budget.remaining() < self.cfg.final_margin_s:
                break
            res = yield from self._do("SAY", line, "education", "환자 설명", [], "closing")
            if res and res[0].accepted:
                self.closing_said.append((line, self.turns))

    def _section_text(self, section: str) -> str:
        return " / ".join(e.reply for e in self.ledger.section(section))

    def _rank_tests(self) -> list[tuple[str, str, float]]:
        """Tests by value of information over the current differential: ordered in the final round, and in the
        preliminary round written into the plan ("the test that would settle it")."""
        diff = self.state.get("differential") or []
        if not diff:
            return []
        try:
            tested = {t["test_id"] for t in self.state.get("test_results", [])}
            ids = [d["disease_id"] for d in diff[: self.settings.top_k_load]]
            acts = [a for a in self.store.actions_for(ids) if a.test and a.test not in tested]
            model = PL._Model(diff, self.state.get("other_prob", 0.0), self.store, self.settings)
            scored = sorted((model.score_test(a) for a in acts), key=lambda s: s.voi, reverse=True)
            out = []
            for s in scored:
                td = self.store.test_def(s.action.test)
                if s.voi > 0 and td:
                    out.append((td.id, td.ko or td.name, round(s.voi, 4)))
            return out[:5]
        except Exception as exc:
            self._error("rank_tests", exc)
            return []

    def _separating_test(self, a_id: str, b_id: str) -> Optional[str]:
        """The registry test whose outcome distributions differ most between two diseases (total variation)."""
        best, name = 0.0, None
        for tid in self.store.tests_for(a_id):
            td = self.store.test_def(tid)
            if td is None:
                continue
            pa, pb = self.store.test_dist_eff(a_id, tid), self.store.test_dist_eff(b_id or None, tid)
            tv = 0.5 * sum(abs(pa.get(o, 0) - pb.get(o, 0)) for o in td.outcomes)
            if tv > best:
                best, name = tv, td.ko or td.name
        return name

    def _confirm_test(self, disease_id: str) -> Optional[str]:
        """The registry test that most sharply confirms a disease: max KL(P(outcome | disease) || background)."""
        import math

        best, name = 0.0, None
        for tid in self.store.tests_for(disease_id):
            td = self.store.test_def(tid)
            if td is None:
                continue
            p = self.store.test_dist_eff(disease_id, tid)
            kl = sum(p[o] * math.log(p[o] / td.background[o]) for o in td.outcomes if p.get(o, 0) > 0 and td.background.get(o, 0) > 0)
            if kl > best:
                best, name = kl, td.ko or td.name
        return name

    def _candidates(self) -> str:
        """The leading diagnoses so far for the assessment prompt: the core's computed differential (core mode), else
        the focus call's (or the topic plan's) leading diagnoses."""
        diff = self.state.get("differential") or []
        if diff:
            return "\n".join(f"- {d['name']}: {d['prob']:.2f}" + ("" if d.get("has_card") else " (no knowledge card)")
                             for d in diff[:6])
        return "\n".join(f"- {d['en']}" + (f" ({d['ko']})" if d.get("ko") else "") + (f": {d['why']}" if d.get("why") else "")
                         + (f"; against: {d['against']}" if d.get("against") else "") for d in self.leading)

    def _concern(self) -> str:
        """The patient's answer to the concern question ("" if not asked or not answered)."""
        e = next((e for e in self.ledger.accepted() if e.item == CL.CONCERN.item and e.section != "bundled"), None)
        reply = (e.reply or "").strip() if e else ""
        yn = yes_no(reply)
        return "" if yn is not None and not yn[0] else reply  # "아니요, 없어요": no stated worry

    def _patient_profile(self) -> str:
        """Who the doctor is talking to, and what they worry about or ask for (for the closing lines)."""
        parts = []
        if self.ctx.get("address"):
            parts.append(f"address as '{self.ctx['address']}'")
        if self.ctx.get("pediatric"):
            parts.append("the speaker is the child's guardian")
        concern = self._concern()
        if concern:
            parts.append(f"stated worry: \"{concern[:120]}\"")
        asks = [r for _, r in self.patient_asks if r != concern][-2:]
        if asks:
            parts.append("the patient asked (answer it in the closing lines): "
                         + " / ".join(f"\"{r[:120]}\"" for r in asks))
        return "; ".join(parts)

    def _assess(self) -> None:
        st = self.state
        diff = st.get("differential") or []
        leader = diff[0] if diff else None
        policy = self.cfg.final_dx
        fixed = leader["name"] if leader and (policy == "core" or (policy == "auto" and self._leader_trusted())) else None
        fixed_id = leader["disease_id"] if fixed else None
        self.tests_ranked = self._rank_tests()
        self.dangerous = []
        for d in diff[:8]:
            if d.get("cant_miss") and d["prob"] >= self.settings.exclusion_threshold and d["disease_id"] != fixed_id:
                t = self._separating_test(d["disease_id"], fixed_id or (leader["disease_id"] if leader else ""))
                self.dangerous.append({"name": d["name"], "disease_id": d["disease_id"], "tests": [t] if t else []})
        a: Optional[Assessment] = None
        if self.budget.remaining() > self.cfg.final_margin_s:
            guides = self._guideline_excerpts(fixed)
            pb = self.playbook
            open_dx = list(dict.fromkeys([d["name"] for d in self.dangerous] + self.cant_miss_open))
            try:
                a = self.llm.structured(
                    Assessment, ASSESS_SYSTEM.format(dx_rule=DX_RULE_FIXED.format(dx=fixed) if fixed else DX_RULE_CHOOSE,
                                                     n_closing=max(1, self.cfg.closing_lines)),
                    assess_user(self.ctx.get("opening", ""), self.ctx.get("vitals_line", ""), self.ledger.transcript(),
                                self._candidates(), open_dx,
                                [n for _, n, _ in self.tests_ranked], self._section_text("allergy"), self.cfg.round,
                                guides, self._presentation_ref(), self._workup(), self._patient_profile(),
                                self._plan_ref()))
            except Exception as exc:
                self._error("assess", exc)
        if a is not None:
            a.primary_diagnosis_en = one_diagnosis(a.primary_diagnosis_en)
            a.primary_diagnosis_ko = one_diagnosis(a.primary_diagnosis_ko)
            # "undetermined" is no diagnosis: the fallbacks below do better (a symptom scores above nothing)
            if _UNDETERMINED.search(a.primary_diagnosis_en):
                a.primary_diagnosis_en = ""
            if _UNDETERMINED.search(a.primary_diagnosis_ko):
                a.primary_diagnosis_ko = ""
            if not a.names_cause:
                self._event("syndrome_diagnosis", diagnosis=a.primary_diagnosis_en)
        self.assessment = a
        # without an assessment: the model's most probable proposal (the focus call's leader, or a hypothesis the
        # model proposed in the core mode), never a disease only the core added
        proposed = [h for h in st.get("hypotheses") or [] if not str(h.get("prior_source", "")).startswith("reverse_index")]
        ids = {h["disease_id"] for h in proposed}
        fallback = next((d["name"] for d in diff if d["disease_id"] in ids), proposed[0]["name"] if proposed else "")
        fallback_ko = ""
        if not fallback and self.leading:
            fallback, fallback_ko = self.leading[0]["en"], self.leading[0].get("ko", "")
        if not fallback and self.playbook is not None and self.playbook.differential:
            top = self.playbook.differential[0]  # the presentation's most likely (or most dangerous) cause
            fallback, fallback_ko = top["en"], top.get("ko", "")
        if not fallback:
            fallback_ko = self._complaint_dx()
        if fixed:
            en, source = fixed, "core"
        elif a is not None and (a.primary_diagnosis_en.strip() or a.primary_diagnosis_ko.strip()):
            en, source = a.primary_diagnosis_en.strip(), "llm"
        else:
            en, source = fallback, "fallback"
        ko = a.primary_diagnosis_ko.strip() if a is not None and source != "fallback" else fallback_ko
        if not en and not ko:
            ko = "진단 미정"
        self.final = {"en": en, "ko": ko, "source": source, "core_leader": leader["name"] if leader else "",
                      "core_prob": round(leader["prob"], 3) if leader else 0.0}
        card = self.store.get(slug(en)) or self.store.get(en)
        names = {en.lower(), ko.lower()} - {""}
        self.dangerous = [d for d in self.dangerous  # the primary diagnosis is never its own "rule-out"
                          if not (card is not None and d["disease_id"] == card.id) and d["name"].lower() not in names]
        self.confirm_test = self._confirm_test(card.id) if card is not None else None
        self._event("assess", **self.final)

    def _complaint_dx(self) -> str:
        """The presenting problem as a diagnosis of last resort: the topic plan's or focus call's phrase, else the
        complaint words of the opening ("코골이", "탈모")."""
        for phrase in (getattr(self.focus, "topic_ko", "") if self.focus else "",
                       getattr(self.topic, "topic_ko", "") if self.topic else ""):
            if phrase and phrase.strip():
                return re.sub(r"^\d+\s*(일|주|개월|달|년)(간|째)?\s*", "", phrase.strip())[:60]
        return complaint_phrase(self.ctx.get("opening", ""))

    def _workup(self) -> list[str]:
        """The playbook's tests as '<test> : <purpose> (for: <diagnoses>)' lines, for the assessment."""
        pb = self.playbook
        if pb is None:
            return []
        out = []
        for t in pb.tests:
            link = pb.test_links.get(t, {})
            dx = ", ".join(link.get("for", ()))
            out.append(pb.test_line(t) + (f" (for: {dx})" if dx else ""))
        return out

    def _plan_ref(self) -> str:
        """The playbook's usual plan (treatment, disposition, follow-up, warning signs), "" without one."""
        plan = self.playbook.plan if self.playbook is not None else {}
        parts = []
        for key, name in (("treatment", "Treatment"), ("disposition", "Disposition"), ("follow_up", "Follow-up"),
                          ("red_flags", "Warning signs")):
            v = plan.get(key)
            if v:
                parts.append(f"{name}: " + ("; ".join(v) if isinstance(v, list) else str(v)))
        return "\n".join(parts)

    def _presentation_ref(self) -> str:
        """The playbook's usual causes and can't-miss diagnoses, for the assessment prompt ("" without a playbook)."""
        pb = self.playbook
        if pb is None:
            return ""
        usual = ", ".join(d["en"] for d in pb.differential if not d.get("cant_miss"))
        return (f"{pb.title_en} ({pb.title_ko}). Usual causes: {usual or 'none listed'}. "
                f"Can't-miss causes to confirm or exclude: {', '.join(pb.cant_miss()) or 'none listed'}.")

    def _library(self) -> Any:
        """The guideline library, loaded (prepare() normally does it); None when unavailable or switched off."""
        cfg = self.cfg
        if not cfg.use_guidelines:
            return None
        lib = self.agent.guidelines
        if lib is None:
            from dxagent.nova.guidelines import GuidelineLibrary

            lib = self.agent.guidelines = GuidelineLibrary.from_config(cfg)
        if not lib.available():
            return None
        if not lib.loaded:
            lib.load(time_budget_s=min(30.0, max(0.0, self.budget.remaining() - cfg.wrap_margin_s)))
        return lib

    def _guideline_text(self, names: list[str], terms: tuple[str, ...], chars: int) -> str:
        """Excerpts of the best document for the leading names, as plain text for a prompt ("" if none)."""
        if not names or chars <= 0:
            return ""
        try:
            lib = self._library()
            if lib is None:
                return ""
            docs = lib.search(names, k=1)
            ex = lib.excerpts(docs, terms=list(names) + list(terms), max_chars=chars)
        except Exception as exc:
            self._error("guidelines", exc)
            return ""
        return "\n".join(f"{g['title']} ({g.get('org', '')} {g.get('year', '')}): {g['text']}" for g in ex)

    def _guideline_excerpts(self, fixed: Optional[str]) -> list[dict[str, str]]:
        """Short excerpts from the contest's guideline library for the leading diagnoses (read at run time; nothing
        derived from the documents is stored or shipped). Queries: the fixed or leading diagnosis, then the
        hypotheses the model proposed (these exist even when no knowledge card does)."""
        cfg = self.cfg
        if not cfg.use_guidelines or cfg.guideline_docs <= 0:
            return []
        try:
            lib = self._library()
            if lib is None:
                return []
            names: list[str] = [fixed] if fixed else []
            names += [d["name"] for d in (self.state.get("differential") or [])[:2]]
            names += [h["name"] for h in (self.state.get("hypotheses") or [])[:3]]
            names += [d["en"] for d in self.leading[:3]]
            queries = list(dict.fromkeys(n for n in names if n))[:5]
            docs = lib.search(queries, k=cfg.guideline_docs)
            if self.playbook is not None and len(docs) < cfg.guideline_docs:  # the presentation's own references
                docs += [d for d in lib.get(self.playbook.guidelines) if d not in docs][:cfg.guideline_docs - len(docs)]
                queries += [self.playbook.title_en]
            out = lib.excerpts(docs, terms=queries[:3], max_chars=cfg.guideline_chars)
        except Exception as exc:
            self._error("guidelines", exc)
            return []
        self.guideline_refs = [{k: g.get(k, "") for k in ("title", "org", "year", "path")} for g in out]
        self._event("guidelines", docs=[g["title"][:80] for g in out])
        return out

    def _closing_lines(self) -> list[str]:
        """The assessment's explanation lines (statements of at most 30 characters, no drug the patient is allergic
        to), completed from the playbook's closing lines and generic ones; the first line greets the patient by name
        when none does and it fits."""
        n = max(0, self.cfg.closing_lines)
        limit = self.cfg.say_max_chars
        lines: list[str] = []
        if self.assessment is not None:
            for ln in self.assessment.closing_lines:
                ln = (ln or "").strip()
                if ln and check_say(ln, limit)[0] and "?" not in ln:
                    lines.append(ln)
                elif ln:
                    lines += split_line(ln, limit)  # "A해요. B해요." too long for one turn: its sentences that fit
        lines = list(dict.fromkeys(lint_lines(lines, self._section_text("allergy"))))
        model_lines = len(lines)
        ko = (self.final.get("ko") or "").split("(")[0].strip()
        if not re.search(r"[가-힣]", ko) or re.search(r"(미정|미상|undetermined|unknown)", ko, re.I) or \
                self.final.get("source") == "fallback":
            ko = ""  # a guess of last resort is never announced to the patient
        fallback = [f"{ko} 가능성이 높아요." if ko else "검사로 원인을 확인할게요."]
        fallback += list(self.playbook.education) if self.playbook is not None else []
        flags = (self.playbook.plan.get("red_flags") or []) if self.playbook is not None else []
        fallback += [f"{f} 있으면 바로 오세요." for f in flags[:2]]
        fallback += ["필요한 검사와 치료를 진행할게요.", "증상이 심해지면 바로 알려주세요.", "궁금한 점은 언제든 물어보세요."]
        for f in fallback:
            if len(lines) >= n:
                break
            if check_say(f, limit)[0] and f not in lines:
                lines.append(f)
        lines = [ln for ln in lines if ln not in self.performed][:n]
        if n >= 2 and lines and not any(_FOLLOW_UP.search(ln) for ln in lines):
            # the follow-up visit is always said: in place of a generic line, else of the last one
            k = len(lines) if len(lines) < n else min(model_lines, n - 1) if model_lines else n - 1
            lines = lines[:k] + [CL.FOLLOW_UP_LINE] + lines[k:]
            lines = lines[:n]
        name = self.ctx.get("address", "")
        if lines and name and not any(name.split()[0] in ln for ln in lines):
            first = f"{name}, {lines[0]}"
            if check_say(first, limit)[0]:
                lines[0] = first
        return lines

    # -- DIAGNOSE -------------------------------------------------------------------

    def _diagnose(self) -> Action:
        from dxagent.nova.soap import build_soap, emergency_soap, minimal_soap

        en, ko = self.final.get("en", ""), self.final.get("ko", "")
        if not en and not ko:
            diff = self.state.get("differential") or []
            en = diff[0]["name"] if diff else "진단 미정"
            self.final.setdefault("en", en)
        diagnosis = f"{ko} ({en})" if ko and en and ko != en else (ko or en)
        try:
            out = build_soap(self)
        except Exception as exc:
            self._error("soap", exc)
            try:
                out = minimal_soap(self)
            except Exception as exc2:  # last resort: never leave the case without a valid submission
                self._error("minimal_soap", exc2)
                out = emergency_soap(self, diagnosis)
        flat = {k: out[k] for k in ("S", "O", "A", "P", "text") if k in out}
        return Action(kind="DIAGNOSE", diagnosis=out.get("diagnosis") or diagnosis, note=out["note"], soap=flat)

    def record(self, action: Action, result: Any = None) -> dict[str, Any]:
        diff = self.state.get("differential") or []
        rec = {
            "case_id": self.opening.case_id, "diagnosis": action.diagnosis, "note": action.note, "soap": action.soap,
            "final": self.final, "turns": self.turns, "budget": self.budget.summary(),
            "llm_trace": getattr(self.llm, "trace", []),
            "differential": [{"name": d["name"], "prob": round(d["prob"], 4), "has_card": d.get("has_card"),
                              "cant_miss": d.get("cant_miss")} for d in diff[:8]],
            "core_committed": self.core_committed, "commit_reason": self.commit_reason,
            "closing_said": self.closing_said, "safety_notes": self.safety_notes,
            "tests_ranked": self.tests_ranked, "dangerous": self.dangerous, "guidelines": self.guideline_refs,
            "playbook": self.playbook.id if self.playbook is not None else "", "leading": self.leading,
            "dropped": self.dropped,
            "ledger": self.ledger.to_list(), "events": self.events, "errors": self.errors,
            "assessment": self.assessment.model_dump() if self.assessment is not None else None,
            "env_result": result,
        }
        if self.cfg.log_dir:
            try:
                d = Path(self.cfg.log_dir)
                d.mkdir(parents=True, exist_ok=True)
                name = (self.opening.case_id or f"case_{int(time.time())}").replace("/", "_")
                (d / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            except Exception as exc:
                log.warning("could not write case log: %s", exc)
        return rec
