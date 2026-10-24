"""Prompts and output schemas used only by the competition runtime. The core's prompts (extraction, hypotheses,
critic) are reused unchanged from dxagent/prompts.py."""

from __future__ import annotations

from typing import Optional, Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BankItem(BaseModel):
    kind: Literal["question", "exam", "test"] = "question"
    text_ko: str = Field(description="Korean. A question: ONE question, at most 30 characters including spaces. "
                                     "An exam or test: ONE maneuver or ONE test as a short sentence.")
    target: str = Field(default="", description="English: the finding this is meant to learn, e.g. 'pain migrating to right lower quadrant'.")
    code: str = Field(default="", description="snake_case finding code: a listed vocabulary code if one matches, else a new one.")
    section: Literal["hpi", "ros", "pmh", "meds", "allergy", "fhx", "shx", "ob"] = "hpi"


class QuestionBank(BaseModel):
    items: list[BankItem] = Field(default_factory=list)


class PhraseOutput(BaseModel):
    text_ko: str = Field(description="The one Korean utterance.")


class DxItem(BaseModel):
    name_ko: str
    name_en: str = ""
    reason_ko: str = Field(default="", description="One short Korean sentence: the findings for and against.")
    test_ko: str = Field(default="", description="Korean: the test that confirms or excludes it, e.g. '복부 초음파'.")


class PlanOut(BaseModel):
    tests: list[str] = Field(default_factory=list, description="Korean lines '<분류> : <검사들> : <목적>', one line per "
                                                               "purpose, 분류 one of 혈액검사, 소변검사, 영상검사, 심전도, "
                                                               "내시경, 기타.")
    treatment: list[str] = Field(default_factory=list, description="Korean: initial treatment (drug class or first-line agent).")
    disposition: str = Field(default="", description="Korean: admit / emergency observation / outpatient follow-up / referral, and why.")
    follow_up: list[str] = Field(default_factory=list, description="Korean, three lines: the interval and its purpose; "
                                                                   "the monitoring; the named warning signs that need "
                                                                   "an immediate visit.")
    education: list[str] = Field(default_factory=list, description="Korean, 4-6 items: what the patient was told or must "
                                                                   "know, including the answer to the stated worry.")


class Assessment(BaseModel):
    primary_diagnosis_en: str = Field(description="The ONE primary diagnosis in English, as specific as the findings "
                                                  "support (cause, type or site), e.g. 'acute cholecystitis' rather than "
                                                  "'abdominal pain'.")
    names_cause: bool = Field(default=True, description="True if the primary diagnosis names a disease or cause; False "
                                                        "if it names only a symptom or syndrome.")
    primary_diagnosis_ko: str = Field(description="The same diagnosis in Korean medical terms.")
    evidence_ko: list[str] = Field(default_factory=list, description="2-5 Korean bullets: collected facts and what "
                                                                     "they mean, chart style.")
    rationale_ko: str = Field(default="", description="Optional one-sentence Korean summary of the reasoning.")
    differentials: list[DxItem] = Field(default_factory=list, description="2-5 differential diagnoses.")
    plan: PlanOut = Field(default_factory=PlanOut)
    closing_lines: list[str] = Field(default_factory=list,
                                     description="Korean sentences to say to the patient, each at most 30 characters.")
    chief_complaint_ko: str = Field(default="", description="The chief complaint in a short Korean clinical phrase, with duration.")


class PlanItem(BaseModel):
    text_ko: str = Field(description="Korean. A question: ONE question, at most 30 characters including spaces. An exam: "
                                     "ONE maneuver as a short sentence, e.g. '복부를 촉진하겠습니다.'")
    target: str = Field(default="", description="English: what it is meant to learn, e.g. 'night sweats'.")
    item: str = Field(default="", description="Chart item name for the note, e.g. 'Night sweats', 'Abdomen', 'Lung'.")
    section: Literal["hpi", "ros", "pmh", "meds", "allergy", "fhx", "shx", "ob", "exam"] = "hpi"


class TopicPlan(BaseModel):
    """The complaint-specific part of a first-visit history and exam (CPX style), planned once at turn 0 when no
    presentation playbook matched the opening."""
    topic_ko: str = Field(default="", description="The presenting problem as a short Korean clinical phrase, e.g. '상복부 통증'.")
    presentation: str = Field(default="", description="When a menu of presentations is given: the id that fits the "
                                                      "presenting problem, or '' if none fits.")
    likely_causes: list[str] = Field(default_factory=list, description="English: 3-5 likely causes, most likely first.")
    questions: list[PlanItem] = Field(default_factory=list)
    exams: list[PlanItem] = Field(default_factory=list)


class LeadDx(BaseModel):
    name_en: str
    name_ko: str = ""
    why: str = Field(default="", description="Korean, a few words: the findings that support it.")
    against: str = Field(default="", description="Korean, a few words: the findings against it or still missing.")


class FocusPlan(BaseModel):
    """Mid-visit focus (one call after the history of the complaint): what is still worth asking and examining."""
    topic_ko: str = Field(default="", description="The presenting problem as a short Korean clinical phrase with duration.")
    leading: list[LeadDx] = Field(default_factory=list, description="3-5 most likely diagnoses, most likely first, "
                                                                   "including any dangerous cause not yet excluded.")
    skip: list[str] = Field(default_factory=list, description="ids of OPTIONAL items not worth asking for this patient.")
    add: list[str] = Field(default_factory=list, description="ids of EXTRA items worth asking for this patient.")
    questions: list[PlanItem] = Field(default_factory=list)
    exams: list[PlanItem] = Field(default_factory=list)


class ReviewPlan(BaseModel):
    """Diagnostic time-out (one call after the history and the review of systems, before the examination)."""
    leading: list[LeadDx] = Field(default_factory=list, description="The 3 most likely diagnoses now, most likely "
                                                                   "first, with findings for (why) and against.")
    cant_miss_open: list[str] = Field(default_factory=list, description="English: dangerous causes not yet excluded.")
    names_cause: bool = Field(default=True, description="True if the first leading diagnosis names a disease or its "
                                                        "cause; False if only a symptom or syndrome.")
    questions: list[PlanItem] = Field(default_factory=list)
    exams: list[PlanItem] = Field(default_factory=list)
    exam_order: list[str] = Field(default_factory=list, description="ids of the PLANNED maneuvers, most "
                                                                    "discriminating first.")
    skip_exams: list[str] = Field(default_factory=list, description="ids of OPTIONAL planned maneuvers whose result "
                                                                    "would not change the assessment.")
    exam_intro_ko: str = Field(default="", description="One Korean sentence to the patient before the examination, at "
                                                       "most 30 characters: what will be examined first.")


class ChartLine(BaseModel):
    id: str = Field(description="The entry id, e.g. e14.")
    text: str = Field(description="'<부위> : <소견>' in chart style, without the turn number.")


class ChartOutput(BaseModel):
    """S and O fields of the official note (development guide, 초진 의무기록 작성 안내)."""
    cc: str = Field(default="", description="'C.C : ...' line, 'Onset : ...' line, and for pain '통증평가 : NRS n점, 부위, 양상, 기간'.")
    hpi: str = Field(default="", description="'<age>세 <sex>.' then short chart lines, one theme each, ending with the "
                                             "patient's worry or request.")
    pmh: str = Field(default="", description="Item-wise (+)/(-): DM / HTN, Tbc / hepatitis, Op. Hx, Medicine history, Allergy, others.")
    fh: str = Field(default="", description="Family history, item-wise, or the general negative.")
    sh: str = Field(default="", description="Smoking, Alcohol, occupation.")
    ros: str = Field(default="", description="(+)/(-) list of symptoms asked about.")
    exam: list[ChartLine] = Field(default_factory=list)
    tests: list[ChartLine] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BANK_SYSTEM = """You assist a doctor agent in a Korean clinical encounter with a simulated patient.
Propose the next questions and examination maneuvers that best tell apart the leading diagnoses.

Rules:
- Korean. A question is ONE question to the patient in plain lay language, at most 30 characters including spaces,
  with a single question mark. Never two questions in one item, never a diagnosis name.
- An exam item is ONE physical examination maneuver as a short sentence, e.g. "복부 촉진을 하겠습니다." Never join two
  maneuvers (no 와/과/및/그리고, commas or "·"), never a whole region or an unspecified method ("손 떨림 확인").
- {tests_rule}
- Never repeat anything already asked or examined (listed below). Prefer items whose answer would differ most
  between the top diagnoses, and items that check dangerous diagnoses that are still possible.
- Order items by importance."""

TESTS_RULE_PRELIM = "Do not propose laboratory or imaging tests: they cannot be ordered in this round."
TESTS_RULE_FINAL = "A test item is ONE test as a short sentence, e.g. \"흉부 X선을 찍겠습니다.\""


def bank_user(problem_rep: str, differential: list[dict[str, Any]], transcript: str, asked: list[str],
              vocab: dict[str, dict[str, str]], n_questions: int, n_exams: int, n_tests: int) -> str:
    diff = "; ".join(f"{d['name']} ({d['prob']:.2f})" for d in differential[:6]) or "(none yet)"
    voc = "\n".join(f"- {c}: {v['label']}" for c, v in list(vocab.items())[:120])
    return (f"Problem representation: {problem_rep}\nLeading diagnoses: {diff}\n\nEncounter so far:\n{transcript}\n\n"
            f"Already asked or examined:\n" + "\n".join(f"- {a}" for a in asked[-60:]) +
            f"\n\nVocabulary codes (reuse when the meaning matches):\n{voc}\n\n"
            f"Give up to {n_questions} questions, {n_exams} exam maneuvers" + (f" and {n_tests} tests." if n_tests else "."))


TOPIC_SYSTEM = """You plan the history and physical examination of a first outpatient visit, like a clinical performance
examination (CPX) checklist, for a doctor agent talking with a Korean simulated patient.

From the opening statement, plan the questions a well-trained doctor asks for THIS presenting problem:
- characteristics of the main symptom (onset, course, pattern, severity, triggers, what was tried) that the standard
  questions listed below do not already cover;
- the associated symptoms and red flags that separate the likely causes (each likely cause should get at least one
  discriminating question);
- the problem-specific past, medication, family, social, obstetric or developmental history (e.g. painkillers for
  epigastric pain, travel and sick contacts for diarrhea, the last menstrual period for lower abdominal pain).
Then the focused physical examination maneuvers for these causes, and the likely causes themselves (likely_causes,
English, most likely first).

Rules:
- Korean. A question is ONE question in plain lay language, at most 30 characters including spaces, one question
  mark, never two topics, never a diagnosis name, not leading. {guardian}
- An exam item is a short sentence asking for ONE maneuver named with its method or site ("슬개건 반사", "자세성 떨림", "폐 청진"), never a symptom
  to look at ("손 떨림 확인"), a whole region ("무릎 진찰", "신경학적 검사"), two parts ("목·어깨", "체중과 키") or anything that
  is not a bedside examination (a test, a record, a question): the examiner refuses those and the turn is lost.
  The examiner matches requests to a table of standard maneuvers; name them as the table does when one fits:
  {exam_names}.
- Never repeat or rephrase anything in the standard list below.
- Most important first. {n_q} questions and {n_e} exams at most."""

EXAM_NAMES = ("전반적 외관 시진, 의식·지남력 평가, 기립성 혈압 측정, 요골동맥 맥박 촉진, 피부 온도·발한 촉진, 결막·공막 시진, "
              "외안근 운동, 안구진탕 확인, 갑상선 안징후 검사, 안면 기형 소견 시진, 경부 림프절 촉진, 갑상선 촉진, "
              "갑상선 잡음 청진, 흉곽 시진, 폐 청진, 심음 청진, 복부 시진, 장음 청진, 복부 촉진, 간·비장 촉진, "
              "사지 근력 검사, 심부건반사, 자세성 떨림, 떨림·서동 관찰, 소뇌 협조운동 검사, 구음장애 확인, 보행 관찰, "
              "롬베르그 검사, 인지기능 선별검사, 하지 말초 진찰, 신체 계측, 성장도표 백분위수 확인, 신체 비율 평가, "
              "사춘기 성숙도 평가")  # the standard maneuver vocabulary of CPX exam tables

GUARDIAN_RULE = ("The patient is a child: questions go to the guardian and name the child ('아이가 ~', e.g. "
                 "'아이가 밥은 잘 먹나요?').")


def topic_user(opening: str, vitals: str, age: Any, sex: Any, hypotheses: list[str], standard: list[str],
               guideline: str = "", menu: Optional[list[tuple[str, str]]] = None) -> str:
    pick = ("\n\nPresentation menu (set `presentation` to the id of the one that fits the presenting problem, "
            "or leave it empty):\n" + ", ".join(f"{i} ({t})" for i, t in menu)) if menu else ""
    return (f"Opening statement: {opening}\nVital signs: {vitals or 'not given'}\nAge: {age}  Sex: {sex}\n"
            f"Likely causes so far: {', '.join(hypotheses) or 'not yet known'}\n\n"
            "Standard questions and maneuvers already planned (do not repeat):\n" + "\n".join(f"- {x}" for x in standard)
            + (f"\n\nReference (history and physical, background only):\n{guideline}" if guideline else "") + pick)


FOCUS_SYSTEM = """You help a doctor agent in the middle of a first outpatient visit with a Korean simulated patient (a
clinical performance examination, CPX). The history of the presenting complaint has been taken. Decide what is still
worth asking and examining for THIS patient: the visit is scored on asking what is necessary AND on avoiding
unnecessary questions.

- leading: the 3-5 most likely diagnoses from what is known, most likely first. Keep every dangerous cause that the
  answers have not excluded.
- skip: ids of OPTIONAL items whose answer would not change the assessment for this patient (already answered in the
  transcript, or irrelevant to the leading diagnoses). Never skip a red flag of a dangerous cause that is still
  possible.
- add: ids of EXTRA items that now matter for the leading diagnoses.
- questions: at most {n_q} new questions that separate the leading diagnoses and that no list covers. Korean, ONE
  question in plain lay language, at most 30 characters including spaces, one question mark, not leading, no
  diagnosis names. {guardian}
- exams: at most {n_e} maneuvers not listed, each a short sentence asking for ONE maneuver named with its method or site ("슬개건 반사", "자세성 떨림", "폐 청진"), never a symptom
  to look at ("손 떨림 확인"), a whole region ("무릎 진찰", "신경학적 검사"), two parts ("목·어깨", "체중과 키") or anything that
  is not a bedside examination (a test, a record, a question): the examiner refuses those and the turn is lost.
  Name them as the examiner's table does when one fits: {exam_names}.
- topic_ko: the presenting problem as a short Korean clinical phrase with its duration (e.g. "2주간 상복부 통증").
Use the ids exactly as listed."""


def focus_user(opening: str, vitals: str, age: Any, sex: Any, presentation: str, transcript: str,
               planned: list[str], optional: list[tuple[str, str]], extra: list[tuple[str, str]],
               turns: Optional[tuple[int, int, int, int]] = None) -> str:
    """`turns`: (used, end of the plan as listed, target, hard limit), all counting the closing explanations."""
    budget = ""
    if turns:
        used, end, target, cap = turns
        budget = (f"\nTurns: {used} used; with every OPTIONAL item asked the visit ends at about turn {end}. A visit "
                  f"usually needs about {target}; {cap} is the hard limit. Skip what this patient does not need, "
                  "but do not skip a needed item only to save turns.\n")
    return (f"Opening statement: {opening}\nVital signs: {vitals or 'not given'}\nAge: {age}  Sex: {sex}\n"
            + (f"Presentation reference (not patient facts): {presentation}\n" if presentation else "")
            + f"\nTranscript so far:\n{transcript}\n\nPlanned and always asked:\n"
            + "\n".join(f"- {x}" for x in planned)
            + "\n\nOPTIONAL (asked unless you skip them):\n" + ("\n".join(f"[{i}] {t}" for i, t in optional) or "(none)")
            + "\n\nEXTRA (asked only if you add them):\n" + ("\n".join(f"[{i}] {t}" for i, t in extra) or "(none)")
            + budget)


REVIEW_SYSTEM = """You are the diagnostic time-out of a doctor agent in a first outpatient visit with a Korean simulated
patient (a clinical performance examination, CPX). The history and the review of systems are done; the physical
examination comes next. Premature closure (stopping at the first diagnosis that fits) is the commonest cause of
diagnostic error: re-read the whole transcript before deciding.

- leading: the 3 most likely diagnoses now, most likely first; why: the findings for it, against: the findings against
  it or still missing (Korean, a few words each).
- cant_miss_open: the dangerous causes of this presentation that the answers have NOT excluded (English).
- names_cause: true if the first leading diagnosis names a disease or its cause; false if it is only a symptom or a
  syndrome (insomnia, dizziness, weight loss, anemia, chronic cough). When false, the questions must look for the cause.
- questions: at most {n_q} questions that would most change the ranking: quantify or characterise a positive answer
  that was not followed up (how much, how often, since when, getting worse, tried to stop, what happens without it),
  check a can't-miss cause, or separate the first two diagnoses. Korean, ONE question in plain lay language, at most 30
  characters including spaces, one question mark, not leading, no diagnosis names, never one already asked or listed
  as still to be asked, nor one whose answer the transcript already gives. {guardian}
- exam_order: the ids of the PLANNED maneuvers, the most discriminating for the leading and can't-miss diagnoses first.
- skip_exams: ids of OPTIONAL planned maneuvers whose result would not change the assessment for this patient.
- exams: at most {n_e} maneuvers not planned that would separate the leading diagnoses: a short sentence asking for ONE
  maneuver named with its method or site ("슬개건 반사", "폐 청진"), never a symptom ("손 떨림 확인"), a whole region
  ("무릎 진찰"), two parts ("목·어깨") or anything that is not a bedside examination; the examiner refuses those and the
  turn is lost. Name them as the examiner's table does when one fits: {exam_names}.
- exam_intro_ko: ONE Korean sentence the doctor says to the patient just before the examination, AT MOST 30
  CHARACTERS including spaces, a statement (no question), plain words: the body parts that will be examined (not one
  maneuver: the order may change) and to say if anything hurts, e.g. "이제 배와 가슴을 진찰할게요. 아프면
  말씀하세요." {guardian}
Use the ids exactly as listed. Leave a list empty rather than fill it: every question and maneuver costs a turn."""


def review_user(opening: str, vitals: str, age: Any, sex: Any, presentation: str, transcript: str,
                leading: str, planned: list[tuple[str, str, int]], turns: tuple[int, int],
                queued: list[str] = ()) -> str:
    """`planned`: (id, maneuver, priority) of the maneuvers still to do; `turns`: (used, last turn for gathering);
    `queued`: history questions still to be asked after the time-out's own, if turns allow."""
    used, end = turns
    return (f"Opening statement: {opening}\nVital signs: {vitals or 'not given'}\nAge: {age}  Sex: {sex}\n"
            + (f"Presentation reference (not patient facts): {presentation}\n" if presentation else "")
            + f"Leading diagnoses at the focus call (a hint): {leading or 'none'}\n"
            + f"\nTranscript so far:\n{transcript}\n\nPLANNED maneuvers (priority 1 always done, 2 OPTIONAL):\n"
            + ("\n".join(f"[{i}] {t} (p{p})" for i, t, p in planned) or "(none)")
            + ("\n\nHistory questions still to be asked after yours if turns allow (do not repeat them):\n"
               + "\n".join(f"- {q}" for q in queued) if queued else "")
            + f"\n\nTurns: {used} used; the history and the examination must end by turn {end}.")


PHRASE_SYSTEM = """Write ONE Korean utterance for a doctor in a clinical encounter.
- History finding: one question to the patient in plain lay language, at most 30 characters including spaces,
  exactly one question mark, no diagnosis names, not leading.
- Exam finding: one maneuver request as a short sentence, e.g. "폐 청진을 하겠습니다." (one maneuver only).
- Test: one test as a short sentence, e.g. "흉부 X선을 찍겠습니다." """


def phrase_user(kind: str, label: str) -> str:
    return f"Kind: {kind}\nWhat to learn: {label}"


ASSESS_SYSTEM = """You write the assessment and plan of a doctor agent's first-visit note (Korean chart style), and the
doctor's explanation to the patient. The encounter is over; use only what the transcript shows. Guideline excerpts and
the presentation reference are background knowledge, never facts about this patient.

- Primary diagnosis: {dx_rule}
- names_cause: true if the primary diagnosis names a disease or its cause; false if it names only a symptom or syndrome.
- evidence_ko: 3-5 bullets "<facts, with the patient's own numbers> : <interpretation> (<turns>)", at least one from
  the history, one from the examination and one decisive negative or family history, each ending with the turns its
  facts come from as the transcript numbers them (T0 is the opening and the vital signs) (e.g. "발열 38.5도, 식후
  우상복부 통증 : 담낭 염증 시사 (T0, T3)"; "머피 징후(+), 반발압통(-) : 국소 담낭 자극, 범발성 복막염 소견 없음
  (T31, T32)"). Never invent findings, values or turns.
- differentials: 3-5 alternatives, including every dangerous cause of this presentation that the visit did not exclude:
  reason_ko "<supporting findings>, <opposing findings>" from this patient; test_ko the test that settles it.
- plan.tests: one line per purpose, "<분류> : <검사들> : <목적>", 분류 one of 혈액검사, 소변검사, 영상검사, 심전도, 내시경,
  기타 (e.g. "영상검사 : 복부 초음파 : 담석·담낭벽 비후 확인"). At least one line for the primary diagnosis and one for
  each differential (name it in the purpose, e.g. "급성 췌장염 감별"), plus the safety tests the treatment needs (a
  pregnancy test before teratogenic drugs or radiation in a woman aged 12-55; baseline CBC and LFT before drugs toxic to
  marrow or liver; ECG before QT- or rate-affecting drugs). Give the expected result when it settles the question. No
  test unrelated to the differential. Choose from the presentation's work-up when it fits. In the preliminary round this
  field is how tests are evaluated.
- plan.treatment: safe initial management: first-line drug class or agent, non-drug measures, and what waits for a
  test ("<검사> 확인 후 <약물> 시작"). Respect the recorded allergies and medications. Give a dose only if it is a
  standard adult dose you are certain of.
- plan.disposition: "<외래 추적 / 전문과 의뢰 / 응급 처치·입원>. 근거 : <활력징후 안정 여부, 위험 징후 유무>", with the
  department when a referral is needed. Unstable vital signs or a dangerous diagnosis not excluded mean emergency care.
- plan.follow_up: three lines: "<기간> 후 재진 : <목적: 검사 결과 확인, 치료 반응 평가 등>"; the monitoring ("<치료>
  시작 후 <간격> <항목> 확인"); "<2-4 named warning signs of this disease> 시 예정일 전이라도 즉시 내원". With emergency
  care, the first line is what happens next ("응급실 : <처치·검사>, <전문과> 협진") and the last "<warning signs> 시 즉시
  의료진 호출".
- plan.education: 4-6 items: the likely cause in plain words, why the tests, the treatment and self-care, the warning
  signs, and the direct answer to the patient's stated worry or request.
- closing_lines: exactly {n_closing} short Korean sentences the doctor now says to the patient, EACH AT MOST 30
  CHARACTERS including spaces (one sentence per item), warm, plain lay words, polite (-요/-ㅂ니다), no jargon, no
  certainty the tests have not given, statements only (no question), in this order as far as the number allows:
  (1) the likely cause of the patient's own main symptom ("~ 때문일 가능성이 높아요");
  (2) the direct answer to the patient's stated worry or question when there is one, else the history or examination
      finding behind the likely cause in plain words ("진찰에서 오른쪽 아랫배가 아프셨어요");
  (3) the tests and what they will show; (4) the treatment or what to do now; (5) the named warning signs that need an
  immediate visit; (6) the follow-up: when to come back and what happens then ("일주일 뒤 결과 보며 치료를 정해요"),
  or with emergency care what happens next ("지금 응급실에서 검사를 시작할게요").
  Use the form of address given, at most once. A child's guardian hears about "아이".
- chief_complaint_ko: the chief complaint as a short clinical phrase with its duration.
Write Korean, using standard Korean medical terms (English in parentheses where usual)."""

DX_RULE_FIXED = "it is fixed: {dx}. Write it in primary_diagnosis_en/primary_diagnosis_ko and justify it."
DX_RULE_CHOOSE = ("choose the single most likely diagnosis from the whole transcript, as specific as the findings "
                  "support. First check whether it names a cause (a disease with its cause, type or site) or only a "
                  "symptom or syndrome; when the transcript points to a cause (a drug or substance, an infection, an "
                  "endocrine, structural or psychiatric cause), name that cause, e.g. 'acute cholecystitis' rather than "
                  "'right upper quadrant pain', 'benign paroxysmal positional vertigo' rather than 'dizziness', 'iron "
                  "deficiency anemia from heavy menstrual bleeding' rather than 'fatigue'; the disease level when it "
                  "does not. One entity: never a symptom, a list, 'A and B', 'r/o' or '의증'. The leading diagnoses so "
                  "far are a hint that may not cover this case. (The diagnosis is graded on a 5-point scale: an exact "
                  "or more specific diagnosis scores highest; a symptom without its cause, an overly broad or a "
                  "wrong-cause diagnosis scores lower, and an unrelated second diagnosis lowers the score.)")


def assess_user(opening: str, vitals: str, transcript: str, candidates: str, dangerous: list[str],
                tests_ranked: list[str], allergies: str, round_name: str, guidelines: list[dict[str, str]] = (),
                presentation: str = "", workup: list[str] = (), patient: str = "", plan_ref: str = "") -> str:
    """`candidates`: the leading diagnoses so far (the time-out's, the focus call's, or the core's differential), as
    lines; `workup`: the presentation's tests with their purpose; `plan_ref`: its usual plan (playbook)."""
    g = "\n\n".join(f"[G{i + 1}] {x['title']} ({x.get('org', '')} {x.get('year', '')}).\n{x['text']}"
                     for i, x in enumerate(guidelines))
    return (f"Opening: {opening}\nVital signs (T0): {vitals or 'not given'}\n"
            + (f"Patient: {patient}\n" if patient else "")
            + f"\nTranscript:\n{transcript}\n\nLeading diagnoses so far (a hint):\n{candidates or '- (none)'}\n\n"
            f"Dangerous diagnoses still possible: {', '.join(dangerous) or 'see the presentation reference'}\n"
            f"Computed test ranking (most informative first): {', '.join(tests_ranked) or 'none'}\n"
            f"Recorded allergies: {allergies or 'not recorded'}\n"
            f"Round: {round_name} ({'tests could not be ordered during the visit; list them in the plan' if round_name == 'prelim' else 'tests could be ordered'})"
            + (f"\n\nPresentation reference (usual causes of this complaint, not patient facts):\n{presentation}"
               if presentation else "")
            + (f"\nStandard work-up for this presentation (choose what fits):\n" + "\n".join(f"- {t}" for t in workup)
               if workup else "")
            + (f"\nUsual plan for this presentation (adapt it to the diagnosis):\n{plan_ref}" if plan_ref else "")
            + (f"\n\nGuideline excerpts (background knowledge, not patient facts):\n{g}" if g else ""))


CHART_SYSTEM = """You write the S and O fields of a Korean first-visit medical record (초진 기록) from a doctor's
encounter with a patient. Use ONLY the questions and answers given. Never add a fact, never guess, never write a value
the patient did not say: a note that differs from what was confirmed is penalized, and items never asked earn nothing.
Korean chart style with standard English abbreviations, concise. Mark an item never asked "( )" (not confirmed),
never "(-)".

- cc: "C.C : <주호소, short clinical phrase>" (the patient's request in parentheses when the visit is for one, e.g.
  "C.C : 두통 (진통제 처방 원함)"), next line "Onset : <short, e.g. 2주 전>", next line
  "통증평가 : NRS <n>점, <부위>, <양상>, <기간>" for pain, else "통증평가 : 해당 없음".
- hpi: "<age>세 <sex>." then short chart lines, one theme per line, in noun phrases rather than narration: onset and
  course with the patient's own numbers, written as digits (e.g. "진통제 주 1회 → 최근 주 4회"), character,
  aggravating/relieving factors, associated symptoms with (+) and the asked negatives with (-), prior episodes and care;
  the last line is the patient's worry or request (e.g. "뇌에 문제가 있는 건 아닌지 걱정함."), or "~하여 내원함." if none
  was voiced.
  Example shape: "58세 남자. 2주 전부터 계단 오를 때 숨참, 점차 악화.\n기좌호흡 (+), 발목 부종 (+). 흉통·객혈 (-).
  \n이전 유사 증상 (-).\n심장병이 아닌지 걱정함."
- pmh (one item per line): "DM / HTN (-/-)", "previous stroke ( )", "Tbc / hepatitis ( / )", other items asked
  for this complaint (e.g. "heart disease (-)  dyslipidemia (+)"), "Op. Hx ( )", "Medicine history : <drugs and
  what they are for> " or "Medicine history (-)", "Allergy (-)", then "Others : <other illnesses or, for a child,
  birth, development and vaccination history>".
- fh: "DM / HTN ( / )" ((+) when any relative has it), "Stroke ( )", "Others : <who has which illness, as the patient
  named it; parents' heights for a child>".
- sh: "Smoking (-)" or "Smoking (+) <amount> p/d <years>", "Alcohol (+) <frequency>" or "Alcohol (-)",
  "Others : <occupation, caffeine, exercise, diet, stress>"; for a child: "Others : <daycare or school, diet, the
  household as the guardian told it>", no smoking or alcohol items.
- ros: these lines, (+) for symptoms reported anywhere in the encounter, (-) ONLY for symptoms asked and denied:
  "General weakness ( )  Fatigue ( )" / "Headache ( )  Dizziness ( )" / "Chest pain ( )  Palpitation ( )" /
  "Fever/Chill/Cough/Sputum/Rhinorrhea ( / / / / )" / "Abdominal pain ( )  Melena ( )  Hematochezia ( )  Hematemesis ( )" /
  "Anorexia/Nausea/Vomiting/Diarrhea/Constipation ( / / / / )" / "Others : <every other symptom asked, (+) or (-)>".
- Rows marked "자발 진술" were not asked: the patient said it while answering another question. Record the fact under
  its item like an answer (e.g. "식은땀 (+)").
- exam / tests: one line per entry id, "<부위> : <maneuver> <결과 exactly as reported>", the part one of V/S, Growth
  (a child's measurements), General, Skin, H/E/E/N/T, Neck, Chest, Abdomen, Back, GU, Extremities / Neuro, Others
  (e.g. "Abdomen : 복부 촉진 우상복부 압통(+), 반발압통(-)"). Keep every reported detail; do not add the turn number (the runtime adds it and groups the
  lines by part).
- A field with nothing asked is ""."""


def chart_user(opening: str, vitals: str, fields: dict[str, list[str]]) -> str:
    parts = [f"Opening statement (T0): {opening}", f"Vital signs (T0): {vitals or 'not given'}"]
    for name, rows in fields.items():
        parts.append(f"\n[{name}]\n" + ("\n".join(rows) if rows else "(nothing asked)"))
    return "\n".join(parts)
