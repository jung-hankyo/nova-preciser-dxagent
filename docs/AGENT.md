# The contest runtime (v0.8.1)

`run.py` and `dxagent/nova/` play one contest case at a time, as a doctor would run an outpatient visit:

- one question or one exam maneuver per turn, in Korean, at most 30 characters, 50 turns at most;
- a **presentation playbook** (one YAML file per chief complaint) gives the history, maneuvers, differential, work-up and
  plan; a *focus call* adapts it to the patient and a *diagnostic time-out* before the exam re-reads the history and
  orders the exam;
- a positive answer is followed up (how much, how long, getting worse) by the playbook's own follow-up questions;
- it talks to the patient by name, acknowledges pain and worry, asks what they worry about and answers it at the end;
- it writes the S/O/A/P note from a **ledger of what was actually asked**, in a CPX chart layout;
- it quotes a guideline library when the environment provides one, and it never fails a case.

The research core (`docs/CORE.md`) is off by default (`NOVA_CORE=0`, decision 80). Decisions: `docs/decisions/v0.5-simple-approach.md`.

## What the code assumes about the contest

The contest branches were developed against the organizers' starter package and development kit (the SDK, a local
simulator, sample cases). None of that material is in this repository, and these documents do not restate the
competition's rules, scoring or cases. What the code relies on:

- **Entry point.** `run.py` ends with the SDK's `run_agent(prepare, consult)`. `prepare()` loads what the visit needs
  without case access; `consult(client, state)` plays the case through the client and returns the server's answer to
  DIAGNOSE. The submission ZIP holds `run.py`, `requirements.txt` (pinned wheels), `NOTICE.md` and `dxagent/`; the SDK
  is not redistributed (`dxagent/nova/package.py` checks the layout).
- **Actions.** One SAY (a question or an explanation) or EXAM (one maneuver) per turn, each at most 30 characters after
  NFC normalization, then DIAGNOSE with the S/O/A/P note (`soap.FIELDS`) and one primary diagnosis on one line.
  `protocol.check_say`, `check_exam` and `single_topic` stop what the server would refuse before it costs a turn.
- **The Doctor model** is reached only through `client.chat()` (`sdk.SdkTransport`), with per-case call and token
  limits; every case makes at least one call (`ensure_chat_call`). JSON is parsed from the text; an empty or cut answer
  is retried once with more room, and no call asks past the case's output budget (`Budget.output_room`).
- **Errors.** `protocol.py` groups the SDK's codes: a refusal is never resent as is; a server model's unusable answer is
  retried under a new `request_id`; a request that may still be running, or may not have arrived, is resent under the
  same `request_id` so it is not executed twice; a note refused for its shape is repaired (`sdk.repair_record`) and
  resent; when the session is over, the agent goes straight to DIAGNOSE.
- **Patient and examiner** are server-side models. A bundled question gets a request to ask one thing at a time, so the
  agent splits such questions into single-topic parts (`Slot.alt`); maneuvers are single actions from a standard
  vocabulary (`playbooks.SAMPLE_EXAMS`, `specific_exam`).
- **Grading** is the organizers'. The note records only what was asked, told or said, and the primary diagnosis is the
  most specific one the findings support.

## The visit

A playbook leads the visit whenever the opening names a known complaint (58 playbooks). Every item has a priority:
1 always asked; 2 asked unless the focus call skips it, the plan would run past its cap (turn 38) or it reaches the
turns kept for later steps; 3 asked only if the focus call adds it.

```
turn 0   session(): opening + vital signs (no turn); playbook chosen from the opening words
         [topic plan, 1 call, only without a playbook: picks one from a menu, or writes 10 questions + 4 maneuvers]
HPI      the complaint's priority-1 items (onset skipped when the opening states it); a positive answer is followed by
         the playbook's follow-ups (up to 4 per visit, each skipped if the answer already says it)
focus    [1 call] leading diagnoses; optional items to skip; priority-3 items to add; <= 3 questions, <= 2 maneuvers;
         told the turns used, where the plan would end, the usual length (44) and the limit (50)
select   priority 1; 2 unless skipped; 3 if added; priority-2 items are cut only if the plan would run past turn 38
HPI      the selected priority-2 items, the focus call's questions, prior episodes, prior care, the patient's worry
history  adult: DM/HTN, other illness, surgery, medications, allergy, family, smoking (+ amount), alcohol, LMP*, plus
         the playbook's background; child: birth, development, vaccination, illness, medications, allergy, family,
         diet, asked of the guardian
ROS      the selected review items
time-out [1 call] the 3 leading diagnoses with findings for and against; can't-miss causes not excluded; whether the
         leader names a cause; <= 3 questions that would most change the ranking (asked now); the planned maneuvers
         ranked by what they separate, optional ones that would not change the assessment dropped, <= 2 new ones
exam     the ranked maneuvers (<= 8); a speculum or bimanual exam only for a sexually active patient**
close    guideline excerpts; [assessment and plan, 1 call]; 5 explanation lines (SAY): likely cause, tests,
         treatment, warning signs, the answer to the patient's worry
DIAGNOSE [chart, 1 call: S and O in chart style] + A and P + one diagnosis (0 turns)
```
\* women aged 12-55. \*\* never a child; an adolescent (13-17) only after a "yes" to the sexual-activity question; an
adult unless she denied it.

Turns are reserved while the history is taken: up to 6 for the planned maneuvers and 4 for the time-out's items. A
priority-2 history item yields when it would use them; nothing uses the 5 closing turns or the spare turn. Turns the
follow-ups and the time-out leave unused go back, after the time-out, to priority-2 items cut only for turns.

Gathering ends at turn 44 at most (50 - 5 closing lines - 1 spare). The usual length the focus call is told (44) is an
aim, not a cut: nothing the plan needs is dropped only to reach it, and there are no filler questions, so a visit ends
when its plan does. Without a playbook, the older complaint groups (chest pain, dyspnea, cough, abdominal, headache,
fever, dizziness, back pain, weight, sleep, growth, fatigue, palpitation; otherwise general) give the plan.

**Talking**, inside the 30-character limit (each a prefix only when it fits, one per utterance):
the first question opens with a greeting and the form of address (the patient's name, or "mother"/"father" for a
child's guardian); an early question opens with one acknowledgment worded from the opening (pain, worry, a child); when
the patient voices worry, distress or severe pain, the next question opens with a short acknowledgment (at most twice,
3 turns apart); signposts introduce the background and the review of systems; "what worries you most?" ends the history;
the closing lines name the likely cause of the patient's own symptom, what happens next, and answer that worry (the
first line addresses the patient by name).

## Presentation playbooks

Contest cases are CPX-style outpatient visits drawn from a known list of chief complaints, and what an examiner expects
depends on the complaint more than on the hidden disease. A playbook is one YAML file per presentation in
`dxagent/nova/playbooks/` (58 files; the format is in the docstring of `playbooks.py`):

- `match`: opening words that select it; `not`: words that exclude it; `population`: any, adult, child or female. Each
  word is tried on the opening as written and with adverbs and durations removed, so a frequency complaint and a
  volume complaint stay distinct; a word that starts with a one-syllable body-part noun must start a word;
- `history`: single-topic questions of at most 30 characters, positively phrased, each with a section (hpi, ros, pmh,
  meds, fhx, shx, ob), a chart item, a priority, optionally `only` (female, male, adult, child, sexually_active) and
  `gate: sexual`, and `then`: up to 3 follow-ups asked after a positive answer, each with a `known` pattern that skips
  it when the answer already says it;
- `exams`: at most 8 bedside maneuvers, from the standard vocabulary where possible, each one specific action;
- `differential` (with can't-miss flags); `tests` as `{name, for, why}`; `plan`: first-line treatment, usual disposition,
  follow-up and 2-4 named warning signs; 2-4 closing lines that are safe whatever the final diagnosis; paths into the
  guideline library, read on the server.

Selection: the playbook whose words appear earliest in the opening; without one, the turn-0 topic plan may name one from
a menu that fits the patient. The playbook is the visit's deterministic floor: the focus call trims and extends it, and
it still works when the model returns broken JSON.

| Group | Playbooks |
|---|---|
| Cardiopulmonary | chest_pain, palpitation, dyspnea, cough, hemoptysis, syncope, edema, high_bp, dyslipidemia |
| Gastrointestinal | abdominal_pain, dyspepsia, vomiting, diarrhea, constipation, upper_gi_bleeding, hematochezia, jaundice, abdominal_distension, dysphagia |
| General and endocrine | fever, fatigue, weight_loss, weight_gain, polyuria, neck_mass |
| Neurology | headache, dizziness, seizure, weakness_numbness, memory_loss, tremor, altered_consciousness |
| Musculoskeletal | joint_pain, back_pain, neck_pain |
| Urology | dysuria, hematuria (red urine), voiding_difficulty, oliguria |
| Women's health | menstrual_irregularity, dysmenorrhea, vaginal_discharge, abnormal_uterine_bleeding, breast_mass |
| Mental health | insomnia, depressed_mood, anxiety |
| Skin and blood | rash, easy_bruising |
| Pediatrics | ped_fever, short_stature, developmental_delay, ped_vomiting_diarrhea, ped_cough, ped_rash |
| ENT | sore_throat, ear_symptoms, nasal_symptoms |

Every file is `status: draft` until a clinician sets `status: reviewed` and `reviewer`; the AI reviews and their open
doubts are in `docs/PLAYBOOK_REVIEW.md`. Not covered (decision 103): counselling and visit-type stations (smoking and
alcohol, substance use, breaking bad news, violence, prenatal care, vaccination).

```bash
python -m dxagent.nova.playbooks                             # validate every file
python -m dxagent.nova.playbooks --budget                    # planned turns per playbook and patient
python -m dxagent.nova.playbooks --sheet docs/PLAYBOOKS.md   # a readable review sheet (generated, not committed)
```

## The note

- **Template.** `nova_sdk.empty_record()` when importable, else `soap.FIELDS` (identical). A field is never added; if
  the server refuses the note, `SdkEnv.submit` repairs it once per error code.
- **S and O come only from the ledger.** One `ChartOutput` call writes them in the reference chart's style. Two guards
  check each field against what was said:
  - *numbers*: every number on a line must come from the answers, the age or the vital signs. Numbers spoken in Korean
    words count as digits ("thirty-two years" = 32), and a difference of two grounded numbers on one line is allowed
    ("usual 72 kg, now 66 kg (6 kg loss)"). A failing line is dropped and the rest of the field kept; a field with no
    line left is written from the raw answers;
  - *signs*: a (+)/(-) on a known item never discussed becomes `( )`, "not confirmed".
- **Layout.** S.cc: chief complaint as a clinical phrase, onset in chart form ("3 months ago"), a pain score. S.hpi: age
  and sex, then short chart lines, one theme each, the patient's worry or request last. S.pmh / fh / sh: the template
  items in order (DM / HTN, TB / hepatitis, operations, medicines, allergy; smoking, alcohol), `( )` when not asked, then
  the complaint's own items and `Others`. S.ros: the fixed block with (+)/(-) only for what was asked. O.exam: a vital
  sign line, then one line per region (Growth, General, Skin, H/E/E/N/T, Neck, Chest, Abdomen, Back, GU, Extremities /
  Neuro, Others), each finding with its turn.
- **A.evidence:** `#1. <diagnosis>`, then 3-5 bullets "facts with the patient's numbers : interpretation" (one from the
  history, one from the exam, one decisive negative); a bullet with a number the visit never produced is dropped. The
  diagnosis is one entity as specific as the findings support, naming the cause when the transcript points to one;
  "rule out" and joined diagnoses are removed. **A.differential:** one `r/o` line per alternative with its separating
  test, including every can't-miss cause the visit did not exclude.
- **P.tests:** one line per purpose, ordered by kind, at least one for the primary diagnosis and one per differential,
  plus safety tests the treatment needs. **P.follow_up:** interval and purpose, monitoring, named warning signs.
  **P.education:** 4-6 items, then the lines actually said, with turns.
- **Without the model** the note is still complete in the same layout: raw answers one item per line (clipped), evidence
  from the ledger, the playbook's can't-miss causes as `r/o` lines with their linked test, and its treatment, follow-up,
  warning signs and closing lines.
- **A and P** come from the assessment call, then `safety.py`: confirmatory and excluding tests (core mode), drug-class
  allergy checks, a disposition raise for unstable vital signs or a primary diagnosis lethal within hours, and a
  pregnancy check before radiation.

The guideline library is read from `NOVA_GUIDELINES_DIR` when the environment provides one (a Markdown corpus, optionally
with a catalog). `prepare()` indexes titles; the topic plan gets up to 1,500 characters of the best document for the two
leading hypotheses and the assessment up to 2,400 characters from up to 3 documents. The documents' licences limit them
to the competition: nothing derived from them is stored, compiled into `kb/` or put in the ZIP.

## Model calls and failure behavior

| Role | When |
|---|---|
| plan (topic) | once at turn 0, only when no playbook matched the opening |
| focus | once, after the priority-1 history of the complaint |
| review (`ReviewPlan`) | once, the diagnostic time-out after the history and the review of systems |
| assess | once: diagnosis, evidence, differential with separating tests, plan, 5 closing lines |
| notes (`ChartOutput`) | once: S and O in chart style, checked by the guards |
| core mode only: perceive, hypothesize, critic, bank, phrase | extraction after each block, hypotheses every 4 turns, critic before closing, questions when the planner has none |

All visit calls run at medium reasoning effort. A playbook visit makes 4 calls (5 without a playbook; one more for each
output that must be repaired); the core mode about 13. Failure handling:

- **Budget.** A call over a cap is refused and the visit continues on templates; the assessment and the note keep a
  reserve (30,000 output tokens, 3 calls).
- **Breaker.** After 4 consecutive failures only the assessment and the note still try.
- **Deadline.** Gathering stops 4 minutes before the deadline and model calls stop 75 s before it.
- **Exceptions.** Any exception moves the visit to closing, and DIAGNOSE is always sent. If even the start fails,
  `run.consult` sends an emergency note.

## Configuration

Every `NovaConfig` field (`dxagent/nova/config.py`) can be set as `NOVA_<FIELD>`: `1`/`0` for booleans, numbers as
written; `NOVA_EFFORT` and `NOVA_EFFORT_<ROLE>` set the reasoning level, `NOVA_MAX_TOKENS_<ROLE>` the output limit.

| Setting | Default | Meaning |
|---|---|---|
| `NOVA_API_BASE`, `NOVA_RUN_TOKEN`, `NOVA_DEV_KEY` | - | server root; official run token (set by the harness); personal development key |
| `NOVA_TRANSPORT` | `auto` | `auto`: `client.chat()` inside consult, HTTP elsewhere; `sdk`; `http` |
| `NOVA_ROUND`, `NOVA_PROFILE` | `prelim`, `standard` | `final` enables TEST; `lean` = a shorter visit (about 30 turns) |
| `NOVA_OFFLINE` | 0 | no model (tests); consult still makes the one required `client.chat()` call |
| `NOVA_GUIDELINES_DIR` | server path, else `./guidelines` | the guideline library (git-ignored locally) |
| `NOVA_MAX_CALLS`, `NOVA_MAX_INPUT_TOKENS`, `NOVA_MAX_OUTPUT_TOKENS` | 60, 300k, 90k | per-case model budget, under the server's limits |
| `NOVA_TIME_LIMIT`, `NOVA_WRAP_MARGIN`, `NOVA_FINAL_MARGIN`, `NOVA_CALL_TIMEOUT` | 1800, 240, 75, 240 s | consult limit (capped by the session deadline); stop gathering; stop calling the model; one call over HTTP |
| `NOVA_TURN_TARGET`, `NOVA_SOFT_TURN_CAP`, `NOVA_CLOSING_LINES` | 44, 38, 5 | usual length (an aim); priority-2 items past the cap are cut; explanation lines |
| `NOVA_PLAYBOOKS`, `NOVA_FOCUS_CALL`, `NOVA_REVIEW_CALL`, `NOVA_NOTES_CALL`, `NOVA_TOPIC_PLAN` | 1 | switches for the playbooks, the focus call, the time-out, the chart call and the turn-0 topic plan |
| `NOVA_PLAYBOOK_EXAM_MAX`, `NOVA_PLAYBOOK_EXAM_RESERVE`, `NOVA_FOLLOWUP_MAX` | 8, 6, 4 | maneuvers per visit; turns kept for them; playbook follow-ups per visit (safety follow-ups are never capped) |
| `NOVA_EMPATHY`, `NOVA_ADDRESS_BY_NAME`, `NOVA_CUE_EMPATHY_MAX`, `NOVA_SIGNPOSTS`, `NOVA_CONCERN_QUESTION` | 1, 1, 2, 1, 1 | communication |
| `NOVA_USE_GUIDELINES`, `NOVA_GUIDELINE_DOCS`, `NOVA_GUIDELINE_CHARS` | 1, 3, 2400 | guideline excerpts in the assessment |
| `NOVA_CORE` | 0 | 1: the Bayesian core around the visit (about 13 calls per case) |
| `NOVA_LOG_DIR` | - | per-case JSON log (development only: the server's file system is read-only) |

## Testing

| Tier | What | Cost |
|---|---|---|
| 0 | `pytest -q` (308 tests; the runtime's 206 need only the pinned wheels) | seconds |
| 1 | `experiments/nova_devkit_check.py` on the development kit's sample cases (its simulator, a scripted patient, `--audit`) | seconds |
| 1b | `experiments/nova_cpx_check.py` on the 18 local CPX-style cases with our own answer keys (history and key-exam coverage, the offline floor) | about 1 min |
| 2 | the kit's own local-model runner; `run.py --bench --sim-llm --judge` on Korean CPX-style cases | about 1 h per version |
| 3 | a sample run with a development key and `experiments/nova_live_report.py` (`docs/LIVE_RUNS.md`); then sample validation | model calls cost money |
| 4 | the official evaluation (a few cases, rationed) | one attempt |

An official evaluation scores only a few cases: a difference under about 10 points between runs is noise.

## Still open

- The official run scores the whole visit, not its parts: the focus call's and the time-out's choices, the assessment's
  quality, JSON reliability and latency per call are unmeasured separately.
- Whether compound questions ("do you have high blood pressure or diabetes?") draw the bundled-question reply on the
  server; the fallback handles either case at the cost of a turn.
- Final round: the team's own model from `NOVA_MODEL_DIR`, loaded in `prepare()`, with no `client.chat()`. `NovaLLM`
  takes any transport with `chat(payload, timeout)`; the local inference transport is not written.
