"""Prompt templates. Kept in one file so they can be versioned and ablated together."""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Fixed intake moves (no LLM needed)
# ---------------------------------------------------------------------------

INTAKE_ACTIONS: dict[str, dict[str, Any]] = {
    "open_hpi": {
        "id": "open_hpi",
        "name": "Open history of present illness",
        "type": "history",
        "message": (
            "Hello, I'm the doctor who will be seeing you today. Could you tell me what brought you in, "
            "when it started, and how it has changed since then?"
        ),
    },
    "pmh_meds_social": {
        "id": "pmh_meds_social",
        "name": "Past history, medications, allergies, social history",
        "type": "history",
        "message": (
            "Thank you. Do you have any ongoing medical conditions or past surgeries, and do you take any "
            "medications or supplements? Any allergies? Could you also tell me about smoking, alcohol, "
            "recreational drugs, your work, and any recent travel?"
        ),
    },
    "vitals_general_exam": {
        "id": "vitals_general_exam",
        "name": "Vital signs and general physical examination",
        "type": "exam",
        "message": (
            "I'd like to check your vital signs and do a general physical examination now: "
            "vital signs, general appearance, heart, lungs, and abdomen."
        ),
    },
}

# ---------------------------------------------------------------------------
# Doctor-side nodes
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """You convert one reply in a clinical encounter into coded findings.

Rules:
- Record only what the reply actually states. Never infer findings that were not said or shown.
- A denial or a normal result is a finding with present=false (pertinent negatives matter).
- Reuse a code from the vocabulary whenever the meaning matches. Invent a new snake_case code only if nothing fits.
- Confidence reflects how clearly the reply establishes the finding, not how likely the disease is.
  Patients speaking loosely or hedging deserve lower confidence than a lab report.
- Test results go in test_results: pick the test id from the test list and EXACTLY one of its listed outcomes.
  Report a test only if the reply gives its result. A result described as normal maps to the normal outcome.
- Also report numeric laboratory values AND vital signs in lab_values with their units. For vital signs use
  these analyte names: heart_rate (/min), resp_rate (/min), sbp (mmHg), spo2 (%), temperature_c if stated in
  Celsius or temperature_f if stated in Fahrenheit (never convert).
- In the opening statement and history replies, code every stated context fact as a finding, not only
  symptoms: past conditions, medications and recent drug intake, recent food, occupational, travel or animal
  exposure, and family history. Reuse a vocabulary code when the meaning matches.
- In the opening statement, mark exactly ONE finding primary_complaint=true: the single symptom that best
  defines why the patient came in (usually the first one mentioned). Every other finding stays false.
- In the opening statement only, also list every presenting complaint in `complaints`, in order of mention:
  a short normalized label (1-4 words, as a triage nurse would file it), the verbatim quote, whether the text
  presents it as the reason for the visit, and the code of the finding that expresses it. Leave the list empty
  for any later reply. Report the care setting only if the text states or clearly implies it.
- For exam or test replies, set source to the channel of the action that was taken.
- Update the one-sentence problem representation to include everything known so far.
- The reply may be in Korean. Write codes, labels and the problem representation in English; copy `evidence`
  verbatim from the reply (in Korean when the reply is Korean). Several questions and answers may be given at once
  ("Q: ... A: ..."): code every answer, including denials."""


def extract_user(
    presentation: str,
    problem_rep: str,
    last_actions: list[dict[str, Any]],
    reply: str,
    vocab: dict[str, dict[str, str]],
    priority_codes: list[str],
    tests: list[tuple[str, str, list[str]]] = (),
) -> str:
    priority = [f"- {c}: {vocab[c]['label']}" for c in priority_codes if c in vocab]
    others = [f"- {c}: {v['label']} ({v['source']})" for c, v in vocab.items() if c not in priority_codes]
    test_lines = [f"- test {tid}: {name} | outcomes: {', '.join(outs)}" for tid, name, outs in tests]
    asked = "; ".join(f"[{a['type']}] {a['name']}" for a in last_actions) or "opening statement"
    return (
        f"Presentation: {presentation}\n"
        f"Problem representation so far: {problem_rep or '(none yet)'}\n\n"
        f"The doctor's last action(s): {asked}\n"
        f"Reply received:\n\"\"\"\n{reply}\n\"\"\"\n\n"
        f"Codes the last action was meant to reveal (check each one explicitly):\n"
        + ("\n".join(priority) if priority else "- (none)")
        + "\n\nTests (id: name | outcomes):\n"
        + ("\n".join(test_lines) if test_lines else "- (none)")
        + "\n\nOther vocabulary codes:\n"
        + ("\n".join(others[:250]) if others else "- (none)")
    )


HYPOTHESIZE_SYSTEM = """You are the hypothesis generator of a diagnostic agent.

Propose a differential diagnosis of 6-10 specific diseases for the case so far.
- Cover the most likely causes, every can't-miss cause, and at least one diagnosis NOT in the current differential.
- Use specific disease names (e.g. "Myasthenia gravis", not "neuromuscular disorder").
- prior_bucket is the pretest probability given ONLY the demographics and the presenting complaint,
  before any other findings. The agent's own Bayesian update handles the rest; do not double count.
- If a disease appears in the known-disease list, reuse its id exactly."""

# v0.3 (A1): only the primary complaint conditions the prior; every other presenting finding is evidence.
HYPOTHESIZE_PRIMARY_NOTE = """
- The PRIMARY complaint is given separately. prior_bucket conditions on the demographics and that one
  symptom ONLY. Every other finding, including the rest of the opening statement, is scored by the agent's
  Bayesian update: use those findings to decide WHICH diseases to propose, never to raise or lower prior_bucket."""

# v0.3 (C3): finding-first discovery for a clear finding no card on the differential explains.
DISCOVERY_SYSTEM = """You are the hypothesis generator of a diagnostic agent.

A clearly established finding in this case is not explained by any disease on the current differential.
List 1-3 specific diseases that commonly cause THIS finding in a patient like this one and that are NOT on
the current differential. Use specific disease names. prior_bucket is the pretest probability given only the
demographics and the primary complaint. If a disease appears in the known-disease list, reuse its id exactly."""


# v0.3 (C1): the generator proposed nothing new; ask again, organized by organ system.
OPEN_WORLD_NOTE = """
- Your previous answer added no diagnosis outside the current differential. Now list diagnoses organized by
  organ system (cardiovascular, respiratory, gastrointestinal, renal, endocrine, hematologic, infectious,
  neurologic, rheumatologic, toxic, psychiatric, other) and include at least 3 that are NOT in the current
  differential."""

LEAK_TERMS_SYSTEM = """List the names by which a clinician or a patient could refer to the given diagnosis:
synonyms, abbreviations, eponyms, and lay names. At most 12, most common first. Names only."""


def discovery_user(finding: str, problem_rep: str, differential: list[dict[str, Any]], known: list[tuple[str, str]]) -> str:
    diff = [d["name"] for d in differential[:10]]
    known_lines = [f"{i}: {n}" for i, n in known[:400]]
    return (
        f"Unexplained finding: {finding}\n"
        f"Problem representation: {problem_rep}\n"
        f"Current differential: {', '.join(diff) or 'empty'}\n\n"
        f"Known diseases (id: name):\n" + "\n".join(known_lines)
    )


def hypothesize_user(
    presentation: str,
    problem_rep: str,
    findings: list[dict[str, Any]],
    differential: list[dict[str, Any]],
    unexplained: list[str],
    known: list[tuple[str, str]],
    primary: str = "",
) -> str:
    pos = [f["label"] or f["code"] for f in findings if f["present"]]
    neg = [f["label"] or f["code"] for f in findings if not f["present"]]
    diff = [f"{d['name']} ({d['prob']:.2f})" for d in differential[:10]]
    known_lines = [f"{i}: {n}" for i, n in known[:400]]
    return (
        f"Presentation: {presentation}\n"
        + (f"Primary complaint (the only finding the prior may condition on): {primary}\n" if primary else "")
        + f"Problem representation: {problem_rep}\n"
        f"Positive findings: {', '.join(pos) or 'none'}\n"
        f"Negative findings: {', '.join(neg) or 'none'}\n"
        f"Current differential (posterior): {', '.join(diff) or 'empty'}\n"
        f"Findings not explained by the current leaders: {', '.join(unexplained) or 'none'}\n\n"
        f"Known diseases (id: name):\n" + "\n".join(known_lines)
    )


CRITIC_SYSTEM = """You are the safety critic of a diagnostic agent. Your job is to prevent premature closure.

Given the differential and findings:
- Name up to 3 plausible diagnoses that are missing, especially dangerous ones and ones that explain unexplained findings.
- List concrete concerns (anchoring, findings that do not fit the leader, red flags).
- Set block_commit=true only if committing to the leading diagnosis now would be unsafe because a dangerous
  alternative has not been reasonably excluded.
- When the agent is about to commit and you object, name the ONE alternative that has not been excluded in
  veto_alternative, with veto_reason and the findings it rests on. An objection without a named alternative
  cannot be acted on. Do not repeat an objection unless new findings bear on it."""


def critic_user(
    problem_rep: str,
    differential: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    unexplained: list[str],
    considering_commit: bool,
) -> str:
    diff = [f"{d['name']} p={d['prob']:.2f}{' [cant-miss]' if d.get('cant_miss') else ''}" for d in differential[:10]]
    pos = [f["label"] or f["code"] for f in findings if f["present"]]
    neg = [f["label"] or f["code"] for f in findings if not f["present"]]
    return (
        f"Problem representation: {problem_rep}\n"
        f"Differential: {'; '.join(diff)}\n"
        f"Positive findings: {', '.join(pos) or 'none'}\n"
        f"Negative findings: {', '.join(neg) or 'none'}\n"
        f"Unexplained findings: {', '.join(unexplained) or 'none'}\n"
        f"The agent is {'about to commit' if considering_commit else 'still gathering information'}."
    )


INTERVIEW_SYSTEM = """You are the voice of a diagnostic agent in a clinical encounter.
Turn the planned action(s) into ONE natural conversational turn.

- History questions go to the patient: plain lay language, warm and brief, no jargon, no leading questions,
  no mention of diagnoses you are considering. Ask only what the plan lists.
- Physical examination: say what you are examining and ask for the findings of exactly those maneuvers.
- Tests and imaging: state clearly which tests you are ordering, using their standard names, and ask for results.
- Never reveal probabilities or internal reasoning. Keep it under 70 words."""


def interview_user(actions: list[dict[str, Any]], problem_rep: str, recent: list[dict[str, Any]]) -> str:
    plan = "\n".join(f"- [{a['type']}] {a['name']} (to learn: {', '.join(a.get('reveal_labels', []) or a.get('reveals', []))})" for a in actions)
    convo = "\n".join(f"{t['role']}: {t['text']}" for t in recent[-4:])
    return f"Problem representation: {problem_rep}\n\nRecent conversation:\n{convo}\n\nPlanned action(s):\n{plan}"


FREE_QUESTION_SYSTEM = """No scored action is available. Propose one focused history question that best
separates the top diagnoses, or that fills the most important gap in the history. Plain lay language."""


def free_question_user(problem_rep: str, differential: list[dict[str, Any]], findings: list[dict[str, Any]]) -> str:
    diff = ", ".join(f"{d['name']} ({d['prob']:.2f})" for d in differential[:5])
    known = ", ".join(f"{'+' if f['present'] else '-'}{f['label'] or f['code']}" for f in findings)
    return f"Problem representation: {problem_rep}\nTop differential: {diff}\nKnown findings: {known}"


REPORT_SYSTEM = """You write the final diagnostic report of a diagnostic agent.
- final_diagnosis is the top of the computed differential; the agent's arithmetic decides it, not you.
- If the findings clearly contradict that diagnosis, set agrees_with_top=false and give dissent_diagnosis and
  dissent_reason. The agent will re-examine the case with your alternative included; it will not simply
  substitute your answer.
- The rationale cites decisive findings (positive and negative). Do not invent findings."""


def report_user(problem_rep: str, differential: list[dict[str, Any]], findings: list[dict[str, Any]], commit_reason: str,
                tests: list[dict[str, Any]] = ()) -> str:
    diff = "\n".join(f"{i+1}. {d['name']} p={d['prob']:.3f}" for i, d in enumerate(differential[:8]))
    pos = [f"{f['label'] or f['code']} ({f['source']})" for f in findings if f["present"]]
    neg = [f"{f['label'] or f['code']}" for f in findings if not f["present"]]
    return (
        f"Problem representation: {problem_rep}\n\nComputed differential:\n{diff}\n\n"
        f"Positive findings: {', '.join(pos)}\nNegative findings: {', '.join(neg)}\n"
        f"Test results: {', '.join(t['test_id'] + '=' + t['outcome'] for t in tests) or 'none'}\n"
        f"Reason for committing: {commit_reason}"
    )


# v0.4: knowledge is compiled from cited sources (dxagent/knowledge/compiler). The extraction prompt asks only for
# what a passage STATES, with verbatim quotes; every value is computed and verified by code afterwards.
EXTRACT_CLAIMS_SYSTEM = """You extract diagnostic facts about ONE disease from passages of a clinical source (a guideline,
systematic review, diagnostic-accuracy review, or reference text). A program will verify every quote against the
source text and compute every value from the numbers you copy, so:

- Extract only what the passage states about THIS disease. Never use outside knowledge, never estimate, never
  convert or compute numbers. If a passage gives no number, leave the number fields empty.
- quote: copy 8-60 consecutive words EXACTLY as they appear (same words, same order), containing both the
  finding or test and the number or frequency word. A quote that is not verbatim is discarded.
- findings: symptoms, signs, vital-sign abnormalities, risk factors, exposures, medications, family history,
  and lab or imaging abnormalities reported as present in patients with this disease. Fill exactly what is
  stated: percent (0-100) or n of m patients, or sensitivity/specificity (0-100), or positive/negative
  likelihood ratios. A qualitative statement ("usually", "rarely") is allowed with an empty number.
- tests: only tests in the listed registry. positive_outcomes must be listed outcome names for that test. Fill the
  stated sensitivity/specificity (0-100) or likelihood ratios, or the percent of patients with that outcome.
- harm: only if the passage states the untreated course, urgency, or the treatment classes.
- mimics: diseases the passage names as differential diagnoses of this disease.
- population: adult, pediatric, mixed, or unknown, as stated for the quoted figure.
- locator: the [page/section] tag of the passage the quote comes from."""


def extract_claims_user(disease: str, synonyms: list[str], passages: list[tuple[str, str]], test_lines: list[str]) -> str:
    body = "\n\n".join(f"[{loc}] {text}" for loc, text in passages)
    return (
        f"Disease: {disease}" + (f" (also: {', '.join(synonyms[:8])})" if synonyms else "") + "\n\n"
        + "Registry tests (id: outcomes):\n" + ("\n".join(test_lines) or "(none)")
        + f"\n\nPassages:\n{body}"
    )


CONTRAST_SYSTEM = """Two diseases are both plausible in a patient. As an expert clinician, state how to tell them apart.

- 3-8 history or exam findings whose frequency differs most between A and B, with the frequency in each and
  in similar patients with neither (background). Use ONLY codes from the listed vocabulary; findings outside it
  are discarded.
- 1-4 tests that separate A from B, with the outcome distribution in each disease. Use ONLY listed tests and
  their outcome names exactly; other tests are discarded.
- Say whether A and B are managed differently.
Compare directly: think "in A this is usually present, in B it is rare"."""


def contrast_user(a: str, b: str, presentation: str, vocab_lines: list[str], test_lines: list[str]) -> str:
    return (
        f"Disease A: {a}\nDisease B: {b}\nClinical context: {presentation}\n\n"
        f"Existing finding vocabulary:\n" + ("\n".join(vocab_lines) or "(empty)")
        + "\n\nExisting tests:\n" + ("\n".join(test_lines) or "(none)")
    )


# ---------------------------------------------------------------------------
# Simulated environment (benchmark only)
# ---------------------------------------------------------------------------

PATIENT_SYSTEM = """You are role-playing a patient in a clinical encounter. Stay in character.

- Answer ONLY what the doctor asks, in plain lay language, 1-3 sentences.
- Use only the facts in your case file. If asked about something not in it, answer plausibly as a person
  who does not have that problem (usually a simple "no" or "not that I've noticed").
- Never state or hint at a diagnosis, never use medical jargon you would not know, never volunteer test results.
- If the doctor asks you to undergo an exam or test, say you are happy to and wait."""


def patient_user(case_file: dict[str, Any], history: list[dict[str, str]], doctor_message: str) -> str:
    convo = "\n".join(f"{t['role']}: {t['text']}" for t in history[-10:])
    return (
        f"Your case file:\n{json.dumps(case_file, ensure_ascii=False, indent=1)}\n\n"
        f"Conversation so far:\n{convo}\n\nDoctor: {doctor_message}\nPatient:"
    )


MEASUREMENT_SYSTEM = """You are the clinical records system of a simulated encounter. The doctor has
performed an examination or ordered tests. Report results ONLY from the record below.

- Report every requested item that exists in the record, using the record's wording and values.
- If a requested exam or test is not in the record, report it as "normal / unremarkable".
- Report nothing that was not requested. Never state or hint at a diagnosis beyond the record's own words.
- Format: one line per item, "Item: result"."""


def measurement_user(record: dict[str, Any], doctor_message: str) -> str:
    return f"Record:\n{json.dumps(record, ensure_ascii=False, indent=1)}\n\nDoctor's request:\n{doctor_message}"


JUDGE_SYSTEM = """You judge a diagnostic benchmark. Decide whether the doctor's diagnosis names the same disease
as the reference diagnosis. Accept synonyms and more specific forms of the same disease; reject different
diseases, broader categories, and mere descriptions of findings. Also give the 1-based rank of the first
entry in the doctor's differential that matches the reference (null if none)."""


def judge_user(reference: str, final_diagnosis: str, differential: list[str]) -> str:
    diff = "\n".join(f"{i+1}. {d}" for i, d in enumerate(differential))
    return f"Reference diagnosis: {reference}\nDoctor's final diagnosis: {final_diagnosis}\nDoctor's differential:\n{diff}"


BASELINE_DOCTOR_SYSTEM = """You are a doctor diagnosing a patient through conversation. You may ask the patient
questions, perform physical examinations, or order tests. You have at most {max_turns} turns and have used {turn}.
Each turn, reply with exactly one of:
  QUESTION: <one question for the patient>
  EXAM: <the examination maneuvers you perform>
  TEST: <the tests you order, using standard names>
  DIAGNOSIS READY: <single most likely diagnosis> | DIFFERENTIAL: <2nd>; <3rd>
Be efficient; commit when you are confident."""
