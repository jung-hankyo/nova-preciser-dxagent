# Changelog

Newest first. Decision numbers refer to `docs/decisions/`; test counts are the full suite at release (the runtime's share
in parentheses where given). Each branch's changelog stops at its own version.

## 0.9.0-pre (2026-10-20, `second-clean-approach`)

A simulated contest, a presubmit gate and the hardening they found. Official run: **66.174**. The package version string
stays 0.8.1.

- **`sim/`:** a simulated contest server, an SDK-faithful client (or the real client over a simulated socket), a scripted
  patient and examiner, Doctor stand-ins (oracle, faulty, record/replay), a virtual clock, 36 scenarios, 65 cases
  (`experiments/data/nova_cpx/`: 18 original, 37 more, 10 special situations). 2,340 visits pass.
- **Presubmit:** `python -m sim.presubmit` (tests, playbooks, cases, the matrix, a ZIP check in an isolated interpreter
  with every file write and network connection recorded); `.github/workflows/presubmit.yml` runs it on every push and
  keeps the ZIP's contents as an artifact.
- **Hardening found by the matrix:** a still-running request is polled under the same `request_id` instead of being
  reopened (a slow model used to yield nothing); every model call, SAY/EXAM and DIAGNOSE runs under a watchdog; the time
  budget follows the case's observed pace; a diagnosis exists even when no model call succeeded; JSON wrapped in its
  schema name is unwrapped; a cut answer is retried once; routing and onset fixes.
- **Packaging:** the SDK file is optional in the ZIP; `dxagent/nova/settings.json` changes defaults without code.
- 350 tests (248).

## 0.8.1 (2026-10-17, `first-simple-approach`)

A new Doctor model. Decision 104. Official run: **63.131**. The model ID is sent on the HTTP route only; temperature is
never sent; effort stays medium; output room per call was raised (assessment 16,000 tokens) with one retry at twice the room
and a case cap that keeps room for the assessment and the chart. 308 tests (206).

## 0.8.0 (2026-10-16)

An evidence review applied to the visit, the playbooks and the note. Decisions 95-103.

- **Diagnostic time-out:** one call after the history (leading diagnoses for and against, open can't-miss causes, up to 3
  questions that would change the ranking, the exam ranked). A visit makes 4 calls.
- **Follow-ups** after positive answers; **5 closing explanations**; gathering ends at 44, the plan is cut at 38.
- **Note:** the number check reads Korean numerals; P.tests per purpose from linked tests; a three-part follow-up.
- **Safety:** pelvic and speculum exams only for sexually active patients. **Playbooks:** tests and a `plan` block in all;
  corrections from the audits; five new ones (dyslipidemia, oliguria, neck_pain, nasal_symptoms, easy_bruising).
- Offline, 18 local cases: history 0.84, key exams 0.83, 42.5 turns. 307 tests.

## 0.7.2 and 0.7.1 (2026-10-12)

Every maneuver one action (decisions 92-94): 18 bundled maneuver names became single actions across 30 playbooks and
model-written maneuvers are filtered the same way. The revised rules (86-91): 40 turns became an aim, not a cut; wrongly
sent actions cost a turn and are counted; `reasoning_effort="medium"`; slow calls are resent under the same
`request_id`; an SDK older than 1.0.1 is refused; the most specific diagnosis; a revised note layout. 280 tests (178).

## 0.7.0 (2026-10-11)

A lean, patient-specific visit and a chart-shaped note. Decisions 80-85. **Model calls 13 to 3** per playbook visit: the
Bayesian core is off by default. One mid-visit focus call; priority tiers; communication inside 30 characters; a note with
chart sections, turns and `( )` for what was not asked. 275 tests.

## 0.6.0 (2026-10-11)

Presentation playbooks. Decisions 72-79. 53 YAML playbooks (one per CPX chief complaint) with a loader, selection,
validator and review sheet; six independent AI clinical reviewers (320 findings, each checked by hand); fixture cards off.
Offline, 18 local cases: history 0.55 to 0.85, key exams 0.36 to 0.86. 258 tests.

## 0.5.2, 0.5.1, 0.5.0 (2026-10-10)

The contest runtime `dxagent/nova` and `run.py` around the unchanged core. Decisions 38-71.

- **0.5.2** aligned with SDK 1.0 and the development kit: a topic plan at turn 0, children interviewed through the
  guardian, error codes by class, a time budget anchored to the session deadline. 144 tests.
- **0.5.1** the official contract: `prepare()` / `consult()`, the SDK adapter and model route through `client.chat()`,
  the 30-character rule, bundled-question handling, the ledger-based note with number and sign guards, the guideline
  library, pinned requirements, the ZIP builder. 140 tests.
- **0.5.0** a runtime with its own model client, per-case budget, one action per turn, a Korean documentation checklist,
  deterministic safety checks and the SOAP note from the ledger. Simulation: one action per turn costs -0.9 pp top-1, no
  tests -18.9 pp. 123 tests.

## 0.4.2 (2026-10-02, `main`)

DeepSeek (OpenAI-compatible endpoint, thinking mode per spec) and OpenAI reasoning models; default profile DeepSeek agent
with OpenAI environment and judge; the usage ledger records cached and reasoning tokens and the report estimates cost per
case. 102 tests.

## 0.4.1 (2026-10-01)

Local-document retrieval as the primary fallback for a disease with no card; `docs/SOURCES.md` and the manifest template;
live-benchmark reporting (`dxagent bench-report`). 96 tests.

## 0.4.0 (2026-10-01)

The knowledge compiler. Decisions 29-35. Cards are compiled offline from a source manifest (the LLM copies verbatim
quotes, code verifies and computes, confidence comes from a fixed tier table); MONDO and HPO ontology spine; HPO
annotations give about 9,400 rare-disease cards without an LLM; card-less diseases inherit their nearest carded ancestor.
Removed: LLM-only card drafting. Held-out simulation: 0.938 top-1 with full coverage; the result is set by coverage and
the share of entries sources state. 91 tests.

## 0.3.1 (2026-10-01)

Prior context (decisions 20-22): the prior conditions only on age, sex, setting and the primary complaint through an
information-hidden assessor; a deterministic primary-complaint rule; context enters as evidence. Neutral in simulation
(+5.5 calls per case), adopted for the structural guarantee. 78 tests.

## 0.3.0

Six fixes, measured together on held-out worlds (top-1 0.887 to 0.930, Brier -0.047, ECE -0.026, 0.46 fewer turns, $27
less test cost per case): the primary complaint alone conditions the prior (+1.4 pp); a deterministic lexicon and
canonical finding codes (+0.7 pp); card prevalence for the reverse index (+1.0 pp); a critic's veto must name an
alternative; the report LLM's dissent is an objection (+3.0 pp over letting its name win).

## 0.2.1

The reverse index: known diseases the LLM never proposed join the differential when the evidence makes them at least 5%
likely. Top-1 +24 / +15 / +4 pp at LLM recall 0.6 / 0.75 / 0.9.

## 0.2.0

Frequencies replace stored likelihood ratios: cards hold `P(finding | disease)` and `P(test result | disease)`, ratios are
formed between the live diseases at run time, tests are categorical, and "A vs B" contrast cards are elicited when two
diseases are both live. Top-1 +0.8 to +2.0 pp and 26-28% less test cost than 0.1.

## 0.1.0

The first agent: a LangGraph interview loop (perceive, hypothesize, update belief, critic, plan, interview, report) with a
Bayesian belief update and a value-of-information planner, and likelihood ratios stored in its knowledge cards.
