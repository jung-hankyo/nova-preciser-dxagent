# Building and checking a submission

The submission is a ZIP with `run.py`, `requirements.txt`, `NOTICE.md` and `dxagent/` at its root. It has no
`nova_sdk.py`: the SDK is not redistributed and the server provides its own copy (to bundle one anyway:
`zip submission.zip nova_sdk.py`).

Live runs on the contest server are scarce, so everything the repository can check without the server is checked before
the ZIP is built: the runtime tests, the playbooks, 65 simulated cases under 36 server, network, model and patient
scenarios (2,340 simulated visits), and the ZIP itself, unpacked and run in isolation. The live runs then only have to
confirm what the simulator cannot know.

## 1. Get the ZIP

**From GitHub.** Every push runs the `presubmit` workflow (Actions tab). On the latest green run of the branch, download
the artifact **submission**; its contents are the ZIP's layout.

**Locally, about two minutes.**

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt pytest
cp <starter>/nova_sdk.py .          # optional: bundled when present (git-ignored)
python -m sim.presubmit             # must end with "GO"; the ZIP is dist/submission.zip
```

`python -m sim.presubmit` prints one line per check:

| Check | What it proves |
|---|---|
| tests | the runtime and simulator tests pass (248) |
| playbooks | the 58 playbooks are valid and within their turn budgets |
| cases | the 65 simulated cases are well-formed and route to their playbooks |
| matrix | every case under every scenario: DIAGNOSE accepted once and before the deadline, the model called, nothing refused for length, nothing sent twice, a clean note (no instruction to a grader, no raw model output), a determined one-line diagnosis |
| zip | the ZIP passes the server's rules, imports its own code in an isolated interpreter, `prepare()` returns, the simulated visits pass, and nothing is written to disk or sent over the network |

The same run writes `runs/sim/presubmit/` (`results.jsonl`, `summary.md`); CI uploads it as the artifact
**simulation-results**.

## 2. The live test: sample validation

Upload the ZIP to the competition's sample validation. It returns a reference score per area (information gathering,
reasoning, tests, safety, communication, efficiency). How to read it:

| What the result shows | Likely cause | What changes |
|---|---|---|
| A case failed, timed out or scored 0 | the runtime on the server: a crash, the deadline, no model call | fix first; nothing else matters until every case completes |
| Efficiency low | too many turns or refused exams | fewer optional items (`turn_target`), or the refused maneuvers' names |
| Gathering low, efficiency high | the visit stops too early | more history (`turn_target`, `soft_turn_cap`) |
| Reasoning or diagnosis weak | the model's assessment | `"effort": {"assess": "high"}` |
| Tests low | the P.tests lines | the assessment prompt's test rules |
| Communication low | the explanations | `closing_lines`, the empathy lines |

## 3. Settings without code

`dxagent/nova/settings.json` (absent by default) changes defaults of `dxagent/nova/config.py` without editing code, and
travels inside the ZIP:

```json
{"effort": {"assess": "high"}, "closing_lines": 5}
```

Keys are the `NovaConfig` field names; unknown keys and values of the wrong type are ignored and logged, never fatal.
Commit the file and push (the workflow builds the new ZIP), or rebuild locally with `python -m sim.presubmit`.

## 4. The official evaluation

Submit the ZIP that passed sample validation, or the next version after its fixes, once its presubmit is GO. Official
evaluations are rationed, and a failed run still counts.

## 5. What the simulator checked, and what only the server can tell

Checked offline (`python -m sim --scenarios all`):

- **The model.** Answers that are bare, fenced, wrapped in prose or in their schema name, with extra keys, with values as
  strings, with trailing commas, empty, cut by the token limit, garbage, 50,000 characters long, carrying instructions
  to a grader, or with over-long, bundled and non-exam items. Calls of 200 s and 420 s (past the server's hold, then
  polled with the same request), a spent token budget, an older server that refuses `reasoning_effort`.
- **The server and network.** Lost requests and lost responses, patient and examiner errors, a 9-minute session, a
  session that expires mid-visit, turns already used, a refused first DIAGNOSE, no opening statement, no vital signs.
- **The patient.** Terse, verbose, strict about compound questions, injecting instructions, partly in English.
- **The SDK.** The real `nova_sdk.Client` (when `nova_sdk.py` is present) with only its socket layer simulated.

Only the server can tell how its patient and examiner models answer our questions and maneuvers, how fast the Doctor
model answers and how it formats its answers, and how the grader scores the note. The per-area scores of a live run are
the only signal.

## 6. If something goes wrong

| Symptom | What to do |
|---|---|
| The workflow run is red | open the run and read the failing step (presubmit prints the failing check) |
| The upload is refused | check the root holds `run.py` and `requirements.txt` (not a folder); rebuild with `python -m sim.presubmit` |
| A sample shows a crash or timeout | do not submit for the official evaluation until it is fixed |
