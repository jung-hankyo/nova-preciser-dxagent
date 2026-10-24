# Live runs with a development key

For participants who hold a development key. Run every step in your own terminal: the key is never given to a coding
tool and never written to a file, a commit or a chat (the competition's rules forbid sharing it).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
cp <starter>/nova_sdk.py .        # the competition's SDK and sample runner: git-ignored, never commit them
cp <starter>/sample.py .
```

The sample runner must sit in the repository root, next to `run.py`, so that it imports this agent. Without a local
guideline library (`./guidelines`, git-ignored) the assessment gets no guideline excerpts.

## Fast checks (no key)

```bash
python -m sim.presubmit       # ends with GO; builds dist/submission.zip
```

It runs the runtime and simulator tests, the playbooks, the simulated cases, the scenario matrix and the isolated ZIP
check (`docs/SUBMISSION.md`).

If a check fails, stop there.

## A live run and its report

```bash
export NOVA_API_BASE=<server root>
read -rs -p 'Development API key: ' NOVA_DEV_KEY; echo; export NOVA_DEV_KEY
export NOVA_LOG_DIR=runs/live/<version>      # one JSON log per case (runs/ is git-ignored)
python sample.py --case <case ID>
python -B experiments/nova_live_report.py runs/live/<version>
grep -lF "$NOVA_DEV_KEY" runs/live/<version>/* || echo "clean: no key in the logs"
```

`read -rs` keeps the key off the screen and out of the shell history; it lasts until the terminal closes. The report
script prints a table row per case and writes `runs/live/<version>/<case>.md`: the outcome (playbook, diagnosis, turns,
refused actions), every model call with tokens, seconds and error, the dialogue, the items dropped for lack of turns, the
visit's events and the submitted note. What to look for:

- **Model calls:** a failed call, an answer cut at its token limit, calls slow enough to endanger the case's time limit.
- **The visit:** turns and refused actions; whether the time-out asked useful questions and the follow-ups fired.
- **The note:** lines dropped by the number check; the diagnosis and plan.

One case is an anecdote. Compare versions on the same cases, each version in its own log folder, and decide only on
clear differences (a failed call, a wrong diagnosis, a missing plan item).

## When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `KeyError: 'NOVA_DEV_KEY'` | the key is not exported in this terminal | export it again |
| HTTP 401 or 403 | a wrong key, or an outdated SDK | re-enter the key; take the SDK from the current starter package |
| a model-limit error code | the case's model budget is spent | the visit finishes without the model; the report says why |
| `no case logs found` | `NOVA_LOG_DIR` was not set when the sample runner ran | export it and rerun |
| a Python traceback | a bug | keep the full traceback with the report |

**Never** commit `nova_sdk.py`, `sample.py`, `guidelines/` or `runs/` (all four are git-ignored, so a plain `git add .`
is safe).
