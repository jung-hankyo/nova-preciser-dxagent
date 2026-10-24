"""N.O.V.A competition runtime (v0.8.1, "simple approach").

A thin, robust layer around the unchanged dxagent core (perceive -> hypothesize -> load_knowledge -> update_belief ->
plan). It speaks the contest protocol (SAY / EXAM / TEST / DIAGNOSE), keeps one question or one maneuver per turn,
asks a documentation checklist the core never would, explains the diagnosis to the patient, writes the SOAP note from
a ledger of what was actually asked, and never fails a case: every LLM call is budgeted, and any error or a closing
deadline still ends in a DIAGNOSE.

    config.py     NovaConfig: contest limits, model endpoint, budgets, encounter shape (all env-overridable)
    llm.py        NovaLLM: OpenAI-compatible client for the Doctor model, JSON extraction and repair, budget metering
    budget.py     per-case call / token / wall-clock budget
    protocol.py   actions, observations, validators (30 characters, one question, one maneuver), injection filter
    korean.py     yes/no fast path, vital-sign and demographic parsing, complaint groups
    checklist.py  documentation slots (HPI, background, ROS) and core exams per complaint group, in Korean
    ledger.py     every action, reply and turn number: the only source of the S and O sections
    prompts.py    nova-only prompts and output schemas (question bank, assessment, chart S/O, phrasing)
    agent.py      NovaAgent: the encounter (pull-style `run(env)` and push-style `StepAgent`)
    soap.py       the official S/O/A/P note with guards;  safety.py  deterministic plan checks
    sdk.py        SdkEnv / SdkTransport: the official client (say, exam, test, diagnose, chat) and error codes
    guidelines.py the contest guideline library at NOVA_GUIDELINES_DIR: title index, excerpts for the assessment
    env.py        LocalSimEnv (contest rules and error codes, simulated Korean patient), StdioEnv (development)
    devclient.py  LocalClient: an SDK stand-in over LocalSimEnv, so run.consult() runs end to end locally
    offline.py    offline stand-ins for every role (tests, demos without an API)
    judge.py      local 6-axis rubric judge;  bench.py  mini-bench;  package.py  submission ZIP
"""

from dxagent.nova.config import NovaConfig

__all__ = ["NovaConfig"]
