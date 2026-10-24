"""NovaConfig: every contest limit and runtime choice, overridable by environment variables (NOVA_*).

The limits follow the competition's rules and its SDK (neither is included in this repository): a bounded number of
turns and minutes per case; SAY, EXAM and TEST of at most 30 code points, one question or one maneuver at a time (a
wrongly sent one costs a turn); DIAGNOSE with the fixed S/O/A/P fields; the Doctor model only through the contest's
metered route, with per-case call and token limits and at least one call per case; a guideline library at
NOVA_GUIDELINES_DIR when the environment provides one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dxagent.config import PACKAGE_DIR, PROJECT_DIR, Settings

ROLES = ("perceive", "hypothesize", "critic", "plan", "focus", "review", "bank", "phrase", "assess", "notes", "complaint",
         "prior", "patient", "judge")

# The visit's calls (topic plan, focus, time-out, assessment, chart) reason at "medium" (given explicitly, it gave
# fewer empty answers than the default in earlier tests, and "low" more). The core's roles (off by default) stay "low". Reasoning tokens count against
# max_tokens and gpt-6-luna may write long answers, hence the room below: the visit's four calls may use 44,000 output
# tokens together, within the case's output budget (an empty answer is retried once with twice the room, up to 32,000).
DEFAULT_EFFORT = {"plan": "medium", "focus": "medium", "review": "medium", "assess": "medium",
                  "notes": "medium"}  # others: "low"
DEFAULT_MAX_TOKENS = {"perceive": 5000, "hypothesize": 4000, "critic": 3000, "plan": 8000, "focus": 8000,
                      "review": 8000, "bank": 3000,
                      "phrase": 1500,
                      "assess": 16000, "notes": 12000, "complaint": 1500, "prior": 2500, "patient": 1500, "judge": 5000}
MAX_TOKENS_CEILING = 32000  # an empty answer is retried once with twice the room, never more than this

PROFILES = {
    # standard: at most 49 turns (the plan to 38, follow-ups and the time-out's items to 44, 5 closing explanations).
    # lean: about 30 (priority-1 items first, 2 closing lines), cut to fit.
    "standard": {},
    "lean": {"diag_history_max": 4, "diag_exam_max": 2, "core_exam_max": 3, "closing_lines": 2,
             "ask_alcohol": False, "ask_tbc_hep": False, "ask_prior_care": False, "smoking_followup": False,
             "topic_questions_max": 6, "topic_exams_max": 2, "soft_turn_cap": 26, "turn_target": 30,
             "review_questions_max": 2, "review_exams_max": 1, "review_reserve": 2, "followup_max": 2,
             "gather_max": 30},
}


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _envb(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def _envi(name: str, default: int) -> int:
    return int(float(_env(name, str(default))))


def _envf(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass
class NovaConfig:
    round: str = "prelim"  # prelim (SAY, EXAM, DIAGNOSE) | final (+ TEST)
    profile: str = "standard"

    # Model endpoint (OpenAI-compatible chat completions). Preliminary round: the contest server's gpt-6-luna.
    # Final round: the team's own model served locally (e.g. vLLM on a local GPU), same client.
    transport: str = "auto"  # auto: SDK client.chat() inside consult(), HTTP elsewhere | sdk | http
    offline: bool = False  # no model at all (tests); consult() still makes the one required client.chat() call
    api_base: str = ""
    api_key: str = ""
    model: str = "openai/gpt-6-luna"  # sent on the HTTP route only; the SDK route omits it (the Doctor either way)
    temperature: Optional[float] = None  # not sent: gpt-6-luna takes only its default (the server ignores the field);
    # NOVA_TEMPERATURE sets one for a local server
    json_mode: str = "none"  # none | json_object | json_schema. The contest's chat endpoint takes only messages,
    # max_tokens, reasoning_effort and temperature (response_format: 400 INVALID_COMPLETION_FIELDS); JSON is parsed
    # leniently from the text. json_object or json_schema for a local server that supports them (auto-downgraded)
    call_timeout: float = 240.0  # the SDK's own wait (1.0.1); the server gives up on a call earlier
    retries: int = 2
    effort: dict = field(default_factory=dict)  # role -> low | medium | high ("" = do not send)
    max_tokens: dict = field(default_factory=dict)  # role -> max output tokens (reasoning tokens included)

    # Contest limits.
    max_turns: int = 50
    time_limit_s: float = 1800.0  # consult() per case (the session deadline is used when the server gives one)
    wrap_margin_s: float = 240.0  # stop gathering when less than this remains
    final_margin_s: float = 75.0  # below this, the SOAP note is written without the LLM
    say_max_chars: int = 30

    # Per-case LLM budget, with a reserve kept for the assessment and the note.
    max_calls: int = 60
    max_input_tokens: int = 300_000
    max_output_tokens: int = 90_000  # the contest's 100,000 less a margin: a call never asks past it (Budget.output_room)
    reserve_calls: int = 3
    reserve_input_tokens: int = 24_000
    reserve_output_tokens: int = 30_000  # room for the assessment and the chart

    # Encounter shape (turns).
    turn_target: int = 44  # the usual length of a visit, closing explanations included: the focus call is told it and
    # trims optional items toward it (efficiency is 20 of 100 points), but nothing is cut only to reach it
    soft_turn_cap: int = 38  # the selection cuts priority-2 items that would plan the visit past this turn, leaving
    # room for the follow-ups and the time-out's items; gathering itself ends at max_turns - closing_lines - 1 (44)
    closing_lines: int = 5  # explanations at the end: likely cause, tests, treatment, warning signs, the worry
    gather_max: int = 0  # the last gathering turn (0: max_turns - closing_lines - 1, i.e. 44)
    # Presentation playbooks (dxagent/nova/playbooks/*.yaml): the complaint's history, exams, differential and plan.
    playbooks: bool = True
    playbook_exam_max: int = 8  # playbook maneuvers (priority order)
    playbook_exam_reserve: int = 6  # turns kept for the maneuvers while the history is taken
    # Mid-visit focus (one call after the history of the complaint): leading diagnoses, which optional items to skip,
    # which priority-3 items to add, and a few case-specific questions and maneuvers.
    focus_call: bool = True
    focus_questions_max: int = 3
    focus_exams_max: int = 2
    # Diagnostic time-out (one call after the history and the review of systems, before the examination): the leading
    # diagnoses with findings for and against, the can't-miss causes still open, whether the leader names a cause, a
    # few questions that would most change the ranking, the planned maneuvers ranked by what they separate.
    review_call: bool = True
    review_questions_max: int = 3
    review_exams_max: int = 2
    review_reserve: int = 4  # turns kept for the time-out's items while the history is taken
    # Playbook follow-ups (`then`): questions asked right after a positive answer (how much, how long, escalating).
    followup_max: int = 4
    topic_plan: bool = True  # one call at turn 0, only when no playbook matched the opening (it may pick one)
    topic_questions_max: int = 10
    topic_exams_max: int = 4
    # Without a playbook (complaint groups) and in the core mode.
    core_exam_max: int = 5  # standard maneuvers for the complaint group
    diag_history_max: int = 6  # core mode: questions chosen by the core planner (or the LLM question bank)
    diag_exam_max: int = 3  # core mode: exam maneuvers chosen by the core planner (or the LLM question bank)
    test_max: int = 4  # final round only
    # Communication.
    empathy: bool = True  # one acknowledgment early in the history, worded from the opening (pain, worry, a child)
    address_by_name: bool = True  # "안녕하세요, 이수진 님." / "안녕하세요, 어머님." when it fits in 30 characters
    cue_empathy_max: int = 2  # acknowledgments of worry, distress or severe pain voiced in a reply
    signposts: bool = True  # "이제 지난 병력을 여쭤볼게요." before the background, when it fits
    concern_question: bool = True  # "가장 걱정되는 점이 무엇인가요?" at the end of the history
    ask_alcohol: bool = True
    ask_job: bool = False
    ask_tbc_hep: bool = True
    ask_prior_care: bool = True
    smoking_followup: bool = True  # "하루에 몇 개비 피우세요?" when the patient smokes (S.sh wants p/d)

    # Guideline library (read-only, at NOVA_GUIDELINES_DIR when the environment provides one).
    use_guidelines: bool = True
    guidelines_dir: str = ""  # default: $NOVA_GUIDELINES_DIR, else ./guidelines
    guideline_docs: int = 3  # documents quoted in the assessment prompt
    guideline_chars: int = 2400  # total excerpt characters in the assessment prompt

    # Knowledge.
    use_fixtures: bool = False  # the 8 illustrative chest-pain/dyspnea cards: off since the playbooks (they misled on
    # unseen cases: experiments/README.md); NOVA_USE_FIXTURES=1 for the core's tests and the AgentClinic bench
    kb_dir: Path = field(default_factory=lambda: PROJECT_DIR / "kb")
    cards_dirs: list = field(default_factory=lambda: [PACKAGE_DIR / "nova" / "cards"])

    # Reasoning. core=False: the visit makes 3 model calls with a playbook (focus, assessment, chart) and 4 without.
    # core=True restores the Bayesian core around it (finding extraction after every block, hypotheses, critic,
    # value-of-information questions, question bank): about 13 calls, and it needs knowledge cards to add anything.
    core: bool = False
    final_dx: str = "auto"  # auto: the core's leader if it is card-backed and >= core_trust, else the LLM's | core | llm
    core_trust: float = 0.5
    # The core's reverse index adds card-backed diseases the model never proposed. With the narrow card set shipped
    # now (8 chest-pain/dyspnea fixtures) and a broad outpatient case mix (insomnia, weight loss, short
    # stature), it mostly adds wrong cardiopulmonary candidates, so it stays off until kb/ is broad.
    reverse_index: bool = False
    hypothesize_every: int = 4
    hypothesize_max: int = 4
    bank_max_calls: int = 2
    critic: bool = True
    notes_call: bool = True  # one LLM call that rewrites raw answers as note phrases (digits checked against the answer)
    yes_no_fast_path: bool = True
    prior_context: bool = False  # v0.3.1 P1 assessor off: no measured gain, +5.5 calls per case

    log_dir: str = ""  # write one JSON log per case here (never needed by the contest)

    @property
    def allow_tests(self) -> bool:
        return self.round == "final"

    def effort_for(self, role: str) -> str:
        return self.effort.get(role, DEFAULT_EFFORT.get(role, "low"))

    def max_tokens_for(self, role: str) -> int:
        return int(self.max_tokens.get(role, DEFAULT_MAX_TOKENS.get(role, 1500)))

    def gather_end(self) -> int:
        """The last turn of history and examination: the closing explanations and one spare turn follow."""
        end = self.max_turns - self.closing_lines - 1
        return min(end, self.gather_max) if self.gather_max > 0 else end

    @classmethod
    def from_env(cls, **overrides) -> "NovaConfig":
        c = cls()
        c.round = _env("NOVA_ROUND", c.round).lower()
        c.profile = _env("NOVA_PROFILE", c.profile).lower()
        for k, v in PROFILES.get(c.profile, {}).items():
            setattr(c, k, v)
        c.transport = _env("NOVA_TRANSPORT", c.transport).lower()
        c.offline = _envb("NOVA_OFFLINE", c.offline)
        c.api_base = _env("NOVA_API_BASE", _env("OPENAI_BASE_URL", ""))
        # official run: the server-issued run token; development: the personal key (both Bearer tokens)
        c.api_key = _env("NOVA_RUN_TOKEN", _env("NOVA_DEV_KEY", _env("NOVA_API_KEY", _env("OPENAI_API_KEY", ""))))
        c.model = _env("NOVA_MODEL", c.model)
        t = _env("NOVA_TEMPERATURE", "" if c.temperature is None else str(c.temperature))
        c.temperature = None if t.lower() in ("", "none", "default") else float(t)
        c.json_mode = _env("NOVA_JSON_MODE", c.json_mode).lower()
        c.call_timeout = _envf("NOVA_CALL_TIMEOUT", c.call_timeout)
        c.retries = _envi("NOVA_RETRIES", c.retries)
        g = os.getenv("NOVA_EFFORT")
        for r in ROLES:
            e = os.getenv(f"NOVA_EFFORT_{r.upper()}") or g
            if e is not None:
                c.effort[r] = "" if e.lower() in ("none", "off") else e.lower()
            m = os.getenv(f"NOVA_MAX_TOKENS_{r.upper()}")
            if m:
                c.max_tokens[r] = int(m)
        c.max_turns = _envi("NOVA_MAX_TURNS", c.max_turns)
        c.time_limit_s = _envf("NOVA_TIME_LIMIT", c.time_limit_s)
        c.wrap_margin_s = _envf("NOVA_WRAP_MARGIN", c.wrap_margin_s)
        c.final_margin_s = _envf("NOVA_FINAL_MARGIN", c.final_margin_s)
        c.say_max_chars = _envi("NOVA_SAY_MAX_CHARS", c.say_max_chars)
        c.max_calls = _envi("NOVA_MAX_CALLS", c.max_calls)
        c.max_input_tokens = _envi("NOVA_MAX_INPUT_TOKENS", c.max_input_tokens)
        c.max_output_tokens = _envi("NOVA_MAX_OUTPUT_TOKENS", c.max_output_tokens)
        for k in ("turn_target", "soft_turn_cap", "diag_history_max", "diag_exam_max", "core_exam_max", "test_max",
                  "closing_lines", "hypothesize_every", "hypothesize_max", "bank_max_calls", "guideline_docs",
                  "guideline_chars", "topic_questions_max", "topic_exams_max", "playbook_exam_max",
                  "playbook_exam_reserve", "focus_questions_max", "focus_exams_max", "cue_empathy_max",
                  "review_questions_max", "review_exams_max", "review_reserve", "followup_max", "gather_max"):
            setattr(c, k, _envi(f"NOVA_{k.upper()}", getattr(c, k)))
        for k in ("empathy", "address_by_name", "signposts", "concern_question", "ask_alcohol", "ask_job", "ask_tbc_hep",
                  "ask_prior_care", "smoking_followup", "topic_plan", "focus_call", "review_call", "core", "reverse_index",
                  "playbooks",
                  "use_guidelines", "use_fixtures", "critic", "notes_call", "yes_no_fast_path", "prior_context"):
            setattr(c, k, _envb(f"NOVA_{k.upper()}", getattr(c, k)))
        c.final_dx = _env("NOVA_FINAL_DX", c.final_dx).lower()
        c.core_trust = _envf("NOVA_CORE_TRUST", c.core_trust)
        c.kb_dir = Path(_env("DX_KB_DIR", str(c.kb_dir)))
        extra = [Path(p) for p in _env("NOVA_CARDS_DIRS", "").split(os.pathsep) if p]
        c.cards_dirs = list(c.cards_dirs) + extra
        c.log_dir = _env("NOVA_LOG_DIR", c.log_dir)
        c.guidelines_dir = _env("NOVA_GUIDELINES_DIR", c.guidelines_dir)
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def core_settings(self) -> Settings:
        """The unchanged core's settings, adapted to the contest: one action per turn, no tests in the preliminary
        round, no intake script (the checklist replaces it), no runtime knowledge drafting or document fallbacks,
        no K = 3 self-consistency calls."""
        s = Settings()
        s.max_batch_history = 1
        s.max_batch_tests = 1
        s.intake_script = ()
        s.max_turns = self.max_turns  # the runtime enforces its own soft cap; the core never forces a commit early
        s.allowed_action_types = ("history", "exam") + (("poc", "lab", "imaging", "advanced_imaging", "procedure")
                                                         if self.allow_tests else ())
        s.contrast_cards = False
        s.local_fallback = False
        s.live_fallback = False
        s.prior_context = self.prior_context
        s.prior_samples = 1
        s.open_world_enforce = False
        s.cardless_discovery = False
        s.conditional_batches = False
        s.critic = self.critic
        s.kb_dir = self.kb_dir
        s.reverse_index = self.reverse_index
        return s
