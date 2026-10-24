"""Per-case budget: LLM calls, tokens and wall-clock time.

A call is refused (BudgetExhausted) when it would break a cap, so the encounter degrades to deterministic behavior
(templates, the yes/no fast path, a template-built SOAP note) instead of failing the case. A reserve is held back for
the closing roles (assessment, note phrasing) so the note is written by the model whenever possible.

Time is planned from the pace actually observed in the case: the seconds per patient answer (turn_s) and per model
call (call_s), running means that start from conservative defaults. They size the end of the visit (close_s: the
closing explanations, the assessment call, DIAGNOSE), when gathering stops, and how long each model call may take
(call_timeout: what is left minus what the rest of the visit needs), so a slow server shortens the visit instead of
pushing DIAGNOSE past the deadline.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from dxagent.nova.config import NovaConfig

CLOSING_ROLES = ("assess", "notes")


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, cfg: NovaConfig, clock: Optional[Callable[[], float]] = None,
                 time_limit_s: Optional[float] = None):
        self.cfg = cfg
        self.clock = clock or time.monotonic
        self.t0 = self.clock()
        # the session deadline when the server gives one (client.session()["deadline"]), minus a network margin
        self.time_limit = cfg.time_limit_s if time_limit_s is None else max(60.0, min(cfg.time_limit_s,
                                                                                       time_limit_s - 15.0))
        self.calls = 0
        self.calls_ok = 0
        self.errors = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.by_role: dict[str, dict[str, float]] = {}
        self.turn_s = cfg.pace_turn_s  # seconds per patient action (running mean)
        self.call_s = cfg.pace_call_s  # seconds per model call (running mean)
        self.turns_left: Callable[[], int] = lambda: 0  # set by the encounter: actions still to come before closing

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def remaining(self) -> float:
        return self.time_limit - self.elapsed()

    def check(self, role: str) -> None:
        """Raise BudgetExhausted if a call for `role` may not be made now."""
        c = self.cfg
        closing = role in CLOSING_ROLES
        rc = 0 if closing else c.reserve_calls
        ri = 0 if closing else c.reserve_input_tokens
        ro = 0 if closing else c.reserve_output_tokens
        if self.calls >= c.max_calls - rc:
            raise BudgetExhausted(f"call cap ({self.calls}/{c.max_calls})")
        if self.input_tokens >= c.max_input_tokens - ri:
            raise BudgetExhausted(f"input-token cap ({self.input_tokens}/{c.max_input_tokens})")
        if self.output_tokens >= c.max_output_tokens - ro:
            raise BudgetExhausted(f"output-token cap ({self.output_tokens}/{c.max_output_tokens})")
        floor = c.final_margin_s * (0.4 if closing else 1.0)
        if self.remaining() < floor:
            raise BudgetExhausted(f"time ({self.remaining():.0f}s left)")
        if self.call_timeout(role) < (20.0 if closing else 30.0):
            # too little time for an answer: a call abandoned midway would keep the server's model busy (REQUEST_PENDING)
            # for the next one, so it is not made at all
            raise BudgetExhausted(f"no time for a {role} call ({self.remaining():.0f}s left)")

    def output_room(self, role: str) -> int:
        """Output tokens a call for `role` may still ask for (the closing roles may use the reserve)."""
        reserve = 0 if role in CLOSING_ROLES else self.cfg.reserve_output_tokens
        return self.cfg.max_output_tokens - reserve - self.output_tokens

    def note_action(self, seconds: float) -> None:
        """One SAY/EXAM took this long: update the pace."""
        if seconds >= 0:
            self.turn_s = min(180.0, max(0.5, 0.7 * self.turn_s + 0.3 * float(seconds)))

    def close_s(self) -> float:
        """Seconds the end of the visit needs: the closing explanations, the assessment call and DIAGNOSE."""
        return self.cfg.closing_lines * self.turn_s + self.call_s + 30.0

    def call_timeout(self, role: str = "other") -> float:
        """The longest one call for `role` may take: what is left minus what the visit still needs after it (the
        remaining actions and the end of the visit mid-visit; the closing explanations after the assessment; DIAGNOSE
        after the chart), never more than cfg.call_timeout."""
        if role == "assess":
            reserve = self.cfg.closing_lines * self.turn_s + 25.0
        elif role in CLOSING_ROLES:
            reserve = 20.0
        else:
            reserve = self.turns_left() * self.turn_s + self.close_s()
        return max(5.0, min(self.cfg.call_timeout, self.remaining() - reserve))

    def charge(self, role: str, input_tokens: int, output_tokens: int, ok: bool, seconds: float) -> None:
        self.calls += 1
        self.calls_ok += int(ok)
        self.errors += int(not ok)
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        r = self.by_role.setdefault(role, {"calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0})
        r["calls"] += 1
        r["errors"] += int(not ok)
        r["input_tokens"] += int(input_tokens)
        r["output_tokens"] += int(output_tokens)
        r["seconds"] = round(r["seconds"] + seconds, 2)
        if seconds >= 1.0 and (ok or seconds > self.call_s):  # a failure that waited long says the model is slow
            self.call_s = min(900.0, 0.5 * self.call_s + 0.5 * float(seconds))

    def wrap_up_due(self, turns_used: int) -> bool:
        """Stop gathering: the turns for the closing explanations, or the time the end of the visit needs."""
        return (self.remaining() < max(self.cfg.wrap_margin_s, self.close_s())
                or turns_used >= self.cfg.gather_end())

    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "calls_ok": self.calls_ok, "errors": self.errors,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "seconds": round(self.elapsed(), 1), "turn_s": round(self.turn_s, 1), "call_s": round(self.call_s, 1),
                "by_role": self.by_role}
