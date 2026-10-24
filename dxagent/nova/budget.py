"""Per-case budget: LLM calls, tokens and wall-clock time.

A call is refused (BudgetExhausted) when it would break a cap, so the encounter degrades to deterministic behavior
(templates, the yes/no fast path, a template-built SOAP note) instead of failing the case. A reserve is held back for
the closing roles (assessment, note phrasing) so the note is written by the model whenever possible.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from dxagent.nova.config import NovaConfig

CLOSING_ROLES = ("assess", "notes")


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, cfg: NovaConfig, clock: Callable[[], float] = time.monotonic,
                 time_limit_s: Optional[float] = None):
        self.cfg = cfg
        self.clock = clock
        self.t0 = clock()
        # the session deadline when the server gives one (client.session()["deadline"]), minus a network margin
        self.time_limit = cfg.time_limit_s if time_limit_s is None else max(60.0, min(cfg.time_limit_s,
                                                                                       time_limit_s - 15.0))
        self.calls = 0
        self.calls_ok = 0
        self.errors = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.by_role: dict[str, dict[str, float]] = {}

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

    def output_room(self, role: str) -> int:
        """Output tokens a call for `role` may still ask for (the closing roles may use the reserve)."""
        reserve = 0 if role in CLOSING_ROLES else self.cfg.reserve_output_tokens
        return self.cfg.max_output_tokens - reserve - self.output_tokens

    def call_timeout(self) -> float:
        """Never let one call run into the deadline: leave room for a template-built DIAGNOSE."""
        return max(5.0, min(self.cfg.call_timeout, self.remaining() - 20.0))

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

    def wrap_up_due(self, turns_used: int) -> bool:
        return self.remaining() < self.cfg.wrap_margin_s or turns_used >= self.cfg.gather_end()

    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "calls_ok": self.calls_ok, "errors": self.errors,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "seconds": round(self.elapsed(), 1), "by_role": self.by_role}
