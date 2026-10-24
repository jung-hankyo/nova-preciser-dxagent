"""LLM usage ledger (v0.4.1): calls, tokens and seconds per (case, role, model, call kind).

`LLM` records every call here. The benchmark runner sets the current case per worker thread (`set_case`), so usage
is attributed to cases even with parallel workers; outside a benchmark the case key is None.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Optional

_local = threading.local()
_lock = threading.Lock()
_ledger: dict[tuple, dict[str, float]] = defaultdict(lambda: {"calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0,
                                                              "cache_read_tokens": 0, "reasoning_tokens": 0, "seconds": 0.0})


def set_case(case: Optional[Any]) -> None:
    _local.case = case


def current_case() -> Optional[Any]:
    return getattr(_local, "case", None)


def record(role: str, model: str, kind: str, usage: Optional[dict] = None, seconds: float = 0.0, error: bool = False) -> None:
    key = (current_case(), role, model, kind)
    with _lock:
        row = _ledger[key]
        row["calls"] += 1
        row["errors"] += int(error)
        row["seconds"] += seconds
        if usage:
            row["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            row["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            row["cache_read_tokens"] += int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
            row["reasoning_tokens"] += int((usage.get("output_token_details") or {}).get("reasoning", 0) or 0)


def case_usage(case: Any) -> dict[str, Any]:
    """{"by_role": {role: totals}, "by_kind": {"role/kind": totals}, "by_model": {model: totals}, "total": totals}."""
    by_role: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_kind: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_model: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total: dict[str, float] = defaultdict(float)
    with _lock:
        rows = [(k, dict(v)) for k, v in _ledger.items() if k[0] == case]
    for (_, role, model, kind), v in rows:
        for f, x in v.items():
            by_role[role][f] += x
            by_kind[f"{role}/{kind}"][f] += x
            by_model[model][f] += x
            total[f] += x
    rnd = lambda d: {k: (round(x, 2) if isinstance(x, float) else x) for k, x in d.items()}  # noqa: E731
    return {"by_role": {k: rnd(v) for k, v in by_role.items()}, "by_kind": {k: rnd(v) for k, v in by_kind.items()},
            "by_model": {k: rnd(v) for k, v in by_model.items()}, "total": rnd(total)}


def reset() -> None:
    with _lock:
        _ledger.clear()


def bind(fn):
    """Wrap `fn` so calls made from a worker thread are attributed to the caller's current case."""
    case = current_case()

    def run(*a, **k):
        prev = current_case()
        set_case(case)
        try:
            return fn(*a, **k)
        finally:
            set_case(prev)

    return run
