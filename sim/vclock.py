"""Virtual time for simulated visits.

Inside `virtual_time(clock)`, the agent's modules see a `time` module whose monotonic(), time() and sleep() read and
advance one shared clock, and a `datetime` whose now() follows it. A model call that "takes" 200 s therefore moves the
clock by 200 s in microseconds of real time, and every deadline decision of the agent (budget, call timeouts, wrap-up,
the session deadline from client.session()) is exercised exactly as it would be on the server.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import importlib
import threading
import time as _time
from typing import Iterator

# Modules of the runtime that read the clock or sleep (late-bound: they call time.x at use).
MODULES = ("dxagent.nova.agent", "dxagent.nova.budget", "dxagent.nova.llm", "dxagent.nova.sdk", "dxagent.nova.env",
           "dxagent.nova.soap", "dxagent.nova.guidelines", "dxagent.nova.watchdog")


class VirtualClock:
    def __init__(self, start: float = 10_000.0, epoch: float | None = None):
        self._t = float(start)
        self.start = float(start)
        self.epoch = _time.time() if epoch is None else float(epoch)  # wall time at `start`
        self._lock = threading.Lock()
        self.slept = 0.0  # total time the agent spent sleeping (retry back-off)

    def monotonic(self) -> float:
        return self._t

    def time(self) -> float:
        return self.epoch + (self._t - self.start)

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._t += max(0.0, float(seconds))

    def sleep(self, seconds: float) -> None:
        self.slept += max(0.0, float(seconds))
        self.advance(seconds)

    def elapsed(self) -> float:
        return self._t - self.start

    def utc(self) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(self.time(), _dt.timezone.utc)


class _TimeModule:
    """Stands in for `time` in a patched module; anything else falls through to the real module."""

    def __init__(self, clock: VirtualClock):
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.monotonic()

    def perf_counter(self) -> float:
        return self._clock.monotonic()

    def time(self) -> float:
        return self._clock.time()

    def sleep(self, seconds: float) -> None:
        self._clock.sleep(seconds)

    def __getattr__(self, name: str):
        return getattr(_time, name)


def _datetime_class(clock: VirtualClock) -> type:
    class VirtualDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102
            t = clock.utc()
            return t.astimezone(tz) if tz is not None else t.astimezone().replace(tzinfo=None)

        @classmethod
        def utcnow(cls):  # noqa: D102
            return clock.utc().replace(tzinfo=None)

    return VirtualDateTime


@contextlib.contextmanager
def virtual_time(clock: VirtualClock, modules: tuple[str, ...] = MODULES) -> Iterator[VirtualClock]:
    saved: list[tuple[object, str, object]] = []
    tm, vdt = _TimeModule(clock), _datetime_class(clock)
    try:
        for name in modules:
            try:
                m = importlib.import_module(name)
            except ImportError:
                continue
            if getattr(m, "time", None) is _time:
                saved.append((m, "time", m.time))
                m.time = tm  # type: ignore[attr-defined]
            if getattr(m, "datetime", None) is _dt.datetime:
                saved.append((m, "datetime", m.datetime))
                m.datetime = vdt  # type: ignore[attr-defined]
        yield clock
    finally:
        for m, attr, value in reversed(saved):
            setattr(m, attr, value)
