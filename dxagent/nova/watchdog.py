"""Hard wall-clock bounds for calls that block on the contest server.

The SDK waits up to 240 s per request and the agent cannot interrupt it; a case that runs past consult()'s limit is
lost however good the visit was. `bounded(fn, timeout)` runs the call in a daemon thread and stops waiting after
`timeout` seconds (CallTimeout): the visit goes on (a deterministic fallback, then DIAGNOSE) while the abandoned
request finishes or fails on its own. Exceptions of the call, BaseException included, are re-raised in the caller.
"""

from __future__ import annotations

import threading
from typing import Any, Callable


class CallTimeout(TimeoutError):
    """The call did not return within its time limit; it may still complete on the server."""


def bounded(fn: Callable[..., Any], timeout: float, *args: Any, **kwargs: Any) -> Any:
    if timeout is None or timeout <= 0:
        raise CallTimeout("no time left for the call")
    box: dict[str, Any] = {}
    done = threading.Event()

    def work() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001  (re-raised in the caller)
            box["error"] = exc
        finally:
            done.set()

    t = threading.Thread(target=work, name="nova-bounded-call", daemon=True)
    t.start()
    if not done.wait(timeout):
        raise CallTimeout(f"no answer within {timeout:.0f} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")
