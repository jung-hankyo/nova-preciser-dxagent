"""Drive one encounter through the graph. Shared by the terminal chat and the benchmark runner."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from langgraph.types import Command

from dxagent.graph.build import build_graph
from dxagent.runtime import Runtime


@dataclass
class TurnResult:
    message: str  # what the doctor says now (a question, an order, or the final diagnosis)
    action_type: str  # history | exam | poc | lab | imaging | ... | diagnosis
    actions: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    done: bool = False
    report: Optional[dict[str, Any]] = None


class DxSession:
    def __init__(self, rt: Runtime, thread_id: Optional[str] = None):
        self.rt = rt
        self.graph = build_graph(rt)
        self.config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": 200}

    @property
    def state(self) -> dict[str, Any]:
        return self.graph.get_state(self.config).values

    def start(self, presentation: str, source: str = "patient") -> TurnResult:
        role = "patient" if source == "patient" else "referral"
        init = {
            "presentation": presentation,
            "presentation_source": source,
            "last_reply": presentation,
            "transcript": [{"role": role, "type": "history", "text": presentation, "turn": 0}],
            "turn": 0,
            "phase": "gathering",
            "findings": [],
            "test_results": [],
            "hypotheses": [],
            "last_actions": [],
            "context": {},
        }
        self.graph.invoke(init, self.config)
        return self._result()

    def reply(self, text: str) -> TurnResult:
        self.graph.invoke(Command(resume=text), self.config)
        return self._result()

    def _result(self) -> TurnResult:
        st = self.state
        if st.get("phase") == "committed":
            return TurnResult(st.get("doctor_message", ""), "diagnosis", [], st.get("turn", 0), True, st.get("final_report"))
        actions = st.get("last_actions", [])
        kind = actions[0]["type"] if actions else "history"
        return TurnResult(st.get("doctor_message", ""), kind, actions, st.get("turn", 0), False, None)
