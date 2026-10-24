"""Rule-based safety checks. The LLM critic in graph/nodes.py is called only when these rules say so."""

from __future__ import annotations

from typing import Any

from dxagent.config import Settings
from dxagent.reasoning.planner import live_cant_miss, threshold_commit_ready


def should_call_critic(state: dict[str, Any], settings: Settings) -> tuple[bool, bool]:
    """Return (call_llm_critic, considering_commit).

    The LLM critic runs at most once per turn, and only when uncertainty is high,
    a can't-miss diagnosis is live, a commit is imminent, or on a fixed cadence.
    """
    turn = state.get("turn", 0)
    if state.get("critic_turn") == turn:
        return False, False
    diff = state.get("differential", [])
    if not diff or turn < len(settings.intake_script):
        return False, False
    considering_commit = (
        turn >= settings.min_turns_before_commit and threshold_commit_ready(diff, settings)
    ) or turn >= settings.max_turns - 1
    high_entropy = state.get("entropy", 1.0) >= settings.critic_entropy_trigger
    cant_miss_live = bool(live_cant_miss(diff, settings))
    cadence = turn % max(settings.critic_every, 1) == 0
    return (considering_commit or (cadence and (high_entropy or cant_miss_live))), considering_commit


def unexplained_findings(state: dict[str, Any]) -> list[str]:
    return list(state.get("unexplained", []))
