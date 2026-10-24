"""Assemble the diagnostic loop as a LangGraph StateGraph.

    START -> perceive -> hypothesize -> load_knowledge -> update_belief -> contrast
    contrast --(new A-vs-B knowledge)--> update_belief
    contrast --> critic
    critic --(added hypotheses)--> load_knowledge
    critic --> plan
    plan --(commit)--> report -> END
    report --(the report LLM dissents: once, the named disease is added)--> load_knowledge
    plan --> interview -> await_reply (interrupt) -> perceive
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from dxagent.graph.nodes import Nodes
from dxagent.runtime import Runtime
from dxagent.schemas import DxState


def build_graph(rt: Runtime, checkpointer=None):
    n = Nodes(rt)
    g = StateGraph(DxState)
    g.add_node("perceive", n.perceive)
    g.add_node("hypothesize", n.hypothesize)
    g.add_node("load_knowledge", n.load_knowledge)
    g.add_node("update_belief", n.update_belief)
    g.add_node("contrast", n.contrast)
    g.add_node("critic", n.critic)
    g.add_node("plan", n.plan)
    g.add_node("interview", n.interview)
    g.add_node("await_reply", n.await_reply)
    g.add_node("report", n.report)

    g.add_edge(START, "perceive")
    g.add_edge("perceive", "hypothesize")
    g.add_edge("hypothesize", "load_knowledge")
    g.add_edge("load_knowledge", "update_belief")
    g.add_edge("update_belief", "contrast")
    g.add_conditional_edges("contrast", n.route_after_contrast, {"update_belief": "update_belief", "critic": "critic"})
    g.add_conditional_edges("critic", n.route_after_critic, {"load_knowledge": "load_knowledge", "plan": "plan"})
    g.add_conditional_edges("plan", n.route_after_plan, {"interview": "interview", "report": "report"})
    g.add_edge("interview", "await_reply")
    g.add_edge("await_reply", "perceive")
    g.add_conditional_edges("report", n.route_after_report, {"end": END, "load_knowledge": "load_knowledge"})
    return g.compile(checkpointer=checkpointer or InMemorySaver())
