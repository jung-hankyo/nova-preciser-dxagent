"""LangGraph assembly of the diagnostic loop.

`build_graph` is imported lazily (v0.5): the competition runtime (dxagent.nova) drives the same nodes without
LangGraph, so importing `dxagent.graph.nodes` must not require it.
"""

__all__ = ["build_graph"]


def __getattr__(name):
    if name == "build_graph":
        from dxagent.graph.build import build_graph

        return build_graph
    raise AttributeError(name)
