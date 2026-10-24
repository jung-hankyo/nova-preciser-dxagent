"""dxagent: a knowledge-first diagnostic agent.

The agent interviews a patient (or a simulated encounter), chooses its own next
question, exam, or test by value of information, updates a probabilistic
differential deterministically from disease cards, and commits to a diagnosis.
The core runs inside a LangGraph graph (dxagent.graph, terminal chat, AgentClinic
bench) or inside the N.O.V.A competition runtime (dxagent.nova, run.py), which
needs only pydantic and pyyaml.
"""

__version__ = "0.8.1"
