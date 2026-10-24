# Demo fixtures (not knowledge)

These eight cards were the hand-written v0.3 seed cards (chest pain and dyspnea). In v0.4 they are retired as
knowledge: the runtime never loads them. They remain only as fixtures for the unit tests and for
`dxagent chat --offline` / `dxagent bench --offline`, which check the plumbing without an API key.

Their values are illustrative and not individually sourced (for example, P(normal chest X-ray | PE) = 0.75 is
higher than published series report). Real cards come from `dxagent kb compile`.
