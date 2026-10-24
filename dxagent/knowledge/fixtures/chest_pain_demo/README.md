# Demo fixtures (not knowledge)

These eight cards were the hand-written v0.3 seed cards (chest pain and dyspnea). Since v0.4 they are retired as
knowledge: they remain as fixtures for the unit tests and for `dxagent chat --offline` / `dxagent bench --offline`,
which check the plumbing without an API key, and the contest runtime loads them only with `NOVA_USE_FIXTURES=1` (off
by default).

Their values are illustrative and not individually sourced (for example, P(normal chest X-ray | PE) = 0.75 is higher
than published series report). Real cards come from `dxagent kb compile`.
