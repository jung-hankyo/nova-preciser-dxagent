"""Simulated contest visits for testing the agent before submission (not part of the submission ZIP).

    python -m sim --help

sim/server.py   the contest server's contract, with faults: sessions, actions, the Doctor model route
sim/client.py   an SDK-faithful client, or the real nova_sdk.Client with its socket layer simulated
sim/actors.py   the scripted patient and examiner
sim/doctor.py   Doctor model stand-ins: oracle, faulty, record/replay
sim/scenarios.py, sim/check.py, sim/runner.py   the scenario matrix, the invariants, the runner
sim/vclock.py   virtual time, so minutes of latency run in milliseconds
"""
