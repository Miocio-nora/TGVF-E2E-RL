"""Import hook that keeps the candidate V1 manager class exactly upstream.

``TaskRunnerV1`` imports the configured manager class in the trainer process
before ``fit``.  Importing the infrastructure objective here registers its
zero-advantage estimator in that same process.  The exported object is an alias
of upstream ``AgentLoopManagerTQ`` rather than a subclass or wrapper, so the
TransferQueue behavior and method contract remain entirely upstream-owned.
"""

from __future__ import annotations

from verl.trainer.ppo.v1 import AgentLoopManagerTQ

from . import sync_gate_objective as _sync_gate_objective  # noqa: F401


SyncGateAgentLoopManagerTQ = AgentLoopManagerTQ


__all__ = ["SyncGateAgentLoopManagerTQ"]
