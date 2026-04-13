"""
core/agents/__init__.py

Agent registries for the C-suite system.

Two tiers of agents:
    CSUITE_AGENTS  — deliberation tier (CFO, COO, CMO, CTO, ...)
    WORKER_AGENTS  — execution tier (CCA, and future workers)

The CEO is separate — it synthesizes but does not deliberate.

─── How to add a new C-suite agent ───────────────────────────────────
1. Create core/agents/<role>.py extending BaseAgent
2. Add the import below
3. Append the class to CSUITE_AGENTS
4. Add agent personality to company config.json

─── How to add a new worker agent ────────────────────────────────────
1. Create core/agents/<role>.py extending BaseWorker
2. Add the import below
3. Append the class to WORKER_AGENTS
4. The spawn_workers node will auto-dispatch based on keywords
"""

# ── C-suite agents (deliberation tier) ───────────────────────────────────────

from core.agents.ceo import CEOAgent
from core.agents.cfo import CFOAgent
from core.agents.coo import COOAgent
from core.agents.cmo import CMOAgent
from core.agents.cto import CTOAgent

# Agents that participate in deliberation rounds.
# CEO is excluded — it synthesizes, not deliberates.
# Order here is execution order (sequential, one GPU).
CSUITE_AGENTS = [CFOAgent, COOAgent, CMOAgent, CTOAgent]

# ── Worker agents (execution tier) ───────────────────────────────────────────

from core.agents.base_worker import BaseWorker
from core.agents.cca import CCAAgent
from core.agents.cwa import CWAAgent
from core.agents.cra import CRAAgent
from core.agents.csa import CSAAgent

# Each entry: the worker class itself.
# Workers declare their own keywords via the `keywords` class attribute.
# spawn_workers matches task text against keywords to decide who runs.
WORKER_AGENTS = [CCAAgent, CWAAgent, CRAAgent, CSAAgent]

# ── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    # C-suite
    "CEOAgent",
    "CFOAgent",
    "COOAgent",
    "CMOAgent",
    "CTOAgent",
    "CSUITE_AGENTS",
    # Workers
    "BaseWorker",
    "CCAAgent",
    "CWAAgent",
    "CRAAgent",
    "CSAAgent",
    "WORKER_AGENTS",
]
