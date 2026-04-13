"""
core/agents/base_worker.py

Abstract base class for all worker-tier agents.

Workers sit below the C-suite. They are invoked after human approval
to execute concrete tasks — coding, writing, research, comms, art, etc.

Every worker must define:
    role      (str)       — short identifier, e.g. "cca"
    title     (str)       — display name, e.g. "Claude Code Agent"
    keywords  (list[str]) — words that trigger this worker when found
                            in the CEO synthesis or task text

Every worker must implement:
    execute(task: str) -> dict
        Returns at minimum: {worker, success, summary, output}
        Workers may add extra keys (e.g. files_changed for CCA).

Workers may set:
    interactive (bool) — if True, the worker manages a multi-turn session
        with the user through the UI. The graph flags it as pending and
        the UI layer (app.py) handles the conversation loop. Default: False.

To add a new worker:
    1. Create core/agents/<name>.py with a class extending BaseWorker
    2. Add it to WORKER_AGENTS in core/agents/__init__.py
    That's it — spawn_workers will pick it up automatically.
"""

from abc import ABC, abstractmethod


class BaseWorker(ABC):

    role:        str
    title:       str
    keywords:    list[str]
    interactive: bool = False

    def __init__(self, company_config: dict):
        self.config = company_config
        self.company = company_config.get("company_name", "the company")

    @abstractmethod
    def execute(self, task: str) -> dict:
        """
        Execute a task and return a result dict.

        Required keys in the returned dict:
            worker  (str)       — self.role
            success (bool)      — whether the task completed
            summary (str)       — human-readable description of what happened
            output  (str)       — raw output for debugging/logging

        Workers may add additional keys relevant to their domain.
        """

    def can_handle(self, text: str) -> bool:
        """
        Returns True if this worker's keywords match the given text.
        Used by spawn_workers to decide which workers to invoke.
        """
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.keywords)
