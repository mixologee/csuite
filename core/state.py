"""
core/state.py

LangGraph state schema for the C-suite session graph.

CompanyState is the single source of truth that flows through every node
in the graph. Fields annotated with operator.add accumulate across node
calls rather than being overwritten — this is how agent outputs from
multiple rounds build up in one list.

All fields are optional at construction time (the task_intake node
initialises the session fields). Company identity fields are injected
by the runner before the graph starts.
"""

import operator
from typing import Annotated, Optional, TypedDict


class AgentOutput(TypedDict):
    agent:          str    # "cfo" | "coo" | "cmo" | "cto" | "*_response"
    analysis:       str    # free-form natural language reasoning
    recommendation: str    # "proceed" | "block" | "modify"
    concerns:       list[str]
    confidence:     float  # 0.0 – 1.0


class Decision(TypedDict):
    decision_id:    str
    session_id:     str
    task:           str
    outcome:        str          # what was decided
    reasoning:      str          # CEO synthesis summary
    votes:          dict         # {"cfo": "proceed", ...}
    human_override: Optional[str]
    timestamp:      str


class CompanyState(TypedDict):

    # ── Identity (loaded from DNA at session start, never mutated) ──────────
    company_id:     str
    company_name:   str
    company_config: dict         # full config.json contents

    # ── Session tracking ─────────────────────────────────────────────────────
    session_id:     str
    session_start:  str

    # ── Current task ─────────────────────────────────────────────────────────
    current_task:   str
    agenda:         list[str]    # remaining items this session
    task_context:   str          # extra background for this specific task

    # ── Retrieved memory (injected at session start) ─────────────────────────
    relevant_memories: list[dict]

    # ── Deliberation tracking ────────────────────────────────────────────────
    debate_round:   int          # increments each time we loop back

    # ── Agent outputs — operator.add means these ACCUMULATE across nodes ─────
    agent_outputs:  Annotated[list[AgentOutput], operator.add]

    # ── CEO deliberation ─────────────────────────────────────────────────────
    ceo_synthesis:        str
    conflicts_identified: list[str]
    consensus_reached:    bool

    # ── Escalation ───────────────────────────────────────────────────────────
    escalate_to_human:  bool
    escalation_reason:  str
    human_decision:     Optional[str]  # None until you respond

    # ── Full message log ─────────────────────────────────────────────────────
    messages: Annotated[list[dict], operator.add]

    # ── Completed decisions — accumulate across tasks in one session ─────────
    decisions_made: Annotated[list[Decision], operator.add]
