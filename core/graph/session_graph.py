"""
core/graph/session_graph.py

Builds and compiles the LangGraph session graph for a given company instance.

Each company gets its own compiled graph. The checkpointer points at that
company's SQLite database, which means:
  - Session state is isolated between companies
  - Sessions can be resumed by thread_id if interrupted
  - Full state history is persisted automatically by LangGraph
"""

from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from core.state import CompanyState
from core.graph.nodes import (
    task_intake,
    memory_retrieval,
    round1_deliberation,
    cross_response,
    ceo_synthesis,
    present_recommendation,
    human_interrupt_node,
    memory_write,
)
from core.graph.edges import conflict_router

DATA_ROOT = Path("G:/csuite_data")


def build_session_graph(company_id: str):
    """
    Constructs and compiles the session graph for the given company.

    Returns a compiled LangGraph graph ready to stream.
    """
    builder = StateGraph(CompanyState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("task_intake",            task_intake)
    builder.add_node("memory_retrieval",       memory_retrieval)
    builder.add_node("round1_deliberation",    round1_deliberation)
    builder.add_node("cross_response",         cross_response)
    builder.add_node("ceo_synthesis",          ceo_synthesis)
    builder.add_node("present_recommendation", present_recommendation)
    builder.add_node("human_interrupt",        human_interrupt_node)
    builder.add_node("memory_write",           memory_write)

    # ── Linear edges ──────────────────────────────────────────────────────
    builder.set_entry_point("task_intake")
    builder.add_edge("task_intake",            "memory_retrieval")
    builder.add_edge("memory_retrieval",       "round1_deliberation")
    builder.add_edge("round1_deliberation",    "cross_response")
    builder.add_edge("cross_response",         "ceo_synthesis")
    builder.add_edge("present_recommendation", "human_interrupt")
    builder.add_edge("human_interrupt",        "memory_write")
    builder.add_edge("memory_write",           END)

    # ── Conditional edge: CEO synthesis → route on conflict state ─────────
    builder.add_conditional_edges(
        "ceo_synthesis",
        conflict_router,
        {
            "resolved":   "present_recommendation",
            "deadlocked": "round1_deliberation",    # loop back for round 2
            "escalate":   "present_recommendation", # max rounds hit → escalate
        },
    )

    # ── Checkpointer: per-company SQLite ──────────────────────────────────
    db_path = DATA_ROOT / company_id / f"{company_id}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver.from_conn_string(str(db_path))

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_interrupt"],
    )
