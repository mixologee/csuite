"""
core/graph/runner.py

Entry point for running a C-suite session from the command line.

Usage:
    python -m core.graph.runner --company acme_corp --task "Should we raise prices by 10%?"
    python -m core.graph.runner --company acme_corp --task "..." --context "Q3 revenue down 8%"

What this does:
    1. Loads company DNA from CSUITE_COMPANY_ROOT/<company_id>/config.json
    2. Builds the compiled LangGraph graph for this company
    3. Streams the graph to the human_interrupt pause point
    4. Displays the full deliberation report
    5. Waits for your input
    6. Resumes the graph, writes memory, ends session
"""

import argparse
import json
import sys

from core.config import COMPANY_ROOT
from core.graph.session_graph import build_session_graph


def run_session(company_id: str, task: str, context: str = "") -> None:
    """
    Runs a complete session for the given company and task.
    Blocks at the human interrupt waiting for input.
    """
    # ── Load company DNA ───────────────────────────────────────────────────
    config_path = COMPANY_ROOT / company_id / "config.json"
    if not config_path.exists():
        print(f"Error: No company found at {config_path}")
        print(f"Run: python scripts/new_company.py --id {company_id} to create one.")
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"\n  Company: {config['company_name']}  |  Task: {task[:60]}\n")

    # ── Build graph ────────────────────────────────────────────────────────
    graph  = build_session_graph(company_id)
    thread = {"configurable": {"thread_id": f"{company_id}_{hash(task)}"}}

    # ── Initial state ──────────────────────────────────────────────────────
    initial_state = {
        "company_id":          company_id,
        "company_name":        config["company_name"],
        "company_config":      config,
        "current_task":        task,
        "task_context":        context,
        "agenda":              [],
        "relevant_memories":   [],
        "agent_outputs":       [],
        "messages":            [],
        "consensus_reached":   False,
        "escalate_to_human":   False,
        "escalation_reason":   "",
        "human_decision":      None,
        "decisions_made":      [],
        "debate_round":        1,
        "session_id":          "",
        "session_start":       "",
        "ceo_synthesis":       "",
        "conflicts_identified":[],
    }

    # ── Stream to interrupt ────────────────────────────────────────────────
    print("  Starting deliberation...\n")
    for _ in graph.stream(initial_state, thread, stream_mode="values"):
        pass   # nodes print their own progress via _log()

    # ── Collect human decision ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  YOUR RESPONSE:")
    print("  Options: approve | override <reason> | more info <question>")
    print("=" * 70)
    human_input = input("  > ").strip()

    if not human_input:
        human_input = "acknowledged"

    # ── Resume graph ───────────────────────────────────────────────────────
    graph.update_state(
        thread,
        {"human_decision": human_input},
        as_node="human_interrupt",
    )

    for _ in graph.stream(None, thread, stream_mode="values"):
        pass  # memory_write runs silently

    print("\n  Session complete. Decisions written to memory.\n")


# ── CLI entrypoint ─────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a C-suite deliberation session."
    )
    parser.add_argument(
        "--company", required=True,
        help="Company ID (must match a folder under CSUITE_COMPANY_ROOT)"
    )
    parser.add_argument(
        "--task", required=True,
        help="The decision or question for the C-suite to deliberate on"
    )
    parser.add_argument(
        "--context", default="",
        help="Optional background context for this specific task"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_session(
        company_id=args.company,
        task=args.task,
        context=args.context,
    )
