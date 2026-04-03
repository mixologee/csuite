"""
core/graph/edges.py

Conditional edge functions for the session graph.

LangGraph conditional edges are plain Python functions that receive
the current state and return a string key. That key maps to the next
node name in the graph's routing table.

Currently defined edges:
    conflict_router — routes after CEO synthesis based on consensus state
"""

MAX_DEBATE_ROUNDS = 2


def conflict_router(state: dict) -> str:
    """
    Called after every ceo_synthesis node execution.

    Decision tree:
      1. Consensus reached → "resolved" → present recommendation to human
      2. No consensus, rounds remaining → "deadlocked" → loop back for round 2
      3. No consensus, rounds exhausted → "escalate" → present options to human

    The graph maps these string keys to node names:
        "resolved"   → present_recommendation
        "deadlocked" → round1_deliberation  (round 2 with CEO conflict framing)
        "escalate"   → present_recommendation (with escalate_to_human=True)
    """
    debate_round = state.get("debate_round", 1)

    if state.get("consensus_reached", False):
        return "resolved"

    if debate_round <= MAX_DEBATE_ROUNDS:
        return "deadlocked"

    return "escalate"
