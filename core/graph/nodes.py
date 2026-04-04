"""
core/graph/nodes.py

Node functions for the session graph.

Each function receives the current CompanyState, performs its work,
and returns a dict of fields to update in state. LangGraph merges
these dicts into the running state — fields not returned are unchanged.

Node execution order:
    task_intake → memory_retrieval → round1_deliberation → cross_response
    → ceo_synthesis → [conflict_router] → present_recommendation
    → human_interrupt → memory_write → END
"""

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from core.agents.ceo import CEOAgent
from core.agents.cfo import CFOAgent
from core.agents.coo import COOAgent
from core.agents.cmo import CMOAgent
from core.agents.cto import CTOAgent
from core.memory.retrieval import retrieve_relevant_memories
from core.memory.writer import write_session_to_db
from core.state import AgentOutput

# Optional callback for UI progress updates.
# Set by the UI layer (app.py) before streaming the graph.
# Signature: callback(event: str, data: dict) -> None
_progress_callback: Optional[Callable] = None


def set_progress_callback(cb: Optional[Callable]) -> None:
    global _progress_callback
    _progress_callback = cb


def _notify(event: str, **data) -> None:
    if _progress_callback:
        _progress_callback(event, data)


# ── Infrastructure nodes ──────────────────────────────────────────────────────

def task_intake(state: dict) -> dict:
    """
    Entry point. Initialises session-level fields.
    Company identity fields are already in state (injected by the runner).
    """
    return {
        "session_id":           str(uuid.uuid4()),
        "session_start":        datetime.now(timezone.utc).isoformat(),
        "debate_round":         1,
        "agent_outputs":        [],
        "messages":             [],
        "consensus_reached":    False,
        "escalate_to_human":    False,
        "escalation_reason":    "",
        "ceo_synthesis":        "",
        "conflicts_identified": [],
        "decisions_made":       [],
    }


def memory_retrieval(state: dict) -> dict:
    """
    Fetches relevant past decisions and injects them into state
    so the CEO can include them in agent briefings.
    """
    memories = retrieve_relevant_memories(
        company_id=state["company_id"],
        query=state["current_task"],
    )
    return {"relevant_memories": memories}


# ── Deliberation nodes ────────────────────────────────────────────────────────

def round1_deliberation(state: dict) -> dict:
    """
    Each C-suite agent independently analyzes the current task.
    Agents cannot see each other's outputs at this stage.
    All four run sequentially (single shared GPU).
    """
    round_num = state.get("debate_round", 1)
    outputs   = []
    briefing  = _build_agent_briefing(state, include_prior_outputs=False,
                                       round_num=round_num)

    agents = [CFOAgent, COOAgent, CMOAgent, CTOAgent]
    for i, AgentClass in enumerate(agents):
        agent  = AgentClass(state["company_config"])
        _notify("agent_start", agent=agent.role, phase="deliberation",
                round=round_num, index=i, total=len(agents))
        result = agent.analyze(briefing)
        outputs.append(AgentOutput(
            agent          = agent.role,
            analysis       = result["analysis"],
            recommendation = result["recommendation"],
            concerns       = result["concerns"],
            confidence     = result["confidence"],
        ))
        _log(f"[{agent.role.upper()}] {result['recommendation'].upper()} "
             f"({result['confidence']:.0%} confidence)")

    return {
        "agent_outputs": outputs,
        "messages": [{"role": "system",
                      "content": f"Round {round_num} deliberation complete."}],
    }


def cross_response(state: dict) -> dict:
    """
    Each agent reads all round-1 outputs and responds to peer positions.
    This is where genuine debate happens — agents may agree, push back,
    or propose modifications based on what their colleagues argued.
    """
    round_num     = state.get("debate_round", 1)
    prior_outputs = state["agent_outputs"]
    briefing      = _build_agent_briefing(state, include_prior_outputs=True,
                                           round_num=round_num)
    outputs = []

    agents = [CFOAgent, COOAgent, CMOAgent, CTOAgent]
    for i, AgentClass in enumerate(agents):
        agent  = AgentClass(state["company_config"])
        _notify("agent_start", agent=agent.role, phase="cross-response",
                round=round_num, index=i, total=len(agents))
        peers  = [o for o in prior_outputs if o["agent"] != agent.role]
        result = agent.respond_to_peers(briefing, peers)
        outputs.append(AgentOutput(
            agent          = f"{agent.role}_response",
            analysis       = result["analysis"],
            recommendation = result["recommendation"],
            concerns       = result["concerns"],
            confidence     = result["confidence"],
        ))
        _log(f"[{agent.role.upper()} cross-response] {result['recommendation'].upper()}")

    return {
        "agent_outputs": outputs,
        "messages": [{"role": "system",
                      "content": f"Round {round_num} cross-response complete."}],
    }


# ── CEO nodes ─────────────────────────────────────────────────────────────────

def ceo_synthesis(state: dict) -> dict:
    """
    CEO reads all agent outputs from this round and attempts synthesis.
    On round 2, is explicitly told this is the final chance to resolve
    internally before forcing an escalation.
    """
    ceo       = CEOAgent(state["company_config"])
    round_num = state.get("debate_round", 1)
    is_final  = round_num >= 2
    _notify("agent_start", agent="ceo", phase="synthesis", round=round_num,
            index=0, total=1)

    result = ceo.synthesize(
        task           = state["current_task"],
        agent_outputs  = state["agent_outputs"],
        memories       = state.get("relevant_memories", []),
        is_final_round = is_final,
    )

    _log(f"[CEO] {'CONSENSUS' if result['consensus'] else 'CONFLICT DETECTED'}")

    return {
        "ceo_synthesis":        result["synthesis"],
        "conflicts_identified": result["conflicts"],
        "consensus_reached":    result["consensus"],
        "escalate_to_human":    result["escalate"],
        "debate_round":         round_num + 1,
        "messages": [{"role": "assistant", "content": result["synthesis"]}],
    }


def present_recommendation(state: dict) -> dict:
    """
    Assembles the full deliberation report for display.
    This is everything you see — every agent's reasoning, cross-responses,
    CEO synthesis, and the final recommendation or escalation options.
    """
    ceo    = CEOAgent(state["company_config"])
    report = ceo.format_presentation(state)
    print(report)   # displayed in terminal / piped to Chainlit

    return {
        "messages": [{"role": "assistant", "content": report}],
    }


def human_interrupt_node(state: dict) -> dict:
    """
    LangGraph pauses here via interrupt_before=[].
    The runner resumes the graph by calling graph.update_state() with your
    decision. This node records the decision once the graph resumes.
    """
    human_decision = state.get("human_decision", "")
    _log(f"[HUMAN] {human_decision[:80]}")
    return {
        "messages": [{"role": "user", "content": human_decision}],
    }


def reconsider_with_info(state: dict) -> dict:
    """
    The human has provided new information via "more info <details>".
    Extract the new context, append it to task_context, and reset
    deliberation state so the C-suite reconsiders from scratch with
    the full picture.

    Previous agent outputs are preserved (operator.add) so the CEO
    can see how positions evolved after receiving new information.
    """
    raw = state.get("human_decision", "")

    # Strip the "more info" prefix to get the actual new information
    new_info = raw.strip()
    for prefix in ("more info", "More info", "MORE INFO"):
        if new_info.startswith(prefix):
            new_info = new_info[len(prefix):].strip().lstrip(":").strip()
            break

    # Append to existing context
    existing = state.get("task_context", "")
    separator = "\n\n" if existing else ""
    updated_context = f"{existing}{separator}[Additional information from owner] {new_info}"

    _log(f"[RECONSIDER] New information received — restarting deliberation")

    return {
        "task_context":         updated_context,
        "debate_round":         1,
        "consensus_reached":    False,
        "escalate_to_human":    False,
        "escalation_reason":    "",
        "ceo_synthesis":        "",
        "conflicts_identified": [],
        "human_decision":       None,
        "messages": [{"role": "system",
                      "content": f"Owner provided new information: {new_info}. "
                                 f"Restarting deliberation with updated context."}],
    }


# ── Memory node ───────────────────────────────────────────────────────────────

def memory_write(state: dict) -> dict:
    """
    Flushes the completed session to SQLite and embeds key decisions
    into ChromaDB for future semantic retrieval. Runs silently.
    """
    write_session_to_db(state)
    _log("[MEMORY] Session written to SQLite and ChromaDB.")
    return {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_agent_briefing(
    state: dict,
    include_prior_outputs: bool,
    round_num: int,
) -> str:
    """
    Assembles the briefing text injected into each agent's prompt.
    On round 2, explicitly frames the CEO-identified conflicts so agents
    know what the sticking points are and can address them directly.
    """
    parts = [
        f"Company: {state['company_name']}",
        f"Task: {state['current_task']}",
        f"Context: {state.get('task_context', 'None provided')}",
    ]

    if state.get("relevant_memories"):
        mem_text = "\n".join(
            f"- {m['task']}: {m['outcome']}"
            for m in state["relevant_memories"]
        )
        parts.append(f"Relevant past decisions:\n{mem_text}")

    # Surface any prior owner decisions from this session so agents
    # do not re-litigate points the owner has already settled.
    owner_directions = _extract_owner_directions(state)
    if owner_directions:
        parts.append(
            "OWNER DIRECTIONS (treat these as settled — do not re-litigate):\n"
            + "\n".join(f"- {d}" for d in owner_directions)
        )

    if include_prior_outputs and state.get("agent_outputs"):
        outputs_text = "\n\n".join(
            f"{o['agent'].upper()}:\n{o['analysis']}\n"
            f"Recommendation: {o['recommendation']}"
            for o in state["agent_outputs"]
        )
        parts.append(f"Your colleagues' positions:\n{outputs_text}")

    if round_num >= 2 and state.get("conflicts_identified"):
        conflict_text = "\n".join(
            f"- {c}" for c in state["conflicts_identified"]
        )
        parts.append(
            f"NOTE — Round 2 of deliberation. The CEO has identified the "
            f"following unresolved conflicts after round 1. Please reconsider "
            f"your position with these specific tensions in mind:\n{conflict_text}"
        )

    return "\n\n".join(parts)


def _extract_owner_directions(state: dict) -> list[str]:
    """
    Pull any human decisions or 'more info' context from the session's
    message history. These represent settled points the owner has already
    weighed in on — agents should incorporate them, not argue against them.
    """
    directions = []
    for msg in state.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content:
                directions.append(content)
    return directions


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"  {ts}  {msg}")
