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

from core.agents import CEOAgent, CSUITE_AGENTS, WORKER_AGENTS
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


def prior_decision_check(state: dict) -> dict:
    """
    Checks whether the current task is asking about something already decided,
    or is essentially a repeat of a prior decision.

    Uses two strategies:
        1. SQLite keyword search — looks for prior decisions with overlapping
           keywords in the task text
        2. ChromaDB similarity (if available) — high cosine similarity match

    If a match is found, the CEO answers from the knowledge document and
    prior decision record, skipping full deliberation.
    """
    import sqlite3
    from core.config import DATA_ROOT

    task = state.get("current_task", "").strip()
    company_id = state.get("company_id", "")
    memories = state.get("relevant_memories", [])

    if not task or not company_id:
        return {"prior_decision_found": False}

    # ── Strategy 1: check if this is a question about past decisions ─────
    question_patterns = [
        "what did we decide", "where do we stand", "what was decided",
        "what happened with", "status of", "update on", "did we",
        "have we decided", "what's the decision on", "recap",
    ]
    is_status_question = any(p in task.lower() for p in question_patterns)

    # ── Strategy 2: SQLite keyword match for prior decisions ─────────────
    best_match = _find_matching_decision(company_id, task)

    # ── Strategy 3: ChromaDB similarity (fallback if available) ──────────
    if not best_match:
        for m in memories:
            score = m.get("similarity_score")
            if score is not None and score >= 0.85:
                if best_match is None or score > best_match.get("similarity_score", 0):
                    best_match = m

    # ── Decide: skip deliberation or proceed ─────────────────────────────
    if not best_match and not is_status_question:
        return {"prior_decision_found": False}

    # If it's a status question and we have a knowledge doc, let the CEO
    # answer from the full knowledge document rather than a single match
    knowledge_doc = ""
    for m in memories:
        if m.get("source") == "distilled_knowledge":
            knowledge_doc = m.get("reasoning", "")
            break

    if is_status_question and knowledge_doc:
        _log("[MEMORY] Status question detected — CEO answering from knowledge document.")

        # Ask the CEO to answer the question from the knowledge doc
        ceo = CEOAgent(state["company_config"])
        prompt = (
            f"The owner is asking: \"{task}\"\n\n"
            f"Answer this question using ONLY the institutional knowledge below. "
            f"Be specific about what was decided, when, and why. If the topic "
            f"hasn't been decided yet, say so clearly.\n\n"
            f"INSTITUTIONAL KNOWLEDGE:\n{knowledge_doc}"
        )
        from core.agents.base import invoke_llm
        synthesis = invoke_llm(ceo.llm, prompt)

        return {
            "prior_decision_found": True,
            "ceo_synthesis":        synthesis,
            "consensus_reached":    True,
            "escalate_to_human":    False,
            "messages": [{"role": "assistant", "content": synthesis}],
        }

    if best_match:
        _log(f"[MEMORY] Prior decision found — skipping deliberation.")

        prior_task = best_match.get("task", "")
        prior_outcome = best_match.get("outcome", "")
        prior_reasoning = best_match.get("reasoning", "")
        prior_override = best_match.get("human_override", "")

        synthesis_parts = [
            "This topic has already been decided in a prior session.",
            "",
            f"**Prior task:** {prior_task}",
            f"**Decision:** {prior_outcome}",
        ]
        if prior_reasoning:
            synthesis_parts.append(f"**Reasoning:** {prior_reasoning}")
        if prior_override:
            synthesis_parts.append(f"**Owner directive:** {prior_override}")
        synthesis_parts.append(
            "\nIf circumstances have changed, provide new information "
            "via 'more info' to trigger a fresh deliberation."
        )

        return {
            "prior_decision_found": True,
            "ceo_synthesis":        "\n".join(synthesis_parts),
            "consensus_reached":    True,
            "escalate_to_human":    False,
            "messages": [{"role": "assistant", "content": "\n".join(synthesis_parts)}],
        }

    return {"prior_decision_found": False}


def _find_matching_decision(company_id: str, task: str) -> dict | None:
    """
    Search SQLite for a prior decision whose task text shares significant
    keywords with the current task. Returns the best match or None.
    """
    import sqlite3
    from core.config import DATA_ROOT

    db_path = DATA_ROOT / company_id / f"{company_id}.db"
    if not db_path.exists():
        return None

    # Extract meaningful words from the task (skip short/common words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "we", "our", "us",
        "did", "do", "does", "what", "where", "how", "when", "about",
        "with", "for", "and", "or", "but", "not", "this", "that", "on",
        "in", "to", "of", "it", "its", "has", "have", "had", "can",
        "should", "would", "could", "will", "been", "being", "from",
    }
    words = [
        w.lower().strip("?.,!\"'")
        for w in task.split()
        if len(w) > 2 and w.lower().strip("?.,!\"'") not in stop_words
    ]

    if not words:
        return None

    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT task, outcome, reasoning, human_override, decided_at
                FROM decisions
                WHERE outcome IS NOT NULL
                ORDER BY decided_at DESC
                """,
            ).fetchall()

        best = None
        best_score = 0

        for row in rows:
            prior_task = (row["task"] or "").lower()
            matches = sum(1 for w in words if w in prior_task)
            score = matches / len(words) if words else 0

            if score > best_score and score >= 0.4:
                best_score = score
                best = {
                    "task":           row["task"],
                    "outcome":        row["outcome"],
                    "reasoning":      row["reasoning"] or "",
                    "human_override": row["human_override"] or "",
                    "source":         "sqlite_keyword_match",
                    "similarity_score": None,
                }

        return best

    except Exception:
        return None


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

    for i, AgentClass in enumerate(CSUITE_AGENTS):
        agent  = AgentClass(state["company_config"])
        _notify("agent_start", agent=agent.role, phase="deliberation",
                round=round_num, index=i, total=len(CSUITE_AGENTS))
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

    for i, AgentClass in enumerate(CSUITE_AGENTS):
        agent  = AgentClass(state["company_config"])
        _notify("agent_start", agent=agent.role, phase="cross-response",
                round=round_num, index=i, total=len(CSUITE_AGENTS))
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


# ── Worker dispatch node ──────────────────────────────────────────────────────

def spawn_workers(state: dict) -> dict:
    """
    After human approval, identifies and dispatches matching worker agents.

    Gates:
        - human_decision must start with 'implement'
        - Each worker's keywords must match the task/synthesis text

    Non-interactive workers execute inline and their results go to
    worker_results. Interactive workers (like CCA) are flagged as
    pending — the UI layer handles their multi-turn sessions after
    the graph completes.
    """
    human_decision = (state.get("human_decision") or "").strip().lower()

    if not human_decision.startswith("implement"):
        return {}

    # Match keywords ONLY against the human's instruction (what they
    # typed after "implement"), not the synthesis or original task.
    # The synthesis contains too much noise — it mentions every topic
    # discussed, which causes false matches.
    human_text = (state.get("human_decision") or "").strip()
    for prefix in ("implement", "Implement", "IMPLEMENT"):
        if human_text.startswith(prefix):
            human_text = human_text[len(prefix):].strip()
            break

    # If the user just typed "implement" with no details, fall back
    # to the original task for matching.
    match_text = human_text if human_text else state.get("current_task", "")

    config = state.get("company_config", {})

    # Build the task briefing for workers
    task_parts = []
    if state.get("current_task"):
        task_parts.append(f"Task: {state['current_task']}")
    if state.get("ceo_synthesis"):
        task_parts.append(f"CEO recommendation: {state['ceo_synthesis']}")
    if state.get("human_decision"):
        task_parts.append(f"Owner direction: {state['human_decision']}")
    task_text = "\n\n".join(task_parts)

    results = []

    for WorkerClass in WORKER_AGENTS:
        if not _worker_matches(WorkerClass, match_text):
            continue

        # Interactive workers are handled by the UI after the graph ends
        if WorkerClass.interactive:
            try:
                WorkerClass(config)  # validate config (e.g. codebase_path)
            except ValueError as e:
                _log(f"[{WorkerClass.role.upper()}] Skipped — {e}")
                continue

            _log(f"[{WorkerClass.role.upper()}] Flagged for interactive session.")
            results.append({
                "worker":  WorkerClass.role,
                "pending": True,
                "task":    task_text,
            })
            continue

        # Non-interactive workers execute inline
        _notify("agent_start", agent=WorkerClass.role, phase="implementation",
                index=0, total=1)

        try:
            worker = WorkerClass(config)
        except ValueError as e:
            _log(f"[{WorkerClass.role.upper()}] Skipped — {e}")
            continue

        _log(f"[{WorkerClass.role.upper()}] Dispatching...")
        result = worker.execute(task_text)
        results.append(result)

        if result.get("success"):
            _log(f"[{WorkerClass.role.upper()}] Complete.")
        else:
            _log(f"[{WorkerClass.role.upper()}] Failed — "
                 f"{result.get('summary', '')[:80]}")

    if not results:
        return {}

    return {
        "worker_results": results,
        "messages": [
            {"role": "assistant",
             "content": f"[{r.get('worker', '?').upper()}] "
                        f"{'Pending interactive session' if r.get('pending') else r.get('summary', '')[:500]}"}
            for r in results
        ],
    }


def _worker_matches(worker_cls, text: str) -> bool:
    """Check if a worker's keywords match the given text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in worker_cls.keywords)


# ── Memory node ───────────────────────────────────────────────────────────────

def memory_write(state: dict) -> dict:
    """
    Flushes the completed session to SQLite and embeds key decisions
    into ChromaDB for future semantic retrieval. Then checks if the
    knowledge indexer should run to update the distilled knowledge doc.
    """
    write_session_to_db(state)
    _log("[MEMORY] Session written to SQLite and ChromaDB.")

    # Check if the indexer should run
    company_id = state.get("company_id", "")
    config = state.get("company_config", {})
    if company_id and config:
        from core.memory.indexer import should_reindex, run_indexer
        if should_reindex(company_id, config):
            _log("[INDEXER] Threshold reached — rebuilding knowledge document...")
            try:
                run_indexer(company_id)
                _log("[INDEXER] Knowledge document updated.")
            except Exception as e:
                _log(f"[INDEXER] Failed (non-fatal): {e}")

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
        # Distilled knowledge doc gets injected as-is (it's already structured)
        distilled = [m for m in state["relevant_memories"]
                     if m.get("source") == "distilled_knowledge"]
        other = [m for m in state["relevant_memories"]
                 if m.get("source") != "distilled_knowledge"]

        if distilled:
            parts.append(
                "COMPANY INSTITUTIONAL KNOWLEDGE:\n"
                + distilled[0].get("reasoning", "")
            )

        if other:
            mem_text = "\n".join(
                f"- {m['task']}: {m['outcome']}"
                for m in other if m.get("task")
            )
            if mem_text:
                parts.append(f"Recent decisions (since last knowledge update):\n{mem_text}")

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
