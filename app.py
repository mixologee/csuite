"""
app.py

Chainlit web interface for the C-suite deliberation system.
Replaces core/graph/runner.py as the primary interaction surface.

Run with:
    chainlit run app.py

Flow:
    1. User selects a company at startup
    2. User submits a task for the C-suite to deliberate on
    3. Agents deliberate — outputs shown progressively as each node completes
    4. CEO synthesizes and presents recommendation
    5. User responds (approve / override / more info)
    6. Decision written to memory, session ready for next task
"""

import asyncio
import json

import chainlit as cl

from core.config import COMPANY_ROOT
from core.graph.session_graph import build_session_graph


# ── Startup: company selection ───────────────────────────────────────────────

@cl.on_chat_start
async def start():
    """List available companies and let the user pick one."""
    companies = _list_companies()

    if not companies:
        await cl.Message(
            content=(
                "No companies found.\n\n"
                "Run `python scripts/new_company.py --id <name>` to create one."
            )
        ).send()
        return

    actions = [
        cl.Action(name="select_company", value=c["id"], label=c["name"])
        for c in companies
    ]

    res = await cl.AskActionMessage(
        content="**Select a company** to start a deliberation session:",
        actions=actions,
    ).send()

    if res:
        company_id = res.get("value") if isinstance(res, dict) else res.value
        await _load_company(company_id)


async def _load_company(company_id: str):
    """Load company config and prompt the user for a task."""
    config_path = COMPANY_ROOT / company_id / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    cl.user_session.set("company_id", company_id)
    cl.user_session.set("company_config", config)
    cl.user_session.set("phase", "awaiting_task")

    priorities = "\n".join(
        f"- {p}" for p in config.get("strategic_priorities", [])
    )

    await cl.Message(
        content=(
            f"## {config['company_name']}\n\n"
            f"*{config.get('mission', '')}*\n\n"
            f"**Strategic priorities:**\n{priorities}\n\n"
            f"---\n\n"
            f"What decision or question should the C-suite deliberate on?"
        )
    ).send()


# ── Message handler ──────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    phase = cl.user_session.get("phase")

    if phase == "awaiting_task":
        await _run_deliberation(message.content)

    elif phase == "awaiting_decision":
        await _resume_with_decision(message.content)

    elif phase == "running":
        await cl.Message(
            content="Deliberation in progress. Please wait for it to finish."
        ).send()

    else:
        await cl.Message(
            content="No active session. Please refresh to start."
        ).send()


# ── Deliberation: stream the graph and display results ───────────────────────

async def _run_deliberation(task: str):
    company_id = cl.user_session.get("company_id")
    config = cl.user_session.get("company_config")

    cl.user_session.set("phase", "running")

    graph = build_session_graph(company_id)
    thread_config = {"configurable": {"thread_id": f"{company_id}_{hash(task)}"}}

    initial_state = {
        "company_id":           company_id,
        "company_name":         config["company_name"],
        "company_config":       config,
        "current_task":         task,
        "task_context":         "",
        "agenda":               [],
        "relevant_memories":    [],
        "agent_outputs":        [],
        "messages":             [],
        "consensus_reached":    False,
        "escalate_to_human":    False,
        "escalation_reason":    "",
        "human_decision":       None,
        "decisions_made":       [],
        "debate_round":         1,
        "session_id":           "",
        "session_start":        "",
        "ceo_synthesis":        "",
        "conflicts_identified": [],
    }

    # Store graph + thread for resuming after human input
    cl.user_session.set("graph", graph)
    cl.user_session.set("thread_config", thread_config)

    await cl.Message(content="Starting deliberation...").send()

    # Stream graph in a background thread, pushing state snapshots to a queue
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _stream():
        try:
            for state in graph.stream(
                initial_state, thread_config, stream_mode="values"
            ):
                asyncio.run_coroutine_threadsafe(
                    queue.put(("state", dict(state))), loop
                )
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", str(exc))), loop
            )

    stream_future = loop.run_in_executor(None, _stream)

    # Track what we've already displayed
    prev_output_count = 0
    last_synthesis = ""
    current_round = 1
    last_state = initial_state

    while True:
        tag, payload = await queue.get()

        if tag == "error":
            await cl.Message(
                content=f"Deliberation failed:\n\n```\n{payload}\n```"
            ).send()
            cl.user_session.set("phase", "awaiting_task")
            await stream_future
            return

        if tag == "done":
            break

        state = payload
        last_state = state

        # ── Detect round change ──────────────────────────────────────────
        round_num = state.get("debate_round", 1)
        if round_num > current_round:
            if round_num == 2:
                conflicts = state.get("conflicts_identified", [])
                conflict_text = "\n".join(f"- {c}" for c in conflicts)
                await cl.Message(
                    content=(
                        "---\n\n**Round 2** — The CEO has identified conflicts "
                        "and is sending the team back for focused deliberation."
                        + (f"\n\n{conflict_text}" if conflict_text else "")
                    ),
                    author="CEO",
                ).send()
            current_round = round_num

        # ── Show new agent outputs ───────────────────────────────────────
        outputs = state.get("agent_outputs", [])
        if len(outputs) > prev_output_count:
            new_outputs = outputs[prev_output_count:]

            # Insert a phase header when we see the first output of a batch
            first = new_outputs[0]
            is_cross = "_response" in first.get("agent", "")
            display_round = current_round
            if is_cross:
                await cl.Message(
                    content=f"### Cross-Response — Peer Debate"
                ).send()
            else:
                await cl.Message(
                    content=f"### Round {display_round} — Independent Analysis"
                ).send()

            for o in new_outputs:
                await _send_agent_output(o)

            prev_output_count = len(outputs)

        # ── Show CEO synthesis ───────────────────────────────────────────
        synthesis = state.get("ceo_synthesis", "")
        if synthesis and synthesis != last_synthesis:
            consensus = state.get("consensus_reached", False)
            label = "Consensus Reached" if consensus else "Conflict Detected"
            await cl.Message(
                content=f"### CEO Synthesis — {label}\n\n{synthesis}",
                author="CEO",
            ).send()
            last_synthesis = synthesis

    await stream_future

    # ── Prompt for human decision ────────────────────────────────────────
    cl.user_session.set("phase", "awaiting_decision")

    escalated = last_state.get("escalate_to_human", False)
    if escalated:
        prompt = (
            "---\n\n"
            "**Escalated — your decision is required.**\n\n"
            "The C-suite could not reach consensus. Please respond with:\n"
            "- Your decision (**proceed** / **block** / **modify**)\n"
            "- Your reasoning (this becomes institutional memory)\n"
            "- Any instructions for the team"
        )
    else:
        prompt = (
            "---\n\n"
            "**Your response:**\n"
            "- **approve** — accept the recommendation\n"
            "- **override** *reason* — override with your decision\n"
            "- **more info** *question* — ask for clarification"
        )

    await cl.Message(content=prompt).send()


# ── Resume after human decision ──────────────────────────────────────────────

async def _resume_with_decision(human_input: str):
    graph = cl.user_session.get("graph")
    thread_config = cl.user_session.get("thread_config")

    if not graph or not thread_config:
        await cl.Message(content="Session expired. Please refresh.").send()
        return

    cl.user_session.set("phase", "writing_memory")

    def _resume():
        graph.update_state(
            thread_config,
            {"human_decision": human_input},
            as_node="human_interrupt",
        )
        for _ in graph.stream(None, thread_config, stream_mode="values"):
            pass

    try:
        await asyncio.get_running_loop().run_in_executor(None, _resume)
        await cl.Message(
            content="Session complete. Decision written to memory."
        ).send()
    except Exception as e:
        await cl.Message(content=f"Error writing to memory: {e}").send()

    # Ready for next task
    cl.user_session.set("phase", "awaiting_task")
    await cl.Message(
        content="You can submit another task or close the session."
    ).send()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _list_companies() -> list[dict]:
    """Scan COMPANY_ROOT for directories containing a valid config.json."""
    if not COMPANY_ROOT.exists():
        return []

    companies = []
    for d in sorted(COMPANY_ROOT.iterdir()):
        config_file = d / "config.json"
        if d.is_dir() and config_file.exists():
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                companies.append({
                    "id": d.name,
                    "name": config.get("company_name", d.name),
                })
            except (json.JSONDecodeError, OSError):
                continue
    return companies


async def _send_agent_output(output: dict):
    """Format and send a single agent output as a Chainlit message."""
    raw_agent = output.get("agent", "unknown")
    is_cross = "_response" in raw_agent
    agent_name = raw_agent.replace("_response", "").upper()
    phase_label = " (cross-response)" if is_cross else ""

    rec = output.get("recommendation", "?").upper()
    conf = output.get("confidence", 0.0)
    analysis = output.get("analysis", "")
    concerns = output.get("concerns", [])

    content = f"**{rec}** · {conf:.0%} confidence{phase_label}\n\n{analysis}"
    if concerns:
        content += "\n\n**Concerns:**\n" + "\n".join(f"- {c}" for c in concerns)

    await cl.Message(content=content, author=agent_name).send()
