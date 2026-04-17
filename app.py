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

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl

from core.config import COMPANY_ROOT
from core.graph.session_graph import build_session_graph
from core.graph.nodes import set_progress_callback


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
        cl.Action(name="select_company", payload={"id": c["id"]}, label=c["name"])
        for c in companies
    ]

    res = await cl.AskActionMessage(
        content="**Select a company** to start a deliberation session:",
        actions=actions,
    ).send()

    if res:
        company_id = res.get("payload", {}).get("id") if isinstance(res, dict) else res.payload["id"]
        await _load_company(company_id)


async def _load_company(company_id: str):
    """Load company config and prompt the user for a task."""
    config_path = COMPANY_ROOT / company_id / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    cl.user_session.set("company_id", company_id)
    cl.user_session.set("company_config", config)
    cl.user_session.set("phase", "ready")

    priorities = "\n".join(
        f"- {p}" for p in config.get("strategic_priorities", [])
    )

    await cl.Message(
        content=(
            f"## {config['company_name']}\n\n"
            f"*{config.get('mission', '')}*\n\n"
            f"**Strategic priorities:**\n{priorities}\n\n"
            f"---\n\n"
            f"You can:\n"
            f"- **Chat** — ask questions, get updates, discuss strategy\n"
            f"- **Deliberate** — say \"should we...\" to get C-suite input on a decision\n"
            f"- **Execute** — say \"implement\" or \"draft/build/research...\" to dispatch workers"
        )
    ).send()


# ── Message handler ──────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    phase = cl.user_session.get("phase")

    if phase == "ready":
        intent = _classify_intent(message.content)
        if intent == "deliberate":
            await _run_deliberation(message.content)
        elif intent == "implement":
            await _run_workers_direct(message.content)
        else:
            await _ceo_chat(message.content)

    elif phase == "awaiting_task":
        # Legacy — redirect to ready
        cl.user_session.set("phase", "ready")
        intent = _classify_intent(message.content)
        if intent == "deliberate":
            await _run_deliberation(message.content)
        elif intent == "implement":
            await _run_workers_direct(message.content)
        else:
            await _ceo_chat(message.content)

    elif phase == "awaiting_decision":
        await _resume_with_decision(message.content)

    elif phase == "cca_session":
        await _continue_cca_session(message.content)

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

    graph, checkpointer_ctx = build_session_graph(company_id)
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
        "prior_decision_found": False,
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
        "worker_results":      [],
    }

    # Store graph + thread + checkpointer for resuming after human input
    cl.user_session.set("graph", graph)
    cl.user_session.set("thread_config", thread_config)
    cl.user_session.set("checkpointer_ctx", checkpointer_ctx)

    AGENT_TITLES = {
        "cfo": "CFO (Financial Risk)",
        "coo": "COO (Operations)",
        "cmo": "CMO (Market & Customer)",
        "cto": "CTO (Technical Risk)",
        "ceo": "CEO (Synthesis)",
    }

    status_msg = None  # created fresh for each phase

    async def _new_status(text: str):
        nonlocal status_msg
        status_msg = cl.Message(content=text)
        await status_msg.send()

    await _new_status("Starting deliberation...")

    # Stream graph in a background thread, pushing state snapshots to a queue
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _on_progress(event: str, data: dict):
        """Called from the graph thread when an agent starts working."""
        if event == "agent_start":
            agent = data.get("agent", "")
            phase = data.get("phase", "")
            idx = data.get("index", 0) + 1
            total = data.get("total", 4)
            title = AGENT_TITLES.get(agent, agent.upper())
            bar = "\u2588" * idx + "\u2591" * (total - idx)
            label = "Analyzing" if phase == "deliberation" else (
                "Cross-response" if phase == "cross-response" else "Synthesizing"
            )
            text = f"**{label}** [{bar}] {idx}/{total} — {title} is thinking..."
            asyncio.run_coroutine_threadsafe(
                queue.put(("progress", text)), loop
            )

    def _stream():
        try:
            set_progress_callback(_on_progress)
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
        finally:
            set_progress_callback(None)

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
            cl.user_session.set("phase", "ready")
            await stream_future
            return

        if tag == "done":
            break

        if tag == "progress":
            if status_msg:
                status_msg.content = payload
                await status_msg.update()
            continue

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
                    content="### Cross-Response — Peer Debate"
                ).send()
            else:
                await cl.Message(
                    content=f"### Round {display_round} — Independent Analysis"
                ).send()

            for o in new_outputs:
                await _send_agent_output(o)

            prev_output_count = len(outputs)

            # Send a fresh status message below the outputs
            await _new_status("Continuing deliberation...")

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

    # Save output count so reconsideration knows where prior outputs end
    cl.user_session.set("prev_output_count", prev_output_count)

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
            "- **approve** — accept the recommendation (no implementation)\n"
            "- **implement** — approve and dispatch workers to execute (e.g. CCA writes code)\n"
            "- **override** *reason* — override with your decision\n"
            "- **more info** *details* — provide new information for reconsideration"
        )

    await cl.Message(content=prompt).send()


# ── Resume after human decision ──────────────────────────────────────────────

async def _resume_with_decision(human_input: str):
    graph = cl.user_session.get("graph")
    thread_config = cl.user_session.get("thread_config")

    if not graph or not thread_config:
        await cl.Message(content="Session expired. Please refresh.").send()
        return

    is_more_info = human_input.strip().lower().startswith("more info")

    if is_more_info:
        # The graph will loop back through deliberation and pause again
        # at human_interrupt — run it the same way as the initial deliberation
        cl.user_session.set("phase", "running")

        await cl.Message(
            content="---\n\n**New information received** — sending back to "
                    "the C-suite for reconsideration...",
        ).send()

        AGENT_TITLES = {
            "cfo": "CFO (Financial Risk)",
            "coo": "COO (Operations)",
            "cmo": "CMO (Market & Customer)",
            "cto": "CTO (Technical Risk)",
            "ceo": "CEO (Synthesis)",
        }

        status_msg = None

        async def _new_status(text: str):
            nonlocal status_msg
            status_msg = cl.Message(content=text)
            await status_msg.send()

        await _new_status("Restarting deliberation with new context...")

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _on_progress(event: str, data: dict):
            if event == "agent_start":
                agent = data.get("agent", "")
                phase = data.get("phase", "")
                idx = data.get("index", 0) + 1
                total = data.get("total", 4)
                title = AGENT_TITLES.get(agent, agent.upper())
                bar = "\u2588" * idx + "\u2591" * (total - idx)
                label = "Analyzing" if phase == "deliberation" else (
                    "Cross-response" if phase == "cross-response" else "Synthesizing"
                )
                text = f"**{label}** [{bar}] {idx}/{total} — {title} is thinking..."
                asyncio.run_coroutine_threadsafe(
                    queue.put(("progress", text)), loop
                )

        def _stream():
            try:
                set_progress_callback(_on_progress)
                graph.update_state(
                    thread_config,
                    {"human_decision": human_input},
                    as_node="human_interrupt",
                )
                for state in graph.stream(
                    None, thread_config, stream_mode="values"
                ):
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("state", dict(state))), loop
                    )
                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", str(exc))), loop
                )
            finally:
                set_progress_callback(None)

        stream_future = loop.run_in_executor(None, _stream)

        prev_output_count = cl.user_session.get("prev_output_count", 0)
        last_synthesis = ""
        current_round = 1
        last_state = {}

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

            if tag == "progress":
                if status_msg:
                    status_msg.content = payload
                    await status_msg.update()
                continue

            state = payload
            last_state = state

            outputs = state.get("agent_outputs", [])
            if len(outputs) > prev_output_count:
                new_outputs = outputs[prev_output_count:]

                first = new_outputs[0]
                is_cross = "_response" in first.get("agent", "")
                if is_cross:
                    await cl.Message(
                        content="### Cross-Response — Peer Debate"
                    ).send()
                else:
                    await cl.Message(
                        content="### Reconsideration — Independent Analysis"
                    ).send()

                for o in new_outputs:
                    await _send_agent_output(o)

                prev_output_count = len(outputs)
                await _new_status("Continuing deliberation...")

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

        # Save output count for potential further reconsiderations
        cl.user_session.set("prev_output_count", prev_output_count)

        # Prompt for decision again
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
                "- **approve** — accept the recommendation (no implementation)\n"
                "- **implement** — approve and dispatch workers to execute (e.g. CCA writes code)\n"
                "- **override** *reason* — override with your decision\n"
                "- **more info** *details* — provide new information for reconsideration"
            )
        await cl.Message(content=prompt).send()

    else:
        # Approve, implement, or override — finalize the session
        cl.user_session.set("phase", "writing_memory")

        is_implement = human_input.strip().lower().startswith("implement")
        if is_implement:
            await cl.Message(content="**Dispatching workers...**").send()

        final_states = []

        def _resume():
            graph.update_state(
                thread_config,
                {"human_decision": human_input},
                as_node="human_interrupt",
            )
            for state in graph.stream(
                None, thread_config, stream_mode="values"
            ):
                final_states.append(dict(state))

        try:
            await asyncio.get_running_loop().run_in_executor(None, _resume)
        except Exception as e:
            await cl.Message(content=f"Error: {e}").send()

        final_state = final_states[-1] if final_states else {}
        worker_results = final_state.get("worker_results", [])

        # Show results from non-interactive workers
        for r in worker_results:
            if r.get("pending"):
                continue
            worker_name = r.get("worker", "unknown").upper()
            success = r.get("success", False)
            summary = r.get("summary", "")
            files = r.get("files_changed", [])
            status = "completed" if success else "failed"
            parts = [f"### {worker_name} — {status}\n"]
            if summary:
                parts.append(summary[:1000])
            if files:
                parts.append(
                    "\n**Files changed:**\n"
                    + "\n".join(f"- `{f}`" for f in files)
                )
            await cl.Message(
                content="\n".join(parts), author=worker_name
            ).send()

        # Check for pending interactive workers (CCA)
        pending = [r for r in worker_results if r.get("pending")]
        if pending:
            # Start interactive CCA session
            cca_pending = next(
                (r for r in pending if r["worker"] == "cca"), None
            )
            if cca_pending:
                await _start_cca_session(
                    cca_pending["task"],
                    final_state.get("company_config", {}),
                )
                return  # don't clean up yet — CCA session is ongoing

        if is_implement and not worker_results:
            await cl.Message(
                content="No matching workers found for this task. "
                        "Decision recorded without implementation."
            ).send()

        await _finalize_session()


# ── Intent classification ────────────────────────────────────────────────────

def _classify_intent(text: str) -> str:
    """
    Classify user message into one of three intents:
        "deliberate" — needs C-suite deliberation (decisions, strategy, should-we)
        "implement"  — direct worker execution (implement X, build X, draft X)
        "chat"       — everything else (questions, updates, conversation)
    """
    lower = text.strip().lower()

    # Explicit implement command
    if lower.startswith("implement"):
        return "implement"

    # Deliberation triggers — the user is asking the C-suite to weigh in
    deliberation_phrases = [
        "should we", "should i", "let's decide", "lets decide",
        "deliberate on", "deliberate about", "i need a decision",
        "what do you recommend", "weigh in on", "c-suite",
        "evaluate whether", "assess whether", "is it worth",
        "pros and cons", "risks of", "make a decision",
    ]
    if any(phrase in lower for phrase in deliberation_phrases):
        return "deliberate"

    # Direct task triggers — user wants something done, not discussed
    task_phrases = [
        "draft ", "write ", "create ", "build ", "research ",
        "analyze ", "post to ", "send ", "fix ", "update ",
        "deploy ", "set up ", "configure ",
    ]
    if any(lower.startswith(phrase) for phrase in task_phrases):
        return "implement"

    # Default: conversational
    return "chat"


# ── CEO conversational chat ─────────────────────────────────────────────────

async def _ceo_chat(message: str):
    """
    The CEO answers conversationally from the knowledge document
    and company context. No deliberation, no formal pipeline.
    """
    company_id = cl.user_session.get("company_id")
    config = cl.user_session.get("company_config")

    if not config:
        await cl.Message(content="No company loaded. Please refresh.").send()
        return

    from core.agents.ceo import CEOAgent
    from core.agents.base import invoke_llm
    from core.memory.indexer import load_knowledge

    ceo = CEOAgent(config)
    knowledge = load_knowledge(company_id) if company_id else ""

    prompt_parts = [
        f"You are the CEO of {config.get('company_name', 'the company')}.",
        f"You are having a normal conversation with the owner.",
        f"Answer naturally and directly. You are NOT in a formal deliberation.",
        f"Do not produce JSON. Do not recommend 'proceed/block/modify'.",
        f"Just talk like a knowledgeable executive having a conversation.",
    ]

    if knowledge:
        prompt_parts.append(
            f"\nYou have access to the company's full institutional knowledge:\n"
            f"{knowledge}"
        )

    prompt_parts.append(f"\n--- OWNER SAYS ---\n{message}")

    prompt = "\n\n".join(prompt_parts)
    response = invoke_llm(ceo.llm, prompt)

    await cl.Message(content=response, author="CEO").send()


# ── Direct worker dispatch (no deliberation) ────────────────────────────────

async def _run_workers_direct(message: str):
    """
    Dispatch workers directly without going through deliberation.
    Used when the user gives a direct instruction like 'implement X'
    or 'draft a blog post about Y'.
    """
    from core.graph.nodes import _worker_matches
    from core.agents import WORKER_AGENTS

    config = cl.user_session.get("company_config")
    if not config:
        await cl.Message(content="No company loaded. Please refresh.").send()
        return

    # Strip "implement" prefix if present
    task_text = message.strip()
    for prefix in ("implement", "Implement", "IMPLEMENT"):
        if task_text.startswith(prefix):
            task_text = task_text[len(prefix):].strip()
            break

    match_text = task_text if task_text else message

    # Find matching workers
    matched = [W for W in WORKER_AGENTS if _worker_matches(W, match_text)]

    if not matched:
        # No workers match — treat as a chat message instead
        await _ceo_chat(message)
        return

    await cl.Message(content="**Dispatching workers...**").send()

    # Run non-interactive workers
    for WorkerClass in matched:
        if WorkerClass.interactive:
            # Start interactive session (CCA)
            try:
                agent = WorkerClass(config)
            except ValueError as e:
                await cl.Message(
                    content=f"{WorkerClass.role.upper()} skipped: {e}"
                ).send()
                continue
            await _start_cca_session(task_text, config)
            return  # CCA takes over the session

        # Non-interactive worker
        try:
            worker = WorkerClass(config)
        except ValueError as e:
            await cl.Message(
                content=f"{WorkerClass.role.upper()} skipped: {e}"
            ).send()
            continue

        await cl.Message(
            content=f"**{WorkerClass.title}** is working...",
        ).send()

        result = await asyncio.get_running_loop().run_in_executor(
            None, worker.execute, task_text
        )

        worker_name = result.get("worker", "unknown").upper()
        success = result.get("success", False)
        output = result.get("output", "")

        if success and output:
            await cl.Message(content=output, author=worker_name).send()
        elif success:
            await cl.Message(
                content=result.get("summary", "Done."), author=worker_name
            ).send()
        else:
            await cl.Message(
                content=f"**Failed:** {result.get('summary', 'Unknown error')}",
                author=worker_name,
            ).send()


# ── Session cleanup ──────────────────────────────────────────────────────────

async def _finalize_session():
    """Clean up and return to ready state."""
    ctx = cl.user_session.get("checkpointer_ctx")
    if ctx:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass

    cl.user_session.set("phase", "ready")
    await cl.Message(
        content="Session complete. Decision written to memory."
    ).send()


# ── Interactive CCA session ──────────────────────────────────────────────────

async def _cca_stream_callback(msg: dict):
    """Called by CCA as each message arrives — streams to the UI in real time."""
    msg_type = msg.get("type", "")
    content = msg.get("content", "")

    if not content:
        return

    if msg_type == "text":
        await cl.Message(content=content, author="CCA").send()
    elif msg_type == "tool_use":
        await cl.Message(content=f"`{content}`", author="CCA").send()
    elif msg_type == "result":
        is_error = msg.get("is_error", False)
        if is_error:
            await cl.Message(
                content=f"**Error:** {content}", author="CCA"
            ).send()
        elif content:
            await cl.Message(content=content, author="CCA").send()


async def _start_cca_session(task: str, company_config: dict):
    """Start an interactive Claude Code Agent session."""
    from core.agents.cca import CCAAgent

    try:
        agent = CCAAgent(company_config)
    except ValueError as e:
        await cl.Message(content=f"CCA error: {e}").send()
        await _finalize_session()
        return

    await cl.Message(
        content="### CCA Session Started\n\n"
                "The Claude Code Agent is working on your task. "
                "You'll see its progress in real time below.\n\n"
                "When it finishes, you can send follow-up instructions "
                "or type **done** to end the session.",
        author="CCA",
    ).send()

    try:
        messages, session_id = await agent.start_session(
            task, on_message=_cca_stream_callback
        )
    except Exception as e:
        await cl.Message(content=f"CCA failed to start: {e}").send()
        await _finalize_session()
        return

    cl.user_session.set("cca_agent", agent)
    cl.user_session.set("cca_session_id", session_id)
    cl.user_session.set("phase", "cca_session")

    await cl.Message(
        content="---\n\n"
                "Send follow-up instructions, or type **done** to finish.",
    ).send()


async def _continue_cca_session(user_input: str):
    """Handle a message during an active CCA session."""
    if user_input.strip().lower() == "done":
        await cl.Message(
            content="### CCA Session Ended", author="CCA"
        ).send()
        await _finalize_session()
        return

    agent = cl.user_session.get("cca_agent")
    session_id = cl.user_session.get("cca_session_id")

    if not agent or not session_id:
        await cl.Message(content="CCA session expired. Ending.").send()
        await _finalize_session()
        return

    try:
        messages, new_session_id = await agent.continue_session(
            session_id, user_input, on_message=_cca_stream_callback
        )
    except Exception as e:
        await cl.Message(content=f"CCA error: {e}").send()
        await _finalize_session()
        return

    if new_session_id:
        cl.user_session.set("cca_session_id", new_session_id)

    await cl.Message(
        content="---\n\n"
                "Send follow-up instructions, or type **done** to finish.",
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


AGENT_FULL_NAMES = {
    "cfo": "CFO — Chief Financial Officer",
    "coo": "COO — Chief Operating Officer",
    "cmo": "CMO — Chief Marketing Officer",
    "cto": "CTO — Chief Technology Officer",
}


async def _send_agent_output(output: dict):
    """Format and send a single agent output as a Chainlit message."""
    raw_agent = output.get("agent", "unknown")
    is_cross = "_response" in raw_agent
    agent_key = raw_agent.replace("_response", "")
    author = AGENT_FULL_NAMES.get(agent_key, agent_key.upper())
    phase_label = " (cross-response)" if is_cross else ""

    rec = output.get("recommendation", "?").upper()
    conf = output.get("confidence", 0.0)
    analysis = output.get("analysis", "")
    concerns = output.get("concerns", [])

    header = f"### {author}{phase_label}\n\n"
    content = header + f"**{rec}** · {conf:.0%} confidence\n\n{analysis}"
    if concerns:
        content += "\n\n**Concerns:**\n" + "\n".join(f"- {c}" for c in concerns)

    await cl.Message(content=content, author=agent_key.upper()).send()
