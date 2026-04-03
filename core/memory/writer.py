"""
core/memory/writer.py

Memory write helpers called by the memory_write graph node.

After every session, this module:
  1. Upserts the session record in SQLite
  2. Writes each decision and its agent votes to SQLite
  3. Embeds each decision's reasoning into ChromaDB for future retrieval

Design principle: write full reasoning, not just outcomes.
"We approved budget X" is useless. The dissenting CFO vote is the
valuable part. Every agent's position is preserved verbatim.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

import chromadb
from langchain_ollama import OllamaEmbeddings

from core.config import COMPANY_ROOT, DATA_ROOT

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE = "http://localhost:11434"

# ── Public interface ──────────────────────────────────────────────────────────

def write_session_to_db(state: dict) -> None:
    """
    Main write entry point. Called once at the end of every session.
    Writes to SQLite first (always), then embeds to ChromaDB (best-effort).
    """
    company_id = state.get("company_id", "unknown")
    _ensure_db_exists(company_id)

    session_id = state.get("session_id", str(uuid.uuid4()))
    now        = datetime.now(timezone.utc).isoformat()

    _write_session(state, session_id, now, company_id)
    _write_decisions(state, session_id, now, company_id)
    _embed_decisions(state, company_id)


# ── Internal: DB initialisation ───────────────────────────────────────────────

def _db_path(company_id: str) -> Path:
    path = DATA_ROOT / company_id
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{company_id}.db"


def _ensure_db_exists(company_id: str) -> None:
    """
    Creates the SQLite database and schema if they don't exist.
    Safe to call on every session — uses CREATE TABLE IF NOT EXISTS.
    """
    db = _db_path(company_id)
    with sqlite3.connect(str(db)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id      TEXT PRIMARY KEY,
                company_id      TEXT NOT NULL,
                started_at      TEXT NOT NULL,
                ended_at        TEXT,
                task_count      INTEGER DEFAULT 0,
                outcome_summary TEXT
            );

            CREATE TABLE IF NOT EXISTS decisions (
                decision_id     TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                company_id      TEXT NOT NULL,
                task            TEXT NOT NULL,
                outcome         TEXT,
                reasoning       TEXT,
                escalated       INTEGER DEFAULT 0,
                human_override  TEXT,
                decided_at      TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS agent_votes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id     TEXT NOT NULL,
                agent           TEXT NOT NULL,
                recommendation  TEXT NOT NULL,
                analysis        TEXT,
                concerns        TEXT,
                confidence      REAL,
                FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
            );

            CREATE TABLE IF NOT EXISTS knowledge (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id  TEXT NOT NULL,
                category    TEXT,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL,
                source      TEXT,
                added_at    TEXT,
                chroma_id   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_decisions_company
                ON decisions(company_id, decided_at DESC);
            CREATE INDEX IF NOT EXISTS idx_votes_decision
                ON agent_votes(decision_id);
            CREATE INDEX IF NOT EXISTS idx_knowledge_company
                ON knowledge(company_id, category);
        """)


# ── Internal: SQLite writers ──────────────────────────────────────────────────

def _write_session(state: dict, session_id: str, now: str, company_id: str) -> None:
    decisions = state.get("decisions_made", [])
    summary   = _build_session_summary(state)

    db = _db_path(company_id)
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, company_id, started_at, ended_at,
                 task_count, outcome_summary)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                company_id,
                state.get("session_start", now),
                now,
                len(decisions),
                summary,
            ),
        )


def _write_decisions(
    state: dict, session_id: str, now: str, company_id: str
) -> None:
    decisions    = state.get("decisions_made", [])
    agent_outputs = state.get("agent_outputs", [])
    db = _db_path(company_id)

    # If no structured decisions were produced (e.g. escalated without resolving),
    # synthesise one from the current task and state.
    if not decisions:
        decisions = [_synthesise_decision(state, session_id, now)]

    with sqlite3.connect(str(db)) as conn:
        for decision in decisions:
            decision_id = decision.get("decision_id", str(uuid.uuid4()))

            conn.execute(
                """
                INSERT OR REPLACE INTO decisions
                    (decision_id, session_id, company_id, task, outcome,
                     reasoning, escalated, human_override, decided_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    session_id,
                    company_id,
                    decision.get("task", state.get("current_task", "")),
                    decision.get("outcome", ""),
                    decision.get("reasoning", state.get("ceo_synthesis", "")),
                    1 if state.get("escalate_to_human") else 0,
                    state.get("human_decision", ""),
                    decision.get("timestamp", now),
                ),
            )

            # Write each agent's round-1 vote (not cross-responses — those are
            # captured in the analysis text of the decision)
            round1_outputs = [
                o for o in agent_outputs
                if "_response" not in o.get("agent", "")
            ]
            for output in round1_outputs:
                conn.execute(
                    """
                    INSERT INTO agent_votes
                        (decision_id, agent, recommendation,
                         analysis, concerns, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        output.get("agent", ""),
                        output.get("recommendation", ""),
                        output.get("analysis", ""),
                        json.dumps(output.get("concerns", [])),
                        output.get("confidence", 0.5),
                    ),
                )


def _synthesise_decision(state: dict, session_id: str, now: str) -> dict:
    """
    Builds a minimal decision record when the session ended without a
    formal Decision object (e.g. pure escalation, session interrupted).
    """
    human = state.get("human_decision", "")
    return {
        "decision_id": str(uuid.uuid4()),
        "session_id":  session_id,
        "task":        state.get("current_task", ""),
        "outcome":     human if human else "escalated — pending human decision",
        "reasoning":   state.get("ceo_synthesis", ""),
        "votes":       {},
        "human_override": human,
        "timestamp":   now,
    }


def _build_session_summary(state: dict) -> str:
    """One-line session summary for the sessions table."""
    task     = state.get("current_task", "")
    rounds   = state.get("debate_round", 1) - 1
    escalate = state.get("escalate_to_human", False)
    decision = state.get("human_decision", "")

    parts = [f"Task: {task[:80]}"]
    if rounds > 1:
        parts.append(f"{rounds} deliberation rounds")
    parts.append("escalated" if escalate else "resolved internally")
    if decision:
        parts.append(f"human decision: {decision[:60]}")
    return " | ".join(parts)


# ── Internal: ChromaDB embedding ─────────────────────────────────────────────

def _embed_decisions(state: dict, company_id: str) -> None:
    """
    Embeds each decision's full reasoning into ChromaDB.
    The document stored is the CEO's synthesis — this is what semantic
    search will match against future tasks.

    Errors here are non-fatal. SQLite is the source of truth.
    """
    chroma_path = COMPANY_ROOT / company_id / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)

    decisions    = state.get("decisions_made", [])
    agent_outputs = state.get("agent_outputs", [])

    # Build the text to embed: CEO synthesis + all agent analyses
    full_deliberation = _build_embed_document(state, agent_outputs)
    if not full_deliberation.strip():
        return

    try:
        embedder   = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)
        client     = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(
            name="decisions",
            metadata={"hnsw:space": "cosine"},
        )

        task = state.get("current_task", "")
        outcome = (
            decisions[-1].get("outcome", "") if decisions
            else state.get("human_decision", "escalated")
        )
        human_override = state.get("human_decision", "")

        doc_id    = str(uuid.uuid4())
        embedding = embedder.embed_documents([full_deliberation])[0]

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[full_deliberation],
            metadatas=[{
                "company_id":     company_id,
                "task":           task,
                "outcome":        outcome,
                "human_override": human_override,
                "escalated":      str(state.get("escalate_to_human", False)),
            }],
        )

    except Exception as e:
        print(f"[memory] ChromaDB embed error (non-fatal): {e}")


def _build_embed_document(state: dict, agent_outputs: list[dict]) -> str:
    """
    Builds the text document to embed. We embed the full deliberation —
    not just the outcome — so future semantic searches can match on the
    reasoning, not just the task description.
    """
    parts = []

    task = state.get("current_task", "")
    if task:
        parts.append(f"Task: {task}")

    synthesis = state.get("ceo_synthesis", "")
    if synthesis:
        parts.append(f"CEO synthesis: {synthesis}")

    for output in agent_outputs:
        if "_response" in output.get("agent", ""):
            continue   # embed only round-1 analyses to keep doc focused
        agent    = output.get("agent", "").upper()
        analysis = output.get("analysis", "")
        rec      = output.get("recommendation", "")
        if analysis:
            parts.append(f"{agent} ({rec}): {analysis}")

    human = state.get("human_decision", "")
    if human:
        parts.append(f"Human decision: {human}")

    return "\n\n".join(parts)
