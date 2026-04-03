"""
core/memory/retrieval.py

Memory retrieval helpers called by the memory_retrieval graph node.

At the start of every session, before any agent sees the task, this module:
  1. Queries ChromaDB for semantically similar past decisions (meaning-based)
  2. Queries SQLite for recent sessions (recency context)
  3. Queries SQLite for any human overrides related to this topic
  4. Returns a combined package injected into CompanyState.relevant_memories

Only the CEO sees this full context. Other agents receive only what the CEO
chooses to include in their briefing — this keeps agent prompts lean and
prevents context window bloat on the worker tier.
"""

import sqlite3

import chromadb
from langchain_ollama import OllamaEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL   = "nomic-embed-text"
OLLAMA_BASE   = "http://localhost:11434"
TOP_K_SEMANTIC = 5    # number of semantically similar decisions to retrieve
RECENT_SESSIONS = 3   # number of recent sessions to surface for recency context
from core.config import COMPANY_ROOT, DATA_ROOT

# ── Public interface ──────────────────────────────────────────────────────────

def retrieve_relevant_memories(
    company_id: str,
    query:      str,
    top_k:      int = TOP_K_SEMANTIC,
) -> list[dict]:
    """
    Main retrieval entry point. Returns a merged list of relevant memories
    ready to be injected into CompanyState.relevant_memories.

    Each item in the returned list is a dict with keys:
        task, outcome, reasoning, human_override, source, similarity_score
    """
    memories = []

    # 1. Semantic search — past decisions similar in meaning to current task
    semantic = _semantic_search(company_id, query, top_k)
    memories.extend(semantic)

    # 2. Recent sessions — last N sessions for recency context
    recent = _recent_decisions(company_id, limit=RECENT_SESSIONS)
    for r in recent:
        # Avoid duplicating anything already found by semantic search
        if not any(m["task"] == r["task"] for m in memories):
            memories.append(r)

    # 3. Human overrides — any time the human owner overruled the agents
    overrides = _human_overrides(company_id, limit=3)
    for o in overrides:
        if not any(m["task"] == o["task"] for m in memories):
            memories.append(o)

    return memories


# ── Internal: ChromaDB semantic search ───────────────────────────────────────

def _semantic_search(company_id: str, query: str, top_k: int) -> list[dict]:
    """
    Embeds the query and retrieves the most semantically similar
    past decisions from this company's ChromaDB store.
    """
    chroma_path = COMPANY_ROOT / company_id / "chroma"
    if not chroma_path.exists():
        return []

    try:
        client     = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(
            name="decisions",
            metadata={"hnsw:space": "cosine"},
        )

        if collection.count() == 0:
            return []

        embedder   = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)
        query_emb  = embedder.embed_query(query)

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            memories.append({
                "task":             meta.get("task", ""),
                "outcome":          meta.get("outcome", ""),
                "reasoning":        doc,
                "human_override":   meta.get("human_override", ""),
                "source":           "semantic_search",
                "similarity_score": round(1 - dist, 3),  # cosine distance → similarity
            })

        return memories

    except Exception as e:
        # Retrieval failure must never crash a session
        print(f"[memory] ChromaDB retrieval error: {e}")
        return []


# ── Internal: SQLite recency queries ─────────────────────────────────────────

def _db_path(company_id: str) -> Path:
    return DATA_ROOT / company_id / f"{company_id}.db"


def _recent_decisions(company_id: str, limit: int = 3) -> list[dict]:
    """
    Fetches the most recent decisions from SQLite for recency context.
    Gives agents awareness of what has been decided lately, even if not
    semantically related to the current task.
    """
    db = _db_path(company_id)
    if not db.exists():
        return []

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT task, outcome, reasoning, human_override, decided_at
                FROM decisions
                WHERE outcome IS NOT NULL
                ORDER BY decided_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "task":           row["task"],
                "outcome":        row["outcome"],
                "reasoning":      row["reasoning"] or "",
                "human_override": row["human_override"] or "",
                "source":         "recent_history",
                "similarity_score": None,
            }
            for row in rows
        ]
    except Exception as e:
        print(f"[memory] SQLite recent decisions error: {e}")
        return []


def _human_overrides(company_id: str, limit: int = 3) -> list[dict]:
    """
    Fetches decisions where the human owner overruled the agent recommendation.
    These are high-value memories — they encode what the human actually values
    vs. what the agents recommended.
    """
    db = _db_path(company_id)
    if not db.exists():
        return []

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT task, outcome, reasoning, human_override, decided_at
                FROM decisions
                WHERE human_override IS NOT NULL AND human_override != ''
                ORDER BY decided_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "task":           row["task"],
                "outcome":        row["outcome"],
                "reasoning":      row["reasoning"] or "",
                "human_override": row["human_override"],
                "source":         "human_override",
                "similarity_score": None,
            }
            for row in rows
        ]
    except Exception as e:
        print(f"[memory] SQLite human overrides error: {e}")
        return []
