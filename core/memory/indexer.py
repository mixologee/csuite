"""
core/memory/indexer.py

Distilled knowledge indexer — Karpathy-style RAG bypass.

Instead of retrieving chunks at query time, the indexer periodically
reads ALL company history from SQLite and distills it into a structured
knowledge document (knowledge.md) that fits entirely in the LLM context.

The indexer runs:
    - Automatically after memory_write when enough new decisions accumulate
    - Manually via: python -m core.memory.indexer --company <id>
    - On first run: indexes everything from scratch

Config fields (in company config.json):
    indexer_model        — model to use for indexing (default: company's model_name)
    index_threshold      — re-index after this many new decisions (default: 5)
    index_version_days   — save versioned snapshot every N days (default: 7)
    context_length       — used to compute max knowledge doc size (50% of context)

Output:
    CSUITE_COMPANY_ROOT/<id>/knowledge.md          — current distilled knowledge
    CSUITE_COMPANY_ROOT/<id>/knowledge_versions/    — timestamped snapshots
    CSUITE_COMPANY_ROOT/<id>/index_meta.json        — indexer state tracking
"""

import json
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.agents.base import build_llm, invoke_llm, DEFAULT_OLLAMA_MODEL
from core.config import COMPANY_ROOT, DATA_ROOT


# ── Indexer prompt ───────────────────────────────────────────────────────────

INDEXER_PROMPT = """You are a knowledge distiller for {company_name}, a company in {industry}.

Your job is to read the complete history of decisions, deliberations, and owner
directives below, and produce a structured knowledge document that captures
the institutional memory of this company.

This document will be loaded into the context window of AI agents at the start
of every future session. It replaces traditional search — the agents will read
this entire document, so it must be comprehensive, well-organized, and concise.

TARGET SIZE: Stay under {max_tokens_estimate} words. Be thorough but not verbose.

Produce the document with exactly these sections:

## Decision Precedents
Group past decisions by topic area. For each, state what was decided, why,
and the outcome. Flag any decisions that were later revisited or reversed.

## Owner Profile
Based on the owner's approvals, overrides, and directives, describe:
- What the owner values most (patterns in approvals)
- What the owner rejects or overrides (patterns in disagreements)
- The owner's decision-making style and risk tolerance

## Executive Dynamics
Which agents tend to agree? Which have recurring tensions?
What types of issues cause deadlock vs. quick consensus?

## Strategic Context
How have the company's priorities evolved over time?
What strategic shifts are visible in the decision history?

## Lessons Learned
Decisions that went well (owner approved, no regrets visible in follow-ups).
Decisions that were problematic (reversed, overridden, or led to problems).

---

COMPANY HISTORY:

{history}

---

Now produce the knowledge document. Write in clear, factual prose.
Do not invent information — only include what is evidenced in the history above.
If the history is thin, say so — do not pad with speculation.
"""


# ── Public interface ─────────────────────────────────────────────────────────

def run_indexer(company_id: str, force: bool = False) -> bool:
    """
    Run the indexer for a company. Returns True if a new knowledge.md
    was generated, False if skipped (not enough new decisions).

    Args:
        company_id: The company to index.
        force: If True, re-index regardless of threshold.
    """
    config = _load_company_config(company_id)
    if not config:
        print(f"[indexer] No config found for company '{company_id}'")
        return False

    meta = _load_meta(company_id)
    threshold = config.get("index_threshold", 5)

    # Count decisions since last index
    current_count = _count_decisions(company_id)
    last_count = meta.get("last_indexed_decision_count", 0)
    new_decisions = current_count - last_count

    if not force and new_decisions < threshold:
        print(f"[indexer] {company_id}: {new_decisions} new decisions "
              f"(threshold: {threshold}) — skipping.")
        return False

    if current_count == 0:
        print(f"[indexer] {company_id}: No decisions in database — skipping.")
        return False

    print(f"[indexer] {company_id}: Indexing {current_count} total decisions "
          f"({new_decisions} new)...")

    # Build full history from SQLite
    history = _build_full_history(company_id)

    # Compute max knowledge doc size (50% of context window, in words ~= tokens * 0.75)
    context_length = config.get("context_length", 32768)
    max_tokens = context_length // 2
    max_words_estimate = int(max_tokens * 0.75)

    # Build the indexer prompt
    company_name = config.get("company_name", company_id)
    industry = config.get("industry", "general")

    prompt = INDEXER_PROMPT.format(
        company_name=company_name,
        industry=industry,
        max_tokens_estimate=max_words_estimate,
        history=history,
    )

    # Build indexer LLM (may use a different model than agents)
    indexer_config = dict(config)
    indexer_model = config.get("indexer_model", "")
    if indexer_model:
        indexer_config["model_name"] = indexer_model

    llm = build_llm(indexer_config, temperature=0.3, max_tokens=max_tokens)
    print(f"[indexer] Generating knowledge document...")
    knowledge_text = invoke_llm(llm, prompt)

    # Write knowledge.md
    knowledge_path = COMPANY_ROOT / company_id / "knowledge.md"
    knowledge_path.write_text(knowledge_text, encoding="utf-8")
    print(f"[indexer] Written: {knowledge_path} ({len(knowledge_text)} chars)")

    # Version snapshot if enough days have passed
    _maybe_version(company_id, knowledge_path, meta, config)

    # Update metadata
    meta["last_indexed_at"] = datetime.now(timezone.utc).isoformat()
    meta["last_indexed_decision_count"] = current_count
    _save_meta(company_id, meta)

    return True


def should_reindex(company_id: str, config: dict) -> bool:
    """
    Quick check: should the indexer run?

    Returns True if:
        - No knowledge.md exists yet and there are decisions in the DB
        - The decision count has crossed the threshold since the last index

    Called by memory_write to decide whether to trigger.
    """
    knowledge_path = COMPANY_ROOT / company_id / "knowledge.md"
    current_count = _count_decisions(company_id)

    # First index: create knowledge.md if any decisions exist
    if not knowledge_path.exists() and current_count > 0:
        return True

    meta = _load_meta(company_id)
    threshold = config.get("index_threshold", 5)
    last_count = meta.get("last_indexed_decision_count", 0)
    return (current_count - last_count) >= threshold


def load_knowledge(company_id: str) -> str:
    """
    Load the distilled knowledge document for a company.
    Returns the full text, or empty string if it doesn't exist.
    """
    knowledge_path = COMPANY_ROOT / company_id / "knowledge.md"
    if knowledge_path.exists():
        return knowledge_path.read_text(encoding="utf-8")
    return ""


# ── Internal: SQLite history extraction ──────────────────────────────────────

def _load_company_config(company_id: str) -> dict:
    config_path = COMPANY_ROOT / company_id / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _count_decisions(company_id: str) -> int:
    db_path = DATA_ROOT / company_id / f"{company_id}.db"
    if not db_path.exists():
        return 0
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return row[0] if row else 0


def _build_full_history(company_id: str) -> str:
    """
    Read the entire decision history from SQLite and format it as
    structured text for the indexer LLM.
    """
    db_path = DATA_ROOT / company_id / f"{company_id}.db"
    if not db_path.exists():
        return "No history available."

    parts = []

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # All decisions with their context
        decisions = conn.execute("""
            SELECT d.task, d.outcome, d.reasoning, d.human_override,
                   d.escalated, d.decided_at, s.outcome_summary
            FROM decisions d
            LEFT JOIN sessions s ON d.session_id = s.session_id
            ORDER BY d.decided_at ASC
        """).fetchall()

        for i, d in enumerate(decisions, 1):
            entry = [f"### Decision {i} ({d['decided_at'] or 'unknown date'})"]
            entry.append(f"**Task:** {d['task']}")
            entry.append(f"**Outcome:** {d['outcome'] or 'pending'}")

            if d["reasoning"]:
                entry.append(f"**CEO Reasoning:** {d['reasoning'][:500]}")
            if d["escalated"]:
                entry.append("**Escalated:** Yes")
            if d["human_override"]:
                entry.append(f"**Owner Override:** {d['human_override']}")

            # Get agent votes for this decision
            votes = conn.execute("""
                SELECT agent, recommendation, confidence, concerns
                FROM agent_votes
                WHERE decision_id = (
                    SELECT decision_id FROM decisions
                    WHERE task = ? AND decided_at = ?
                    LIMIT 1
                )
            """, (d["task"], d["decided_at"])).fetchall()

            if votes:
                vote_lines = []
                for v in votes:
                    concerns = v["concerns"] or "[]"
                    vote_lines.append(
                        f"  - {v['agent'].upper()}: {v['recommendation']} "
                        f"(confidence: {v['confidence']:.0%})"
                    )
                entry.append("**Agent Votes:**\n" + "\n".join(vote_lines))

            parts.append("\n".join(entry))

        # Knowledge entries
        knowledge = conn.execute("""
            SELECT category, title, content, source, added_at
            FROM knowledge
            ORDER BY added_at ASC
        """).fetchall()

        if knowledge:
            parts.append("\n---\n\n## Company Knowledge Entries\n")
            for k in knowledge:
                parts.append(
                    f"**{k['title']}** ({k['category'] or 'general'}, "
                    f"{k['added_at'] or 'unknown date'})\n{k['content']}"
                )

    if not parts:
        return "No history available — this is a new company with no prior decisions."

    return "\n\n".join(parts)


# ── Internal: metadata tracking ──────────────────────────────────────────────

def _meta_path(company_id: str) -> Path:
    return COMPANY_ROOT / company_id / "index_meta.json"


def _load_meta(company_id: str) -> dict:
    path = _meta_path(company_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_meta(company_id: str, meta: dict) -> None:
    path = _meta_path(company_id)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ── Internal: versioning ─────────────────────────────────────────────────────

def _maybe_version(company_id: str, knowledge_path: Path, meta: dict,
                    config: dict) -> None:
    """
    Save a versioned snapshot of knowledge.md if enough days have passed.
    """
    version_days = config.get("index_version_days", 7)
    last_version = meta.get("last_version_at", "")

    should_version = True
    if last_version:
        try:
            last_dt = datetime.fromisoformat(last_version)
            elapsed = datetime.now(timezone.utc) - last_dt
            should_version = elapsed >= timedelta(days=version_days)
        except (ValueError, TypeError):
            pass

    if not should_version:
        return

    version_dir = COMPANY_ROOT / company_id / "knowledge_versions"
    version_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_path = version_dir / f"knowledge_{timestamp}.md"
    shutil.copy2(str(knowledge_path), str(version_path))

    meta["last_version_at"] = datetime.now(timezone.utc).isoformat()
    print(f"[indexer] Version saved: {version_path}")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    parser = argparse.ArgumentParser(
        description="Run the knowledge indexer for a company."
    )
    parser.add_argument("--company", required=True, help="Company ID")
    parser.add_argument("--force", action="store_true",
                        help="Re-index regardless of threshold")
    args = parser.parse_args()

    success = run_indexer(args.company, force=args.force)
    sys.exit(0 if success else 1)
