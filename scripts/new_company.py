"""
scripts/new_company.py

Scaffolds a new company instance — creates the folder structure,
generates a config.json template, and initialises the SQLite database.

Usage:
    python scripts/new_company.py --id acme_corp --name "Acme Corp" --industry "B2B SaaS"
    python scripts/new_company.py --id widget_co  (uses defaults, edit config.json after)

After running this script:
    1. Edit CSUITE_COMPANY_ROOT/<id>/config.json to fill in company details
    2. Run a test session: python -m core.graph.runner --company <id> --task "Hello"
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import COMPANY_ROOT, DATA_ROOT, LOG_ROOT


def scaffold_company(company_id: str, company_name: str, industry: str) -> None:
    print(f"\n  Scaffolding company: {company_name} ({company_id})\n")

    # ── Folder structure ──────────────────────────────────────────────────
    ssd_path   = COMPANY_ROOT / company_id
    chroma_path = ssd_path / "chroma"
    data_path  = DATA_ROOT / company_id
    log_path   = LOG_ROOT / company_id / "sessions"

    for path in [ssd_path, chroma_path, data_path, log_path]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {path}")

    # ── config.json ────────────────────────────────────────────────────────
    config = {
        "company_id":   company_id,
        "company_name": company_name,
        "industry":     industry,
        "stage":        "growth",
        "founded":      "2024",

        "mission": f"[Edit this] The mission of {company_name}.",
        "strategic_priorities": [
            "[Edit this] Priority 1",
            "[Edit this] Priority 2",
            "[Edit this] Priority 3",
        ],
        "constraints": [
            "[Edit this] Constraint 1",
            "[Edit this] Constraint 2",
        ],

        "model_provider":  "ollama",
        "model_name":      "gpt-oss:20b",
        "context_length":  32768,

        "indexer_model":       "",
        "index_threshold":     5,
        "index_version_days":  7,

        "risk_profile":   "moderate",
        "decision_style": "data-driven with bias toward action",

        "escalation_rules": {
            "always_escalate": [
                "Any spend over $10,000",
                "Hiring decisions",
                "Pricing changes",
                "Partnership agreements",
            ],
            "escalate_if_deadlock": True,
            "ceo_can_decide_alone": [
                "Tactical priorities within approved budget",
                "Agent task assignments",
                "Meeting agendas",
            ],
        },

        "_comment_codebase_path": "Absolute path to the codebase this company manages. Required for CCA (Claude Code Agent) to execute implementation tasks. Leave empty if not applicable.",
        "codebase_path": "",

        "agent_personalities": {
            "ceo": "Decisive and synthesis-focused. Comfortable with ambiguity "
                   "but demands clarity before committing.",
            "cfo": "Conservative and data-driven. Always asks about runway and "
                   "unit economics. Skeptical of growth-at-all-costs thinking.",
            "coo": "Process-oriented and execution-focused. Asks how before why. "
                   "Resistant to plans that haven't stress-tested their assumptions.",
            "cmo": "Customer-empathy-driven and brand-conscious. Advocates for "
                   "user experience over short-term revenue optimisation.",
            "cto": "Pragmatic and reliability-focused. Prefers boring technology "
                   "that works. Resistant to rewrites and premature optimisation.",
        },
    }

    config_path = ssd_path / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Created: {config_path}")

    # ── SQLite database ────────────────────────────────────────────────────
    db_path = data_path / f"{company_id}.db"
    with sqlite3.connect(str(db_path)) as conn:
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
    print(f"  Created: {db_path}")

    # ── Done ───────────────────────────────────────────────────────────────
    print(f"""
  Done. Next steps:

    1. Edit the config:
       {config_path}

    2. Fill in mission, priorities, constraints, and personalities.

    3. Run a test session:
       python -m core.graph.runner --company {company_id} --task "Introduce yourself and describe your priorities."
""")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Scaffold a new company instance."
    )
    parser.add_argument("--id",       required=True,  help="Company ID (snake_case, no spaces)")
    parser.add_argument("--name",     default="",     help="Company display name")
    parser.add_argument("--industry", default="General", help="Industry description")
    return parser.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    name    = args.name or args.id.replace("_", " ").title()
    scaffold_company(args.id, name, args.industry)
