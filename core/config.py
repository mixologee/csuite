"""
core/config.py

Centralised path configuration for the C-suite system.

All external data paths are read from environment variables so that
company data, logs, and databases never need to live inside the repo.

Set these before running:

    CSUITE_COMPANY_ROOT  — where company folders live (config.json, chroma/)
                           Default: G:/csuite_data/companies
    CSUITE_DATA_ROOT     — where SQLite databases live
                           Default: G:/csuite_data
    CSUITE_LOG_ROOT      — where session logs are written
                           Default: F:/csuite_logs
"""

import os
from pathlib import Path

COMPANY_ROOT = Path(os.environ.get("CSUITE_COMPANY_ROOT", "G:/csuite_data/companies"))
DATA_ROOT    = Path(os.environ.get("CSUITE_DATA_ROOT",    "G:/csuite_data"))
LOG_ROOT     = Path(os.environ.get("CSUITE_LOG_ROOT",     "F:/csuite_logs"))

# ── Default tunables (overridable per company in config.json) ────────────────

DEFAULTS = {
    "chat_history_length":    20,     # max messages kept in conversation history
    "chat_message_cap":       10000,  # max chars per message in history context
    "cca_max_turns":          50,     # max turns per session
    "worker_max_tokens":      4096,   # max output tokens for non-interactive workers
    "ceo_chat_max_tokens":    2048,   # max output tokens for CEO conversational replies
    "knowledge_max_pct":      50,     # max % of context_length for knowledge.md
}


def get_tunable(company_config: dict, key: str):
    """
    Read a tunable setting from company config, falling back to DEFAULTS.
    Company config values override defaults.
    """
    return company_config.get(key, DEFAULTS.get(key))


def load_agent_prompt(company_id: str, role: str, config: dict) -> str:
    """
    Load an agent's personality/behavioral prompt.

    Checks for a markdown file first:
        CSUITE_COMPANY_ROOT/<company_id>/prompts/<role>.md

    Falls back to the one-liner in config.json:
        config["agent_personalities"]["<role>"]

    Returns the prompt text (may be multi-paragraph markdown from .md
    or a single sentence from config.json).
    """
    prompt_file = COMPANY_ROOT / company_id / "prompts" / f"{role}.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()

    return (
        config.get("agent_personalities", {})
              .get(role, f"You are the {role.upper()} of this company.")
    )
