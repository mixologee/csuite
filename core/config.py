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
