# Installation & Setup Guide

Step-by-step instructions for setting up the Agentic C-Suite System on a
Windows machine with an NVIDIA GPU.

Throughout this guide, replace these placeholders with your actual paths:

| Placeholder | Description | Example |
|---|---|---|
| `{PROJECT}` | Drive/path for the project repo | `D:\csuite` |
| `{MODELS}` | Drive/path for Ollama model cache | `D:\models\ollama` |
| `{VENV}` | Drive/path for the Python virtual environment | `E:\venvs\csuite` |
| `{DATA}` | Drive/path for company databases | `G:\csuite_data` |
| `{COMPANIES}` | Drive/path for company configs + knowledge | `G:\csuite_data\companies` |
| `{LOGS}` | Drive/path for session logs | `F:\csuite_logs` |

**Tip:** Use SSDs for `{PROJECT}`, `{MODELS}`, and `{VENV}` (fast read/write).
HDDs are fine for `{DATA}`, `{COMPANIES}`, and `{LOGS}` (sequential writes).

---

## Prerequisites

| Component | Requirement |
|---|---|
| OS | Windows 10/11 |
| GPU | NVIDIA with 20+ GB VRAM (RTX 3090 recommended) |
| RAM | 32 GB minimum, 64 GB recommended |
| Storage | 40+ GB free on SSD for models and code |
| Python | **3.11.x** (not 3.12, not 3.13, not 3.14) |
| Node.js | Required for Claude Code CLI (CCA worker) |

---

## Step 1: NVIDIA Drivers

Make sure your NVIDIA drivers are installed and working:

```
nvidia-smi
```

You should see your GPU listed with driver version and VRAM. If this
command fails, install drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers).

---

## Step 2: Install Python 3.11

Download Python 3.11 from [python.org](https://www.python.org/downloads/release/python-31111/).

During installation:
- Check "Add Python to PATH"
- Use the default install location or note where you install it

Verify:
```powershell
python --version
# Should show: Python 3.11.x
```

**Important:** The system requires Python 3.11 specifically. Python 3.14
has compatibility issues with the async stack (anyio, uvicorn, chainlit).
Python 3.12+ may work but is untested.

---

## Step 3: Create the Virtual Environment

```powershell
python -m venv {VENV}
{VENV}\Scripts\Activate.ps1
pip install --upgrade pip
```

---

## Step 4: Install Ollama

**Before installing Ollama**, set the model storage location so models
are stored on your SSD, not the default location:

1. Open **System Properties** → **Advanced** → **Environment Variables**
2. Add a new **System variable**:
   - Variable: `OLLAMA_MODELS`
   - Value: `{MODELS}`

Then install Ollama from [ollama.com](https://ollama.com).

Verify:
```bash
ollama --version
```

---

## Step 5: Pull Models

```bash
ollama pull gpt-oss:20b               # primary model (~12 GB)
ollama pull nomic-embed-text           # embedding model (~250 MB)
```

Verify both are available:
```bash
ollama list
```

You should see both models listed.

---

## Step 6: Clone the Repository

```powershell
git clone <your-repo-url> {PROJECT}
cd {PROJECT}
```

---

## Step 7: Install Python Dependencies

```powershell
# Make sure the venv is active
{VENV}\Scripts\Activate.ps1

cd {PROJECT}
pip install -r requirements.txt
```

This installs LangGraph, Chainlit, ChromaDB, and all other dependencies.

---

## Step 8: Install Claude Code CLI (for CCA Worker)

The Claude Code Agent (CCA) requires the Claude Code CLI:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify:
```bash
claude --version
```

**Note:** CCA is optional. The system works without it — you just won't be
able to dispatch code implementation tasks via the `implement` command.

---

## Step 9: Set Data Path Environment Variables

Company data is stored outside the repository. Set these environment
variables so the system knows where to find and store data:

1. Open **System Properties** → **Advanced** → **Environment Variables**
2. Add these **System variables**:

| Variable | Value | Purpose |
|---|---|---|
| `CSUITE_COMPANY_ROOT` | `{COMPANIES}` | Company configs, prompts, knowledge docs |
| `CSUITE_DATA_ROOT` | `{DATA}` | SQLite databases |
| `CSUITE_LOG_ROOT` | `{LOGS}` | Session logs |

All three have defaults in `core/config.py`. You can use any paths you
like — just make sure they exist or can be created.

**Note:** You may need to restart your terminal or IDE after setting
environment variables.

---

## Step 10: Create Your First Company

```powershell
# Make sure venv is active
{VENV}\Scripts\Activate.ps1

cd {PROJECT}
python scripts/new_company.py --id my_company --name "My Company" --industry "Your Industry"
```

This creates:
- `{COMPANIES}/my_company/config.json` — company configuration
- `{COMPANIES}/my_company/prompts/*.md` — agent personality prompts
- `{COMPANIES}/my_company/chroma/` — vector store (starts empty)
- `{DATA}/my_company/my_company.db` — SQLite database
- `{LOGS}/my_company/sessions/` — log directory

---

## Step 11: Configure Your Company

### Edit config.json

Open `{COMPANIES}/my_company/config.json` and set:

- `mission` — one sentence describing the company's purpose
- `strategic_priorities` — current top priorities (list)
- `constraints` — hard limits the agents must respect (list)
- `risk_profile` — "conservative", "moderate", or "aggressive"
- `escalation_rules.always_escalate` — topics that always require your approval
- `codebase_path` — absolute path to your codebase (required for CCA worker, leave empty if not applicable)

### Edit Agent Prompts

Each agent's personality is defined in a markdown file:

```
{COMPANIES}/my_company/prompts/
    ├── ceo.md
    ├── cfo.md
    ├── coo.md
    ├── cmo.md
    └── cto.md
```

The default prompts are functional but generic. For best results, customize
each prompt with:
- The agent's thinking frameworks and decision-making style
- Company-specific context and priorities
- Behavioral rules and communication style

See the `templates/prompts/` directory in the repo for the starter format.

---

## Step 12: Run the Application

```powershell
# Make sure venv is active
{VENV}\Scripts\Activate.ps1

cd {PROJECT}
chainlit run app.py
```

Open your browser to `http://localhost:8000`.

1. Select your company from the list
2. Start chatting — the CEO responds conversationally
3. Say "should we..." to trigger a full C-suite deliberation
4. Say "draft...", "build...", "research..." to dispatch workers

---

## Optional: Configure the .env File

If you plan to use the Anthropic API (Claude) instead of Ollama for
any company, add your API key to `{PROJECT}\.env`:

```
ANTHROPIC_API_KEY=your-key-here
```

Then set `"model_provider": "anthropic"` in that company's `config.json`.

---

## Verifying the Installation

Run these commands to verify everything is working:

```powershell
# 1. Check Python version
python --version
# Expected: Python 3.11.x

# 2. Check Ollama is running
ollama list
# Expected: gpt-oss:20b and nomic-embed-text listed

# 3. Check GPU
nvidia-smi
# Expected: GPU listed with driver info

# 4. Check imports
cd {PROJECT}
python -c "from core.graph.session_graph import build_session_graph; print('OK')"
# Expected: OK

# 5. Check Claude Code CLI (optional)
claude --version
# Expected: version number
```

---

## Troubleshooting

### "No module named 'core'"
Make sure you're running from the project root (`{PROJECT}`) with the
virtual environment activated.

### "ModuleNotFoundError: langgraph.checkpoint.sqlite"
Run: `pip install langgraph-checkpoint-sqlite`

### Chainlit crashes with "NoEventLoopError" or "AsyncLibraryNotFoundError"
You're likely running Python 3.14. Downgrade to Python 3.11 and rebuild
the venv.

### "Claude Code CLI not found"
Install it: `npm install -g @anthropic-ai/claude-code`
If installed but not found, CCA resolves the path from
`%APPDATA%\npm\claude.cmd` automatically.

### CCA errors with "Missing required field 'signature'"
This is a known compatibility issue between the Claude Code SDK and Ollama.
The system handles it gracefully — work completed before the error is
preserved, and the session remains usable.

### "No companies found" on startup
Make sure `CSUITE_COMPANY_ROOT` points to the right directory and you've
run `new_company.py` to create at least one company.

---

## Directory Reference

```
{PROJECT}\              ← project root (this repo)
{MODELS}\               ← Ollama model cache (OLLAMA_MODELS env var)
{VENV}\                 ← Python virtual environment

{DATA}\                 ← CSUITE_DATA_ROOT — SQLite databases
{COMPANIES}\            ← CSUITE_COMPANY_ROOT — company data
    └── <company_id>\
        ├── config.json
        ├── prompts\        ← agent personality prompts
        ├── knowledge.md    ← distilled memory (auto-generated)
        ├── knowledge_versions\
        └── chroma\

{LOGS}\                 ← CSUITE_LOG_ROOT — session logs
```
