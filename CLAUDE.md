# Project Context for Claude Code
## Agentic C-Suite System

This file gives Claude Code the context it needs to work on this project
without requiring the full design conversation history.

---

## What This Project Is

A locally-hosted multi-agent system simulating a full C-suite executive team.
Multiple independent "company instances" can be spun up, each with its own
C-suite that deliberates on decisions and brings structured recommendations
to the human owner (the user). The human is always the final decision-maker.

---

## Key Design Decisions (Do Not Change Without Discussion)

1. **LangGraph** for orchestration — chosen specifically for native
   human-in-the-loop interrupt support and clean multi-instance state isolation.
   Do not swap for CrewAI or AutoGen.

2. **One shared Ollama inference server** — all agents queue through a single
   server. No parallelism. The RTX 3090 runs one model at a time by design.

3. **CEO is sole interface to user** — the CEO does not participate in
   deliberation rounds. It synthesizes and presents. This is intentional.

4. **Two deliberation rounds maximum** — round 1 (independent), cross-response,
   CEO synthesis. If conflict: round 2 with CEO conflict framing, CEO synthesis.
   If still deadlocked: escalate. Never more than 2 full rounds.

5. **Hybrid output format** — agents return structured JSON with one free-form
   field (`analysis`). The envelope (recommendation, confidence, concerns) is
   strict and parsed; the analysis is natural language.

6. **One SQLite DB per company** — never mix company data. Separate files,
   separate blast radius.

7. **Human overrides are first-class data** — every time the user overrules
   agents, it's stored with reasoning. This is the highest-signal memory.

8. **operator.add on agent_outputs** — this is intentional. Outputs accumulate
   across rounds. Do not change to a replace pattern.

---

## Hardware Context

- OS: Windows 10 Pro
- CPU: AMD Ryzen 9 5950X (16-core)
- RAM: 64 GB
- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- SSD: D:\ and E:\ (CT2000BX500SSD1)
- HDD: F:\ and G:\ (WDC WD100EDAZ)

## Drive Layout

```
D:\csuite\           ← project root (this repo) — SSD
D:\models\           ← model cache (set via OLLAMA_MODELS env var) — SSD
E:\venvs\csuite\     ← Python virtual environment — SSD
F:\csuite_logs\      ← session logs — HDD
G:\csuite_data\      ← company SQLite databases — HDD
```

---

## Python Environment

```powershell
# Activate before any Python work
E:\venvs\csuite\Scripts\Activate.ps1
```

Python version: 3.11.x (specific — not 3.12/3.13)

---

## Inference Stack

- Runtime: Ollama (http://localhost:11434)
- Primary model: qwen2.5:32b-instruct-q4_K_M
- Embedding model: nomic-embed-text (768 dimensions)

Verify both are available:
```bash
ollama list
```

---

## What Has Been Built (Complete)

- [x] core/state.py              — CompanyState TypedDict with operator.add fields
- [x] core/agents/base.py        — BaseAgent: LLM call, hybrid parser, retry logic
- [x] core/agents/ceo.py         — CEO: synthesis, presentation, escalation enforcement
- [x] core/agents/cfo.py         — CFO: financial risk lens
- [x] core/agents/coo.py         — COO: operational feasibility lens
- [x] core/agents/cmo.py         — CMO: market and customer lens
- [x] core/agents/cto.py         — CTO: technical risk lens
- [x] core/graph/session_graph.py — LangGraph graph builder + compiler
- [x] core/graph/nodes.py         — All node functions
- [x] core/graph/edges.py         — conflict_router conditional edge
- [x] core/graph/runner.py        — CLI entry point
- [x] core/memory/retrieval.py    — ChromaDB semantic search + SQLite queries
- [x] core/memory/writer.py       — SQLite write + ChromaDB embed
- [x] scripts/new_company.py      — Company scaffolding script
- [x] companies/example_company/config.json — Example DNA config

## What Has NOT Been Built Yet (Next Steps in Order)

- [ ] Chainlit UI layer (replace terminal runner with web chat interface)
- [ ] Live end-to-end test (first real deliberation session)
- [ ] core/tools/ (tools agents can call — placeholder currently)
- [ ] Worker agent tier (subordinate to C-suite, executes tasks)
- [ ] Knowledge ingestion pipeline (load docs into semantic memory)
- [ ] Multi-task agenda handling (queue of tasks in one session)

---

## Current Next Step

Build the Chainlit UI layer in `app.py` at the project root.
It should replace `core/graph/runner.py` as the primary interaction surface,
wrapping the same graph logic with a web chat interface that:
- Shows each agent's output as it arrives (streaming feel)
- Formats the deliberation report with clear visual structure
- Handles the human interrupt as a chat input
- Lets the user select a company at startup

---

## Memory Architecture Summary

| Layer | Technology | Location | Lifespan |
|---|---|---|---|
| Working | LangGraph state (RAM) | In-process | Session only |
| Episodic | SQLite | G:\csuite_data\ | Permanent |
| Semantic | ChromaDB | D:\csuite\companies\<id>\chroma\ | Permanent |
| DNA | JSON config | D:\csuite\companies\<id>\config.json | Until edited |

---

## Agent Output Schema

All agents except CEO return:
```json
{
  "analysis":       "<free-form natural language>",
  "recommendation": "proceed | block | modify",
  "concerns":       ["specific concern"],
  "confidence":     0.0
}
```

CEO synthesis returns:
```json
{
  "synthesis":      "<free-form natural language>",
  "consensus":      true,
  "conflicts":      ["specific conflict between named executives"],
  "recommendation": "proceed | block | modify | escalate",
  "escalate":       false,
  "reasoning":      "<one to three sentences>"
}
```
