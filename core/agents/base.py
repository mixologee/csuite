"""
core/agents/base.py

Base class for all C-suite agents. Handles LLM communication,
hybrid JSON/natural-language response parsing, and retry logic.

Output format used by all agents:
{
    "analysis":       "<free-form natural language — as long as needed>",
    "recommendation": "proceed" | "block" | "modify",
    "concerns":       ["specific concern", ...],
    "confidence":     0.0 – 1.0
}

The 'analysis' field is intentionally unstructured. The three remaining
fields are strict and parsed into typed values. This gives the CEO
reliable data to work with while keeping reasoning human-readable.
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from langchain_ollama import OllamaLLM

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME    = "qwen2.5:32b-instruct-q4_K_M"
OLLAMA_BASE   = "http://localhost:11434"
MAX_RETRIES   = 3
RETRY_DELAY   = 2.0  # seconds between retries

VALID_RECOMMENDATIONS = {"proceed", "block", "modify"}

# ── Response schema shown to every agent in their prompt ────────────────────

RESPONSE_SCHEMA = """
Respond with a single JSON object. No preamble, no markdown fences, no text
outside the JSON. Use this exact schema:

{
  "analysis": "<your full reasoning here — write as many sentences as needed,
               this field is free-form natural language>",
  "recommendation": "<exactly one of: proceed | block | modify>",
  "concerns": ["<specific concern>", "<specific concern>"],
  "confidence": <float between 0.0 and 1.0>
}

Rules:
- analysis: Write your complete reasoning in natural language. Be specific.
  Reference the company's goals, constraints, and any relevant past decisions.
  This is the most important field — do not truncate it.
- recommendation: Must be exactly "proceed", "block", or "modify". No other values.
- concerns: List the specific risks or objections you hold, even if recommending
  "proceed". Empty list [] only if you have zero concerns.
- confidence: How certain you are in your recommendation. 0.9+ = very sure.
  0.5 = genuinely uncertain. Be honest.
"""


class BaseAgent(ABC):
    """
    Abstract base for all C-suite agents.

    Subclasses must define:
        role        (str)  — e.g. "cfo"
        title       (str)  — e.g. "Chief Financial Officer"
        _build_system_prompt(config) -> str
    """

    role:  str
    title: str

    def __init__(self, company_config: dict):
        self.config  = company_config
        self.company = company_config.get("company_name", "the company")
        self.llm     = OllamaLLM(
            model       = MODEL_NAME,
            base_url    = OLLAMA_BASE,
            temperature = 0.7,
            num_predict = 2048,
        )
        self.system_prompt = self._build_system_prompt(company_config)

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def _build_system_prompt(self, config: dict) -> str:
        """
        Build the system prompt that defines this agent's identity,
        expertise, and behavioral constraints.
        """

    # ── Public methods ───────────────────────────────────────────────────────

    def analyze(self, briefing: str) -> dict:
        """
        Round 1: Independent analysis of the task.
        The agent has not yet seen any peer outputs.
        """
        prompt = self._wrap_prompt(
            instruction=(
                f"You are analyzing this task independently. "
                f"No other executive has weighed in yet. "
                f"Apply your expertise as {self.title} and give your honest assessment.\n\n"
                f"{briefing}"
            )
        )
        return self._call_with_retry(prompt)

    def respond_to_peers(self, briefing: str, peer_outputs: list[dict]) -> dict:
        """
        Cross-response round: The agent reads all peer outputs and responds.
        This is where genuine debate happens — agents may agree, push back,
        or propose modifications to the emerging consensus.
        """
        peer_section = self._format_peer_outputs(peer_outputs)
        prompt = self._wrap_prompt(
            instruction=(
                f"You have now read your colleagues' positions on this task. "
                f"As {self.title}, respond to their reasoning. "
                f"You may agree, disagree, or propose modifications. "
                f"Be specific about which colleague's point you are addressing "
                f"and why. Do not simply restate your round-1 position.\n\n"
                f"{briefing}\n\n"
                f"--- YOUR COLLEAGUES' POSITIONS ---\n\n"
                f"{peer_section}"
            )
        )
        return self._call_with_retry(prompt)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _wrap_prompt(self, instruction: str) -> str:
        """
        Combines system prompt, instruction, and response schema
        into a single prompt string for Ollama.
        """
        return (
            f"{self.system_prompt}\n\n"
            f"--- TASK ---\n\n"
            f"{instruction}\n\n"
            f"--- RESPONSE FORMAT ---\n"
            f"{RESPONSE_SCHEMA}"
        )

    def _call_with_retry(self, prompt: str) -> dict:
        """
        Calls the LLM with retry logic. On parse failure, asks the model
        to correct its output rather than discarding it entirely.
        """
        last_error = None
        raw        = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if attempt == 1:
                    raw = self.llm.invoke(prompt)
                else:
                    # Ask the model to fix its previous output
                    fix_prompt = (
                        f"Your previous response could not be parsed as valid JSON.\n"
                        f"Error: {last_error}\n"
                        f"Your response was:\n{raw}\n\n"
                        f"Please rewrite it as a valid JSON object following this schema:\n"
                        f"{RESPONSE_SCHEMA}"
                    )
                    raw = self.llm.invoke(fix_prompt)

                return self._parse_response(raw)

            except (ValueError, KeyError) as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue

        # All retries exhausted — return a degraded output rather than crashing
        return self._fallback_output(raw or "", str(last_error))

    def _parse_response(self, raw: str) -> dict:
        """
        Hybrid parser. Extracts the JSON object from the model's response,
        then validates and coerces each field.

        Raises ValueError if the response cannot be parsed at all.
        """
        # Strip markdown fences if the model included them despite instructions
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        # Find the outermost JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in response: {cleaned[:200]}")

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error: {e}")

        # Validate and coerce fields
        analysis       = str(data.get("analysis", "")).strip()
        recommendation = str(data.get("recommendation", "")).strip().lower()
        concerns_raw   = data.get("concerns", [])
        confidence_raw = data.get("confidence", 0.5)

        if not analysis:
            raise ValueError("Empty 'analysis' field")

        if recommendation not in VALID_RECOMMENDATIONS:
            # Attempt fuzzy recovery
            for valid in VALID_RECOMMENDATIONS:
                if valid in recommendation:
                    recommendation = valid
                    break
            else:
                raise ValueError(
                    f"Invalid recommendation '{recommendation}'. "
                    f"Must be one of: {VALID_RECOMMENDATIONS}"
                )

        concerns = [str(c).strip() for c in concerns_raw if str(c).strip()]

        try:
            confidence = float(confidence_raw)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        return {
            "analysis":       analysis,
            "recommendation": recommendation,
            "concerns":       concerns,
            "confidence":     confidence,
        }

    def _fallback_output(self, raw: str, error: str) -> dict:
        """
        Last resort when all retries fail. Returns a flagged output
        rather than crashing the whole session.
        """
        return {
            "analysis": (
                f"[PARSE FAILURE — {self.title}] "
                f"The agent produced a response that could not be parsed after "
                f"{MAX_RETRIES} attempts. Raw output preserved for review:\n\n"
                f"{raw[:1000]}\n\nError: {error}"
            ),
            "recommendation": "modify",
            "concerns":       ["Agent output could not be parsed — human review required"],
            "confidence":     0.0,
        }

    @staticmethod
    def _format_peer_outputs(peer_outputs: list[dict]) -> str:
        """
        Formats peer outputs into readable text for the cross-response prompt.
        """
        sections = []
        for output in peer_outputs:
            agent = output.get("agent", "unknown").upper().replace("_RESPONSE", "")
            rec   = output.get("recommendation", "unknown")
            conf  = output.get("confidence", 0.0)
            analysis = output.get("analysis", "")
            concerns = output.get("concerns", [])

            concern_text = (
                "\n".join(f"  • {c}" for c in concerns)
                if concerns else "  None stated"
            )

            sections.append(
                f"{agent} [{rec.upper()} · confidence {conf:.0%}]\n"
                f"{analysis}\n"
                f"Concerns:\n{concern_text}"
            )

        return "\n\n" + ("─" * 40 + "\n\n").join(sections)
