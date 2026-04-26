"""
core/agents/ceo.py

Chief Executive Officer agent.

The CEO is architecturally distinct from all other agents. It does not
participate in rounds 1 or 2 of deliberation — it reads the outputs of
those rounds and synthesizes them into a decision or escalation.

The CEO has three jobs:
  1. Synthesize — find the path through the C-suite's competing positions
  2. Arbitrate  — identify genuine conflicts vs. surface disagreements
  3. Present    — deliver the full deliberation to the human in a structured,
                  readable format that makes the decision easy to evaluate

The CEO is the only agent that talks directly to the human. All other agents'
outputs flow through the CEO before the human ever sees them.

CEO response schema (different from base agents):
{
    "synthesis":       "<free-form natural language — full reasoning>",
    "consensus":       true | false,
    "conflicts":       ["<conflict description>", ...],
    "recommendation":  "proceed" | "block" | "modify" | "escalate",
    "escalate":        true | false,
    "reasoning":       "<why the CEO reached this conclusion specifically>"
}
"""

import json
import re
import time
from typing import Optional

from core.agents.base import build_llm, invoke_llm, MAX_RETRIES, RETRY_DELAY

# ── CEO response schema ───────────────────────────────────────────────────────

CEO_RESPONSE_SCHEMA = """
Respond with a single JSON object. No preamble, no markdown fences, no text
outside the JSON. Use this exact schema:

{
  "synthesis": "<your full synthesis of the deliberation — free-form natural language,
                as long as needed. This is the most important field.>",
  "consensus": <true if the C-suite has reached a workable agreement, false if not>,
  "conflicts": ["<specific unresolved conflict between named executives>", ...],
  "recommendation": "<exactly one of: proceed | block | modify | escalate>",
  "escalate": <true if this decision must go to the human owner, false if you can resolve it>,
  "reasoning": "<concise explanation of why you reached this recommendation specifically>"
}

Rules:
- synthesis: Write your complete read of the room. Name which executives agree,
  which disagree, and where the core tension lies. Be specific.
- consensus: true when executives agree on the general direction, even if they
  differ on details, conditions, or emphasis. "Proceed with caution" and
  "proceed if budget allows" are consensus — the direction is the same.
  Only set false when executives have genuinely incompatible positions
  (e.g. one says proceed and another says block for irreconcilable reasons).
  Differences in framing, emphasis, or suggested conditions are NOT conflict.
- conflicts: Empty list [] when consensus is true. When false, list only
  genuinely irreconcilable tensions — not minor differences in emphasis.
- recommendation: "escalate" when consensus is false after round 2, or when the
  decision falls under the company's escalation rules regardless of consensus.
- escalate: true whenever recommendation is "escalate" OR when the decision
  meets any of the company's always-escalate criteria.
- reasoning: One to three sentences. Why THIS recommendation and not another.
"""


class CEOAgent:
    """
    The CEO agent. Does not inherit from BaseAgent because its methods,
    prompt structure, and response schema differ significantly.
    """

    role  = "ceo"
    title = "Chief Executive Officer"

    def __init__(self, company_config: dict):
        self.config  = company_config
        self.company = company_config.get("company_name", "the company")
        self.llm     = build_llm(
            company_config,
            temperature = 0.6,   # slightly lower than agents — CEO is more deliberate
            max_tokens  = 3072,  # CEO synthesis is longer than agent outputs
        )
        self.system_prompt = self._build_system_prompt(company_config)

    # ── Public methods ────────────────────────────────────────────────────────

    def synthesize(
        self,
        task:           str,
        agent_outputs:  list[dict],
        memories:       list[dict],
        is_final_round: bool = False,
    ) -> dict:
        """
        Reads all agent outputs and produces a synthesis.

        On is_final_round=True, the CEO knows this is the last chance to
        resolve internally before escalating to the human. The prompt changes
        to reflect that urgency and to force a clear position even under
        uncertainty.
        """
        formatted_outputs = self._format_all_outputs(agent_outputs)
        memory_context    = self._format_memories(memories)
        escalation_rules  = self._format_escalation_rules()

        round_instruction = ""
        if is_final_round:
            round_instruction = (
                "\n\nIMPORTANT: This is the final synthesis round. The C-suite has "
                "already debated once. If genuine conflict remains, you must either "
                "make a clear executive decision and own it, or formally escalate to "
                "the human owner. You cannot send this back for another round. Choose."
            )

        prompt = (
            f"{self.system_prompt}\n\n"
            f"--- CURRENT TASK ---\n{task}\n\n"
            f"--- RELEVANT COMPANY HISTORY ---\n{memory_context}\n\n"
            f"--- ESCALATION RULES ---\n{escalation_rules}\n\n"
            f"--- FULL C-SUITE DELIBERATION ---\n\n{formatted_outputs}"
            f"{round_instruction}\n\n"
            f"--- RESPONSE FORMAT ---\n{CEO_RESPONSE_SCHEMA}"
        )

        raw    = self._call_with_retry(prompt)
        parsed = self._parse_ceo_response(raw)

        # Apply hard escalation rules regardless of CEO's synthesized preference
        parsed = self._apply_escalation_rules(parsed, task)

        return parsed

    def format_presentation(self, state: dict) -> str:
        """
        Assembles the full deliberation into a clean, readable report
        for display to the human. This is what you see.

        Structure:
          1. Task summary
          2. Each agent's round-1 analysis (named, with recommendation badge)
          3. Cross-response highlights (where agents directly addressed each other)
          4. CEO synthesis and recommendation
          5. If escalating: the specific options with trade-offs laid out
        """
        task         = state.get("current_task", "")
        outputs      = state.get("agent_outputs", [])
        synthesis    = state.get("ceo_synthesis", "")
        conflicts    = state.get("conflicts_identified", [])
        consensus    = state.get("consensus_reached", False)
        escalate     = state.get("escalate_to_human", False)
        round_num    = state.get("debate_round", 1)
        memories     = state.get("relevant_memories", [])

        lines = []

        # ── Header ──────────────────────────────────────────────────────────
        lines.append("=" * 70)
        lines.append(f"  C-SUITE DELIBERATION REPORT")
        lines.append(f"  {self.company}")
        lines.append("=" * 70)
        lines.append(f"\nTASK: {task}\n")

        # ── Relevant history ─────────────────────────────────────────────────
        if memories:
            lines.append("─" * 70)
            lines.append("RELEVANT PAST DECISIONS")
            lines.append("─" * 70)
            for m in memories:
                lines.append(f"  • {m.get('task', '')}")
                lines.append(f"    → {m.get('outcome', '')}")
            lines.append("")

        # ── Round 1 outputs ──────────────────────────────────────────────────
        round1 = [o for o in outputs if "_response" not in o.get("agent", "")]
        if round1:
            lines.append("─" * 70)
            lines.append("ROUND 1 — INDEPENDENT ANALYSIS")
            lines.append("─" * 70)
            for output in round1:
                lines.extend(self._format_agent_block(output))
                lines.append("")

        # ── Cross-responses ──────────────────────────────────────────────────
        cross = [o for o in outputs if "_response" in o.get("agent", "")]
        if cross:
            lines.append("─" * 70)
            lines.append("CROSS-RESPONSE ROUND — PEER DEBATE")
            lines.append("─" * 70)
            for output in cross:
                lines.extend(self._format_agent_block(output))
                lines.append("")

        # ── Second round if it happened ──────────────────────────────────────
        if round_num > 2:
            lines.append("─" * 70)
            lines.append("NOTE: A second deliberation round was conducted.")
            lines.append("─" * 70)
            lines.append("")

        # ── CEO synthesis ────────────────────────────────────────────────────
        lines.append("─" * 70)
        lines.append("CEO SYNTHESIS")
        lines.append("─" * 70)
        lines.append(synthesis)
        lines.append("")

        # ── Conflicts (if any) ───────────────────────────────────────────────
        if conflicts:
            lines.append("─" * 70)
            lines.append("UNRESOLVED CONFLICTS")
            lines.append("─" * 70)
            for conflict in conflicts:
                lines.append(f"  • {conflict}")
            lines.append("")

        # ── Decision block ───────────────────────────────────────────────────
        lines.append("─" * 70)
        if escalate:
            lines.append("ESCALATED TO HUMAN OWNER — YOUR DECISION REQUIRED")
            lines.append("─" * 70)
            lines.append("")
            lines.append("The C-suite has presented its analysis. This decision")
            lines.append("requires your direct call. Please respond with:")
            lines.append("")
            lines.append("  • Your decision (proceed / block / modify)")
            lines.append("  • Brief reasoning (this becomes institutional memory)")
            lines.append("  • Any instructions for the team going forward")
        else:
            decisions = state.get("decisions_made", [])
            rec = decisions[-1].get("outcome", "") if decisions else ""
            lines.append(f"CEO RECOMMENDATION: {rec.upper() if rec else 'SEE SYNTHESIS'}")
            lines.append("─" * 70)
            lines.append("")
            lines.append("The CEO recommends the above. Please respond with:")
            lines.append("  • Approve — I accept this recommendation")
            lines.append("  • Override — followed by your decision and reasoning")
            lines.append("  • More info — followed by what you need clarified")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ── Internal: LLM call ────────────────────────────────────────────────────

    def _call_with_retry(self, prompt: str) -> str:
        last_raw = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = invoke_llm(self.llm, prompt)
                # Quick sanity check — if it parses, return immediately
                self._parse_ceo_response(raw)
                return raw
            except (ValueError, KeyError):
                last_raw = raw if 'raw' in dir() else ""
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        return last_raw

    # ── Internal: parsing ─────────────────────────────────────────────────────

    def _parse_ceo_response(self, raw: str) -> dict:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in CEO response: {cleaned[:200]}")

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"CEO JSON decode error: {e}")

        valid_recs = {"proceed", "block", "modify", "escalate"}
        rec = str(data.get("recommendation", "escalate")).strip().lower()
        if rec not in valid_recs:
            rec = "escalate"

        return {
            "synthesis":      str(data.get("synthesis", "")).strip(),
            "consensus":      bool(data.get("consensus", False)),
            "conflicts":      [str(c) for c in data.get("conflicts", []) if c],
            "recommendation": rec,
            "escalate":       bool(data.get("escalate", rec == "escalate")),
            "reasoning":      str(data.get("reasoning", "")).strip(),
        }

    # ── Internal: escalation rule enforcement ─────────────────────────────────

    def _apply_escalation_rules(self, parsed: dict, task: str) -> dict:
        """
        Checks always-escalate rules from company DNA.
        If any trigger, forces escalation regardless of CEO synthesis.
        """
        always_escalate = (
            self.config
                .get("escalation_rules", {})
                .get("always_escalate", [])
        )
        task_lower = task.lower()
        for rule in always_escalate:
            if any(word.lower() in task_lower for word in rule.split()):
                parsed["escalate"]       = True
                parsed["recommendation"] = "escalate"
                if not any("escalation rule" in c.lower() for c in parsed["conflicts"]):
                    parsed["conflicts"].append(
                        f"Escalation rule triggered: '{rule}' — "
                        f"this decision requires human owner approval regardless of consensus."
                    )
                break
        return parsed

    # ── Internal: formatting helpers ──────────────────────────────────────────

    def _build_system_prompt(self, config: dict) -> str:
        company      = config.get("company_name", "the company")
        industry     = config.get("industry", "our industry")
        stage        = config.get("stage", "current stage")
        mission      = config.get("mission", "Not specified")
        priorities   = config.get("strategic_priorities", [])
        constraints  = config.get("constraints", [])
        risk_profile = config.get("risk_profile", "moderate")
        dec_style    = config.get("decision_style", "balanced")
        from core.config import load_agent_prompt
        personality  = load_agent_prompt(
            config.get("company_id", ""), "ceo", config
        )

        priorities_text = (
            "\n".join(f"  - {p}" for p in priorities)
            if priorities else "  Not specified"
        )
        constraints_text = (
            "\n".join(f"  - {c}" for c in constraints)
            if constraints else "  Not specified"
        )

        return f"""You are the Chief Executive Officer of {company}, a {stage}-stage
company in {industry}.

PERSONALITY & BEHAVIORAL STYLE:
{personality}

YOUR ROLE IN THIS SYSTEM:
You do not participate in the deliberation rounds. You read the outputs of
your C-suite and synthesize them into a decision. You are the arbitration
layer — your job is to find the path through competing positions that best
serves the company's mission and constraints.

You are also the sole interface to the human owner. Every decision either
passes through you or is escalated by you. The human never reads raw agent
outputs — they read your synthesis of them.

COMPANY CONTEXT:
Company: {company}
Mission: {mission}
Industry: {industry}
Stage: {stage}
Risk profile: {risk_profile}
Decision style: {dec_style}

Strategic priorities:
{priorities_text}

Binding constraints:
{constraints_text}

SYNTHESIS PRINCIPLES:
1. Name the disagreements specifically. "The CFO and CMO disagree" is useful.
   "There is some tension" is not.
2. Distinguish between substantive conflicts and positioning differences.
   Executives often phrase the same concern differently — don't treat
   that as conflict requiring resolution.
3. When you find consensus, articulate the shared reasoning clearly.
   The human should understand WHY the team agrees, not just that they do.
4. When you escalate, present the options cleanly with their trade-offs.
   Do not present a preferred option disguised as objective information.
   If you have a lean, state it explicitly as YOUR lean, not as fact.
5. Own your decisions. If you resolve a conflict, take responsibility for
   that call. Do not hide behind "the team decided."
6. Company memory matters. Past decisions set precedent. If you are departing
   from precedent, say so and explain why this situation is different."""

    def _format_all_outputs(self, outputs: list[dict]) -> str:
        """Groups and formats all agent outputs for the CEO synthesis prompt."""
        round1 = [o for o in outputs if "_response" not in o.get("agent", "")]
        cross  = [o for o in outputs if "_response" in o.get("agent", "")]

        sections = []
        if round1:
            sections.append("ROUND 1 — INDEPENDENT ANALYSIS\n")
            for o in round1:
                sections.append(self._agent_output_to_text(o))

        if cross:
            sections.append("CROSS-RESPONSE ROUND\n")
            for o in cross:
                sections.append(self._agent_output_to_text(o))

        return "\n".join(sections)

    @staticmethod
    def _agent_output_to_text(output: dict) -> str:
        agent  = output.get("agent", "unknown").upper().replace("_RESPONSE", " (cross-response)")
        rec    = output.get("recommendation", "?").upper()
        conf   = output.get("confidence", 0.0)
        analysis = output.get("analysis", "")
        concerns = output.get("concerns", [])
        concern_text = (
            "\n".join(f"  • {c}" for c in concerns) if concerns else "  None"
        )
        return (
            f"{agent} — {rec} (confidence: {conf:.0%})\n"
            f"{analysis}\n"
            f"Concerns:\n{concern_text}\n"
        )

    @staticmethod
    def _format_memories(memories: list[dict]) -> str:
        if not memories:
            return "No relevant past decisions found."
        lines = []
        for m in memories:
            task    = m.get("task", "")
            outcome = m.get("outcome", "")
            reason  = m.get("reasoning", "")
            override = m.get("human_override", "")
            line = f"• Task: {task}\n  Outcome: {outcome}"
            if reason:
                line += f"\n  Reasoning: {reason}"
            if override:
                line += f"\n  Human override: {override}"
            lines.append(line)
        return "\n\n".join(lines)

    def _format_escalation_rules(self) -> str:
        rules = self.config.get("escalation_rules", {})
        always   = rules.get("always_escalate", [])
        can_solo = rules.get("ceo_can_decide_alone", [])

        lines = []
        if always:
            lines.append("Always escalate to human owner:")
            lines.extend(f"  • {r}" for r in always)
        if can_solo:
            lines.append("CEO may decide without escalation:")
            lines.extend(f"  • {r}" for r in can_solo)
        if rules.get("escalate_if_deadlock"):
            lines.append("Always escalate if C-suite reaches deadlock after round 2.")
        return "\n".join(lines) if lines else "No specific escalation rules configured."

    @staticmethod
    def _format_agent_block(output: dict) -> list[str]:
        """Formats a single agent output block for the human-facing report."""
        raw_agent = output.get("agent", "unknown")
        is_cross  = "_response" in raw_agent
        agent     = raw_agent.upper().replace("_RESPONSE", "")
        label     = f"{agent} (cross-response)" if is_cross else agent

        rec    = output.get("recommendation", "?").upper()
        conf   = output.get("confidence", 0.0)
        analysis = output.get("analysis", "")
        concerns = output.get("concerns", [])

        lines = [f"[ {label} · {rec} · {conf:.0%} confidence ]"]
        lines.append(analysis)
        if concerns:
            lines.append("Concerns raised:")
            for c in concerns:
                lines.append(f"  • {c}")
        return lines
