"""
core/agents/cra.py

Research Agent (CRA) — a worker that performs analysis and research
after C-suite approval: competitive analysis, market research, pricing
studies, technology evaluations, audience analysis, etc.

Non-interactive. Uses the company's configured LLM to synthesize
research based on the task briefing, company context, and the model's
training knowledge.

Future enhancement: integrate web search tools for live data.
"""

from core.agents.base_worker import BaseWorker
from core.agents.base import build_llm, invoke_llm


class CRAAgent(BaseWorker):

    role        = "cra"
    title       = "Research Agent"
    interactive = False
    keywords    = [
        "research", "analyze", "analysis", "compare", "competitive",
        "market", "pricing", "evaluate", "study", "investigate",
        "benchmark", "survey", "report", "assess", "review",
    ]

    def __init__(self, company_config: dict):
        super().__init__(company_config)
        self.llm = build_llm(company_config, temperature=0.4, max_tokens=4096)
        self.config = company_config

    def build_prompt(self, task: str) -> str:
        company_name = self.config.get("company_name", "the company")
        industry = self.config.get("industry", "")
        mission = self.config.get("mission", "")
        priorities = self.config.get("strategic_priorities", [])
        constraints = self.config.get("constraints", [])

        priorities_text = "\n".join(f"- {p}" for p in priorities) if priorities else "None specified"
        constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else "None specified"

        return (
            f"You are a research analyst working for {company_name}, "
            f"a company in {industry}.\n\n"
            f"Company mission: {mission}\n"
            f"Strategic priorities:\n{priorities_text}\n"
            f"Constraints:\n{constraints_text}\n\n"
            f"--- RESEARCH TASK ---\n\n"
            f"{task}\n\n"
            f"--- INSTRUCTIONS ---\n\n"
            f"Produce a structured research report. Include:\n"
            f"1. **Executive Summary** — key findings in 2-3 sentences\n"
            f"2. **Findings** — detailed analysis organized by topic\n"
            f"3. **Recommendations** — actionable next steps tied to findings\n"
            f"4. **Risks & Unknowns** — what you couldn't determine or what needs validation\n\n"
            f"Be specific and factual. Clearly distinguish between established "
            f"facts, reasonable inferences, and speculation. If you don't have "
            f"enough information on a point, say so rather than guessing."
        )

    def execute(self, task: str) -> dict:
        try:
            findings = invoke_llm(self.llm, self.build_prompt(task))
            return {
                "worker":  self.role,
                "success": True,
                "summary": f"Research report generated ({len(findings)} chars)",
                "output":  findings,
            }
        except Exception as e:
            return {
                "worker":  self.role,
                "success": False,
                "summary": f"Research failed: {e}",
                "output":  "",
            }
