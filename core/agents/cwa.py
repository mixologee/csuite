"""
core/agents/cwa.py

Content Writer Agent (CWA) — a worker that drafts written content
after C-suite approval: blog posts, game descriptions, social media
copy, newsletter content, press releases, documentation, etc.

Non-interactive. Uses the company's configured LLM to generate content
based on the task briefing and company DNA.
"""

from core.agents.base_worker import BaseWorker
from core.agents.base import build_llm, invoke_llm


class CWAAgent(BaseWorker):

    role        = "cwa"
    title       = "Content Writer Agent"
    interactive = False
    keywords    = [
        "write", "draft", "blog", "post", "copy", "content",
        "newsletter", "press release", "description", "article",
        "announcement", "documentation", "readme", "guide",
    ]

    def __init__(self, company_config: dict):
        super().__init__(company_config)
        self.llm = build_llm(company_config, temperature=0.8, max_tokens=4096)
        self.config = company_config

    def execute(self, task: str) -> dict:
        company_name = self.config.get("company_name", "the company")
        industry = self.config.get("industry", "")
        mission = self.config.get("mission", "")
        personality = (
            self.config.get("agent_personalities", {})
                       .get("cmo", "")
        )

        prompt = (
            f"You are a professional content writer for {company_name}, "
            f"a company in {industry}.\n\n"
            f"Company mission: {mission}\n\n"
            f"Brand voice guidance: {personality}\n\n"
            f"--- TASK ---\n\n"
            f"{task}\n\n"
            f"--- INSTRUCTIONS ---\n\n"
            f"Write the requested content. Match the company's brand voice. "
            f"Be specific, engaging, and ready to publish with minimal editing. "
            f"If the task asks for multiple pieces (e.g. several social posts), "
            f"produce all of them clearly separated.\n\n"
            f"Output the content directly — no preamble or meta-commentary."
        )

        try:
            content = invoke_llm(self.llm, prompt)
            return {
                "worker":  self.role,
                "success": True,
                "summary": f"Content drafted ({len(content)} chars)",
                "output":  content,
            }
        except Exception as e:
            return {
                "worker":  self.role,
                "success": False,
                "summary": f"Content generation failed: {e}",
                "output":  "",
            }
