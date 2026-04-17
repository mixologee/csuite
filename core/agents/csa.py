"""
core/agents/csa.py

Communications / Social Agent (CSA) — a worker that drafts
communications and social media content after C-suite approval:
social media posts, Discord announcements, community updates,
email campaigns, player outreach, etc.

Non-interactive. Uses the company's configured LLM to generate
platform-appropriate content based on the task briefing and brand voice.
"""

from core.agents.base_worker import BaseWorker
from core.agents.base import build_llm, invoke_llm


class CSAAgent(BaseWorker):

    role        = "csa"
    title       = "Communications Agent"
    interactive = False
    keywords    = [
        "social media", "social post", "tweet", "discord",
        "community update", "email campaign", "outreach",
        "promote on", "share on", "post on", "post to",
        "discord announcement", "newsletter email",
    ]

    def __init__(self, company_config: dict):
        super().__init__(company_config)
        self.llm = build_llm(company_config, temperature=0.8, max_tokens=3072)
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
            f"You are a communications specialist for {company_name}, "
            f"a company in {industry}.\n\n"
            f"Company mission: {mission}\n\n"
            f"Brand voice: {personality}\n\n"
            f"--- TASK ---\n\n"
            f"{task}\n\n"
            f"--- INSTRUCTIONS ---\n\n"
            f"Produce the requested communications content. For each piece:\n"
            f"1. **Platform** — which platform/channel this is for\n"
            f"2. **Content** — the actual post/message, ready to send\n"
            f"3. **Timing** — when to post (if relevant)\n"
            f"4. **Hashtags / Tags** — if applicable\n\n"
            f"Match the tone and format to each platform. A Discord announcement "
            f"reads differently from a tweet. Keep the brand voice consistent "
            f"across all pieces. If the task asks for a campaign, produce "
            f"a series of posts with a coherent narrative arc.\n\n"
            f"Output the content directly — ready to copy and post."
        )

        try:
            content = invoke_llm(self.llm, prompt)
            return {
                "worker":  self.role,
                "success": True,
                "summary": f"Communications drafted ({len(content)} chars)",
                "output":  content,
            }
        except Exception as e:
            return {
                "worker":  self.role,
                "success": False,
                "summary": f"Communications generation failed: {e}",
                "output":  "",
            }
