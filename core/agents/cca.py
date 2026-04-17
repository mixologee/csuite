"""
core/agents/cca.py

Claude Code Agent (CCA) — an interactive worker that executes implementation
tasks via the Claude Code Python SDK.

Supports multi-turn conversations: the user can review CCA's work and
send follow-up instructions within the same session.

Required config.json field:
    codebase_path — absolute path to the codebase this company manages.
"""

from pathlib import Path

from core.agents.base_worker import BaseWorker


class CCAAgent(BaseWorker):

    role        = "cca"
    title       = "Claude Code Agent"
    interactive = True
    keywords    = [
        "code", "codebase", "build app", "build page", "develop app",
        "deploy", "write code", "create file", "configure server",
        "install", "migrate", "refactor", "fix bug", "patch",
        "repository", "frontend", "backend", "api", "database",
        "script", "website", "landing page", "web page",
    ]

    def __init__(self, company_config: dict):
        super().__init__(company_config)
        self.model = company_config.get("model_name", "gpt-oss:20b")

        codebase_path = company_config.get("codebase_path", "")
        if not codebase_path:
            raise ValueError(
                f"CCAAgent requires 'codebase_path' in company config for "
                f"'{self.company}'. Set it to the absolute path of the "
                f"codebase this company manages."
            )
        self.codebase_path = Path(codebase_path)
        if not self.codebase_path.is_dir():
            raise ValueError(
                f"codebase_path '{codebase_path}' does not exist or is not "
                f"a directory."
            )

    def execute(self, task: str) -> dict:
        """Sync fallback for non-interactive contexts (CLI runner)."""
        import asyncio
        messages, _ = asyncio.run(self.start_session(task))
        result_msg = next(
            (m for m in reversed(messages) if m["type"] == "result"), None
        )
        return {
            "worker":        self.role,
            "success":       not result_msg["is_error"] if result_msg else False,
            "summary":       result_msg["content"][:2000] if result_msg else "",
            "files_changed": [m.get("file", "") for m in messages
                              if m["type"] == "tool_use" and m.get("file")],
            "output":        "\n".join(m["content"] for m in messages
                                       if m.get("content")),
        }

    def _build_options(self, resume: str | None = None):
        from claude_code_sdk import ClaudeCodeOptions
        opts = ClaudeCodeOptions(
            model=self.model,
            cwd=str(self.codebase_path),
            permission_mode="acceptEdits",
            max_turns=25,
            env={
                "ANTHROPIC_AUTH_TOKEN": "ollama",
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_BASE_URL": "http://localhost:11434",
                "OLLAMA_CONTEXT_LENGTH": "65536",
            },
        )
        if resume:
            opts.resume = resume
        return opts

    async def start_session(self, task: str,
                             on_message=None) -> tuple[list[dict], str]:
        """
        Start a new CCA session.

        Args:
            task: The implementation instruction.
            on_message: Optional async callback(msg_dict) called as each
                        message arrives, for real-time UI streaming.

        Returns (all_messages, session_id).
        """
        return await self._run_query(task, self._build_options(),
                                      on_message=on_message)

    async def continue_session(self, session_id: str, user_input: str,
                                on_message=None) -> tuple[list[dict], str]:
        """
        Continue an existing CCA session with user follow-up.
        Returns (all_messages, session_id).
        """
        return await self._run_query(
            user_input, self._build_options(resume=session_id),
            on_message=on_message,
        )

    @staticmethod
    async def _run_query(prompt, options, on_message=None):
        """
        Run a Claude Code SDK query. Yields parsed message dicts to
        on_message (if provided) as they arrive, and also collects
        them into a list returned at the end.
        """
        from claude_code_sdk import (
            query, ResultMessage, AssistantMessage,
            TextBlock, ToolUseBlock,
        )

        messages = []
        session_id = ""

        try:
            async for msg in query(prompt=prompt, options=options):
                parsed = None

                if isinstance(msg, ResultMessage):
                    session_id = msg.session_id or session_id
                    parsed = {
                        "type":     "result",
                        "content":  msg.result or "",
                        "is_error": msg.is_error,
                    }
                elif isinstance(msg, AssistantMessage):
                    if not hasattr(msg, "content"):
                        continue
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            parsed = {
                                "type":    "text",
                                "content": block.text,
                            }
                        elif isinstance(block, ToolUseBlock):
                            name = block.name if hasattr(block, "name") else "tool"
                            inp = block.input if hasattr(block, "input") else {}
                            file_path = ""
                            if isinstance(inp, dict):
                                file_path = inp.get("file_path",
                                                    inp.get("path", ""))
                            parsed = {
                                "type":    "tool_use",
                                "tool":    name,
                                "file":    file_path,
                                "content": f"Using {name}"
                                           + (f" on `{file_path}`"
                                              if file_path else ""),
                            }

                        if parsed:
                            messages.append(parsed)
                            if on_message:
                                await on_message(parsed)
                            parsed = None
                            continue

                if parsed:
                    messages.append(parsed)
                    if on_message:
                        await on_message(parsed)

        except Exception as e:
            # Ollama backend may not include all fields the SDK expects
            # (e.g. 'signature'). If we already have messages, treat the
            # work done so far as the result rather than crashing.
            # Suppress the async generator cleanup noise on stderr.
            import sys
            _orig_unraisable = sys.unraisablehook
            def _suppress_generator_exit(unraisable):
                if (unraisable.exc_type is RuntimeError
                        and "GeneratorExit" in str(unraisable.exc_value)):
                    return
                _orig_unraisable(unraisable)
            sys.unraisablehook = _suppress_generator_exit

            err_str = str(e)
            if messages:
                parsed = {
                    "type":    "result",
                    "content": f"Session ended early: {err_str}",
                    "is_error": False,
                }
            else:
                parsed = {
                    "type":    "result",
                    "content": f"CCA failed: {err_str}",
                    "is_error": True,
                }
            messages.append(parsed)
            if on_message:
                await on_message(parsed)

        return messages, session_id
