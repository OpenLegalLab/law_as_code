"""Custom DSPy LM provider backed by the Claude Agent SDK.

The Agent SDK drives the local ``claude`` CLI as a subprocess and reuses its
credentials, so no API key is needed.  This adapter constrains it to a single
tool-less turn so it behaves as a plain text completion engine, which is what
the DSPy signatures in :mod:`case_classifier.classifier` expect.

Mirrors the structure of :mod:`case_classifier.openjustice_lm` — only the
transport differs.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import dspy

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

_DEFAULT_MODEL = "claude-opus-5"


# ---------------------------------------------------------------------------
# Lightweight containers that mimic the OpenAI response shape expected by
# ``dspy.BaseLM._process_lm_response``.
# ---------------------------------------------------------------------------

@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __iter__(self):
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


@dataclass
class _Response:
    choices: list[_Choice]
    model: str = ""
    usage: _Usage = field(default_factory=_Usage)


def _run_sync(make_coro: Callable[[], Awaitable[str]]) -> str:
    """Run *make_coro* to completion from synchronous code.

    ``forward`` is normally reached through ``asyncio.to_thread`` (see
    ``oll_api_server.pipeline._run_classification``), so the calling thread has
    no running loop and ``asyncio.run`` is safe.  The ThreadPoolExecutor branch
    only matters if the LM is ever driven straight from an event loop thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coro())).result()


# ---------------------------------------------------------------------------
# The custom LM class
# ---------------------------------------------------------------------------

class ClaudeAgentLM(dspy.BaseLM):
    """DSPy language-model adapter for the Claude Agent SDK."""

    def __init__(
        self,
        model: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model or os.environ.get("CLAUDE_MODEL") or _DEFAULT_MODEL,
            model_type="chat",
            **kwargs,
        )
        # Run the CLI outside the repo so no project settings or CLAUDE.md are
        # in scope even if a future SDK default re-enables setting discovery.
        self.cwd = cwd or os.environ.get("CLAUDE_AGENT_CWD") or "/tmp"

    # -- core interface -----------------------------------------------------

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _Response:
        text = self._messages_to_prompt(prompt, messages)
        full_text = _run_sync(lambda: self._complete(text))
        return _Response(
            choices=[_Choice(message=_Message(content=full_text))],
            model=self.model,
        )

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _Response:
        text = self._messages_to_prompt(prompt, messages)
        full_text = await self._complete(text)
        return _Response(
            choices=[_Choice(message=_Message(content=full_text))],
            model=self.model,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _messages_to_prompt(
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> str:
        """Flatten DSPy messages into a single prompt string."""
        if prompt:
            return prompt
        if not messages:
            return ""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def _options(self) -> ClaudeAgentOptions:
        """Constrain the agent to one tool-less completion turn.

        ``system_prompt=None`` keeps Claude Code's coding-agent persona out of
        the prompt, and ``setting_sources=[]`` stops the SDK from loading
        ``~/.claude`` or this repo's own CLAUDE.md/settings into a legal
        classification request.
        """
        return ClaudeAgentOptions(
            system_prompt=None,
            model=self.model,
            max_turns=1,
            tools=[],
            allowed_tools=[],
            permission_mode="dontAsk",
            setting_sources=[],
            cwd=self.cwd,
        )

    async def _complete(self, prompt_text: str) -> str:
        """Drive one Agent SDK turn and collect the assistant text."""
        chunks: list[str] = []

        async for message in query(prompt=prompt_text, options=self._options()):
            if isinstance(message, AssistantMessage):
                chunks.extend(
                    block.text
                    for block in message.content
                    if isinstance(block, TextBlock)
                )
            elif isinstance(message, ResultMessage):
                if message.subtype not in ("success", None):
                    raise RuntimeError(
                        f"Claude Agent SDK error (model={self.model}): "
                        f"{message.subtype} — {message.result}"
                    )

        return "".join(chunks)
