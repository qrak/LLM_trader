"""Estimated-token budgets for brain context and provider-bound requests."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.logger.logger import Logger
    from src.utils.token_counter import TokenCounter


class PromptBudgetExceeded(ValueError):
    """Protected context cannot fit the configured estimated-token budget."""


class PromptBudgetPolicy:
    """Keep complete protected sections while removing lower-priority evidence."""

    DEFAULT_CONTEXT_WINDOWS = {  # noqa: RUF012
        "deepseek/deepseek-flash": 1000000,
        "deepseek/deepseek-v4-pro": 1000000,
        "googleai/gemini-3.8-flash": 1048576,
        "googleai/gemini-3-flash-preview": 1048576,
        "openrouter/google/gemini-3-flash-preview": 1048576,
        "openrouter/deepseek/deepseek-v4.1-flash": 1048576,
    }
    BRAIN_NOTICE = "[Brain context reduced: estimated token budget reached; recent lessons and risk guidance preserved.]"
    OPTIONAL_SECTIONS = (
        ("## EXTERNAL MARKET CONTEXT (UNTRUSTED DATA)", "## TASK", "news"),
        ("## Recent Trading History", "## Overall Performance", "recent decisions"),
        ("## PREVIOUS ANALYSIS CONTEXT", "### DETERMINISTIC TIME CHECK", "previous analysis"),
        ("RELEVANT PAST EXPERIENCES", "### Learned Stats for This Context", "similar experiences"),
        ("### Learned Trading Rules", "### Trade Journal", "semantic rules"),
    )
    PROTECTED_HEADERS = (
        "## CURRENT POSITION", "## Current Position", "### Last Action Outcome",
        "## Capital Status", "## Response Format", "## Decision Rules",
        "## EXPECTED VALUE FRAMEWORK", "## Decision Protocol", "## Core Principles",
        "### Trade Journal", "## ACTIVE RISK PROFILE", "## TASK",
        "### PYTHON R/R FEEDBACK", "## SL Tightening Policy", "### SL Tightening Policy",
    )

    def __init__(
        self, token_counter: TokenCounter, logger: Logger, *,
        brain_max_tokens: int = 50000, request_max_tokens: int = 100000,
        context_windows: dict[str, int] | None = None,
        safety_margin_percent: int = 10, image_reserve_tokens: int = 16384,
    ) -> None:
        self.token_counter = token_counter
        self.logger = logger
        self.brain_max_tokens = brain_max_tokens
        self.request_max_tokens = request_max_tokens
        self.context_windows = dict(self.DEFAULT_CONTEXT_WINDOWS if context_windows is None else context_windows)
        self.safety_margin_percent = safety_margin_percent
        self.image_reserve_tokens = image_reserve_tokens

    def fit_brain(self, lines: list[str], optional_ranges: list[tuple[int, int]]) -> str:
        original = "\n".join(lines)
        original_tokens = self.token_counter.count_tokens(original)
        if original_tokens <= self.brain_max_tokens:
            return original
        excluded: set[int] = set()
        for begin, end in optional_ranges:
            excluded.update(range(begin, end))
            text = "\n".join(line for index, line in enumerate(lines) if index not in excluded)
            result = text + "\n\n" + self.BRAIN_NOTICE
            tokens = self.token_counter.count_tokens(result)
            if tokens <= self.brain_max_tokens:
                self.logger.warning(
                    "Brain context reduced: estimated tokens %d -> %d (limit=%d)",
                    original_tokens, tokens, self.brain_max_tokens,
                )
                return result
        raise PromptBudgetExceeded(
            f"Protected brain context exceeds {self.brain_max_tokens} estimated tokens; no lesson was cut"
        )

    def input_limit(self, provider: str, model: str, model_config: dict, *, chart: bool = False) -> int:
        key = f"{provider}/{model}"
        window = self.context_windows.get(key)
        if window is None:
            raise PromptBudgetExceeded(f"Context window is not configured for {key}; request not sent")
        output_tokens = model_config.get("max_tokens", model_config.get("max_output_tokens"))
        if type(output_tokens) is not int or output_tokens <= 0:
            raise PromptBudgetExceeded(f"Invalid output token reserve for {key}")
        margin = window * self.safety_margin_percent // 100
        image_tokens = self.image_reserve_tokens if chart else 0
        limit = min(self.request_max_tokens, window - output_tokens - margin - image_tokens - 1024)
        if limit <= 0:
            raise PromptBudgetExceeded(f"Output/image/safety reserves exhaust the context window for {key}")
        return limit

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return sum(self.token_counter.count_tokens(message["content"]) + 8 for message in messages)

    @classmethod
    def _remove_optional_section(cls, text: str, start: str, end: str, label: str) -> str:
        starts = list(re.finditer(r"^" + re.escape(start) + r"[^\n]*\n", text, re.MULTILINE))
        ends = list(re.finditer(r"^" + re.escape(end) + r"[^\n]*(?:\n|$)", text, re.MULTILINE))
        if len(starts) != 1 or len(ends) != 1 or ends[0].start() <= starts[0].start():
            return text
        begin, finish = starts[0].start(), ends[0].start()
        section = text[begin:finish]
        if any(re.search(r"^" + re.escape(header), section, re.MULTILINE) for header in cls.PROTECTED_HEADERS):
            return text
        return text[:begin] + f"[Prompt budget: optional {label} omitted.]\n\n" + text[finish:]

    def prepare_messages(
        self, messages: list[dict[str, str]], provider: str, model: str,
        model_config: dict, *, chart: bool = False,
    ) -> list[dict[str, str]]:
        limit = self.input_limit(provider, model, model_config, chart=chart)
        original_tokens = self.count_messages(messages)
        if original_tokens <= limit:
            return messages
        result = [dict(message) for message in messages]
        for start, end, label in self.OPTIONAL_SECTIONS:
            for message in result:
                message["content"] = self._remove_optional_section(message["content"], start, end, label)
            tokens = self.count_messages(result)
            if tokens <= limit:
                self.logger.warning(
                    "Prompt reduced for %s/%s: estimated tokens %d -> %d (input limit=%d)",
                    provider, model, original_tokens, tokens, limit,
                )
                return result
        raise PromptBudgetExceeded(
            f"Protected request for {provider}/{model} exceeds {limit} estimated input tokens; request not sent"
        )
