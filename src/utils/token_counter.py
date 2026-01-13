"""Token counting and cost tracking for AI model usage."""
from typing import Dict, Optional, Any

import tiktoken


class TokenCounter:
    """Handles token counting and tracking for AI model usage with cost support."""
    PROVIDER_COST_MESSAGES = {
        "openrouter": "Cost: ${cost:.4f}",
        "google": "Pricing not calculated for Gemini models using Google AI servers",
        "lmstudio": "Free - local model",
    }

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the TokenCounter.

        Args:
            encoding_name: The tokenizer encoding name to use (default: cl100k_base for OpenAI models)
        """
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.session_tokens = {
            "prompt": 0,
            "completion": 0,
            "system": 0,
            "total": 0
        }
        self.session_costs: Dict[str, float] = {
            "openrouter": 0.0,
            "google": 0.0,
            "lmstudio": 0.0,
        }
        self.request_usage: Optional[Dict[str, Any]] = None

    def count_tokens(self, text: str) -> int:
        """
        Count the tokens in the provided text.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def track_prompt_tokens(self, text: str, message_type: str = "prompt") -> int:
        """
        Count tokens in prompt text and track them in session stats.

        Args:
            text: The prompt text
            message_type: Type of message ("prompt", "system", or "completion")

        Returns:
            Number of tokens
        """
        token_count = self.count_tokens(text)
        if message_type in self.session_tokens:
            self.session_tokens[message_type] += token_count
        else:
            self.session_tokens[message_type] = token_count
        self.session_tokens["total"] += token_count
        return token_count

    def record_api_usage(
        self,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: Optional[float] = None
    ) -> None:
        """
        Record actual token usage from API response (replaces tiktoken estimates).

        Args:
            provider: Provider name (openrouter, google, lmstudio)
            prompt_tokens: Actual prompt token count from API
            completion_tokens: Actual completion token count from API
            cost: Optional cost from API (only OpenRouter provides this)
        """
        self.request_usage = {
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
        }
        self.session_tokens["prompt"] += prompt_tokens
        self.session_tokens["completion"] += completion_tokens
        self.session_tokens["total"] += prompt_tokens + completion_tokens
        if cost and provider in self.session_costs:
            self.session_costs[provider] += cost

    @staticmethod
    def format_cost(cost: float) -> str:
        """Format cost in human-readable format."""
        if cost == 0 or cost is None:
            return "Free"
        elif cost < 0.0001:
            return f"${cost:.8f} ({cost * 100:.6f}¢)"
        elif cost < 0.01:
            return f"${cost:.6f} ({cost * 100:.4f}¢)"
        elif cost < 1:
            return f"${cost:.4f}"
        else:
            return f"${cost:.2f}"

    def process_response_usage(
        self,
        usage: Optional[Dict[str, Any]],
        provider: str = "unknown",
        logger=None,
        fallback_text: Optional[str] = None
    ) -> None:
        """
        Process API response usage data: record and optionally log.

        Args:
            usage: Usage dict from API response (with prompt_tokens, completion_tokens, cost)
            provider: Provider name for cost tracking
            logger: Optional logger instance to output token counts
            fallback_text: If usage is None, estimate tokens from this text
        """
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            total_tokens = prompt_tokens + completion_tokens
            cost = usage.get("cost")
            self.record_api_usage(provider, prompt_tokens, completion_tokens, cost)
            if logger:
                logger.info(f"Prompt token count: {prompt_tokens:,}")
                logger.info(f"Response token count: {completion_tokens:,}")
                logger.info(f"Total tokens used: {total_tokens:,}")
                if cost is not None:
                    logger.info(f"Request cost: {self.format_cost(cost)}")
        elif fallback_text:
            response_tokens = self.track_prompt_tokens(fallback_text, "completion")
            if logger:
                logger.info(f"Response token count (estimated): {response_tokens}")
                stats = self.get_usage_stats()
                logger.info(f"Total tokens used: {stats['total']:,}")

    def record_cost(self, provider: str, cost: float) -> None:
        """
        Record cost for a provider (used when fetching from separate API call).

        Args:
            provider: Provider name
            cost: Cost in dollars
        """
        if provider in self.session_costs:
            self.session_costs[provider] += cost
        if self.request_usage and self.request_usage.get("provider") == provider:
            self.request_usage["cost"] = cost

    def get_last_request_usage(self) -> Optional[Dict[str, Any]]:
        """Get usage data from the last recorded API request."""
        return self.request_usage

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get current token usage statistics.

        Returns:
            Dictionary with token usage counts
        """
        return self.session_tokens.copy()

    def get_session_costs(self) -> Dict[str, float]:
        """
        Get cumulative costs per provider for the session.

        Returns:
            Dictionary with provider -> cost mapping
        """
        return self.session_costs.copy()

    def get_total_session_cost(self) -> float:
        """Get total cost across all providers."""
        return sum(self.session_costs.values())

    def format_usage_display(self, provider: str, prompt_tokens: int, completion_tokens: int, cost: Optional[float] = None) -> str:
        """
        Format token usage and cost for display.

        Args:
            provider: Provider name
            prompt_tokens: Input token count
            completion_tokens: Output token count
            cost: Optional cost (for OpenRouter)

        Returns:
            Formatted string for display
        """
        token_str = f"Tokens: {prompt_tokens:,} in / {completion_tokens:,} out"
        if provider == "openrouter" and cost is not None:
            cost_str = self.PROVIDER_COST_MESSAGES["openrouter"].format(cost=cost)
        elif provider in self.PROVIDER_COST_MESSAGES:
            cost_str = self.PROVIDER_COST_MESSAGES[provider]
        else:
            cost_str = "Cost: unknown provider"
        return f"{token_str} │ {cost_str}"

    def reset_session_stats(self) -> None:
        """Reset session token tracking statistics."""
        self.session_tokens = {
            "prompt": 0,
            "completion": 0,
            "system": 0,
            "total": 0
        }
        self.session_costs = {
            "openrouter": 0.0,
            "google": 0.0,
            "lmstudio": 0.0,
        }
        self.request_usage = None

    def reset_costs_only(self) -> None:
        """Reset only cost tracking (for dashboard reset button)."""
        self.session_costs = {
            "openrouter": 0.0,
            "google": 0.0,
            "lmstudio": 0.0,
        }


class CostStorage:
    """Persistent storage for API costs loaded from/saved to JSON file."""

    def __init__(self, file_path: str = "data/trading/api_costs.json"):
        """
        Initialize cost storage.

        Args:
            file_path: Path to the api_costs.json file
        """
        import os
        self.file_path = file_path
        self._ensure_directory()
        self._costs = self._load_or_create()

    def _ensure_directory(self) -> None:
        """Ensure the directory exists."""
        import os
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_or_create(self) -> Dict[str, Any]:
        """Load existing costs or create default structure."""
        import json
        import os
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._default_costs()

    def _default_costs(self) -> Dict[str, Any]:
        """Return default cost structure."""
        return {
            "openrouter": {"total_cost": 0.0, "total_input_tokens": 0, "total_output_tokens": 0},
            "google": {"total_input_tokens": 0, "total_output_tokens": 0},
            "lmstudio": {"total_input_tokens": 0, "total_output_tokens": 0},
            "last_reset": None
        }

    def save(self) -> None:
        """Save current costs to file."""
        import json
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self._costs, f, indent=2)

    def record_usage(
        self,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: Optional[float] = None
    ) -> None:
        """
        Record usage for a provider and save to disk.

        Args:
            provider: Provider name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            cost: Cost in dollars (only for OpenRouter)
        """
        if provider not in self._costs:
            self._costs[provider] = {"total_input_tokens": 0, "total_output_tokens": 0}
        self._costs[provider]["total_input_tokens"] += prompt_tokens
        self._costs[provider]["total_output_tokens"] += completion_tokens
        if cost is not None and "total_cost" in self._costs[provider]:
            self._costs[provider]["total_cost"] += cost
        self.save()

    def get_costs(self) -> Dict[str, Any]:
        """Get all stored costs."""
        return self._costs.copy()

    def get_provider_costs(self, provider: str) -> Dict[str, Any]:
        """Get costs for a specific provider."""
        return self._costs.get(provider, {}).copy()

    def get_total_openrouter_cost(self) -> float:
        """Get total OpenRouter cost."""
        return self._costs.get("openrouter", {}).get("total_cost", 0.0)

    def reset(self) -> None:
        """Reset all costs and update last_reset timestamp."""
        from datetime import datetime
        self._costs = self._default_costs()
        self._costs["last_reset"] = datetime.utcnow().isoformat() + "Z"
        self.save()
