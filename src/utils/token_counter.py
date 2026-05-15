"""Token counting and cost tracking for AI model usage."""
from __future__ import annotations

import json
import os
import threading
import time
import atexit
import tempfile
from typing import Any

import tiktoken

from src.trading.data_models import ProviderCostStats, SessionCosts

class ModelPricing:
    """Loads and provides model pricing from config/model_pricing.json."""
    _instance: "ModelPricing | None" = None
    _pricing: dict[str, Any] | None = None

    def __new__(cls) -> "ModelPricing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ModelPricing._pricing is None:
            ModelPricing._pricing = self._load_pricing()

    def _load_pricing(self) -> dict[str, Any]:
        """Load pricing data from JSON file."""
        pricing_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "model_pricing.json")
        pricing_path = os.path.normpath(pricing_path)
        try:
            with open(pricing_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"google": {}, "openrouter": {}}

    def get_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """
        Calculate cost for a request based on token counts.

        Args:
            provider: Provider name (google, openrouter)
            model: Model name/identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD or None if pricing not available
        """
        provider_pricing = self._pricing.get(provider, {})
        model_key = self._normalize_model_key(model)
        model_pricing = provider_pricing.get(model_key)
        if not model_pricing:
            for key in provider_pricing:
                if key.startswith("_"):
                    continue
                if model_key in key or key in model_key:
                    model_pricing = provider_pricing[key]
                    break
        if not model_pricing:
            return None
        input_cost = (input_tokens / 1_000_000) * model_pricing.get("input_per_million", 0)
        output_cost = (output_tokens / 1_000_000) * model_pricing.get("output_per_million", 0)
        return input_cost + output_cost

    def _normalize_model_key(self, model: str) -> str:
        """Normalize model name for lookup."""
        return model.lower().replace("models/", "")


class TokenCounter:
    """Handles token counting and tracking for AI model usage with cost support."""

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
        self.session_costs = SessionCosts()

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
        cost: float | None = None
    ) -> None:
        """
        Record actual token usage from API response (replaces tiktoken estimates).

        Args:
            provider: Provider name (openrouter, google, lmstudio)
            prompt_tokens: Actual prompt token count from API
            completion_tokens: Actual completion token count from API
            cost: Optional cost from API (only OpenRouter provides this)
        """
        self.session_tokens["prompt"] += prompt_tokens
        self.session_tokens["completion"] += completion_tokens
        self.session_tokens["total"] += prompt_tokens + completion_tokens
        if cost:
            if provider == "openrouter":
                self.session_costs.openrouter += cost
            elif provider == "google":
                self.session_costs.google += cost
            elif provider == "lmstudio":
                self.session_costs.lmstudio += cost

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
        usage: dict[str, Any] | None,
        provider: str = "unknown",
        logger=None,
        fallback_text: str | None = None
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
                logger.info("Prompt token count: %s", f"{prompt_tokens:,}")
                logger.info("Response token count: %s", f"{completion_tokens:,}")
                logger.info("Total tokens used: %s", f"{total_tokens:,}")
                if cost is not None:
                    logger.info("Request cost: %s", self.format_cost(cost))
        elif fallback_text:
            response_tokens = self.track_prompt_tokens(fallback_text, "completion")
            if logger:
                logger.info("Response token count (estimated): %s", response_tokens)
                stats = self.get_usage_stats()
                logger.info("Total tokens used: %s", f"{stats['total']:,}")

    def get_usage_stats(self) -> dict[str, int]:
        """
        Get current token usage statistics.

        Returns:
            Dictionary with token usage counts
        """
        return self.session_tokens.copy()

    def reset_session_stats(self) -> None:
        """Reset session token tracking statistics."""
        self.session_tokens = {
            "prompt": 0,
            "completion": 0,
            "system": 0,
            "total": 0
        }
        self.session_costs = SessionCosts()


class CostStorage:
    """Persistent storage for API costs loaded from/saved to JSON file."""
    PROVIDERS = ("openrouter", "google", "lmstudio")

    def __init__(self, file_path: str = "data/trading/api_costs.json"):
        """
        Initialize cost storage.

        Args:
            file_path: Path to the api_costs.json file
        """
        self.file_path = file_path
        self._ensure_directory()
        self._providers: dict[str, ProviderCostStats] = {}
        self._last_reset: str | None = None

        self._lock = threading.RLock()
        self._last_save_time = 0.0
        self._dirty = False
        atexit.register(self.save)

        self._load_or_create()

    def _ensure_directory(self) -> None:
        """Ensure the directory exists."""
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_or_create(self) -> None:
        """Load existing costs or create default structure."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._last_reset = data.get("last_reset")
                for provider in self.PROVIDERS:
                    if provider in data:
                        p_data = data[provider]
                        self._providers[provider] = ProviderCostStats(
                            total_cost=p_data.get("total_cost", 0.0),
                            total_input_tokens=p_data.get("total_input_tokens", 0),
                            total_output_tokens=p_data.get("total_output_tokens", 0)
                        )
                    else:
                        self._providers[provider] = ProviderCostStats()
                return
            except (json.JSONDecodeError, IOError):
                pass
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Initialize default provider stats."""
        for provider in self.PROVIDERS:
            self._providers[provider] = ProviderCostStats()
        self._last_reset = None

    def save(self) -> None:
        """Save current costs to file."""
        with self._lock:
            if not self._dirty:
                return

            data = {"last_reset": self._last_reset}
            for provider, stats in self._providers.items():
                data[provider] = stats.to_dict()

            # Atomic write using temporary file
            temp_path = None
            try:
                directory = os.path.dirname(self.file_path)
                with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', dir=directory) as f:
                    temp_path = f.name
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(temp_path, self.file_path)
                self._last_save_time = time.time()
                self._dirty = False
            except Exception as e:
                # Fallback to direct print if logging is not available here
                print(f"Error saving cost storage: {e}")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    def record_usage(
        self,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float | None = None
    ) -> None:
        """
        Record usage for a provider and save to disk (buffered).

        Args:
            provider: Provider name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            cost: Cost in dollars (for OpenRouter and Google)
        """
        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = ProviderCostStats()
            stats = self._providers[provider]
            stats.total_input_tokens += prompt_tokens
            stats.total_output_tokens += completion_tokens
            if cost is not None:
                stats.total_cost += cost

            self._dirty = True

            # Save if enough time has passed
            if time.time() - self._last_save_time >= 5.0:
                self.save()

    def get_provider_costs(self, provider: str) -> ProviderCostStats:
        """Get costs for a specific provider."""
        return self._providers.get(provider, ProviderCostStats())

