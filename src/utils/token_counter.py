"""Token counting and cost tracking for AI model usage."""
import json
import os
import threading
import time
import atexit
import tempfile
from typing import Dict, Optional, Any

import tiktoken

from src.utils.dataclasses import ProviderCostStats, SessionCosts, TokenUsageStats


class ModelPricing:
    """Loads and provides model pricing from config/model_pricing.json."""
    _instance: Optional["ModelPricing"] = None
    _pricing: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ModelPricing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ModelPricing._pricing is None:
            ModelPricing._pricing = self._load_pricing()

    def _load_pricing(self) -> Dict[str, Any]:
        """Load pricing data from JSON file."""
        pricing_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "model_pricing.json")
        pricing_path = os.path.normpath(pricing_path)
        try:
            with open(pricing_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"google": {}, "openrouter": {}}

    def get_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
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
    PROVIDER_COST_MESSAGES = {
        "openrouter": "Cost: ${cost:.4f}",
        "google": "Cost: ${cost:.4f} (estimated)",
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
        self.session_costs = SessionCosts()
        self.request_usage: Optional[TokenUsageStats] = None

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
        self.request_usage = TokenUsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost
        )
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
        if provider == "openrouter":
            self.session_costs.openrouter += cost
        elif provider == "google":
            self.session_costs.google += cost
        elif provider == "lmstudio":
            self.session_costs.lmstudio += cost
        if self.request_usage:
            self.request_usage.cost = cost

    def get_last_request_usage(self) -> Optional[TokenUsageStats]:
        """Get usage data from the last recorded API request."""
        return self.request_usage

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get current token usage statistics.

        Returns:
            Dictionary with token usage counts
        """
        return self.session_tokens.copy()

    def get_session_costs(self) -> SessionCosts:
        """
        Get cumulative costs per provider for the session.

        Returns:
            SessionCosts dataclass with provider costs
        """
        return self.session_costs

    def get_total_session_cost(self) -> float:
        """Get total cost across all providers."""
        return self.session_costs.total

    def format_usage_display(self, provider: str, prompt_tokens: int, completion_tokens: int, cost: Optional[float] = None) -> str:
        """
        Format token usage and cost for display.

        Args:
            provider: Provider name
            prompt_tokens: Input token count
            completion_tokens: Output token count
            cost: Optional cost (for OpenRouter or Google)

        Returns:
            Formatted string for display
        """
        token_str = f"Tokens: {prompt_tokens:,} in / {completion_tokens:,} out"
        if cost is not None and provider in ("openrouter", "google"):
            cost_str = self.PROVIDER_COST_MESSAGES[provider].format(cost=cost)
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
        self.session_costs = SessionCosts()
        self.request_usage = None

    def reset_costs_only(self) -> None:
        """Reset only cost tracking (for dashboard reset button)."""
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
        self._providers: Dict[str, ProviderCostStats] = {}
        self._last_reset: Optional[str] = None

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
        cost: Optional[float] = None
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

    def get_costs(self) -> Dict[str, Any]:
        """Get all stored costs as dict (for backward compatibility)."""
        result: Dict[str, Any] = {"last_reset": self._last_reset}
        for provider, stats in self._providers.items():
            result[provider] = stats.to_dict()
        return result

    def get_provider_costs(self, provider: str) -> ProviderCostStats:
        """Get costs for a specific provider."""
        return self._providers.get(provider, ProviderCostStats())

    def get_total_openrouter_cost(self) -> float:
        """Get total OpenRouter cost."""
        return self._providers.get("openrouter", ProviderCostStats()).total_cost

    def reset(self) -> None:
        """Reset all costs and update last_reset timestamp."""
        from datetime import datetime, timezone
        self._init_defaults()
        self._last_reset = datetime.now(timezone.utc).isoformat()
        self.save()
