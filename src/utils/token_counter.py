
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from datetime import datetime
from typing import Any

import tiktoken

from src.trading.data_models import ProviderCostStats, SessionCosts
from src.utils.data_utils import atomic_write_json
from src.utils.peak_rates import PeakRates


class ModelPricing:
    """Loads model pricing from config/model_pricing.json and applies the active billing window."""

    def __init__(self, peak_rates: PeakRates | None = None):
        self._pricing = self._load_pricing()
        self.peak_rates = peak_rates if peak_rates is not None else PeakRates()

    def _load_pricing(self) -> dict[str, Any]:

        pricing_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "model_pricing.json")
        pricing_path = os.path.normpath(pricing_path)
        try:
            with open(pricing_path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"google": {}, "openrouter": {}}

    def get_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int,
                 at: datetime | None = None) -> float | None:

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
        return (input_cost + output_cost) * self.peak_rates.multiplier(provider, model, at)

    def cost_note(self, provider: str, model: str, at: datetime | None = None) -> str | None:

        return self.peak_rates.tier_label(provider, model, at)

    def _normalize_model_key(self, model: str) -> str:

        return model.lower().replace("models/", "")


class TokenCounter:
    """Handles token counting and tracking for AI model usage with cost support."""

    def __init__(self, encoding_name: str = "cl100k_base"):

        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.session_tokens = {
            "prompt": 0,
            "completion": 0,
            "system": 0,
            "total": 0
        }
        self.session_costs = SessionCosts()

    def count_tokens(self, text: str) -> int:

        if not text:
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def track_prompt_tokens(self, text: str, message_type: str = "prompt") -> int:

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
            elif provider == "deepseek":
                self.session_costs.deepseek += cost

    @staticmethod
    def format_cost(cost: float) -> str:

        if cost == 0 or cost is None:
            return "Free"
        if cost < 0.0001:
            return f"${cost:.8f}"
        if cost < 0.01:
            return f"${cost:.6f}"
        if cost < 1:
            return f"${cost:.4f}"
        return f"${cost:.2f}"

    def process_response_usage(
        self,
        usage: dict[str, Any] | None,
        provider: str = "unknown",
        logger=None,
        fallback_text: str | None = None,
        cost_note: str | None = None
    ) -> None:

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
                    window = f" ({cost_note})" if cost_note else ""
                    logger.info("Request cost: %s%s", self.format_cost(cost), window)
        elif fallback_text:
            response_tokens = self.track_prompt_tokens(fallback_text, "completion")
            if logger:
                logger.info("Response token count (estimated): %s", response_tokens)
                stats = self.get_usage_stats()
                logger.info("Total tokens used: %s", f"{stats['total']:,}")

    def get_usage_stats(self) -> dict[str, int]:

        return self.session_tokens.copy()

    def reset_session_stats(self) -> None:

        self.session_tokens = {
            "prompt": 0,
            "completion": 0,
            "system": 0,
            "total": 0
        }
        self.session_costs = SessionCosts()


class CostStorage:
    """Persistent storage for API costs loaded from/saved to JSON file."""
    PROVIDERS = ("openrouter", "google", "lmstudio", "deepseek")

    def __init__(self, file_path: str = "data/trading/api_costs.json"):

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

        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_or_create(self) -> None:

        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, encoding="utf-8") as f:
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
            except (OSError, json.JSONDecodeError):
                pass
        self._init_defaults()

    def _init_defaults(self) -> None:

        for provider in self.PROVIDERS:
            self._providers[provider] = ProviderCostStats()
        self._last_reset = None

    def save(self) -> None:

        with self._lock:
            if not self._dirty:
                return

            data = {"last_reset": self._last_reset}
            for provider, stats in self._providers.items():
                data[provider] = stats.to_dict()  # type: ignore[reportAttributeAccessIssue]

            try:
                atomic_write_json(self.file_path, data, fsync=True)
                self._last_save_time = time.time()
                self._dirty = False
            except Exception as e:  # noqa: BLE001
                print(f"Error saving cost storage: {e}")

    def record_usage(
        self,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float | None = None
    ) -> None:

        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = ProviderCostStats()
            stats = self._providers[provider]
            stats.total_input_tokens += prompt_tokens
            stats.total_output_tokens += completion_tokens
            if cost is not None:
                stats.total_cost += cost

            self._dirty = True

            if time.time() - self._last_save_time >= 5.0:
                self.save()

    def get_provider_costs(self, provider: str) -> ProviderCostStats:

        return self._providers.get(provider, ProviderCostStats())

    def snapshot(self) -> dict[str, float]:

        with self._lock:
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            for stats in self._providers.values():
                total_cost += stats.total_cost
                total_input_tokens += stats.total_input_tokens
                total_output_tokens += stats.total_output_tokens
            return {
                "cost_usd": float(total_cost),
                "input_tokens": float(total_input_tokens),
                "output_tokens": float(total_output_tokens),
            }

    def total_cost(self) -> float:

        return self.snapshot()["cost_usd"]

