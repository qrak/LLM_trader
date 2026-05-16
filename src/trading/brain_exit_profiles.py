"""Exit profile resolution helpers for trading brain rules and context."""

from collections import Counter
from typing import Any

from src.utils.indicator_classifier import build_exit_execution_context, format_exit_execution_context

from .data_models import ExitExecutionContext


class ExitProfileResolver:
    """Resolve, format, and normalize SL/TP execution profiles."""

    UNKNOWN_EXIT_PROFILE = "SL unknown/unknown | TP unknown/unknown"
    UNKNOWN_EXIT_PROFILE_KEY = "sl_unknown_unknown|tp_unknown_unknown"

    def __init__(self, default_exit_execution_context: ExitExecutionContext):
        """Initialize the resolver with the configured fallback execution context."""
        self.default_exit_execution_context = default_exit_execution_context

    @staticmethod
    def safe_key_part(value: Any, default: str = "unknown") -> str:
        """Normalize metadata values for deterministic semantic rule IDs."""
        if value is None:
            return default
        normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        return normalized or default

    def sanitize_rule_key(self, value: str) -> str:
        """Create a stable Chroma ID suffix from a composite rule key."""
        return self.safe_key_part(value.replace("|", "_"))

    def resolve_exit_execution_context(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> ExitExecutionContext:
        """Return SL/TP execution metadata with configured defaults filled in."""
        raw = metadata or {}
        resolved = build_exit_execution_context(
            stop_loss_type=raw.get("stop_loss_type"),
            stop_loss_check_interval=raw.get("stop_loss_check_interval"),
            take_profit_type=raw.get("take_profit_type"),
            take_profit_check_interval=raw.get("take_profit_check_interval"),
        )
        return resolved.with_defaults(self.default_exit_execution_context)

    def resolve_rule_exit_execution_context(self, metadata: dict[str, Any]) -> ExitExecutionContext:
        """Resolve exit execution context from semantic-rule metadata."""
        return self.resolve_exit_execution_context({
            "stop_loss_type": metadata.get("dominant_stop_loss_type") or metadata.get("stop_loss_type"),
            "stop_loss_check_interval": (
                metadata.get("dominant_stop_loss_interval") or metadata.get("stop_loss_check_interval")
            ),
            "take_profit_type": metadata.get("dominant_take_profit_type") or metadata.get("take_profit_type"),
            "take_profit_check_interval": (
                metadata.get("dominant_take_profit_interval") or metadata.get("take_profit_check_interval")
            ),
        })

    def format_exit_profile_from_context(self, context: ExitExecutionContext) -> str:
        """Format a normalized SL/TP execution context."""
        return self.format_exit_profile(
            context.stop_loss_type,
            context.stop_loss_check_interval,
            context.take_profit_type,
            context.take_profit_check_interval,
        )

    @staticmethod
    def format_exit_profile(
        stop_type: str,
        stop_interval: str,
        take_profit_type: str,
        take_profit_interval: str,
    ) -> str:
        """Format a hard/soft SL/TP profile for rules and dashboard metadata."""
        profile = format_exit_execution_context(
            ExitExecutionContext(
                stop_loss_type=stop_type,
                stop_loss_check_interval=stop_interval,
                take_profit_type=take_profit_type,
                take_profit_check_interval=take_profit_interval,
            ),
            include_unknown=True,
        )
        return profile.removeprefix("Exit Execution: ")

    def replace_unknown_exit_profile_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Replace legacy unknown exit-profile text with resolved rule/default metadata."""
        if self.UNKNOWN_EXIT_PROFILE not in text:
            return text
        if metadata:
            replacement_profile = self.format_exit_profile_from_context(
                self.resolve_rule_exit_execution_context(metadata)
            )
        else:
            replacement_profile = self.format_exit_profile_from_context(self.default_exit_execution_context)
        if replacement_profile == self.UNKNOWN_EXIT_PROFILE:
            return text
        return text.replace(self.UNKNOWN_EXIT_PROFILE, replacement_profile)

    def render_rule_text(self, rule: dict[str, Any]) -> str:
        """Return rule text with legacy unknown exit profile corrected for display."""
        metadata = rule.get("metadata", {})
        return self.replace_unknown_exit_profile_text(rule.get("text", ""), metadata)

    def dominant_exit_execution_context(self, metas: list[dict[str, Any]]) -> ExitExecutionContext:
        """Return the most common resolved SL/TP execution context for a trade group."""
        contexts = [self.resolve_exit_execution_context(meta) for meta in metas]
        if not contexts:
            return self.resolve_exit_execution_context({})

        def most_common(values: list[str]) -> str:
            return Counter(values).most_common(1)[0][0]

        return ExitExecutionContext(
            stop_loss_type=most_common([context.stop_loss_type for context in contexts]),
            stop_loss_check_interval=most_common([context.stop_loss_check_interval for context in contexts]),
            take_profit_type=most_common([context.take_profit_type for context in contexts]),
            take_profit_check_interval=most_common([context.take_profit_check_interval for context in contexts]),
        )

    def legacy_unknown_exit_pattern_key(self, pattern_key: str) -> str:
        """Return the equivalent rule key that used the old unknown exit profile."""
        parts = pattern_key.split("|")
        if len(parts) < 2:
            return pattern_key
        parts[-2:] = self.UNKNOWN_EXIT_PROFILE_KEY.split("|")
        return "|".join(parts)

    def build_exit_profile_key(self, meta: dict[str, Any]) -> str:
        """Build a deterministic key for hard/soft SL/TP execution settings."""
        context = self.resolve_exit_execution_context(meta)
        stop_type = self.safe_key_part(context.stop_loss_type)
        stop_interval = self.safe_key_part(context.stop_loss_check_interval)
        take_profit_type = self.safe_key_part(context.take_profit_type)
        take_profit_interval = self.safe_key_part(context.take_profit_check_interval)
        return f"sl_{stop_type}_{stop_interval}|tp_{take_profit_type}_{take_profit_interval}"