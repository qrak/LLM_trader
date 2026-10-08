"""Peak/off-peak billing windows for models whose per-token rates vary by hour.

Per-token rates stay in ``config/model_pricing.json`` and are treated as the BASE
(peak) rates. This module only decides which billing window a moment falls into and
how that window scales those base rates, so a provider that bills off-peak cheaper is
reported at the price actually charged. The mapping is editable through the optional
``config/peak_rates.json`` (see ``config/peak_rates.example.json``); without that file
the built-in defaults below apply, and a provider nobody configured is billed flat.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

DAY_ORDER = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

FLAT_ENTRY: dict[str, Any] = {
    "peak_multiplier": 1.0,
    "off_peak_multiplier": 1.0,
    "peak_windows_utc": [],
}

DEFAULT_PEAK_RATES: dict[str, Any] = {
    "_default": FLAT_ENTRY,
    "deepseek": {
        "models": ["deepseek-flash", "deepseek-v4-flash-vision-exp", "deepseek-v4-pro"],
        "peak_multiplier": 1.0,
        "off_peak_multiplier": 0.5,
        "peak_windows_utc": [
            {"days": ["mon-fri"], "start_utc": "01:00", "end_utc": "04:00"},
            {"days": ["mon-fri"], "start_utc": "06:00", "end_utc": "10:00"},
        ],
    },
}


class PeakRates:
    """Peak/off-peak rules resolved per provider and model."""

    def __init__(self, file_path: str | None = None):
        """Load the rules file, falling back to the built-in defaults when it is absent or broken."""
        self.file_path = file_path or self.default_path()
        self._rates = self._load()

    @staticmethod
    def default_path() -> str:
        """Return the path of the optional user-editable rules file."""
        return os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "peak_rates.json")
        )

    def multiplier(self, provider: str, model: str, at: datetime | None = None) -> float:
        """Return the multiplier to apply to the base rates at the given moment."""
        entry = self.resolve(provider, model)
        if not entry["peak_windows_utc"]:
            return 1.0
        chosen = entry["peak_multiplier"] if self._is_peak(entry, at) else entry["off_peak_multiplier"]
        return float(chosen)

    def tier_label(self, provider: str, model: str, at: datetime | None = None) -> str | None:
        """Return the billing window applied to a model, or None when the model is flat-rated."""
        entry = self.resolve(provider, model)
        if not entry["peak_windows_utc"]:
            return None
        if self._is_peak(entry, at):
            return f"peak x{entry['peak_multiplier']:g}"
        return f"off-peak x{entry['off_peak_multiplier']:g}"

    def resolve(self, provider: str, model: str) -> dict[str, Any]:
        """Return the normalized rule entry that applies to a provider/model pair."""
        entry = self._rates.get(self._normalize(provider))
        if isinstance(entry, dict) and self._covers(entry, model):
            return self._normalize_entry(entry)
        fallback = self._rates.get("_default")
        return self._normalize_entry(fallback if isinstance(fallback, dict) else {})

    def _load(self) -> dict[str, Any]:
        """Return the configured rules merged over the built-in defaults, provider by provider."""
        merged = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in DEFAULT_PEAK_RATES.items()
        }
        for provider, entry in self._read_file().items():
            if isinstance(entry, dict) and isinstance(merged.get(provider), dict):
                merged[provider] = {**merged[provider], **entry}
            else:
                merged[provider] = entry
        return merged

    def _read_file(self) -> dict[str, Any]:
        """Read the rules file; an absent, unreadable or malformed file is not an error."""
        try:
            with open(self.file_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _is_peak(self, entry: dict[str, Any], at: datetime | None) -> bool:
        """Return True when the moment falls inside any of the entry's peak windows."""
        moment = self._as_utc(at)
        for window in entry["peak_windows_utc"]:
            if self._window_matches(window, moment):
                return True
        return False

    def _window_matches(self, window: Any, moment: datetime) -> bool:
        """Return True when a window's days and UTC clock range contain the moment."""
        if not isinstance(window, dict):
            return False
        if moment.weekday() not in self._days(window.get("days")):
            return False
        start = self._minutes_of_day(window.get("start_utc"))
        end = self._minutes_of_day(window.get("end_utc"))
        if start is None or end is None:
            return False
        current = moment.hour * 60 + moment.minute
        return start <= current < end

    def _covers(self, entry: dict[str, Any], model: str) -> bool:
        """Return True when an entry applies to the model (no ``models`` list means every model)."""
        models = entry.get("models")
        if not models:
            return True
        if not isinstance(models, (list, tuple)):
            models = [models]
        target = self._normalize(model)
        return any(self._same_model(target, self._normalize(str(item))) for item in models)

    def _days(self, spec: Any) -> tuple[int, ...]:
        """Expand a day spec such as ``["mon-fri"]`` into weekday indexes."""
        tokens = spec if isinstance(spec, (list, tuple)) else [spec] if spec else []
        indexes: list[int] = []
        for token in tokens:
            if not isinstance(token, str):
                continue
            for index in self._expand_day(token):
                if index not in indexes:
                    indexes.append(index)
        return tuple(indexes)

    @staticmethod
    def _expand_day(token: str) -> list[int]:
        """Expand one token, either a single day name or an inclusive range like ``mon-fri``."""
        value = token.strip().lower()
        if "-" in value:
            start_name, _, end_name = value.partition("-")
            return PeakRates._day_span(start_name.strip(), end_name.strip())
        index = PeakRates._day_index(value)
        return [] if index is None else [index]

    @staticmethod
    def _day_span(start_name: str, end_name: str) -> list[int]:
        """Return every weekday index between two names, wrapping when the range crosses Sunday."""
        start = PeakRates._day_index(start_name)
        end = PeakRates._day_index(end_name)
        if start is None or end is None:
            return []
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, len(DAY_ORDER))) + list(range(end + 1))

    @staticmethod
    def _day_index(name: str) -> int | None:
        """Return the weekday index for a three-letter day name, or None when unknown."""
        return DAY_ORDER.index(name) if name in DAY_ORDER else None

    @staticmethod
    def _minutes_of_day(value: Any) -> int | None:
        """Parse a ``HH:MM`` UTC clock value into minutes since midnight."""
        if not isinstance(value, str) or ":" not in value:
            return None
        hours, _, minutes = value.partition(":")
        try:
            return int(hours.strip()) * 60 + int(minutes.strip())
        except ValueError:
            return None

    @staticmethod
    def _as_utc(at: datetime | None) -> datetime:
        """Return the moment in UTC; a naive datetime is read as UTC."""
        moment = at if at is not None else datetime.now(timezone.utc)
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    @staticmethod
    def _normalize(model: str) -> str:
        """Normalize a provider or model name for lookup."""
        return str(model).strip().lower().replace("models/", "")

    @staticmethod
    def _same_model(target: str, candidate: str) -> bool:
        """Match a configured model name exactly or as a substring of the reported name."""
        return bool(candidate) and (candidate == target or candidate in target or target in candidate)

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
        """Fill every key of a rule entry so consumers never read a missing field."""
        windows = entry.get("peak_windows_utc")
        return {
            "peak_multiplier": PeakRates._multiplier_value(entry.get("peak_multiplier"), 1.0),
            "off_peak_multiplier": PeakRates._multiplier_value(entry.get("off_peak_multiplier"), 1.0),
            "peak_windows_utc": list(windows) if isinstance(windows, (list, tuple)) else [],
        }

    @staticmethod
    def _multiplier_value(value: Any, default: float) -> float:
        """Return a non-negative multiplier, or the default when the value is unusable."""
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default
