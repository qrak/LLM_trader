
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

        self.file_path = file_path or self.default_path()
        self._rates = self._load()

    @staticmethod
    def default_path() -> str:

        return os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "peak_rates.json")
        )

    def multiplier(self, provider: str, model: str, at: datetime | None = None) -> float:

        entry = self.resolve(provider, model)
        if not entry["peak_windows_utc"]:
            return 1.0
        chosen = entry["peak_multiplier"] if self._is_peak(entry, at) else entry["off_peak_multiplier"]
        return float(chosen)

    def tier_label(self, provider: str, model: str, at: datetime | None = None) -> str | None:

        entry = self.resolve(provider, model)
        if not entry["peak_windows_utc"]:
            return None
        if self._is_peak(entry, at):
            return f"peak x{entry['peak_multiplier']:g}"
        return f"off-peak x{entry['off_peak_multiplier']:g}"

    def resolve(self, provider: str, model: str) -> dict[str, Any]:

        entry = self._rates.get(self._normalize(provider))
        if isinstance(entry, dict) and self._covers(entry, model):
            return self._normalize_entry(entry)
        fallback = self._rates.get("_default")
        return self._normalize_entry(fallback if isinstance(fallback, dict) else {})

    def _load(self) -> dict[str, Any]:

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

        try:
            with open(self.file_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _is_peak(self, entry: dict[str, Any], at: datetime | None) -> bool:

        moment = self._as_utc(at)
        for window in entry["peak_windows_utc"]:
            if self._window_matches(window, moment):
                return True
        return False

    def _window_matches(self, window: Any, moment: datetime) -> bool:

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

        models = entry.get("models")
        if not models:
            return True
        if not isinstance(models, (list, tuple)):
            models = [models]
        target = self._normalize(model)
        return any(self._same_model(target, self._normalize(str(item))) for item in models)

    def _days(self, spec: Any) -> tuple[int, ...]:

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

        value = token.strip().lower()
        if "-" in value:
            start_name, _, end_name = value.partition("-")
            return PeakRates._day_span(start_name.strip(), end_name.strip())
        index = PeakRates._day_index(value)
        return [] if index is None else [index]

    @staticmethod
    def _day_span(start_name: str, end_name: str) -> list[int]:

        start = PeakRates._day_index(start_name)
        end = PeakRates._day_index(end_name)
        if start is None or end is None:
            return []
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, len(DAY_ORDER))) + list(range(end + 1))

    @staticmethod
    def _day_index(name: str) -> int | None:

        return DAY_ORDER.index(name) if name in DAY_ORDER else None

    @staticmethod
    def _minutes_of_day(value: Any) -> int | None:

        if not isinstance(value, str) or ":" not in value:
            return None
        hours, _, minutes = value.partition(":")
        try:
            return int(hours.strip()) * 60 + int(minutes.strip())
        except ValueError:
            return None

    @staticmethod
    def _as_utc(at: datetime | None) -> datetime:

        moment = at if at is not None else datetime.now(timezone.utc)
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    @staticmethod
    def _normalize(model: str) -> str:

        return str(model).strip().lower().replace("models/", "")

    @staticmethod
    def _same_model(target: str, candidate: str) -> bool:

        return bool(candidate) and (candidate == target or candidate in target or target in candidate)

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:

        windows = entry.get("peak_windows_utc")
        return {
            "peak_multiplier": PeakRates._multiplier_value(entry.get("peak_multiplier"), 1.0),
            "off_peak_multiplier": PeakRates._multiplier_value(entry.get("off_peak_multiplier"), 1.0),
            "peak_windows_utc": list(windows) if isinstance(windows, (list, tuple)) else [],
        }

    @staticmethod
    def _multiplier_value(value: Any, default: float) -> float:

        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default
