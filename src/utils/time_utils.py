"""One place that decides which clock a timestamp is shown on.

Trading arithmetic and persisted data stay in UTC. Only text a human reads
(log lines, notifications, operator messages, chart labels) is rendered in the
configured display zone, and it always carries an explicit UTC offset so a DST
change can never make two distinct moments look identical.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, tzinfo
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DISPLAY_DATETIME_FORMAT = "%d.%m.%Y %H:%M:%S"
DISPLAY_DAY_FORMAT = "%Y_%m_%d"
ISO_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
UTC_STAMP_FORMAT = "%Y-%m-%d %H:%M:%S UTC"


def system_timezone() -> tzinfo:
    """Best-effort OS zone used when ``[general] timezone`` is not configured.

    ``TZ`` (the usual Linux/WSL setting) is exact. Otherwise the platform only
    reports the offset that applies *now*, which is right for current timestamps
    but not for instants on the other side of a DST change — set the config key
    to get DST-correct labels on Windows.
    """
    name = os.environ.get("TZ", "").strip()
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def resolve_timezone(name: str | None) -> tzinfo:
    """Return the zone for an IANA name; empty means the OS zone."""
    if not name:
        return system_timezone()
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Invalid timezone {name!r} in [general] timezone — expected an IANA name "
            f"such as 'Europe/Warsaw'"
        ) from exc


def configured_timezone_name() -> str:
    """Return the configured display zone name, or '' when unset or unavailable."""
    try:
        from src.config.loader import config

        return str(config.TIMEZONE or "").strip()
    except Exception:  # noqa: BLE001
        return ""


@lru_cache(maxsize=8)
def _cached_zone(name: str) -> tzinfo:
    return resolve_timezone(name or None)


def local_timezone() -> tzinfo:
    """Display zone for operator-facing text (cached per configured name)."""
    return _cached_zone(configured_timezone_name())


def reset_timezone_cache() -> None:
    """Forget the cached display zone after the configuration changed."""
    _cached_zone.cache_clear()


def as_utc(moment: datetime) -> datetime:
    """Interpret a naive timestamp as UTC and return an aware UTC datetime."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def format_offset(moment: datetime) -> str:
    """Render the UTC offset of an aware datetime as ``+HH:MM``."""
    offset = moment.utcoffset()
    if offset is None:
        return "+00:00"
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    hours, remainder = divmod(abs(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if seconds:
        return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{sign}{hours:02d}:{minutes:02d}"


def utc_and_local(moment: datetime, tz: tzinfo | None = None) -> str:
    """Render one instant as ``<UTC> UTC / <display zone> <abbreviation>``."""
    utc_moment = as_utc(moment)
    local_moment = utc_moment.astimezone(tz or local_timezone())
    return (
        f"{utc_moment.strftime(ISO_DATETIME_FORMAT)} UTC "
        f"/ {local_moment.strftime(ISO_DATETIME_FORMAT)} {local_moment.tzname() or 'local'}"
    )


def local_stamp(moment: datetime) -> str:
    """Render an instant for a log line: display zone plus explicit offset."""
    local_moment = as_utc(moment).astimezone(local_timezone())
    return f"{local_moment.strftime(DISPLAY_DATETIME_FORMAT)} {format_offset(local_moment)}"


def day_stamp(moment: datetime | None = None) -> str:
    """Folder name of the display day an instant belongs to."""
    instant = moment or datetime.now(timezone.utc)
    return as_utc(instant).astimezone(local_timezone()).strftime(DISPLAY_DAY_FORMAT)


def utc_stamp(moment: datetime | None) -> str:
    """Render an instant in UTC for messages that must never be ambiguous."""
    if moment is None:
        return "unknown"
    return as_utc(moment).strftime(UTC_STAMP_FORMAT)
