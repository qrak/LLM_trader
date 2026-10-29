

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

    name = os.environ.get("TZ", "").strip()
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def resolve_timezone(name: str | None) -> tzinfo:

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

    try:
        from src.config.loader import config

        return str(config.TIMEZONE or "").strip()
    except Exception:  # noqa: BLE001
        return ""


@lru_cache(maxsize=8)
def _cached_zone(name: str) -> tzinfo:
    return resolve_timezone(name or None)


def local_timezone() -> tzinfo:

    return _cached_zone(configured_timezone_name())


def reset_timezone_cache() -> None:

    _cached_zone.cache_clear()


def as_utc(moment: datetime) -> datetime:

    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def format_offset(moment: datetime) -> str:

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

    utc_moment = as_utc(moment)
    local_moment = utc_moment.astimezone(tz or local_timezone())
    return (
        f"{utc_moment.strftime(ISO_DATETIME_FORMAT)} UTC "
        f"/ {local_moment.strftime(ISO_DATETIME_FORMAT)} {local_moment.tzname() or 'local'}"
    )


def local_stamp(moment: datetime) -> str:

    local_moment = as_utc(moment).astimezone(local_timezone())
    return f"{local_moment.strftime(DISPLAY_DATETIME_FORMAT)} {format_offset(local_moment)}"


def day_stamp(moment: datetime | None = None) -> str:

    instant = moment or datetime.now(timezone.utc)
    return as_utc(instant).astimezone(local_timezone()).strftime(DISPLAY_DAY_FORMAT)


def utc_stamp(moment: datetime | None) -> str:

    if moment is None:
        return "unknown"
    return as_utc(moment).strftime(UTC_STAMP_FORMAT)
