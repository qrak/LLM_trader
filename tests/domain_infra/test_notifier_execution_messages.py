

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord

from src.notifiers.notifier import (
    NOT_EXECUTED_PREFIX,
    DiscordNotifier,
    format_utc_stamp,
)
from tests.conftest import make_config, make_position


class DummyBot:
    """Bot double resolving a single channel."""

    def __init__(self) -> None:
        self.user = SimpleNamespace(name="DummyBot")

    def event(self, fn: Any) -> Any:
        return fn

    def get_channel(self, _channel_id: int) -> Any:
        return MagicMock()


def notifier() -> DiscordNotifier:

    unified_parser = MagicMock()
    unified_parser.extract_text_before_json.return_value = "Reasoning text"
    formatter = MagicMock()
    formatter.fmt.side_effect = lambda value: str(value)
    return DiscordNotifier(
        logger=MagicMock(),
        config=make_config(QUOTE_CURRENCY="USDC"),
        unified_parser=unified_parser,
        formatter=formatter,
        bot=DummyBot(),
        file_handler=MagicMock(),
    )


def sent_embed(subject: DiscordNotifier) -> discord.Embed:

    return subject._send_embed.await_args.args[0]


def field_value(embed: discord.Embed, name: str) -> str | None:
    for field in embed.fields:
        if field.name == name:
            return field.value
    return None


class TestRecommendationNotExecuted:
    """An unexecuted recommendation must not look like a performed action."""

    async def test_banner_message_and_embed_mark_the_recommendation(self) -> None:
        subject = notifier()
        subject.send_message = AsyncMock()
        subject._send_embed = AsyncMock()

        await subject.send_analysis_notification(
            result={"analysis": {"signal": "UPDATE", "confidence": 70, "reasoning": "Move the stop."}, "raw_response": "text"},
            symbol="BTC/USDC",
            timeframe="4h",
            channel_id=123,
            execution_note="Recommendation UPDATE NOT executed: the position is closed.",
        )

        banner = subject.send_message.await_args_list[0].kwargs["message"]
        assert NOT_EXECUTED_PREFIX in banner
        assert "the position is closed" in banner

        embed = sent_embed(subject)
        assert NOT_EXECUTED_PREFIX in embed.title
        assert "UPDATE" in embed.title
        assert NOT_EXECUTED_PREFIX in embed.description
        assert "the position is closed" in (field_value(embed, NOT_EXECUTED_PREFIX) or "")

    async def test_without_a_note_the_card_stays_a_plain_analysis(self) -> None:
        subject = notifier()
        subject.send_message = AsyncMock()
        subject._send_embed = AsyncMock()

        await subject.send_analysis_notification(
            result={"analysis": {"signal": "BUY", "confidence": 70, "reasoning": "Breakout."}, "raw_response": "text"},
            symbol="BTC/USDC",
            timeframe="4h",
            channel_id=123,
        )

        assert subject.send_message.await_count == 1
        assert NOT_EXECUTED_PREFIX not in sent_embed(subject).title

    def test_embed_builder_is_backwards_compatible(self) -> None:
        subject = notifier()
        plain = subject._create_analysis_embed({"signal": "UPDATE", "confidence": 70, "reasoning": "x"}, "BTC/USDC", "4h")
        noted = subject._create_analysis_embed(
            {"signal": "UPDATE", "confidence": 70, "reasoning": "x"}, "BTC/USDC", "4h", execution_note="not executed"
        )

        assert plain is not None and noted is not None
        assert plain.title == "📊 BTC/USDC - UPDATE"
        assert noted.title.startswith(NOT_EXECUTED_PREFIX)


class TestPositionStatusVerification:
    """The OPEN card states where its position evidence comes from."""

    async def test_exchange_verified_card_uses_the_clean_open_title(self) -> None:
        subject = notifier()
        subject._send_embed = AsyncMock()
        checked_at = datetime(2026, 9, 22, 17, 32, tzinfo=timezone.utc)

        await subject.send_position_status(
            position=make_position(symbol="BTC/USDC"),
            current_price=86100.0,
            channel_id=123,
            verification="exchange_verified",
            verified_at=checked_at,
        )

        embed = sent_embed(subject)
        assert embed.title == "📈 Open LONG Position - BTC/USDC"
        description = embed.description or ""
        assert "position and its protection are active" in description
        state = field_value(embed, "Position State") or ""
        assert state == f"✅ Exchange-verified {format_utc_stamp(checked_at)} — protection active"
        assert "executor-reported" not in embed.title
        assert "NOT exchange-verified" not in description

    async def test_executor_reported_card_never_claims_exchange_verification(self) -> None:

        subject = notifier()
        subject._send_embed = AsyncMock()
        checked_at = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)

        await subject.send_position_status(
            position=make_position(symbol="BTC/USDC"),
            current_price=81100.0,
            channel_id=123,
            verification="executor_reported",
            verified_at=checked_at,
        )

        embed = sent_embed(subject)
        assert "executor-reported" in embed.title
        assert "NOT exchange-verified" in embed.description
        state = field_value(embed, "Position State") or ""
        assert "executor-reported" in state
        assert "NOT exchange-verified" in state
        assert format_utc_stamp(checked_at) in state
        assert "✅" not in state
        assert "Exchange State" not in [f.name for f in embed.fields]

    async def test_legacy_verified_value_is_downgraded_to_unverified(self) -> None:

        subject = notifier()
        subject._send_embed = AsyncMock()

        await subject.send_position_status(
            position=make_position(), current_price=81100.0, channel_id=123,
            verification="verified",
            verified_at=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc),
        )

        embed = sent_embed(subject)
        assert "UNVERIFIED" in embed.title
        state = field_value(embed, "Position State") or ""
        assert "UNVERIFIED" in state
        assert "✅" not in state
        assert "Exchange State" not in [f.name for f in embed.fields]

    async def test_unverified_card_is_an_explicit_warning(self) -> None:
        subject = notifier()
        subject._send_embed = AsyncMock()
        last_confirmed = datetime(2026, 9, 21, 11, 0, tzinfo=timezone.utc)

        await subject.send_position_status(
            position=make_position(symbol="BTC/USDC"),
            current_price=81100.0,
            channel_id=123,
            verification="unverified",
            verified_at=last_confirmed,
            verification_detail="executor position query failed",
        )

        embed = sent_embed(subject)
        assert "UNVERIFIED" in embed.title
        assert "NOT proof of an open position" in embed.description
        assert "executor position query failed" in embed.description
        assert "UNVERIFIED" in (field_value(embed, "Position State") or "")
        assert format_utc_stamp(last_confirmed) in (field_value(embed, "Position State") or "")

    async def test_call_without_verification_defaults_to_unverified(self) -> None:

        subject = notifier()
        subject._send_embed = AsyncMock()

        await subject.send_position_status(make_position(), 81100.0, 123)

        embed = sent_embed(subject)
        assert embed.title.startswith("⚠️ Position Status UNVERIFIED")
        state = field_value(embed, "Position State") or ""
        assert "UNVERIFIED" in state
        assert "executor tracker only" in state

    def test_stamp_formatter_treats_naive_timestamps_as_utc(self) -> None:
        naive_stamp = datetime(2026, 9, 21, 9, 32, 51)  # noqa: DTZ001
        assert format_utc_stamp(naive_stamp) == "2026-09-21 09:32:51 UTC"
        assert format_utc_stamp(None) == "unknown"
