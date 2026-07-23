"""Tests for CryptoTradingBot wait/timeframe logic after DRY refactor."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app import CryptoTradingBot


class TestCalculateNextCheck:
    """Tests for _calculate_next_check helper extracted from the two wait methods."""

    @pytest.fixture
    def bot(self):
        """Create a minimal bot with mocked dependencies."""
        services = MagicMock()
        services.logger = MagicMock()
        services.config = MagicMock()
        services.dashboard_state = None
        services.force_analysis_event = None
        services.discord_task = None
        services.position_monitor_factory = None
        services.shutdown_manager = None
        services.exchange_manager = MagicMock()
        services.market_analyzer = MagicMock()
        services.trading_strategy = MagicMock()
        services.discord_notifier = None
        services.keyboard_handler = MagicMock()
        services.rag_engine = MagicMock()
        services.coingecko_api = None
        services.market_api = None
        services.alternative_me_api = None
        services.http_session = None
        services.persistence = MagicMock()
        services.model_manager = MagicMock()
        services.brain_service = MagicMock()
        services.statistics_service = MagicMock()
        services.memory_service = MagicMock()
        services.exit_monitor = MagicMock()
        services.executor_handler = None

        bot = CryptoTradingBot(services)
        bot.current_timeframe = "15m"
        return bot

    @patch("src.app.time.time")
    @patch("src.app.TimeframeValidator")
    def test_returns_delay_and_next_check_time(self, mock_validator, mock_time, bot):
        """_calculate_next_check returns delay_seconds and next_check_time UTC."""
        # Source is 100s ago, next candle is 200s from now → delay ~200s + buffer
        now_seconds = 1_000_000.0
        mock_time.return_value = now_seconds
        source_ms = int((now_seconds - 100) * 1000)  # 100s ago
        next_candle_ms = int((now_seconds + 200) * 1000)  # 200s from now
        mock_validator.calculate_next_candle_time.return_value = next_candle_ms

        delay_seconds, next_check_time = bot._calculate_next_check(source_ms)

        assert delay_seconds > 0
        assert isinstance(next_check_time, datetime)
        assert next_check_time.tzinfo == timezone.utc

    @patch("src.app.TimeframeValidator")
    def test_clamps_negative_delay_to_zero(self, mock_validator, bot):
        """Delay is clamped to 0 when next candle has already started."""
        mock_validator.calculate_next_candle_time.return_value = 1_747_000_000_000  # in the past

        delay_seconds, _ = bot._calculate_next_check(1_746_000_000_000)

        assert delay_seconds == 0.0

    def test_raises_when_timeframe_not_set(self, bot):
        """Raises ValueError when current_timeframe is None."""
        bot.current_timeframe = None

        with pytest.raises(ValueError, match="current timeframe is not set"):
            bot._calculate_next_check(1_000_000_000_000)


class TestWaitForNextTimeframe:
    """Tests for _wait_for_next_timeframe after DRY refactor."""

    @pytest.fixture
    def bot(self):
        services = MagicMock()
        services.logger = MagicMock()
        services.config = MagicMock()
        services.dashboard_state = None
        services.force_analysis_event = None
        services.discord_task = None
        services.position_monitor_factory = None
        services.shutdown_manager = None
        services.exchange_manager = MagicMock()
        services.market_analyzer = MagicMock()
        services.trading_strategy = MagicMock()
        services.discord_notifier = None
        services.keyboard_handler = MagicMock()
        services.rag_engine = MagicMock()
        services.coingecko_api = None
        services.market_api = None
        services.alternative_me_api = None
        services.http_session = None
        services.persistence = MagicMock()
        services.model_manager = MagicMock()
        services.brain_service = MagicMock()
        services.statistics_service = MagicMock()
        services.memory_service = MagicMock()
        services.exit_monitor = MagicMock()
        services.executor_handler = None

        bot = CryptoTradingBot(services)
        bot.current_timeframe = "15m"
        bot._interruptible_sleep = AsyncMock(return_value=False)
        return bot

    @pytest.mark.asyncio
    @patch("src.app.TimeframeValidator")
    async def test_logs_next_check_time_and_sleeps(self, mock_validator, bot):
        """Logs next check info, updates dashboard, calls _interruptible_sleep."""
        mock_validator.calculate_next_candle_time.return_value = 1_750_000_000_000

        result = await bot._wait_for_next_timeframe()

        bot._interruptible_sleep.assert_awaited_once()
        bot.logger.info.assert_called()
        assert "Next check" in bot.logger.info.call_args[0][0]
        assert result is False  # _interruptible_sleep returned False

    @pytest.mark.asyncio
    @patch("src.app.TimeframeValidator")
    async def test_updates_dashboard_state_when_present(self, mock_validator, bot):
        """Updates dashboard_state when it's available."""
        mock_validator.calculate_next_candle_time.return_value = 1_750_000_000_000
        bot.dashboard_state = MagicMock()
        bot.dashboard_state.update_next_check = AsyncMock()

        await bot._wait_for_next_timeframe()

        bot.dashboard_state.update_next_check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self, bot):
        """Sleeps ERROR_WAIT_LONG on calculation error and returns False."""
        bot.current_timeframe = None  # triggers ValueError

        with patch("src.app.ERROR_WAIT_LONG", 1):
            result = await bot._wait_for_next_timeframe()

        assert result is False
        bot._interruptible_sleep.assert_awaited_with(1)  # ERROR_WAIT_LONG


class TestWaitUntilNextTimeframeAfter:
    """Tests for _wait_until_next_timeframe_after after DRY refactor."""

    @pytest.fixture
    def bot(self):
        services = MagicMock()
        services.logger = MagicMock()
        services.config = MagicMock()
        services.dashboard_state = None
        services.force_analysis_event = None
        services.discord_task = None
        services.position_monitor_factory = None
        services.shutdown_manager = None
        services.exchange_manager = MagicMock()
        services.market_analyzer = MagicMock()
        services.trading_strategy = MagicMock()
        services.discord_notifier = None
        services.keyboard_handler = MagicMock()
        services.rag_engine = MagicMock()
        services.coingecko_api = None
        services.market_api = None
        services.alternative_me_api = None
        services.http_session = None
        services.persistence = MagicMock()
        services.model_manager = MagicMock()
        services.brain_service = MagicMock()
        services.statistics_service = MagicMock()
        services.memory_service = MagicMock()
        services.exit_monitor = MagicMock()
        services.executor_handler = None

        bot = CryptoTradingBot(services)
        bot.current_timeframe = "15m"
        bot._interruptible_sleep = AsyncMock(return_value=False)
        return bot

    def _make_last_time(self, hours_ago: float = 1.0):
        """Helper to create a past datetime."""
        from datetime import timedelta
        return datetime.now(timezone.utc) - timedelta(hours=hours_ago)

    @pytest.mark.asyncio
    @patch("src.app.TimeframeValidator")
    async def test_returns_early_when_candle_already_passed(self, mock_validator, bot):
        """Returns immediately when current time is past the next candle time."""
        mock_validator.calculate_next_candle_time.return_value = 1  # very old candle (1ms epoch)

        result = await bot._wait_until_next_timeframe_after(self._make_last_time())

        assert result is None  # early return
        bot._interruptible_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.app.TimeframeValidator")
    async def test_waits_when_candle_not_yet_passed(self, mock_validator, bot):
        """Sleeps when the next candle hasn't started yet."""
        mock_validator.calculate_next_candle_time.return_value = 9_999_999_999_999_999  # far future
        mock_validator.is_same_candle.return_value = True

        await bot._wait_until_next_timeframe_after(self._make_last_time())

        bot._interruptible_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self, bot):
        """Sleeps ERROR_WAIT_SHORT on calculation error."""
        bot.current_timeframe = None  # triggers ValueError

        with patch("src.app.ERROR_WAIT_SHORT", 1):
            await bot._wait_until_next_timeframe_after(self._make_last_time())

        bot._interruptible_sleep.assert_awaited_with(1)  # ERROR_WAIT_SHORT
