"""Shared fixtures and builders for the LLM_trader test suite.

Every domain suite imports its doubles from here instead of rebuilding config
namespaces, positions and dependency mocks per module. Defaults mirror the real
values in src/config/loader.py, so a test only states what it changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

PRODUCTION_DEFAULTS: dict[str, Any] = {
    "PROVIDER": "googleai",
    "LM_STUDIO_BASE_URL": "http://localhost:1234/v1",
    "LM_STUDIO_MODEL": "local-model",
    "LM_STUDIO_STREAMING": True,
    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
    "OPENROUTER_BASE_MODEL": "google/gemini-3-flash-preview",
    "OPENROUTER_FALLBACK_MODEL": "deepseek/deepseek-v4.1-flash",
    "DEEPSEEK_API_KEY": "test-deepseek-key",
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    "DEEPSEEK_MODEL": "deepseek-flash",
    "GOOGLE_STUDIO_API_KEY": "test-google-key",
    "GOOGLE_STUDIO_PAID_API_KEY": "",
    "GOOGLE_STUDIO_MODEL": "gemini-3.8-flash",
    "BLOCKRUN_BASE_URL": "https://blockrun.ai/api",
    "BLOCKRUN_MODEL": "deepseek/deepseek-reasoner",
    "BLOCKRUN_WALLET_KEY": "",
    "MODEL_VERBOSITY": "high",
    "LOGGER_DEBUG": False,
    "CRYPTO_PAIR": "BTC/USDT",
    "DISCORD_BOT_ENABLED": False,
    "MAIN_CHANNEL_ID": 0,
    "BOT_TOKEN_DISCORD": "",
    "TIMEFRAME": "1h",
    "CANDLE_LIMIT": 999,
    "AI_CHART_CANDLE_LIMIT": 200,
    "MARKET_TYPE": "spot",
    "ENTRY_ORDER_TYPE": "market",
    "INCLUDE_COIN_DESCRIPTION": False,
    "DEBUG_SAVE_CHARTS": False,
    "DEBUG_CHART_SAVE_PATH": "test_images",
    "LOG_DIR": "logs",
    "DATA_DIR": "data",
    "DASHBOARD_ENABLED": True,
    "DASHBOARD_HOST": "0.0.0.0",
    "DASHBOARD_PORT": 8000,
    "DASHBOARD_ENABLE_CORS": False,
    "DASHBOARD_CORS_ORIGINS": [],
    "ADMIN_USERNAME": "",
    "ADMIN_PASSWORD_HASH": "",
    "ADMIN_SIGNING_KEY": "",
    "RESEARCH_TEAM_ENABLED": False,
    "SOCIAL_SENTIMENT_ENABLED": False,
    "RAG_UPDATE_INTERVAL_HOURS": 4,
    "RAG_NEWS_LIMIT": 5,
    "RAG_ARTICLE_MAX_TOKENS": 1000,
    "RAG_NEWS_FETCH_TIMEOUT": 20,
    "RAG_NEWS_ENRICH_TIMEOUT": 120,
    "RAG_UPDATE_TIMEOUT": 180,
    "RAG_NEWS_SOURCES": None,
    "SUPPORTED_EXCHANGES": ["binance", "kucoin", "gateio"],
    "MARKET_REFRESH_HOURS": 24,
    "TRANSACTION_FEE_PERCENT": 0.00075,
    "DEMO_QUOTE_CAPITAL": 10000.0,
    "MAX_POSITION_SIZE": 0.10,
    "MIN_RR_ENTRY": 1.0,
    "POSITION_SIZE_FALLBACK_LOW": 0.01,
    "POSITION_SIZE_FALLBACK_MEDIUM": 0.02,
    "POSITION_SIZE_FALLBACK_HIGH": 0.03,
    "SL_TIGHTENING_SCALPING": 0.25,
    "SL_TIGHTENING_INTRADAY": 0.20,
    "SL_TIGHTENING_SWING": 0.15,
    "SL_TIGHTENING_POSITION": 0.10,
    "SL_TIGHTENING_FLOOR": 0.05,
    "SL_TIGHTENING_CEILING": 0.40,
    "SL_TIGHTENING_MIN_SAMPLES": 10,
    "STOP_LOSS_TYPE": "soft",
    "TAKE_PROFIT_TYPE": "soft",
    "EXECUTOR_API_ENABLED": False,
    "EXECUTOR_API_URL": "http://127.0.0.1:9199/decision",
    "EXECUTOR_VERDICT_PATH": "data/trading/executor_verdicts.jsonl",
    "EXECUTOR_MAX_POSITION_USDC": 0.0,
    "FILE_MESSAGE_EXPIRY": 604800,
}


def make_config(**overrides: Any) -> SimpleNamespace:
    """Config double carrying the production defaults plus explicit overrides."""
    values = dict(PRODUCTION_DEFAULTS)
    values.update(overrides)
    timeframe = values["TIMEFRAME"]
    values.setdefault("STOP_LOSS_CHECK_INTERVAL", timeframe)
    values.setdefault("TAKE_PROFIT_CHECK_INTERVAL", timeframe)
    values["QUOTE_CURRENCY"] = overrides.get("QUOTE_CURRENCY", values["CRYPTO_PAIR"].split("/")[1])
    return SimpleNamespace(**values)


def null_logger() -> MagicMock:
    """Logger double: silent, records calls for assertions."""
    return MagicMock()


def mock_brain() -> MagicMock:
    """TradingBrainService double.

    Only the awaited surface (``get_context``, ``get_vector_context``) is async:
    ``update_from_closed_trade`` and ``track_position_update`` are synchronous in
    production and are called directly / through ``asyncio.to_thread``, so an
    ``AsyncMock`` there records nothing and leaks an un-awaited coroutine.
    """
    brain = MagicMock()
    brain.get_context = AsyncMock(return_value="")
    brain.get_vector_context = AsyncMock(return_value="")
    brain.update_from_closed_trade = MagicMock()
    brain.track_position_update = MagicMock()
    brain.record_closed_trade = AsyncMock()
    return brain


def mock_statistics() -> MagicMock:
    """TradingStatisticsService double (production ``recalculate`` is synchronous)."""
    stats = MagicMock()
    stats.get_statistics = MagicMock(return_value={})
    stats.recalculate = MagicMock()
    return stats


def mock_persistence() -> MagicMock:
    """PersistenceManager double.

    Both async persistence seams are ``AsyncMock``: production awaits
    ``async_save_trade_decision`` (every decision) and ``async_save_position`` (state
    writes). The sync counterparts stay ``MagicMock``.
    """
    persistence = MagicMock()
    persistence.async_save_trade_decision = AsyncMock()
    persistence.async_save_position = AsyncMock()
    persistence.save_position = MagicMock()
    persistence.load_position = MagicMock(return_value=None)
    return persistence


def mock_extractor() -> MagicMock:
    """PositionExtractor double returning a HOLD decision by default."""
    extractor = MagicMock()
    extractor.extract_trading_info = MagicMock(return_value=("HOLD", "MEDIUM", None, None, None, ""))
    return extractor


def make_market_conditions(**overrides: Any):
    """MarketConditions instance built from real production defaults."""
    from src.trading.data_models import MarketConditions

    values: dict[str, Any] = {
        "trend_direction": "NEUTRAL",
        "adx": 20.0,
        "rsi": 50.0,
        "rsi_level": "NEUTRAL",
        "volatility": "MEDIUM",
        "atr": 500.0,
        "atr_percentage": 1.0,
        "macd_signal": "NEUTRAL",
        "bb_position": "MIDDLE",
        "volume_state": "NORMAL",
        "market_sentiment": "NEUTRAL",
        "order_book_bias": "BALANCED",
        "choppiness": 50.0,
    }
    values.update(overrides)
    return MarketConditions(**values)


def make_position(**overrides: Any):
    """Position instance built on a MarketConditions snapshot."""
    from src.trading.data_models import Position

    values: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "confidence": "HIGH",
        "entry_price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit": 52000.0,
        "size": 0.01,
        "entry_time": datetime(2026, 9, 17, tzinfo=timezone.utc),
        "conditions_at_entry": make_market_conditions(),
    }
    values.update(overrides)
    return Position(**values)


@pytest.fixture
def config() -> SimpleNamespace:
    return make_config()


@pytest.fixture
def logger_double() -> MagicMock:
    return null_logger()


@pytest.fixture
def format_utils():
    from src.utils.format_utils import FormatUtils

    return FormatUtils()


@pytest.fixture
def brain_double() -> MagicMock:
    return mock_brain()


@pytest.fixture
def statistics_double() -> MagicMock:
    return mock_statistics()


@pytest.fixture
def persistence_double() -> MagicMock:
    return mock_persistence()


@pytest.fixture
def extractor_double() -> MagicMock:
    return mock_extractor()


@pytest.fixture
def position():
    return make_position()


@pytest.fixture
def market_conditions():
    return make_market_conditions()


@pytest.fixture
def sqlite_history(tmp_path):
    from src.managers.sqlite_trade_history import SQLiteTradeHistory

    return SQLiteTradeHistory(logger=null_logger(), db_path=str(tmp_path / "trade_history.db"))
