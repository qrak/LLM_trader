"""Tests for bugs that are easy to miss while the app keeps running."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.managers.persistence_manager import PersistenceManager
from src.parsing.unified_parser import UnifiedParser
from src.rag.market_components import MarketDataCache
from src.trading.position_extractor import PositionExtractor
from src.trading.statistics_calculator import TradingStatistics
from src.utils.format_utils import FormatUtils


def test_statistics_persistence_sanitizes_nonfinite_values(tmp_path) -> None:
    manager = PersistenceManager(logger=MagicMock(), data_dir=str(tmp_path))
    stats = TradingStatistics(
        total_trades=4,
        winning_trades=4,
        sortino_ratio=float("inf"),
        profit_factor=float("nan"),
    )

    manager.save_statistics(stats)

    raw_json = manager.statistics_file.read_text(encoding="utf-8")
    assert "Infinity" not in raw_json
    assert "NaN" not in raw_json
    saved = json.loads(raw_json)
    assert saved["sortino_ratio"] is None
    assert saved["profit_factor"] is None

    restored = manager.load_statistics()
    assert restored.sortino_ratio == 0.0
    assert restored.profit_factor == 0.0


def test_unified_parser_sanitizes_nonfinite_ai_numeric_fields() -> None:
    parser = UnifiedParser(logger=MagicMock(), format_utils=FormatUtils())
    parsed = parser.parse_ai_response(
        """```json
{
  "analysis": {
    "signal": "BUY",
    "confidence": "NaN",
    "entry_price": "Infinity",
    "stop_loss": "76500",
    "take_profit": "-Infinity",
    "position_size": "NaN",
    "risk_reward_ratio": "inf",
    "confidence_score": "nan",
    "confluence_factors": {
      "trend_alignment": "inf",
      "momentum_strength": "nan",
      "volume_support": "71"
    },
    "key_levels": {
      "support": ["76500", "inf", "nan", "bad"],
      "resistance": ["79000"]
    }
  }
}
```"""
    )

    analysis = parsed["analysis"]
    assert analysis["confidence"] == 50
    assert analysis["entry_price"] is None
    assert analysis["stop_loss"] == 76500.0
    assert analysis["take_profit"] is None
    assert analysis["position_size"] is None
    assert analysis["risk_reward_ratio"] is None
    assert analysis["confidence_score"] == 50
    assert analysis["confluence_factors"]["trend_alignment"] == 50.0
    assert analysis["confluence_factors"]["momentum_strength"] == 50.0
    assert analysis["confluence_factors"]["volume_support"] == 71.0
    assert analysis["key_levels"]["support"] == [76500.0]
    assert analysis["key_levels"]["resistance"] == [79000.0]
    assert parsed["response_validation"]["status"] == "invalid"


def test_position_extractor_rejects_nonfinite_json_trade_fields() -> None:
    parser = UnifiedParser(logger=MagicMock(), format_utils=FormatUtils())
    extractor = PositionExtractor(logger=MagicMock(), unified_parser=parser)

    signal, confidence, stop_loss, take_profit, position_size, reasoning = extractor.extract_trading_info(
        """```json
{
  "analysis": {
    "signal": "BUY",
    "confidence": NaN,
    "stop_loss": "NaN",
    "take_profit": "Infinity",
    "position_size": "NaN",
    "reasoning": "Bad numeric payload"
  }
}
```"""
    )

    assert signal == "BUY"
    assert confidence == "MEDIUM"
    assert stop_loss is None
    assert take_profit is None
    assert position_size is None
    assert reasoning == "Bad numeric payload"


def test_market_data_cache_unparseable_timestamp_is_stale_without_crashing() -> None:
    cache = MarketDataCache(logger=MagicMock(), file_handler=MagicMock())
    cache.current_market_overview = {"published_on": "not-a-timestamp"}

    assert cache.is_overview_stale(max_age_hours=1) is True


def test_market_data_cache_nonfinite_timestamp_is_stale_without_crashing() -> None:
    cache = MarketDataCache(logger=MagicMock(), file_handler=MagicMock())
    cache.current_market_overview = {"published_on": "NaN"}

    assert cache.is_overview_stale(max_age_hours=1) is True


def test_market_data_cache_recent_timestamp_stays_fresh() -> None:
    cache = MarketDataCache(logger=MagicMock(), file_handler=MagicMock())
    recent_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()
    cache.current_market_overview = {"published_on": recent_timestamp}

    assert cache.is_overview_stale(max_age_hours=1) is False