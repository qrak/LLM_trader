"""Silent-failure tests for Statistics calculator and Data Model serialization.

Focus: scenarios that produce wrong results without raising exceptions.
"""

import pytest
import math
import json
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
# Statistics calculator edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestStatisticsEdgeCases:
    """StatisticsCalculator scenarios that silently produce wrong results."""

    @pytest.fixture
    def calc(self):
        from src.trading.statistics_calculator import StatisticsCalculator
        return StatisticsCalculator()

    def test_empty_history_returns_defaults(self, calc):
        result = calc.calculate_from_history([])
        assert result.total_trades == 0
        assert result.win_rate == 0.0

    def test_all_winning_trades_sortino_is_inf(self, calc):
        """All-winning trades → Sortino = float('inf') in memory.
        This is still produced by the calculator but _convert_value now
        defends against it on deserialization (inf → null → 0.0)."""
        history = [
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 110.0, "quantity": 1.0},
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 105.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert math.isinf(result.sortino_ratio)
        assert result.sortino_ratio > 0

    def test_all_winning_trades_profit_factor_is_inf(self, calc):
        history = [
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 110.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert math.isinf(result.profit_factor)

    def test_initial_capital_zero_is_safe(self, calc):
        history = [
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 110.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history, initial_capital=0.0)
        assert result.total_pnl_pct == 0.0

    def test_closed_trade_entry_price_zero_is_skipped(self, calc):
        """FIXED: entry_price=0 now skips the trade entirely instead of
        producing phantom profit through unguarded pnl_quote."""
        history = [
            {"action": "BUY", "price": 0.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 110.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert result.total_trades == 0  # trade skipped

    def test_close_before_open_silently_dropped(self, calc):
        history = [
            {"action": "CLOSE", "price": 110.0, "quantity": 1.0},
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 105.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert result.total_trades == 1

    def test_double_close_silently_drops_second(self, calc):
        history = [
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 110.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 115.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert result.total_trades == 1

    def test_only_buys_no_closes(self, calc):
        history = [
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "BUY", "price": 95.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert result.total_trades == 0

    def test_short_trade_pnl_calculation(self, calc):
        history = [
            {"action": "SELL", "price": 110.0, "quantity": 1.0},
            {"action": "CLOSE_SHORT", "price": 100.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert result.total_trades == 1
        assert result.winning_trades == 1
        assert result.total_pnl_quote == 10.0

    def test_sharpe_with_single_trade_returns_zero(self, calc):
        history = [
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 110.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert result.sharpe_ratio == 0.0

    def test_drawdown_with_single_trade(self, calc):
        history = [
            {"action": "BUY", "price": 100.0, "quantity": 1.0},
            {"action": "CLOSE", "price": 90.0, "quantity": 1.0},
        ]
        result = calc.calculate_from_history(history)
        assert result.max_drawdown_pct < 0
        assert result.worst_trade_pct < 0


# ═══════════════════════════════════════════════════════════════════════
# Statistics → JSON persistence round-trip
# ═══════════════════════════════════════════════════════════════════════

class TestStatisticsPersistenceRoundTrip:
    """Statistics with inf/NaN values survive JSON serialization round-trip."""

    def test_sortino_inf_roundtrip_recovered_as_zero(self):
        """FIXED: inf Sortino → serialize_for_json → null → from_dict → 0.0.
        Non-Optional float fields now default to 0.0 instead of None."""
        from src.trading.statistics_calculator import TradingStatistics
        from src.utils.data_utils import serialize_for_json

        stats = TradingStatistics(
            total_trades=5,
            winning_trades=5,
            sortino_ratio=float('inf'),
            profit_factor=float('inf'),
        )

        raw = stats.to_dict()
        safe = serialize_for_json(raw)
        json_str = json.dumps(safe)
        loaded_raw = json.loads(json_str)

        # inf becomes null in JSON
        assert loaded_raw["sortino_ratio"] is None
        assert loaded_raw["profit_factor"] is None

        restored = TradingStatistics.from_dict(loaded_raw)
        # FIXED: non-Optional float fields default to 0.0, not None
        assert restored.sortino_ratio == 0.0
        assert restored.profit_factor == 0.0

    def test_nan_in_metrics_roundtrip_recovered_as_zero(self):
        """FIXED: NaN → serialize_for_json → null → from_dict → 0.0."""
        from src.trading.statistics_calculator import TradingStatistics
        from src.utils.data_utils import serialize_for_json

        stats = TradingStatistics(win_rate=float('nan'), sharpe_ratio=float('nan'))
        raw = stats.to_dict()
        safe = serialize_for_json(raw)
        json_str = json.dumps(safe)
        loaded_raw = json.loads(json_str)
        restored = TradingStatistics.from_dict(loaded_raw)
        # FIXED: non-Optional float fields default to 0.0 instead of None
        assert restored.win_rate == 0.0
        assert restored.sharpe_ratio == 0.0

    def test_optional_float_still_accepts_none(self):
        """Optional[float] fields should still allow None."""
        from src.trading.data_models import TradeDecision
        from src.utils.data_utils import serialize_for_json

        td = TradeDecision(
            timestamp=datetime(2026, 5, 25, 12, 0, 0),
            symbol="BTC/USDC",
            action="HOLD",
            confidence="MEDIUM",
            price=50000.0,
            stop_loss=None,
            take_profit=None,
            position_size=0.0,
            reasoning="Test",
        )

        raw = td.to_dict()
        safe = serialize_for_json(raw)
        json_str = json.dumps(safe)
        loaded = json.loads(json_str)
        restored = TradeDecision.from_dict(loaded)
        # stop_loss/take_profit are float | None → should stay None
        assert restored.stop_loss is None
        assert restored.take_profit is None


# ═══════════════════════════════════════════════════════════════════════
# Data Model serialization
# ═══════════════════════════════════════════════════════════════════════

class TestDataModelSerialization:
    """Serialization correctness tests."""

    def test_datetime_from_invalid_string_now_raises(self, cfg=None):
        """FIXED: invalid datetime now raises ValueError instead of silently
        keeping the corrupt string that would crash on .isoformat()."""
        from src.utils.data_utils import SerializableMixin
        from dataclasses import dataclass

        @dataclass
        class HasDatetime(SerializableMixin):
            ts: datetime = None
            name: str = ""

        with pytest.raises(ValueError, match="Cannot convert"):
            HasDatetime.from_dict({"ts": "not-a-date", "name": "test"})

    def test_position_full_roundtrip(self):
        from src.trading.data_models import Position
        from src.utils.data_utils import serialize_for_json

        pos = Position(
            symbol="BTC/USDC",
            direction="LONG",
            confidence="HIGH",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            size=0.001,
            entry_time=datetime(2026, 5, 25, 12, 0, 0),
            entry_fee=3.75,
            size_pct=0.05,
            tp_distance_pct=4.0,
            sl_distance_pct=2.0,
            confluence_factors=(("trend", 0.8), ("momentum", 0.6)),
        )

        raw = pos.to_dict()
        safe = serialize_for_json(raw)
        json_str = json.dumps(safe)
        loaded = json.loads(json_str)

        restored = Position.from_dict(loaded)

        assert restored.symbol == "BTC/USDC"
        assert restored.direction == "LONG"
        assert restored.entry_price == 50000.0
        assert restored.stop_loss == 49000.0
        assert restored.size_pct == 0.05
        assert isinstance(restored.entry_time, datetime)

    def test_trade_decision_roundtrip_with_none_fields(self):
        from src.trading.data_models import TradeDecision
        from src.utils.data_utils import serialize_for_json

        td = TradeDecision(
            timestamp=datetime(2026, 5, 25, 12, 0, 0),
            symbol="BTC/USDC",
            action="HOLD",
            confidence="MEDIUM",
            price=50000.0,
            stop_loss=None,
            take_profit=None,
            position_size=0.0,
            reasoning="Test reasoning",
        )

        raw = td.to_dict()
        safe = serialize_for_json(raw)
        json_str = json.dumps(safe)
        loaded = json.loads(json_str)
        restored = TradeDecision.from_dict(loaded)
        assert restored.stop_loss is None
        assert restored.take_profit is None
        assert restored.action == "HOLD"

    def test_confluence_factors_deserialization_preserves_tuple(self):
        """FIXED: plain tuple without type args now converts list→tuple."""
        from src.trading.data_models import Position

        pos_data = {
            "symbol": "BTC/USDC",
            "direction": "LONG",
            "confidence": "HIGH",
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "size": 0.001,
            "entry_time": "2026-05-25T12:00:00",
            "confluence_factors": [["trend", 0.8], ["momentum", 0.6]],
        }

        pos = Position.from_dict(pos_data)
        assert isinstance(pos.confluence_factors, tuple), (
            f"Expected tuple, got {type(pos.confluence_factors).__name__}"
        )
        assert pos.confluence_factors == (("trend", 0.8), ("momentum", 0.6))
        assert all(isinstance(factor, tuple) for factor in pos.confluence_factors)

    def test_int_bool_str_fields_default_from_none(self):
        """Non-Optional int/bool/str fields recover from None with zero-values."""
        from src.trading.statistics_calculator import TradingStatistics

        # Simulate corrupted persistence: null for every non-Optional field
        corrupted = {
            "total_trades": None,
            "winning_trades": None,
            "losing_trades": None,
            "win_rate": None,
            "total_pnl_pct": None,
            "total_pnl_quote": None,
            "initial_capital": None,
            "current_capital": None,
            "avg_trade_pct": None,
            "best_trade_pct": None,
            "worst_trade_pct": None,
            "max_drawdown_pct": None,
            "avg_drawdown_pct": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "profit_factor": None,
        }

        restored = TradingStatistics.from_dict(corrupted)
        assert restored.total_trades == 0  # int → 0
        assert restored.win_rate == 0.0    # float → 0.0
        assert restored.initial_capital == 0.0
