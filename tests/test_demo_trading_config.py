"""Test demo trading configuration (capital and fees)."""

import sys
import importlib
import pytest
from unittest.mock import patch


class MockConfig:
    """Mock config with demo trading settings."""

    def __init__(
        self,
        demo_capital: float = 10000.0,
        fee_percent: float = 0.00075
    ):
        self._demo_capital = demo_capital
        self._fee_percent = fee_percent

    @property
    def DEMO_QUOTE_CAPITAL(self) -> float:
        return self._demo_capital

    @property
    def TRANSACTION_FEE_PERCENT(self) -> float:
        return self._fee_percent

    @property
    def DEFAULT_POSITION_SIZE(self) -> float:
        return 0.02

    @property
    def DEFAULT_STOP_LOSS_PCT(self) -> float:
        return 0.02

    @property
    def DEFAULT_TAKE_PROFIT_PCT(self) -> float:
        return 0.04


class TestDemoTradingConfig:
    """Test demo trading configuration values."""

    def test_default_capital_value(self):
        """Test default capital is 10000."""
        config = MockConfig()
        assert config.DEMO_QUOTE_CAPITAL == 10000.0

    def test_default_fee_percent(self):
        """Test default fee is 0.075%."""
        config = MockConfig()
        assert config.TRANSACTION_FEE_PERCENT == 0.00075

    def test_custom_capital_value(self):
        """Test custom capital configuration."""
        config = MockConfig(demo_capital=50000.0)
        assert config.DEMO_QUOTE_CAPITAL == 50000.0

    def test_custom_fee_percent(self):
        """Test custom fee percentage configuration."""
        config = MockConfig(fee_percent=0.001)  # 0.1%
        assert config.TRANSACTION_FEE_PERCENT == 0.001


class TestPositionSizeCalculation:
    """Test position size calculations using demo capital."""

    def test_quantity_calculation_with_default_capital(self):
        """Test quantity = (capital * size_pct) / price."""
        config = MockConfig(demo_capital=10000.0)
        current_price = 100000.0  # BTC price
        size_pct = 0.02  # 2% of capital

        capital = config.DEMO_QUOTE_CAPITAL
        allocation = capital * size_pct
        quantity = allocation / current_price

        assert capital == 10000.0
        assert allocation == 200.0  # 2% of 10000
        assert quantity == 0.002  # 200 / 100000

    def test_quantity_calculation_with_custom_capital(self):
        """Test quantity with different capital."""
        config = MockConfig(demo_capital=50000.0)
        current_price = 100000.0
        size_pct = 0.10  # 10% of capital

        capital = config.DEMO_QUOTE_CAPITAL
        allocation = capital * size_pct
        quantity = allocation / current_price

        assert capital == 50000.0
        assert allocation == 5000.0  # 10% of 50000
        assert quantity == 0.05  # 5000 / 100000

    def test_allocation_varies_with_position_size(self):
        """Test different position sizes produce correct allocations."""
        config = MockConfig(demo_capital=10000.0)
        current_price = 50000.0

        test_cases = [
            (0.01, 100.0, 0.002),    # 1% -> $100 -> 0.002 BTC
            (0.02, 200.0, 0.004),    # 2% -> $200 -> 0.004 BTC
            (0.05, 500.0, 0.010),    # 5% -> $500 -> 0.010 BTC
            (0.10, 1000.0, 0.020),   # 10% -> $1000 -> 0.020 BTC
            (0.50, 5000.0, 0.100),   # 50% -> $5000 -> 0.100 BTC
            (1.00, 10000.0, 0.200),  # 100% -> $10000 -> 0.200 BTC
        ]

        for size_pct, expected_alloc, expected_qty in test_cases:
            allocation = config.DEMO_QUOTE_CAPITAL * size_pct
            quantity = allocation / current_price
            assert allocation == pytest.approx(expected_alloc), f"Failed for size_pct={size_pct}"
            assert quantity == pytest.approx(expected_qty), f"Failed for size_pct={size_pct}"


class TestFeeCalculation:
    """Test transaction fee calculations."""

    def test_entry_fee_calculation(self):
        """Test entry fee = allocation * fee_percent."""
        config = MockConfig(demo_capital=10000.0, fee_percent=0.00075)
        size_pct = 0.02  # 2%

        allocation = config.DEMO_QUOTE_CAPITAL * size_pct
        entry_fee = allocation * config.TRANSACTION_FEE_PERCENT

        assert allocation == 200.0
        assert entry_fee == pytest.approx(0.15)  # 200 * 0.00075 = 0.15

    def test_larger_position_higher_fee(self):
        """Test larger positions have proportionally higher fees."""
        config = MockConfig(demo_capital=10000.0, fee_percent=0.00075)

        small_alloc = config.DEMO_QUOTE_CAPITAL * 0.01  # $100
        large_alloc = config.DEMO_QUOTE_CAPITAL * 0.10  # $1000

        small_fee = small_alloc * config.TRANSACTION_FEE_PERCENT
        large_fee = large_alloc * config.TRANSACTION_FEE_PERCENT

        assert small_fee == pytest.approx(0.075)
        assert large_fee == pytest.approx(0.75)
        assert large_fee == 10 * small_fee

    def test_round_trip_fees(self):
        """Test total fees for entry + exit (round trip)."""
        config = MockConfig(demo_capital=10000.0, fee_percent=0.00075)
        size_pct = 0.02  # 2%

        allocation = config.DEMO_QUOTE_CAPITAL * size_pct
        entry_fee = allocation * config.TRANSACTION_FEE_PERCENT
        exit_fee = allocation * config.TRANSACTION_FEE_PERCENT
        total_fees = entry_fee + exit_fee

        assert total_fees == pytest.approx(0.30)  # 2 * 0.15

    def test_fee_impact_on_pnl(self):
        """Test fees reduce net PnL."""
        config = MockConfig(demo_capital=10000.0, fee_percent=0.00075)
        size_pct = 0.02
        entry_price = 100000.0
        exit_price = 102000.0  # 2% profit

        allocation = config.DEMO_QUOTE_CAPITAL * size_pct  # $200
        quantity = allocation / entry_price  # 0.002 BTC

        gross_pnl = (exit_price - entry_price) * quantity  # $4.00
        entry_fee = allocation * config.TRANSACTION_FEE_PERCENT  # $0.15
        exit_value = exit_price * quantity  # $204
        exit_fee = exit_value * config.TRANSACTION_FEE_PERCENT  # ~$0.153
        net_pnl = gross_pnl - entry_fee - exit_fee

        assert gross_pnl == pytest.approx(4.0)
        assert entry_fee == pytest.approx(0.15)
        assert exit_fee == pytest.approx(0.153)
        assert net_pnl == pytest.approx(3.697, rel=0.01)


class TestConfigLoaderIntegration:
    """Integration tests with actual Config loader."""

    @pytest.fixture
    def real_config_class(self):
        """Fixture to recover the real Config class, bypassing conftest mocking."""
        # Save the mock module
        mock_module = sys.modules.get('src.config.loader')
        
        # Temporarily remove it from sys.modules to allow real import
        if 'src.config.loader' in sys.modules:
            del sys.modules['src.config.loader']
            
        try:
            # Import and reload the real module
            import src.config.loader
            importlib.reload(src.config.loader)
            config_class = src.config.loader.Config
            yield config_class
        finally:
            # Restore the mock for other tests
            if mock_module:
                sys.modules['src.config.loader'] = mock_module

    def test_config_loader_parses_demo_capital(self, real_config_class):
        """Test Config.DEMO_QUOTE_CAPITAL returns float from config data."""
        Config = real_config_class

        with patch.object(Config, '_load_environment'):
            with patch.object(Config, '_load_ini_config'):
                with patch.object(Config, '_build_dynamic_urls'):
                    with patch.object(Config, '_build_model_configs'):
                        config = Config()
                        config._config_data = {
                            'demo_trading': {
                                'demo_quote_capital': 25000.0,
                                'transaction_fee_percent': 0.001,
                            }
                        }

                        assert config.DEMO_QUOTE_CAPITAL == 25000.0
                        assert config.TRANSACTION_FEE_PERCENT == 0.001

    def test_config_defaults_when_section_missing(self, real_config_class):
        """Test defaults are used when demo_trading section is missing."""
        Config = real_config_class

        with patch.object(Config, '_load_environment'):
            with patch.object(Config, '_load_ini_config'):
                with patch.object(Config, '_build_dynamic_urls'):
                    with patch.object(Config, '_build_model_configs'):
                        config = Config()
                        config._config_data = {}

                        assert config.DEMO_QUOTE_CAPITAL == 10000.0
                        assert config.TRANSACTION_FEE_PERCENT == 0.00075


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_position_size(self):
        """Test extremely small position sizes."""
        config = MockConfig(demo_capital=10000.0)
        size_pct = 0.001  # 0.1%
        current_price = 100000.0

        allocation = config.DEMO_QUOTE_CAPITAL * size_pct
        quantity = allocation / current_price
        fee = allocation * config.TRANSACTION_FEE_PERCENT

        assert allocation == 10.0
        assert quantity == 0.0001
        assert fee == pytest.approx(0.0075)

    def test_full_position_size(self):
        """Test 100% position size."""
        config = MockConfig(demo_capital=10000.0)
        size_pct = 1.0
        current_price = 100000.0

        allocation = config.DEMO_QUOTE_CAPITAL * size_pct
        quantity = allocation / current_price
        fee = allocation * config.TRANSACTION_FEE_PERCENT

        assert allocation == 10000.0
        assert quantity == 0.1
        assert fee == pytest.approx(7.5)

    def test_zero_fee_configuration(self):
        """Test with zero transaction fee."""
        config = MockConfig(fee_percent=0.0)
        allocation = 1000.0

        fee = allocation * config.TRANSACTION_FEE_PERCENT
        assert fee == 0.0

    def test_high_fee_configuration(self):
        """Test high transaction fee (e.g., 0.5%)."""
        config = MockConfig(fee_percent=0.005)
        allocation = 1000.0

        fee = allocation * config.TRANSACTION_FEE_PERCENT
        assert fee == 5.0
class TestDynamicCapital:
    """Test dynamic capital calculation."""

    def test_current_capital_after_loss(self):
        """Current capital = initial + pnl_quote (should decrease after loss)."""
        from src.trading.statistics_calculator import TradingStatistics
        
        stats = TradingStatistics(
            total_trades=1,
            total_pnl_quote=-200.0,
            initial_capital=10000.0,
            current_capital=9800.0
        )
        
        assert stats.current_capital == 9800.0
        assert stats.current_capital == stats.initial_capital + stats.total_pnl_quote

    def test_current_capital_after_wins(self):
        """Current capital = initial + pnl_quote (should increase after wins)."""
        from src.trading.statistics_calculator import TradingStatistics
        
        stats = TradingStatistics(
            total_trades=3,
            total_pnl_quote=500.0,
            initial_capital=10000.0,
            current_capital=10500.0
        )
        
        assert stats.current_capital == 10500.0
        assert stats.current_capital == stats.initial_capital + stats.total_pnl_quote

    def test_statistics_from_dict_backward_compatibility(self):
        """Test from_dict handles missing capital fields gracefully."""
        from src.trading.statistics_calculator import TradingStatistics
        
        old_data = {
            "total_trades": 2,
            "winning_trades": 1,
            "losing_trades": 1,
            "win_rate": 50.0,
            "total_pnl_pct": -1.5,
            "total_pnl_quote": -150.0,
            "avg_trade_pct": -0.75,
            "best_trade_pct": 2.0,
            "worst_trade_pct": -3.5,
            "max_drawdown_pct": -3.5,
            "avg_drawdown_pct": -2.0,
            "sharpe_ratio": 0.5,
            "sortino_ratio": 0.6,
            "profit_factor": 0.8,
            "last_updated": "2026-01-04T12:00:00"
        }
        
        stats = TradingStatistics.from_dict(old_data)
        
        assert stats.initial_capital == 10000.0
        assert stats.current_capital == 10000.0  # Default value when not present in dict

    def test_statistics_to_dict_includes_capital(self):
        """Test to_dict includes capital fields."""
        from src.trading.statistics_calculator import TradingStatistics
        
        stats = TradingStatistics(
            total_trades=1,
            initial_capital=10000.0,
            current_capital=9800.0
        )
        
        data = stats.to_dict()
        
        assert "initial_capital" in data
        assert "current_capital" in data
        assert data["initial_capital"] == 10000.0
        assert data["current_capital"] == 9800.0

    def test_calculate_from_history_sets_capital(self):
        """Test calculate_from_history computes current_capital correctly."""
        from src.trading.statistics_calculator import StatisticsCalculator
        
        trade_history = [
            {
                "timestamp": "2026-01-01T10:00:00",
                "symbol": "BTC/USDC",
                "action": "BUY",
                "price": 100000.0,
                "quantity": 0.001,
                "confidence": "HIGH",
            },
            {
                "timestamp": "2026-01-01T12:00:00",
                "symbol": "BTC/USDC",
                "action": "CLOSE_LONG",
                "price": 98000.0,
                "quantity": 0.001,
                "confidence": "HIGH",
            }
        ]
        
        initial_capital = 10000.0
        stats = StatisticsCalculator.calculate_from_history(trade_history, initial_capital)
        
        assert stats.initial_capital == initial_capital
        assert stats.current_capital == pytest.approx(initial_capital + stats.total_pnl_quote)
        assert stats.current_capital < initial_capital


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
