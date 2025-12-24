"""Test brain learning features for V2 upgrade."""

import pytest
from datetime import datetime, timedelta

from src.trading.dataclasses import (
    Position, TradingBrain, TradingInsight, FactorStats, ConfidenceStats
)


class TestFactorStats:
    """Test FactorStats dataclass."""
    
    def test_factor_stats_update(self):
        """Test that factor stats update correctly."""
        stats = FactorStats(factor_name="volume_support", bucket="HIGH")
        
        # First trade: win
        stats.update(is_win=True, pnl_pct=2.5, score=85)
        assert stats.total_trades == 1
        assert stats.winning_trades == 1
        assert stats.win_rate == 100.0
        assert stats.avg_score == 85.0
        assert stats.avg_pnl_pct == 2.5
        
        # Second trade: loss
        stats.update(is_win=False, pnl_pct=-1.5, score=75)
        assert stats.total_trades == 2
        assert stats.winning_trades == 1
        assert stats.win_rate == 50.0
        assert stats.avg_score == 80.0  # (85+75)/2
        assert stats.avg_pnl_pct == 0.5  # (2.5-1.5)/2
    
    def test_factor_stats_serialization(self):
        """Test FactorStats to_dict and from_dict."""
        stats = FactorStats(factor_name="trend_alignment", bucket="MEDIUM")
        stats.update(is_win=True, pnl_pct=3.0, score=55)
        
        data = stats.to_dict()
        restored = FactorStats.from_dict(data)
        
        assert restored.factor_name == stats.factor_name
        assert restored.bucket == stats.bucket
        assert restored.total_trades == stats.total_trades
        assert restored.win_rate == stats.win_rate


class TestTradingBrainWithFactors:
    """Test TradingBrain with factor performance tracking."""
    
    def test_brain_factor_performance_serialization(self):
        """Test that factor_performance is serialized and restored."""
        brain = TradingBrain()
        
        # Add factor stats
        key = "volume_support_HIGH"
        brain.factor_performance[key] = FactorStats(
            factor_name="volume_support", bucket="HIGH"
        )
        brain.factor_performance[key].update(is_win=True, pnl_pct=2.0, score=80)
        
        # Serialize and restore
        data = brain.to_dict()
        restored = TradingBrain.from_dict(data)
        
        assert key in restored.factor_performance
        assert restored.factor_performance[key].win_rate == 100.0
        assert restored.factor_performance[key].total_trades == 1
    
    def test_brain_min_sample_size_default(self):
        """Test that min_sample_size defaults to 5."""
        brain = TradingBrain()
        assert brain.min_sample_size == 5


class TestPositionWithConfluenceFactors:
    """Test Position dataclass with confluence_factors."""
    
    def test_position_with_confluence_factors(self):
        """Test Position creation with confluence factors."""
        factors = (
            ("volume_support", 85.0),
            ("trend_alignment", 72.0),
            ("pattern_quality", 65.0),
        )
        
        position = Position(
            entry_price=100000.0,
            stop_loss=98000.0,
            take_profit=105000.0,
            size=0.02,
            entry_time=datetime.now(),
            confidence="HIGH",
            direction="LONG",
            symbol="BTC/USDT",
            confluence_factors=factors,
        )
        
        assert position.confluence_factors == factors
        assert len(position.confluence_factors) == 3
        assert position.confluence_factors[0] == ("volume_support", 85.0)
    
    def test_position_default_empty_factors(self):
        """Test Position defaults to empty tuple for confluence_factors."""
        position = Position(
            entry_price=100000.0,
            stop_loss=98000.0,
            take_profit=105000.0,
            size=0.02,
            entry_time=datetime.now(),
            confidence="MEDIUM",
            direction="SHORT",
            symbol="ETH/USDT",
        )
        
        assert position.confluence_factors == ()


class TestTimeDecay:
    """Test time decay calculations for insights."""
    
    def test_time_decay_calculation(self):
        """Test that decay calculation works correctly."""
        now = datetime.now()
        
        # Insight from today: weight should be ~1.0
        recent_insight = TradingInsight(
            lesson="Test lesson",
            category="STOP_LOSS",
            condition="Test",
            trade_count=5,
            last_validated=now,
            confidence_impact="HIGH"
        )
        
        weeks_old = (now - recent_insight.last_validated).days / 7
        decay_weight = 0.95 ** weeks_old
        assert decay_weight > 0.99  # Should be very close to 1.0
        
        # Insight from 4 weeks ago: weight should be ~0.81
        old_insight = TradingInsight(
            lesson="Old lesson",
            category="ENTRY_TIMING",
            condition="Test",
            trade_count=5,
            last_validated=now - timedelta(weeks=4),
            confidence_impact="MEDIUM"
        )
        
        weeks_old = (now - old_insight.last_validated).days / 7
        decay_weight = 0.95 ** weeks_old
        assert 0.80 < decay_weight < 0.83  # ~0.81 for 4 weeks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
