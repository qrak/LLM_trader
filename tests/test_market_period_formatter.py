from unittest.mock import MagicMock

from src.analyzer.formatters.market_period_formatter import MarketPeriodFormatter


def _make_formatter():
    format_utils = MagicMock()
    format_utils.fmt.side_effect = lambda value: f"{value:.2f}" if isinstance(value, float) else str(value)
    return MarketPeriodFormatter(format_utils=format_utils)


def test_format_market_period_metrics_shows_partial_period_label():
    formatter = _make_formatter()
    market_metrics = {
        "30D": {
            "metrics": {
                "period": "30D (Partial)",
                "avg_price": 77123.66,
                "lowest_price": 74135.99,
                "highest_price": 79515.00,
                "price_change_percent": 2.08416868,
                "total_volume": 60209.40,
            },
            "indicator_changes": {},
        }
    }

    result = formatter.format_market_period_metrics(market_metrics)

    assert "30D (PARTIAL):" in result
    assert "\n30D:" not in result


def test_format_market_period_metrics_keeps_full_period_key_when_no_override():
    formatter = _make_formatter()
    market_metrics = {
        "7D": {
            "metrics": {
                "avg_price": 77311.09,
                "lowest_price": 74922.58,
                "highest_price": 79505.79,
                "price_change_percent": -2.19448997,
                "total_volume": 33472.38,
            },
            "indicator_changes": {},
        }
    }

    result = formatter.format_market_period_metrics(market_metrics)

    assert "7D:" in result
    assert "7D (PARTIAL):" not in result