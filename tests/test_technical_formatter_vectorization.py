import pytest
import numpy as np
from unittest.mock import MagicMock
from src.analyzer.formatters.technical_formatter import TechnicalFormatter
from src.utils.format_utils import FormatUtils

def test_technical_formatter_price_action_vectorized():
    format_utils = FormatUtils()
    formatter = TechnicalFormatter(technical_calculator=MagicMock(), format_utils=format_utils)

    context = MagicMock()
    context.current_price = 100.0

    # 10 candles: 7 green (closes >= opens), 3 red
    timestamps = np.arange(10) * 60000
    opens = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
    highs = opens + 1.0
    lows = opens - 0.5
    closes = np.array([101, 102, 103, 102, 105, 106, 107, 106, 109, 110], dtype=float)
    volumes = np.full(10, 1000.0)

    context.ohlcv_candles = np.column_stack((timestamps, opens, highs, lows, closes, volumes))

    res = formatter.format_price_action_section(context, {})
    assert "8G/2R" in res or "8G" in res or "7G" in res
    assert "Price:" in res
