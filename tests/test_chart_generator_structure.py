"""Structural characterization tests for ChartGenerator._create_simple_candlestick_chart.

The method is being split into helpers; these tests pin the figure it returns
(traces, annotation/shape counts, axis layout) on synthetic data so the split
cannot silently change the chart. Expected values were recorded from the
pre-split implementation — a difference means the refactor changed behaviour.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.analyzer.pattern_engine.chart_generator import ChartGenerator

BASELINE = json.loads(r"""{
 "explicit_timestamps": {
  "annotation_texts_head": [
   "104.50",
   "95.51",
   "95.51",
   "104.50",
   "104.50",
   "MAX: 104.50"
  ],
  "annotations": 20,
  "height": 1080,
  "shapes": 14,
  "trace_names": [
   "Price",
   "SMA 50",
   "SMA 200",
   "RSI (14)",
   "Volume",
   "CMF (20)",
   "OBV"
  ],
  "trace_types": [
   "candlestick",
   "scatter",
   "scatter",
   "scatter",
   "bar",
   "scatter",
   "scatter"
  ],
  "traces": 7,
  "width": 1920,
  "xaxis_count": 4,
  "yaxis_count": 5,
  "yaxis_tickformats": [
   ".2f"
  ]
 },
 "full": {
  "annotation_texts_head": [
   "104.50",
   "95.51",
   "95.51",
   "104.50",
   "104.50",
   "MAX: 104.50"
  ],
  "annotations": 20,
  "height": 1080,
  "shapes": 14,
  "trace_names": [
   "Price",
   "SMA 50",
   "SMA 200",
   "RSI (14)",
   "Volume",
   "CMF (20)",
   "OBV"
  ],
  "trace_types": [
   "candlestick",
   "scatter",
   "scatter",
   "scatter",
   "bar",
   "scatter",
   "scatter"
  ],
  "traces": 7,
  "width": 1920,
  "xaxis_count": 4,
  "yaxis_count": 5,
  "yaxis_tickformats": [
   ".2f"
  ]
 },
 "no_history": {
  "annotation_texts_head": [
   "104.50",
   "95.51",
   "95.51",
   "104.50",
   "104.50",
   "MAX: 104.50"
  ],
  "annotations": 17,
  "height": 1080,
  "shapes": 6,
  "trace_names": [
   "Price",
   "Volume"
  ],
  "trace_types": [
   "candlestick",
   "bar"
  ],
  "traces": 2,
  "width": 1920,
  "xaxis_count": 4,
  "yaxis_count": 5,
  "yaxis_tickformats": [
   ".2f"
  ]
 },
 "short_series": {
  "annotation_texts_head": [
   "MAX: 104.50",
   "MIN: 98.50",
   "$100",
   "2K",
   "2K",
   "<span style='color:#ff8c00'>\u2501</span> SMA 50 (Short-term trend)<br><span style='color:#9932cc'>\u2501</span> SMA 200 (Long-term trend)<br><b>Golden Cross:</b> SMA50 crosses above SMA200 = Bullish<br><b>Death Cross:</b> SMA50 crosses below SMA200 = Bearish"
  ],
  "annotations": 8,
  "height": 900,
  "shapes": 6,
  "trace_names": [
   "Price",
   "SMA 50",
   "SMA 200",
   "RSI (14)",
   "Volume",
   "CMF (20)",
   "OBV"
  ],
  "trace_types": [
   "candlestick",
   "scatter",
   "scatter",
   "scatter",
   "bar",
   "scatter",
   "scatter"
  ],
  "traces": 7,
  "width": 1600,
  "xaxis_count": 4,
  "yaxis_count": 5,
  "yaxis_tickformats": [
   ".2f"
  ]
 }
}""")


def _make_ohlcv(n=60, base=100.0):
    rows = []
    t0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
    for i in range(n):
        drift = np.sin(i / 5.0) * 3.0
        o = base + drift
        c = base + np.sin((i + 1) / 5.0) * 3.0
        h = max(o, c) + 1.5
        l = min(o, c) - 1.5
        v = 1000.0 + (i % 7) * 250.0
        ts = int((t0 + timedelta(hours=i)).timestamp() * 1000)
        rows.append([ts, o, h, l, c, v])
    return np.array(rows, dtype=float)


def _make_history(n=60):
    idx = np.arange(n, dtype=float)
    return {
        "rsi": 30.0 + 40.0 * np.abs(np.sin(idx / 6.0)),
        "sma_50": 100.0 + np.sin(idx / 9.0),
        "sma_200": 99.0 + np.cos(idx / 11.0),
        "cmf": np.sin(idx / 4.0) * 0.2,
        "obv": np.cumsum(np.sin(idx / 3.0) * 120.0),
    }


def _digest(fig):
    layout_keys = fig.layout.to_plotly_json()
    return {
        "traces": len(fig.data),
        "trace_types": [t.type for t in fig.data],
        "trace_names": [t.name for t in fig.data],
        "annotations": len(fig.layout.annotations or []),
        "annotation_texts_head": [a.text for a in (fig.layout.annotations or [])[:6]],
        "shapes": len(fig.layout.shapes or []),
        "height": fig.layout.height,
        "width": fig.layout.width,
        "xaxis_count": len([k for k in layout_keys if k.startswith("xaxis")]),
        "yaxis_count": len([k for k in layout_keys if k.startswith("yaxis")]),
        "yaxis_tickformats": [
            fig.layout[k].tickformat
            for k in sorted(layout_keys)
            if k.startswith("yaxis") and fig.layout[k].tickformat
        ],
    }


@pytest.fixture
def generator():
    return ChartGenerator()


def test_full_history_chart_matches_baseline(generator):
    fig = generator._create_simple_candlestick_chart(_make_ohlcv(), "BTC/USDC", "4h", 1080, 1920, None, _make_history())
    assert _digest(fig) == BASELINE["full"]


def test_chart_without_technical_history_matches_baseline(generator):
    fig = generator._create_simple_candlestick_chart(_make_ohlcv(), "BTC/USDC", "4h", 1080, 1920, None, None)
    assert _digest(fig) == BASELINE["no_history"]


def test_explicit_timestamps_chart_matches_baseline(generator):
    ohlcv = _make_ohlcv()
    ts_list = [datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(len(ohlcv))]
    fig = generator._create_simple_candlestick_chart(ohlcv, "BTC/USDC", "4h", 1080, 1920, ts_list, _make_history())
    assert _digest(fig) == BASELINE["explicit_timestamps"]


def test_short_series_chart_matches_baseline(generator):
    fig = generator._create_simple_candlestick_chart(_make_ohlcv(8), "ETH/USDC", "1h", 900, 1600, None, _make_history(8))
    assert _digest(fig) == BASELINE["short_series"]
