from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, ClassVar, Tuple

import numpy as np


@dataclass
class SentimentData:
    timestamp: datetime
    fear_greed_index: int
    value_classification: str
    sentiment_label: str


@dataclass
class MarketData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    sentiment: Optional[SentimentData] = None
    
@dataclass
class TechnicalSnapshot:
    vwap_5m: float
    twap: float
    mfi_14: float
    obv: float
    cmf: float
    force_index: float
    
    rsi_5m_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    
    adx: float
    plus_di: float
    minus_di: float
    supertrend: float
    supertrend_direction: float
    psar: float
    
    atr_5m_14: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    
    hurst: float
    kurtosis: float
    zscore: float

@dataclass
class Position:
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    entry_time: datetime
    confidence: str
    direction: str

@dataclass
class TradeDecision:
    timestamp: datetime
    action: str
    price: float
    confidence: str
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str

@dataclass
class MarketPeriod:
    data: List[MarketData]
    period_name: str
    metrics: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.metrics = self._calculate_period_metrics()

    def _calculate_period_metrics(self) -> Dict[str, float]:
        if not self.data:
            return {}

        closes = [d.close for d in self.data]
        volumes = [d.volume for d in self.data]
        highs = [d.high for d in self.data]
        lows = [d.low for d in self.data]

        return {
            "price_change": (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] != 0 else 0.0,
            "volume_avg": sum(volumes) / len(volumes),
            "price_volatility": (max(closes) - min(closes)) / (min(closes) if min(closes) != 0 else 1) * 100,
            "highest_price": max(highs),
            "lowest_price": min(lows),
            "avg_range": sum([h - l for h, l in zip(highs, lows)]) / len(highs),
            "price_momentum": sum([c2 - c1 for c1, c2 in zip(closes[:-1], closes[1:])]) / len(closes),
            "candle_count": len(closes)
        }

@dataclass
class PromptContext:
    symbol: str
    ohlcv_candles: np.ndarray[MarketData]
    current_price: float
    technical_data: TechnicalSnapshot
    market_metrics: Dict[str, MarketPeriod]
    current_position: Optional[Position]
    trade_history: List[Dict[str, Any]]
    previous_response: Optional[str]
    sentiment: Optional[SentimentData]


@dataclass
class ResponseBuffer:
    full_response: str = ""
    current_chunk: str = ""
    thinking_mode: bool = False
    analysis_mode: bool = False
    buffer_size: int = 100
    timestamp_logged: bool = False
    last_header: Optional[str] = None
    last_header_time: Optional[datetime] = None
    _pending_mode_switch: bool = False

    def should_flush(self) -> bool:
        return bool(
            len(self.current_chunk) >= self.buffer_size
            or "</think>" in self.current_chunk
            or "\n" in self.current_chunk
        )

    def _should_show_header(self, header_type: str) -> bool:
        current_time = datetime.now()
        if not self.last_header_time:
            self.last_header_time = current_time
            return True
        if header_type != self.last_header:
            time_diff = (current_time - self.last_header_time).total_seconds()
            if time_diff >= 1:
                self.last_header_time = current_time
                return True
        return False

    def process_chunk(self, content: str) -> Tuple[str, bool]:
        if "<think>" in content and not self.thinking_mode:
            self.thinking_mode = True
            self.timestamp_logged = False
            if self._should_show_header("Thinking Process"):
                self.last_header = "Thinking Process"
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return f"\n[bold cyan]=== Thinking Process ({timestamp}) ===[/bold cyan]\n", True
            return "", False
        if "</think>" in content and self.thinking_mode:
            self.thinking_mode = False
            self.analysis_mode = True
            if self._should_show_header("Analysis Results"):
                self.last_header = "Analysis Results"
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return f"\n[bold green]=== Analysis Results ({timestamp}) ===[/bold green]\n", False
            return "", False
        return content, False

    def reset(self) -> None:
        self.current_chunk = ""
        self.thinking_mode = False
        self.analysis_mode = False
        self.timestamp_logged = False
        self._pending_mode_switch = False

@dataclass
class TimeframeConfig:
    TIMEFRAME_MAP: ClassVar[Dict[str, int]] = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "3h": 10800,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
        "1w": 604800,
    }

    @classmethod
    def get_seconds(cls, timeframe: str) -> int:
        return cls.TIMEFRAME_MAP.get(timeframe, 300)
