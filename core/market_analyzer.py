import configparser
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import aiohttp
import tiktoken
from ccxt import async_support as ccxt

from core.data_fetcher import DataFetcher
from core.data_persistence import DataPersistence
from core.model_manager import ModelManager
from core.trading_prompt import TradingPromptBuilder
from indicators.base.technical_indicators import TechnicalIndicators
from logger.logger import Logger
from utils.dataclass import MarketData, SentimentData, ResponseBuffer, PromptContext, \
    MarketPeriod, TechnicalSnapshot


class MarketAnalyzer:
    def __init__(self, logger: Logger, config_path: str = "config/config.ini") -> None:
        self.config = self._load_analyzer_config(config_path)
        self.exchange = ccxt.binance()
        self.symbol = self.config.get("exchange", "symbol")
        self.logger = logger
        self.model_manager = ModelManager(logger, config_path)
        self.timeframe = self.config.get("exchange", "timeframe")
        self.limit = self.config.getint("exchange", "limit")
        self.ohlcv_candles = None
        self.periods: Dict[str, MarketPeriod] = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.sentiment_data: Optional[SentimentData] = None
        self.sentiment_refresh_interval = timedelta(
            seconds=self.config.getint("trading", "sentiment_refresh_interval")
        )
        self.data_persistence = DataPersistence(logger=self.logger)
        self.data_fetcher = DataFetcher(exchange=self.exchange, logger=self.logger)
        self.ti = TechnicalIndicators()
        self.latest_technical = None

    async def close(self) -> None:
        try:
            if hasattr(self, 'exchange'):
                await self.exchange.close()
            if hasattr(self, 'model_manager'):
                await self.model_manager.close()
        except Exception as e:
            self.logger.error(f"Error during MarketAnalyzer cleanup: {e}")

    def _load_analyzer_config(self, config_path: str) -> configparser.ConfigParser:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    async def _fetch_fear_greed_index(self, limit: int = 0) -> List[Dict[str, Any]]:
        try:
            params = {"limit": limit, "format": "json"} if limit > 0 else {}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://api.alternative.me/fng/",
                        params=params,
                        timeout=10
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    if data["metadata"]["error"]:
                        raise ValueError(f"API Error: {data['metadata']['error']}")
                    return data["data"]
        except Exception as e:
            self.logger.error(f"Fear & Greed index fetch failed: {e}")
            return []

    def _map_sentiment_label(self, value: int, classification: str) -> str:
        if classification == "Extreme Greed":
            return "extremely_bullish"
        elif classification == "Greed":
            return "bullish"
        elif classification == "Fear":
            return "bearish"
        elif classification == "Extreme Fear":
            return "extremely_bearish"
        elif value > 60:
            return "slightly_bullish"
        elif value < 40:
            return "slightly_bearish"
        return "neutral"

    async def fetch_ohlcv(self) -> List[MarketData]:
        try:
            result = await self.data_fetcher.fetch_candlestick_data(
                pair=self.symbol,
                timeframe=self.timeframe,
                limit=self.limit
            )

            if result is None:
                raise RuntimeError("No data returned from exchange")

            self.ohlcv_candles, _ = result
            await self._calculate_technical_indicators()
            fear_greed_data = await self._fetch_fear_greed_index(limit=7)
            
            data = self._process_market_data(fear_greed_data)
            self._update_periods(data)
            return data

        except Exception as e:
            self.logger.exception(f"OHLCV fetch failed: {str(e)}")
            raise RuntimeError(f"OHLCV fetch failed: {str(e)}")

    async def _calculate_technical_indicators(self) -> None:
        self.ti.get_data(self.ohlcv_candles)
        
        # Calculate all indicators
        vwap_values = self.ti.vol.rolling_vwap(length=14)
        twap_values = self.ti.vol.twap(length=14)
        mfi_values = self.ti.vol.mfi(length=14)
        obv_values = self.ti.vol.obv(length=14)
        cmf_values = self.ti.vol.chaikin_money_flow(length=20)
        force_index_values = self.ti.vol.force_index(length=13)
        rsi_values = self.ti.momentum.rsi(length=14)
        macd_line, macd_signal, macd_hist = self.ti.momentum.macd()
        stoch_k, stoch_d = self.ti.momentum.stochastic()
        williams_r_values = self.ti.momentum.williams_r(length=14)
        adx, plus_di, minus_di = self.ti.trend.adx()
        supertrend, supertrend_direction = self.ti.trend.supertrend()
        psar_values = self.ti.trend.parabolic_sar()
        atr_values = self.ti.volatility.atr(length=14)
        bb_upper, bb_middle, bb_lower = self.ti.volatility.bollinger_bands()
        kurtosis_values = self.ti.statistical.kurtosis(length=30)
        zscore_values = self.ti.statistical.zscore(length=20)
        hurst_values = self.ti.statistical.hurst(max_lag=20)

        # Update technical snapshot
        self.latest_technical = TechnicalSnapshot(
            vwap_5m=vwap_values[-1],
            twap=twap_values[-1],
            mfi_14=mfi_values[-1],
            obv=obv_values[-1],
            cmf=cmf_values[-1],
            force_index=force_index_values[-1],
            rsi_5m_14=rsi_values[-1],
            macd_line=macd_line[-1],
            macd_signal=macd_signal[-1],
            macd_hist=macd_hist[-1],
            stoch_k=stoch_k[-1],
            stoch_d=stoch_d[-1],
            williams_r=williams_r_values[-1],
            adx=adx[-1],
            plus_di=plus_di[-1],
            minus_di=minus_di[-1],
            supertrend=supertrend[-1],
            supertrend_direction=supertrend_direction[-1],
            psar=psar_values[-1],
            atr_5m_14=atr_values[-1],
            bb_upper=bb_upper[-1],
            bb_middle=bb_middle[-1],
            bb_lower=bb_lower[-1],
            hurst=hurst_values[-1],
            kurtosis=kurtosis_values[-1],
            zscore=zscore_values[-1]
        )

    def _process_market_data(self, fear_greed_data: List[Dict[str, Any]]) -> List[MarketData]:
        fear_greed_map = {
            int(item["timestamp"]): {
                "value": int(item["value"]),
                "classification": item["value_classification"]
            }
            for item in fear_greed_data
        }

        data: List[MarketData] = []
        for idx in range(len(self.ohlcv_candles)):
            timestamp = datetime.fromtimestamp(float(self.ohlcv_candles[idx, 0]) / 1000.0)
            day_start = datetime(timestamp.year, timestamp.month, timestamp.day)
            day_timestamp = int(day_start.timestamp())

            sentiment = self._process_sentiment(day_timestamp, fear_greed_map)
            market_data = MarketData(
                timestamp=timestamp,
                open=float(self.ohlcv_candles[idx, 1]),
                high=float(self.ohlcv_candles[idx, 2]),
                low=float(self.ohlcv_candles[idx, 3]),
                close=float(self.ohlcv_candles[idx, 4]),
                volume=float(self.ohlcv_candles[idx, 5]),
                sentiment=sentiment
            )
            data.append(market_data)
        return data

    def _process_sentiment(
            self,
            day_timestamp: int,
            fear_greed_map: Dict[int, Dict[str, Any]]
    ) -> Optional[SentimentData]:
        sentiment = None
        for fg_timestamp, fg_data in fear_greed_map.items():
            if abs(fg_timestamp - day_timestamp) < 86400:  # Within 24 hours
                sentiment = SentimentData(
                    timestamp=datetime.fromtimestamp(fg_timestamp),
                    fear_greed_index=int(fg_data["value"]),
                    value_classification=fg_data["classification"],
                    sentiment_label=self._map_sentiment_label(
                        int(fg_data["value"]),
                        fg_data["classification"]
                    )
                )
                break
        return sentiment

    def _update_periods(self, data: List[MarketData]) -> None:
        if len(data) >= 288:
            self.periods["1D"] = MarketPeriod(data[-288:], "1D")

        if len(data) >= 432:
            self.periods["2D"] = MarketPeriod(data[-432:], "2D")

        self.periods["3D"] = MarketPeriod(data, "3D")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _clean_response(self, text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    async def analyze_trend(self, data: List[MarketData]) -> str:
        if not data:
            return "HOLD"
            
        prompt_context = PromptContext(
            symbol=self.symbol,
            current_price=data[-1].close,
            current_position=getattr(self, 'current_position', None),
            trade_history=self.data_persistence.load_last_n_decisions(5),
            previous_response=self.data_persistence.load_previous_response() or "No previous analysis available.",
            technical_data=self.latest_technical,
            market_metrics=self.periods,
            ohlcv_candles=self.ohlcv_candles,
            sentiment=next((d.sentiment for d in reversed(data) if d.sentiment is not None), None)
        )

        prompt = TradingPromptBuilder(self.logger).build_prompt(prompt_context)
        self.logger.info(f"Prompt Token Count({self.count_tokens(prompt)})")
        
        try:
            return await self._send_data_to_ai_model(prompt)
        except Exception as e:
            self.logger.exception(f"Analysis failed: {e}")
            return "HOLD"

    async def _send_data_to_ai_model(self, prompt: str) -> str:
        try:
            buffer = ResponseBuffer()
            response = await self.model_manager.send_prompt(prompt, buffer)
            cleaned_response = self._clean_response(response)
            response_token_count = self.count_tokens(response)
            self.logger.info(f"\nResponse Token Count: {response_token_count}")
            self.data_persistence.save_previous_response(cleaned_response)
            return cleaned_response
        except Exception as e:
            self.logger.exception(f"Model analysis failed: {str(e)}")
            return "HOLD"