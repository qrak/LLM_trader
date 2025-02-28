from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from logger.logger import Logger
from utils.retry_decorator import retry_async


class DataFetcher:
    def __init__(self, exchange, logger: Logger):
        self.exchange = exchange
        self.logger: Logger = logger

    @retry_async()
    async def fetch_candlestick_data(self,
                                     pair: str,
                                     timeframe: str,
                                     limit: int,
                                     start_time: Optional[int] = None
                                     ) -> Optional[Tuple[NDArray, float]]:
        ohlcv = await self.exchange.fetch_ohlcv(pair, timeframe, since=start_time, limit=limit + 1)

        if ohlcv is None or len(ohlcv) == 0:
            self.logger.warning(f"No data returned for {pair} on {self.exchange.id}")
            return None

        ohlcv_array = np.array(ohlcv)
        closed_candles = ohlcv_array[:-1]
        latest_close = float(ohlcv_array[-1, 4])

        return closed_candles, latest_close

