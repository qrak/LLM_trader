from typing import Union, List

import numpy as np
import pandas as pd

from .indicator_base import IndicatorBase
from .indicator_categories import (
    MomentumIndicators, OverlapIndicators, PriceTransformIndicators,
    SentimentIndicators, StatisticalIndicators, SupportResistanceIndicators,
    TrendIndicators, VolatilityIndicators, VolumeIndicators
)


class TechnicalIndicators:
    def __init__(self, measure_time: bool = False, save_to_csv: bool = False) -> None:
        self._base = IndicatorBase(measure_time=measure_time, save_to_csv=save_to_csv)
        self.overlap = OverlapIndicators(self._base)
        self.momentum = MomentumIndicators(self._base, self.overlap)
        self.price = PriceTransformIndicators(self._base)
        self.sentiment = SentimentIndicators(self._base)
        self.statistical = StatisticalIndicators(self._base)
        self.support_resistance = SupportResistanceIndicators(self._base)
        self.trend = TrendIndicators(self._base)
        self.volatility = VolatilityIndicators(self._base)
        self.vol = VolumeIndicators(self._base)

    @property
    def open(self) -> np.ndarray:
        return self._base.open

    @property
    def high(self) -> np.ndarray:
        return self._base.high

    @property
    def low(self) -> np.ndarray:
        return self._base.low

    @property
    def close(self) -> np.ndarray:
        return self._base.close

    @property
    def volume(self) -> np.ndarray:
        return self._base.volume

    def get_data(self, data: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]) -> None:
        self._base.get_data(data)
