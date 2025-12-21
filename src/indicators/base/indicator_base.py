import timeit
from dataclasses import dataclass
from typing import Union, Tuple, Optional, TypeVar, Generic, Callable, List, Any, Dict

import numpy as np
import pandas as pd

T = TypeVar('T')


@dataclass(kw_only=True)
class IndicatorBase:
    measure_time: bool = False
    save_to_csv: bool = False
    NUM_COLUMNS: int = 5

    def __post_init__(self) -> None:
        self.timestamp: Optional[np.ndarray] = None
        self._initialize_arrays()

    def _initialize_arrays(self) -> None:
        self.open = np.array([], dtype=np.float64)
        self.high = np.array([], dtype=np.float64)
        self.low = np.array([], dtype=np.float64)
        self.close = np.array([], dtype=np.float64)
        self.volume = np.array([], dtype=np.float64)

    def get_data(self, new_data: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]) -> None:
        if isinstance(new_data, pd.DataFrame):
            self._handle_dataframe(new_data)
        elif isinstance(new_data, np.ndarray):
            self._handle_numpy_array(new_data)
        elif isinstance(new_data, list):
            self._handle_list(new_data)
        else:
            raise TypeError("Data must be a Pandas DataFrame, NumPy array, or List")

    def calculate_indicator(self, func: Callable, *args: Any, required_length: int = 0, **kwargs: Any) -> Any:
        if not len(self.close):
            raise ValueError("Data not initialized. Call get_data() first.")

        if len(self.close) < required_length:
            raise ValueError(
                f"Insufficient data. Need at least {required_length} data points, but only have {len(self.close)}.")

        if self.measure_time:
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            print(f"{func.__name__} took {timeit.default_timer() - start_time:.4f} seconds")
        else:
            result = func(*args, **kwargs)

        if self.save_to_csv:
            self._save_indicator_result_to_csv(func.__name__, result)

        return result

    def _save_indicator_result_to_csv(self, indicator_name: str, indicator_result: Union[np.ndarray, Tuple]) -> None:
        data: Dict[str, np.ndarray] = {}
        n = len(self.close)

        if self.timestamp is not None:
            data['timestamp'] = self.timestamp
        data['open'] = self.open.ravel()  # Ensure 1D array
        data['high'] = self.high.ravel()
        data['low'] = self.low.ravel()
        data['close'] = self.close.ravel()
        data['volume'] = self.volume.ravel()

        def add_indicator_to_data(key: str, array: np.ndarray) -> None:
            array = np.asarray(array)
            if array.ndim == 1:
                if len(array) == n:
                    data[key] = array
                elif len(array) == 1:
                    data[key] = np.full(n, array[0])
                else:
                    raise ValueError(f"Indicator result for {key} has invalid length {len(array)}; expected {n}")
            elif array.ndim == 2:
                if array.shape[0] == n:
                    for idx in range(array.shape[1]):
                        data[f"{key}_{idx}"] = array[:, idx]
                elif array.shape[1] == n:
                    for idx in range(array.shape[0]):
                        data[f"{key}_{idx}"] = array[idx, :]
                else:
                    raise ValueError(
                        f"Indicator result for {key} has invalid shape {array.shape}; expected ({n}, m) or (m, {n})")
            else:
                raise ValueError(f"Indicator result for {key} has invalid number of dimensions: {array.ndim}")

        if isinstance(indicator_result, tuple):
            for i, result_array in enumerate(indicator_result):
                add_indicator_to_data(f"{indicator_name}_{i}", result_array)
        else:
            add_indicator_to_data(indicator_name, indicator_result)

        df = pd.DataFrame(data)
        csv_filename = f"{indicator_name}_results.csv"
        df.to_csv(csv_filename, index=False)

    def _handle_list(self, data: List[List[Union[int, float]]]) -> None:
        if not data or not isinstance(data[0], list):
            raise ValueError("Input must be a non-empty list of lists")

        num_cols = len(data[0])
        expected_cols = self.NUM_COLUMNS
        expected_cols_with_timestamp = self.NUM_COLUMNS + 1

        array = np.array(data, dtype=np.float64)

        if num_cols == expected_cols_with_timestamp:
            self.timestamp = array[:, 0]
            data_slice = array[:, 1:]
        elif num_cols == expected_cols:
            self.timestamp = None
            data_slice = array
        else:
            raise ValueError(f"Each list must contain {expected_cols} or {expected_cols_with_timestamp} elements")

        self.open = data_slice[:, 0]
        self.high = data_slice[:, 1]
        self.low = data_slice[:, 2]
        self.close = data_slice[:, 3]
        self.volume = data_slice[:, 4]

    def _handle_dataframe(self, dataframe: pd.DataFrame) -> None:
        col_mapping = {col.lower(): col for col in dataframe.columns}
        expected_cols = {'open', 'high', 'low', 'close', 'volume'}
        timestamp_cols = {'timestamp', 'date', 'datetime', 'time'}

        if missing_cols := expected_cols - set(col_mapping.keys()):
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        arrays = np.column_stack([
            dataframe[col_mapping[col]].to_numpy(dtype=np.float64).reshape(-1)
            for col in ['open', 'high', 'low', 'close', 'volume']
        ])

        self.open, self.high, self.low, self.close, self.volume = arrays.T

        if timestamp_col := next((col_mapping[col] for col in timestamp_cols if col in col_mapping), None):
            self.timestamp = pd.to_datetime(dataframe[timestamp_col]).to_numpy()

    def _handle_numpy_array(self, array: np.ndarray) -> None:
        if array.ndim != 2:
            raise ValueError("NumPy array must be 2-dimensional")

        expected_cols = self.NUM_COLUMNS
        num_cols = array.shape[1]

        if num_cols == expected_cols + 1:
            self.timestamp = array[:, 0]
            data = array[:, 1:]
        elif num_cols == expected_cols:
            self.timestamp = None
            data = array
        else:
            raise ValueError(f"NumPy array must have {expected_cols} or {expected_cols + 1} columns")

        self.open = data[:, 0].reshape(-1)
        self.high = data[:, 1].reshape(-1)
        self.low = data[:, 2].reshape(-1)
        self.close = data[:, 3].reshape(-1)
        self.volume = data[:, 4].reshape(-1)


class IndicatorCategory(Generic[T]):
    def __init__(self, base: IndicatorBase) -> None:
        self._base = base

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
