import dataclasses
import numpy as np
from datetime import datetime
from typing import Any, Dict, Type, TypeVar, get_args, get_origin, List, Union, Optional
from numpy.typing import NDArray

T = TypeVar("T", bound="SerializableMixin")


def get_last_valid_value(
    arr: Union[NDArray, float, int, None],
    default: Optional[float] = None
) -> Optional[float]:
    """Extract the last non-NaN value from an array or return scalar directly.

    Args:
        arr: Numpy array, scalar value, or None.
        default: Value to return if no valid value found.

    Returns:
        Last valid (non-NaN) float value, or default if none found.
    """
    if arr is None:
        return default

    if not hasattr(arr, '__iter__') or isinstance(arr, str):
        if isinstance(arr, (int, float)) and not np.isnan(arr):
            return float(arr)
        return default

    if len(arr) == 0:
        return default

    try:
        if isinstance(arr, list):
            arr = np.array(arr, dtype=float)
        elif hasattr(arr, 'dtype') and arr.dtype == object:
            arr = arr.astype(float)
    except (ValueError, TypeError):
        return default

    valid_indices = np.where(~np.isnan(arr))[0]
    if len(valid_indices) > 0:
        return float(arr[valid_indices[-1]])
    return default


def get_last_n_valid(arr: NDArray, n: int) -> NDArray:
    """Extract last N valid (non-NaN) values from array.

    Args:
        arr: Numpy array with potential NaN values.
        n: Number of valid values to extract from the end.

    Returns:
        Array containing up to n valid values from the end.
    """
    if arr is None or len(arr) == 0:
        return np.array([])

    try:
        if isinstance(arr, list):
            arr = np.array(arr, dtype=float)
        elif hasattr(arr, 'dtype') and arr.dtype == object:
            arr = arr.astype(float)
    except (ValueError, TypeError):
        return np.array([])

    valid_mask = ~np.isnan(arr)
    valid_data = arr[valid_mask]
    return valid_data[-n:] if len(valid_data) >= n else valid_data


def safe_array_to_scalar(
    arr: Union[NDArray, float, int, None],
    index: int = -1,
    default: Optional[float] = None
) -> Optional[float]:
    """Safely extract a scalar value from an array at given index.

    Args:
        arr: Numpy array, scalar value, or None.
        index: Index to extract from array (default: -1 for last element).
        default: Value to return if extraction fails.

    Returns:
        Float value at index, or default if invalid.
    """
    if arr is None:
        return default

    if not hasattr(arr, '__iter__') or isinstance(arr, str):
        if isinstance(arr, (int, float)) and not np.isnan(arr):
            return float(arr)
        return default

    if len(arr) == 0:
        return default

    try:
        value = arr[index]
        if value is None:
            return default
        
        val_float = float(value)
        if np.isnan(val_float):
            return default
        return val_float
    except (IndexError, TypeError, ValueError):
        return default


class SerializableMixin:
    """Mixin to add JSON serialization/deserialization to dataclasses.
    
    Handles:
    - datetime objects (ISO format)
    - Nested dataclasses (recursive conversion)
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary with ISO format dates."""
        def _dict_factory(data: List[tuple[str, Any]]) -> Dict[str, Any]:
            result = {}
            for key, value in data:
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
            return result

        return dataclasses.asdict(self, dict_factory=_dict_factory)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create dataclass instance from dictionary, handling types."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass to use SerializableMixin")

        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        init_args = {}

        for key, value in data.items():
            if key not in field_types:
                continue

            target_type = field_types[key]
            init_args[key] = cls._convert_value(value, target_type)

        return cls(**init_args)

    @staticmethod
    def _convert_value(value: Any, target_type: Type) -> Any:
        """Recursively convert values to match target types."""
        if value is None:
            return None

        origin = get_origin(target_type)
        args = get_args(target_type)

        # Handle Optional[T] (Union[T, None])
        if origin is Union and type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return SerializableMixin._convert_value(value, non_none_args[0])

        # Handle List[T]
        if origin is list or origin is List:
            item_type = args[0]
            if isinstance(value, list):
                return [SerializableMixin._convert_value(item, item_type) for item in value]

        # Handle datetime
        if target_type is datetime and isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return value

        # Handle nested dataclasses
        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            if issubclass(target_type, SerializableMixin):
                return target_type.from_dict(value)
            else:
                return target_type(**value)

        return value
