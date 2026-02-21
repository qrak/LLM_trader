import dataclasses
import numpy as np
from datetime import datetime
from typing import Any, Dict, Type, TypeVar, get_args, get_origin, List, Union, Optional
from numpy.typing import NDArray

T = TypeVar("T")


def get_last_valid_value(
    arr: NDArray,
    default: Optional[float] = None
) -> Optional[float]:
    """Extract the last non-NaN value from a numpy array.

    Args:
        arr: Numpy array (float or object dtype).
        default: Value to return if no valid value found.

    Returns:
        Last valid (non-NaN) float value, or default if none found.
    """
    # Handle scalar values directly
    if isinstance(arr, (int, float)):
        return float(arr) if not np.isnan(arr) else default

    if len(arr) == 0:
        return default

    # If it's an object array, ensure we try to convert to float
    if arr.dtype == object:
        try:
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
    if len(arr) == 0:
        return np.array([])

    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except (ValueError, TypeError):
            return np.array([])

    valid_mask = ~np.isnan(arr)
    valid_data = arr[valid_mask]

    return valid_data[-n:] if len(valid_data) >= n else valid_data


def safe_array_to_scalar(
    arr: NDArray,
    index: int = -1,
    default: Optional[float] = None
) -> Optional[float]:
    """Safely extract a scalar value from an array at given index.

    Args:
        arr: Numpy array.
        index: Index to extract from array (default: -1 for last element).
        default: Value to return if extraction fails.

    Returns:
        Float value at index, or default if invalid/NaN.
    """
    if len(arr) == 0:
        return default

    try:
        val = arr[index]
        val_float = float(val)

        if np.isnan(val_float):
            return default

        return val_float
    except (IndexError, TypeError, ValueError):
        return default


def get_indicator_value(td: dict, key: str) -> Union[float, str]:
    """Get indicator value with proper type checking and error handling.

    Args:
        td: Technical data dictionary
        key: Indicator key to retrieve

    Returns:
        float or str: Indicator value or 'N/A' if invalid
    """
    try:
        value = td[key]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return float(value[0])
        if isinstance(value, (list, tuple)) and len(value) > 1:
            return float(value[-1])
        return 'N/A'
    except (KeyError, TypeError, ValueError, IndexError):
        return 'N/A'


def serialize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(v) for v in obj]

    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            # Fallback for complex/mixed arrays
            return [serialize_for_json(v) for v in obj]

    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)

    # Handle NaN/Inf float values which are standard in NumPy but invalid in JSON
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    # Primitive types pass through
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj

    # Last resort fallback
    try:
        return str(obj)
    except Exception:
        return None


def safe_tolist(obj: Any) -> Union[List, Any]:
    """Safely convert an object to a list if it has a .tolist() method."""
    if hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj


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
