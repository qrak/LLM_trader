"""Array utility functions for handling numpy arrays and extracting valid values.

Consolidates common patterns found across formatters and calculators.
"""

import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray


def get_last_valid_value(
    arr: Union[NDArray, float, int, None],
    default: Optional[float] = None
) -> Optional[float]:
    """Extract the last non-NaN value from an array or return scalar directly.

    Consolidates the pattern `hasattr(__iter__)` + `np.where(~np.isnan())` found in:
    - technical_formatter.py (_format_sma_structure, _format_td_sequential, etc.)
    - technical_calculator.py (_add_signal_interpretations, etc.)

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

    # Ensure array is float type to handle None as NaN
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
        Returns fewer if insufficient valid values exist.
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

    Handles both array and scalar inputs uniformly.

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
