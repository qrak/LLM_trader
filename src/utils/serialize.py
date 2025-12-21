"""
JSON serialization utilities for converting Python/NumPy objects to JSON-serializable types.

This module provides utilities for recursively converting complex Python objects
(particularly NumPy arrays and scalars) into JSON-serializable formats.
"""

from typing import Any, Dict, List, Union


def serialize_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy arrays/scalars and other non-JSON types to JSON-serializable types.
    
    This function handles:
    - Dictionaries (recursively processes values)
    - Lists and tuples (recursively processes elements)
    - NumPy arrays (converts to list via .tolist())
    - NumPy scalars (converts to Python primitives via .item())
    - Primitive types (str, int, float, bool, None) - returned as-is
    - Other objects - converted to string representation
    
    Args:
        obj: Object to serialize (can be of any type)
        
    Returns:
        JSON-serializable representation of the object
        
    Examples:
        >>> import numpy as np
        >>> serialize_for_json(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> serialize_for_json({"data": np.float64(3.14)})
        {"data": 3.14}
        >>> serialize_for_json({"nested": {"values": [np.int32(42)]}})
        {"nested": {"values": [42]}}
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    # Dict - recursively process values
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}

    # List/Tuple - recursively process elements
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(v) for v in obj]

    # Numpy array - convert to list
    if _np is not None and isinstance(obj, _np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            # Fallback: recursively process elements
            return [serialize_for_json(v) for v in obj]

    # Numpy scalar - convert to Python primitive
    if _np is not None and isinstance(obj, _np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)

    # Primitive types - return as-is
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fallback to string representation for any other type
    try:
        return str(obj)
    except Exception:
        return None


def safe_tolist(obj: Any) -> Union[List, Any]:
    """
    Safely convert an object to a list if it has a .tolist() method, otherwise return as-is.
    
    This is a convenience function for handling NumPy arrays and similar objects
    that may or may not have a .tolist() method.
    
    Args:
        obj: Object to convert
        
    Returns:
        List representation if .tolist() is available, otherwise the original object
        
    Examples:
        >>> import numpy as np
        >>> safe_tolist(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> safe_tolist([1, 2, 3])
        [1, 2, 3]
        >>> safe_tolist("not_an_array")
        "not_an_array"
    """
    if hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj
