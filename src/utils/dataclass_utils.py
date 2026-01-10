"""Utility for dataclass serialization and handling."""

import dataclasses
from datetime import datetime
from typing import Any, Dict, Type, TypeVar, get_args, get_origin, List, Union

T = TypeVar("T", bound="SerializableMixin")


class SerializableMixin:
    """Mixin to add JSON serialization/deserialization to dataclasses.
    
    Handles:
    - datetime objects (ISO format)
    - Nested dataclasses (recursive conversion)
    - Enum validation (optional, can be expanded)
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

        # Handle nested dataclasses containing the mixin or just standard dataclasses
        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            if issubclass(target_type, SerializableMixin):
                return target_type.from_dict(value)
            else:
                # Basic dataclass fallback (simple kwargs construction)
                # Ideally, they should also use the mixin.
                return target_type(**value)

        return value
