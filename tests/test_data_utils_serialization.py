"""Test Data Utils Serialization unit tests.

Tests for test_data_utils_serialization.py.
"""
from dataclasses import dataclass

from src.utils.data_utils import SerializableMixin


@dataclass
class TupleModel(SerializableMixin):
    pairs: tuple[tuple[str, float], ...]


def test_from_dict_restores_nested_tuple_fields():
    model = TupleModel.from_dict({"pairs": [["trend", 80.0], ["volume", 55.5]]})

    assert model.pairs == (("trend", 80.0), ("volume", 55.5))
    assert isinstance(model.pairs, tuple)
    assert isinstance(model.pairs[0], tuple)


def test_serialize_for_json_primitives_and_numpy():
    import numpy as np

    from src.utils.data_utils import serialize_for_json

    data = {
        "str": "hello",
        "int": 10,
        "bool": True,
        "none": None,
        "float": 3.14,
        "nan": float("nan"),
        "inf": float("inf"),
        "np_int": np.int64(42),
        "np_float": np.float64(3.14),
        "np_nan": np.float64(np.nan),
        "array": np.array([1.0, 2.0]),
    }

    serialized = serialize_for_json(data)

    assert serialized["str"] == "hello"
    assert serialized["int"] == 10
    assert serialized["bool"] is True
    assert serialized["none"] is None
    assert serialized["float"] == 3.14
    assert serialized["nan"] is None
    assert serialized["inf"] is None
    assert serialized["np_int"] == 42
    assert serialized["np_float"] == 3.14
    assert serialized["np_nan"] is None
    assert serialized["array"] == [1.0, 2.0]


def test_get_last_valid_value():
    import numpy as np

    from src.utils.data_utils import get_last_valid_value

    arr = np.array([1.0, np.nan, 3.5, np.nan])
    assert get_last_valid_value(arr) == 3.5

    arr_all_nan = np.array([np.nan, np.nan])
    assert get_last_valid_value(arr_all_nan, default=0.0) == 0.0

    arr_empty = np.array([])
    assert get_last_valid_value(arr_empty, default=-1.0) == -1.0


def test_get_last_n_valid():
    import numpy as np

    from src.utils.data_utils import get_last_n_valid

    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan])
    res = get_last_n_valid(arr, 3)
    np.testing.assert_array_equal(res, np.array([1.0, 2.0, 4.0, 5.0])[-3:])

    empty_res = get_last_n_valid(np.array([]), 5)
    assert len(empty_res) == 0


