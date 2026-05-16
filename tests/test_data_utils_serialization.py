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
