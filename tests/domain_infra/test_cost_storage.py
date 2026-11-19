

import pytest

from src.utils.token_counter import CostStorage


def test_snapshot_of_empty_storage_is_zero(tmp_path):
    storage = CostStorage(file_path=str(tmp_path / "api_costs.json"))

    assert storage.snapshot() == {
        "cost_usd": 0.0,
        "input_tokens": 0.0,
        "output_tokens": 0.0,
    }
    assert storage.total_cost() == 0.0


def test_snapshot_sums_providers_and_ignores_missing_cost(tmp_path):
    storage = CostStorage(file_path=str(tmp_path / "api_costs.json"))

    storage.record_usage("openrouter", 100, 50, 0.01)
    storage.record_usage("google", 200, 100, 0.02)
    storage.record_usage("deepseek", 10, 5, None)

    snapshot = storage.snapshot()

    assert snapshot["cost_usd"] == pytest.approx(0.03)
    assert snapshot["input_tokens"] == 310.0
    assert snapshot["output_tokens"] == 155.0
    assert storage.total_cost() == pytest.approx(0.03)


def test_commandcode_usage_is_recorded_under_its_own_provider(tmp_path):
    storage = CostStorage(file_path=str(tmp_path / "api_costs.json"))

    storage.record_usage("commandcode", 1_000, 500, 0.004)

    assert storage.get_provider_costs("commandcode").total_cost == pytest.approx(0.004)
    assert storage.snapshot()["cost_usd"] == pytest.approx(0.004)
