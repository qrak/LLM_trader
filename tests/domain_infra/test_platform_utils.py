"""Dense domain tests for platform plumbing: trade-history persistence, the
composition root, the launcher requirement check, the SARIF repair script and
exchange symbol preloading.
"""

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.check_requirements import find_missing
from scripts.fix_sarif import fix_sarif, main, sanitize_level
from src.managers.sqlite_trade_history import SQLiteTradeHistory
from src.platforms.exchange_manager import ExchangeManager


def decision(timestamp: str, action: str = "BUY", symbol: str = "BTC/USDC") -> dict:
    """Trade decision row as the strategy persists it."""
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "action": action,
        "confidence": "HIGH",
        "price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "position_size": 0.05,
        "quote_amount": 500.0,
        "quantity": 5.0,
        "fee": 0.4,
        "reasoning": "test",
    }


def make_history(tmp_path) -> SQLiteTradeHistory:
    """Real SQLite trade-history store on a temp file."""
    return SQLiteTradeHistory(logger=MagicMock(), db_path=str(tmp_path / "trade_history.db"))


def make_exchange_manager(exchanges: list[str] | None = None) -> ExchangeManager:
    """ExchangeManager double config with a fixed exchange priority order."""
    config = MagicMock()
    config.SUPPORTED_EXCHANGES = ["binance", "kucoin", "gate"] if exchanges is None else exchanges
    return ExchangeManager(logger=MagicMock(), config=config)


async def attempted_exchanges(manager: ExchangeManager, loader: AsyncMock) -> list[str]:
    """Run ensure_symbols_loaded with a patched loader and report the venues touched."""
    manager._ensure_exchange_loaded = loader
    await manager.ensure_symbols_loaded()
    return [call.args[0] for call in loader.await_args_list]


def test_trade_history_store_contract_and_query_guards(tmp_path):
    store = make_history(tmp_path)
    now = datetime.now(timezone.utc)

    assert store.count() == 0
    assert store.query() == []
    assert store.get_last_execution_timestamp() is None
    assert store.get_last_execution_timestamp(actions=()) is None

    first = store.insert(decision((now - timedelta(hours=2)).isoformat(), "BUY"))
    store.insert(decision((now - timedelta(hours=1)).isoformat(), "CLOSE_LONG", "BTC/USDC"))
    store.insert(decision(now.isoformat(), "SELL", "ETH/USDC"))

    assert (first, store.count()) == (1, 3)
    assert store.count(symbol="BSD/USDC") == 0
    assert len(store.query(limit=-100, offset=-5)) == 1
    assert len(store.query(limit=0)) == 1
    assert [row["action"] for row in store.query(order="ASC")] == ["BUY", "CLOSE_LONG", "SELL"]
    assert [row["action"] for row in store.query(order="desc")] == ["SELL", "CLOSE_LONG", "BUY"]
    assert len(store.query(symbol="BTC/USDC")) == 2
    assert len(store.query(action="SELL")) == 1
    assert len(store.query(since=(now - timedelta(minutes=90)).isoformat())) == 2
    assert len(store.query(until=(now - timedelta(minutes=90)).isoformat())) == 1
    assert len(store.export_json()) == 3
    assert store.get_last_execution_timestamp(actions=("BUY", "SELL")) == now.isoformat()
    assert store.get_last_execution_timestamp(actions=("CLOSE_LONG",)) == (now - timedelta(hours=1)).isoformat()

    with pytest.raises(ValueError, match="Invalid order"):
        store.query(order="DROP TABLE")


def test_trade_history_insert_of_empty_decision_is_rejected(tmp_path):
    store = make_history(tmp_path)

    assert store.insert({}) == 0
    assert store.count() == 0


def test_trade_history_write_fails_loudly_when_file_is_readonly(tmp_path):
    store = make_history(tmp_path)
    store.insert(decision("2026-05-21T00:00:00+00:00"))
    (tmp_path / "trade_history.db").chmod(0o444)

    assert store.count() == 1
    assert store.insert(decision("2026-05-22T00:00:00+00:00")) == 0
    assert store.count() == 1


def test_start_module_exposes_composition_root_and_single_instance_lock():
    start_path = Path("start.py").resolve()
    assert start_path.exists()
    assert ast.parse(start_path.read_text(encoding="utf-8"), filename=str(start_path)) is not None

    import start

    assert start.CryptoTradingBot is not None
    assert start.SingleInstanceLock is not None

    lock = start.SingleInstanceLock(app_name=".test_llm_trader.lock")
    assert lock.lock_file_path == Path.home() / ".test_llm_trader.lock"
    assert callable(lock._acquire_windows_mutex)
    assert callable(lock._release_windows_mutex)
    assert callable(lock.acquire)
    assert callable(lock.release)


def test_launcher_requirements_check_reports_only_unsatisfied_lines(tmp_path):
    cases = [
        ("pytest>=1\n# comment\n\npackaging\n", []),
        ("definitely-not-installed-xyz==1.0\n", ["definitely-not-installed-xyz==1.0"]),
        ("pytest>=9999\n", ["pytest>=9999"]),
        ("pytest<1\n", ["pytest<1"]),
        ("this is not a requirement!!!\n", ["this is not a requirement!!!"]),
        ("pytest>=9999\npackaging\n# note\n", ["pytest>=9999"]),
        ("", []),
    ]

    for content, expected in cases:
        requirements = tmp_path / "requirements.txt"
        requirements.write_text(content, encoding="utf-8")
        assert find_missing(requirements) == expected


def test_sarif_levels_are_normalized():
    expectations = [
        ("none", "none"),
        ("note", "note"),
        ("warning", "warning"),
        ("error", "error"),
        ("info", "note"),
        ("informational", "note"),
        ("debug", "note"),
        ("low", "note"),
        ("notice", "note"),
        ("style", "note"),
        ("convention", "note"),
        ("medium", "warning"),
        ("warn", "warning"),
        ("high", "error"),
        ("critical", "error"),
        ("fatal", "error"),
        ("off", "none"),
        ("WARNING", "warning"),
        ("  ERROR  ", "error"),
        (None, "warning"),
        ("unknown_custom_level", "warning"),
        ("", "warning"),
        (7, "warning"),
    ]

    for raw_level, expected in expectations:
        assert sanitize_level(raw_level) == expected


def test_fix_sarif_repairs_rules_levels_and_artifact_paths():
    data = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": None, "rules": None}},
                "results": [
                    {
                        "ruleId": "TEST01",
                        "level": "informational",
                        "locations": [
                            {"physicalLocation": {"artifactLocation": {"uri": "/src/src/trading/brain.py"}}},
                            {"physicalLocation": {"artifactLocation": {"uri": "already/relative.py"}}},
                        ],
                        "relatedLocations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": "/home/runner/work/LLM_trader/LLM_trader/src/app.py"
                                    }
                                }
                            }
                        ],
                    },
                    {"ruleId": "TEST02", "level": "CRITICAL"},
                    "not-a-result",
                ],
            },
            {
                "tool": {
                    "driver": {
                        "name": "Other",
                        "rules": [{"id": "R1", "defaultConfiguration": {"level": "high"}}, "not-a-rule"],
                    }
                }
            },
            "not-a-run",
        ],
    }

    fixed = fix_sarif(data)

    driver = fixed["runs"][0]["tool"]["driver"]
    assert driver["name"] == "Codacy"
    assert driver["rules"] == []

    first_result = fixed["runs"][0]["results"][0]
    assert first_result["level"] == "note"
    assert first_result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "src/trading/brain.py"
    assert first_result["locations"][1]["physicalLocation"]["artifactLocation"]["uri"] == "already/relative.py"
    assert (
        first_result["relatedLocations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "src/app.py"
    )
    assert fixed["runs"][0]["results"][1]["level"] == "error"
    assert fixed["runs"][0]["results"][2] == "not-a-result"

    second_driver = fixed["runs"][1]["tool"]["driver"]
    assert second_driver["name"] == "Other"
    assert second_driver["rules"][0]["defaultConfiguration"]["level"] == "error"
    assert second_driver["rules"][1] == "not-a-rule"
    assert fixed["runs"][1]["results"] == []
    assert fixed["runs"][2] == "not-a-run"


def test_fix_sarif_cli_rewrites_file_in_place_and_reports_missing_input(monkeypatch, tmp_path, capsys):
    sarif_file = tmp_path / "results.sarif"
    sarif_file.write_text(
        json.dumps({"version": "2.1.0", "runs": [{"tool": {"driver": {"rules": None}}, "results": [{"level": "high"}]}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["fix_sarif.py", str(sarif_file)])
    assert main() == 0

    fixed = json.loads(sarif_file.read_text(encoding="utf-8"))
    assert fixed["runs"][0]["tool"]["driver"]["rules"] == []
    assert fixed["runs"][0]["tool"]["driver"]["name"] == "Codacy"
    assert fixed["runs"][0]["results"][0]["level"] == "error"

    monkeypatch.setattr("sys.argv", ["fix_sarif.py", str(tmp_path / "missing.sarif")])
    assert main() == 1
    assert "No SARIF file found" in capsys.readouterr().err


async def test_exchange_manager_preloads_only_the_first_reachable_exchange():
    loaded = MagicMock()

    assert await attempted_exchanges(make_exchange_manager(), AsyncMock(return_value=loaded)) == ["binance"]

    assert await attempted_exchanges(
        make_exchange_manager(), AsyncMock(side_effect=[None, loaded])
    ) == ["binance", "kucoin"]

    preloaded = make_exchange_manager()
    preloaded.symbols_by_exchange["binance"] = {"BTC/USDT"}
    assert await attempted_exchanges(preloaded, AsyncMock(return_value=loaded)) == []

    assert await attempted_exchanges(
        make_exchange_manager(), AsyncMock(return_value=None)
    ) == ["binance", "kucoin", "gate"]

    assert await attempted_exchanges(make_exchange_manager(exchanges=[]), AsyncMock(return_value=loaded)) == []


async def test_exchange_manager_symbol_lookup_returns_exchange_and_identifier():
    manager = make_exchange_manager()
    manager.exchanges["binance"] = MagicMock()
    manager.symbols_by_exchange["binance"] = {"BTC/USDT", "ETH/USDT"}

    exchange, exchange_id = await manager.find_symbol_exchange("BTC/USDT")
    assert exchange_id == "binance"
    assert exchange is manager.exchanges["binance"]

    manager._ensure_exchange_loaded = AsyncMock(return_value=None)
    assert await manager.find_symbol_exchange("DOGE/USDT") == (None, None)

    manager.symbols_by_exchange["binance"] = {"BTC/USDT"}
    assert manager.get_all_symbols() == {"BTC/USDT"}
