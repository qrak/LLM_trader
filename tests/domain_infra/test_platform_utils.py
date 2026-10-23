"""Dense domain tests for platform plumbing: trade-history persistence, the
composition root, the launcher requirement check, the SARIF repair script and
exchange symbol preloading.
"""

import ast
import json
import os
import sqlite3
from dataclasses import MISSING, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.check_requirements import find_missing, resolve_constraints_path
from scripts.fix_sarif import fix_sarif, main, sanitize_level
from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.formatters import (
    LongTermFormatter,
    MarketFormatter,
    MarketOverviewFormatter,
    TechnicalFormatter,
)
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.prompts.template_manager import TemplateManager
from src.analyzer.technical_calculator import TechnicalCalculator
from src.app import BotServices
from src.composition import provisioners
from src.managers.sqlite_trade_history import SQLiteTradeHistory
from src.platforms.exchange_manager import ExchangeManager
from src.trading.data_models import MarketSnapshot
from src.utils.format_utils import FormatUtils
from src.utils.timeframe_validator import TimeframeValidator
from start import CompositionRoot, CryptoTradingBot
from tests.conftest import make_config


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


def test_startup_console_width_follows_the_live_terminal(monkeypatch):
    """A console opened in a narrow window must widen with the terminal, not stay pinned."""
    import start as start_module

    size = {"value": os.terminal_size((60, 30))}
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.delenv("LINES", raising=False)
    monkeypatch.setattr(start_module, "Logger", MagicMock())
    monkeypatch.setattr(os, "get_terminal_size", lambda *args, **kwargs: size["value"])

    root = start_module.CompositionRoot()
    size["value"] = os.terminal_size((175, 45))

    assert root.console.width == 175 - int(root.console.legacy_windows)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, 120.0),
        ({"reconcile_interval_seconds": 45}, 45.0),
        ({"reconcile_interval_seconds": "45.5"}, 45.5),
        ({"reconcile_interval_seconds": "invalid"}, 120.0),
        ({"reconcile_interval_seconds": None}, 120.0),
        ({"reconcile_interval_seconds": object()}, 120.0),
    ],
)
async def test_composition_root_wires_reconcile_interval_from_config(
    settings, expected, monkeypatch
):
    root = object.__new__(CompositionRoot)
    root.config = MagicMock()
    root.config.DASHBOARD_ENABLED = False
    root.config.get_config.side_effect = lambda section, key, default: settings.get(key, default)
    root.logger = MagicMock()
    root.loop = None
    root.shutdown_manager = None
    dependencies = {
        field.name: MagicMock()
        for field in fields(BotServices)
        if field.default is MISSING and field.name not in {"logger", "config", "shutdown_manager"}
    }
    root.build_dependencies = AsyncMock(return_value=dependencies)
    monitors = []

    async def initialize(bot):
        monitors.append(bot.position_monitor)

    monkeypatch.setattr(CryptoTradingBot, "initialize", initialize)
    monkeypatch.setattr(CryptoTradingBot, "run", AsyncMock())

    await root.run_async()

    root.config.get_config.assert_called_once_with(
        "executor_api", "reconcile_interval_seconds", 120.0
    )
    assert len(monitors) == 1
    assert monitors[0].reconcile_interval_seconds == expected


@pytest.mark.parametrize("lesson_count", [0, 6])
@pytest.mark.parametrize("vector_context_chars", [0, 18000, 48000, 100000])
def test_trading_provisioning_connects_saved_lessons_to_the_analysis_prompt(
    tmp_path, monkeypatch, lesson_count, vector_context_chars
):
    monkeypatch.chdir(tmp_path)
    root = object.__new__(CompositionRoot)
    root.config = make_config(DATA_DIR="data", TIMEFRAME="4h", CRYPTO_PAIR="BTC/USDC")
    root.logger = MagicMock()
    vector = MagicMock()
    vector.trade_count = 0
    vector.experience_count = 0
    vector.prune_aged_documents.return_value = {}
    vector.get_blocked_trade_feedback.return_value = ""
    vector.get_context_for_prompt.return_value = (
        "Relevant past experiences:\n" + "x" * vector_context_chars
        if vector_context_chars else ""
    )
    vector.get_stats_for_context.return_value = {"total_trades": 0}
    vector.get_relevant_rules.return_value = []
    vector.compute_optimal_thresholds.return_value = {}
    monkeypatch.setattr(provisioners, "configure_hf_hub_auth", lambda: None)
    monkeypatch.setattr(provisioners, "cleanup_legacy_embedding_cache", lambda logger: None)
    monkeypatch.setattr(provisioners, "get_best_device", lambda: "cpu")
    monkeypatch.setattr(provisioners.chromadb, "PersistentClient", MagicMock())
    monkeypatch.setattr(provisioners, "SentenceTransformer", MagicMock())
    monkeypatch.setattr(provisioners, "VectorMemoryService", lambda *args, **kwargs: vector)

    utils = root._provision_utilities()
    trading = root._provision_trading_layer(utils, {"manager": MagicMock()})
    brain = trading["brain_service"]
    repository = trading["post_mortem_repo"]
    assert brain.context_provider.post_mortem_repo is repository
    assert brain.context_provider.prompt_budget is utils["prompt_budget"]
    assert utils["prompt_budget"].token_counter is utils["token_counter"]
    assert brain.persistence is trading["persistence"]

    for index in range(lesson_count):
        row_id = repository.insert_post_mortem(
            trade_id=None, symbol="BTC/USDC", direction="SHORT", verdict="good_exit",
            llm_analysis="Confirmed exit", expected_vs_actual=None,
            lesson_learned=f"Stored lesson {index}", pnl_pct=2.04,
            close_reason="take_profit_filled",
        )
        with sqlite3.connect(repository._db_path) as conn:
            conn.execute(
                "UPDATE trade_post_mortem SET created_at = ? WHERE id = ?",
                (f"2026-10-13 00:00:{index:02d}", row_id),
            )

    formatter = FormatUtils()
    overview = MarketOverviewFormatter(root.logger, formatter)
    long_term = LongTermFormatter(root.logger, formatter)
    builder = PromptBuilder(
        timeframe="4h", logger=root.logger, config=root.config, format_utils=formatter,
        overview_formatter=overview, long_term_formatter=long_term,
        technical_formatter=TechnicalFormatter(
            TechnicalCalculator(root.logger, formatter), root.logger, formatter
        ),
        market_formatter=MarketFormatter(
            root.logger, formatter, root.config, MagicMock(), overview, long_term
        ),
        timeframe_validator=TimeframeValidator,
        template_manager=TemplateManager(root.config, root.logger, TimeframeValidator),
    )
    context = brain.get_context(MarketSnapshot(adx=19.49, trend_direction="BEARISH"))
    assert "Brain context reduced" not in context
    prompt = builder.build_system_prompt(
        "BTC/USDC", AnalysisContext(symbol="BTC/USDC", current_price=82000.0),
        brain_context=context, dynamic_thresholds=brain.get_dynamic_thresholds(),
    )
    marker = "### Trade Journal (Recent Post-Mortem Lessons):"
    if lesson_count:
        assert marker in prompt
        assert "Stored lesson 0" not in prompt
        for index in range(1, lesson_count):
            assert prompt.count(f"Stored lesson {index}") == 1
        assert prompt.count("P&L: +2.0%") == 5
        assert prompt.index("Stored lesson 5") < prompt.index("Stored lesson 1")
        repository.insert_post_mortem(
            trade_id=None, symbol="BTC/USDC", direction="SHORT", verdict="good_exit",
            llm_analysis="Another confirmed exit", expected_vs_actual=None,
            lesson_learned="New lesson after the first context", pnl_pct=1.0,
            close_reason="take_profit_filled",
        )
        assert "New lesson after the first context" in brain.get_context(MarketSnapshot())
    else:
        assert marker not in prompt
        assert "Stored lesson" not in prompt
    vector.store_experience.assert_not_called()


def test_model_provisioning_connects_request_budget_to_the_shared_token_counter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = object.__new__(CompositionRoot)
    root.logger = MagicMock()
    root.config = make_config(OPENROUTER_API_KEY="", GOOGLE_STUDIO_API_KEY="", LM_STUDIO_BASE_URL="")
    root.config.get_model_config = lambda model: {"max_tokens": 1024}
    monkeypatch.setattr(provisioners, "DeepSeekClient", MagicMock())
    utils = root._provision_utilities()
    models = root._provision_model_layer(utils)
    manager = models["manager"]
    assert manager.token_counter is utils["token_counter"]
    assert manager._orchestrator.prompt_budget is utils["prompt_budget"]
    assert manager._orchestrator.prompt_budget.token_counter is manager.token_counter


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


def test_launcher_requirements_check_includes_constraints_file(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pytest>=1\n", encoding="utf-8")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(
        "# floors and caps for packages pulled in as dependencies\npytest>=9999\ndefinitely-not-installed-xyz==1.0\n",
        encoding="utf-8",
    )

    # The unmet floor is reported; a constrained package that is not installed is not
    # (pip does not install constraints, so that line cannot be the reason for a pip run).
    assert find_missing(requirements, constraints) == ["pytest>=9999"]
    assert find_missing(requirements) == []


def test_launcher_requirements_check_resolves_constraints_beside_requirements(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pytest>=1\n", encoding="utf-8")

    assert resolve_constraints_path(requirements) is None

    (tmp_path / "constraints.txt").write_text("pytest<1\n", encoding="utf-8")
    assert resolve_constraints_path(requirements) == tmp_path / "constraints.txt"
    assert resolve_constraints_path(requirements, "explicit.txt") == Path("explicit.txt")
    assert find_missing(requirements, tmp_path / "absent.txt") == []


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
