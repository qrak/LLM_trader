

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import start as start_module
from src.composition import startup_support
from src.composition.startup_support import EMBEDDING_MODEL, configure_hf_hub_auth
from start import CompositionRoot

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module_level_imports(relative: str) -> set[str]:
    tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "relative",
    ["src/composition/provisioners.py", "src/composition/startup_support.py"],
)
def test_huggingface_hub_is_not_imported_at_module_level(relative):
    assert "sentence_transformers" not in _module_level_imports(relative)
    assert "huggingface_hub" not in _module_level_imports(relative)


def _write_cache(root: Path, model_id: str, with_snapshot: bool) -> Path:
    snapshots = root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    snapshots.mkdir(parents=True)
    if with_snapshot:
        (snapshots / "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a").mkdir()
    return root


def test_a_cached_model_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(_write_cache(tmp_path, EMBEDDING_MODEL, True)))

    assert startup_support._is_model_cached(EMBEDDING_MODEL) is True


def test_an_empty_snapshot_folder_is_not_a_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(_write_cache(tmp_path, EMBEDDING_MODEL, False)))

    assert startup_support._is_model_cached(EMBEDDING_MODEL) is False


def test_a_missing_cache_is_not_a_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "absent"))

    assert startup_support._is_model_cached(EMBEDDING_MODEL) is False


def test_a_cached_model_switches_the_hub_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(startup_support, "_is_model_cached", lambda _model: True)

    configure_hf_hub_auth()

    assert startup_support.os.environ["HF_HUB_OFFLINE"] == "1"


def test_an_uncached_model_stays_online(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(startup_support, "_is_model_cached", lambda _model: False)

    configure_hf_hub_auth()

    assert "HF_HUB_OFFLINE" not in startup_support.os.environ


def _probe(name: str, daemon: bool) -> threading.Thread:
    thread = threading.Thread(target=time.sleep, args=(0.5,), name=name, daemon=daemon)
    thread.start()
    return thread


def test_live_threads_skips_the_main_thread_and_daemons():
    daemon = _probe("probe-daemon", True)
    try:
        live = CompositionRoot._live_threads()
        assert threading.main_thread() not in live
        assert daemon not in live
    finally:
        daemon.join()


def test_live_threads_keeps_a_non_daemon():
    worker = _probe("probe-worker", False)
    try:
        assert worker in CompositionRoot._live_threads()
    finally:
        worker.join()


def test_describe_threads_marks_daemons():
    daemon = _probe("probe-daemon", True)
    worker = _probe("probe-worker", False)
    try:
        assert CompositionRoot._describe_threads([daemon, worker]) == "probe-daemon*, probe-worker"
    finally:
        daemon.join()
        worker.join()


def test_finish_exit_returns_when_nothing_is_stuck(monkeypatch):
    root = CompositionRoot.__new__(CompositionRoot)
    root.logger = MagicMock()
    monkeypatch.setattr(CompositionRoot, "_live_threads", staticmethod(list))

    root._finish_exit(42)

    root.logger.warning.assert_not_called()


def test_finish_exit_forces_the_process_out_when_a_thread_is_stuck(monkeypatch, tmp_path):
    root = CompositionRoot.__new__(CompositionRoot)
    root.logger = MagicMock()
    stuck = _probe("probe-worker", False)
    diag = tmp_path / "exit_diag.log"
    monkeypatch.setattr(CompositionRoot, "_live_threads", staticmethod(lambda: [stuck]))
    monkeypatch.setattr(
        CompositionRoot,
        "_open_exit_diag",
        staticmethod(lambda: open(diag, "a", encoding="utf-8")),  # noqa: SIM115
    )
    monkeypatch.setattr(start_module, "EXIT_THREAD_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(start_module.logging, "shutdown", lambda: None)
    monkeypatch.setattr(start_module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    try:
        with pytest.raises(SystemExit) as exit_info:
            root._finish_exit(42)
    finally:
        stuck.join()

    assert exit_info.value.code == 42
    assert "probe-worker" in root.logger.warning.call_args.args[2]
    assert "forced exit" in diag.read_text(encoding="utf-8")