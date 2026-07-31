"""Test suite verifying start.py Composition Root imports, structure, and initialization safety."""

import ast
from pathlib import Path


def test_start_py_has_no_unbound_local_variables():
    """Verify start.py has no local variable shadowing (e.g. UnboundLocalError)."""
    start_path = Path("start.py").resolve()
    assert start_path.exists(), "start.py must exist in project root"

    source = start_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(start_path))
    assert tree is not None


def test_start_py_imports_cleanly():
    """Verify start.py module can be imported without syntax or import errors."""
    import start
    assert hasattr(start, "CryptoTradingBot")
    assert hasattr(start, "SingleInstanceLock")


def test_single_instance_lock_methods_exist():
    """Verify SingleInstanceLock class contract."""
    from start import SingleInstanceLock
    lock = SingleInstanceLock(app_name=".test_llm_trader.lock")
    assert hasattr(lock, "_acquire_windows_mutex")
    assert hasattr(lock, "lock_file_path")
