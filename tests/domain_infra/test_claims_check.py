from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_claims.py"


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_claims(root: Path, collected: int = 100, closed_trades: int = 14) -> None:
    claims = {
        "schema": 1,
        "generated": "2026-11-01",
        "tests": {"collected": collected, "runnable": collected, "deselected": 0},
        "ledger": {"closed_trades": closed_trades, "cycles": 83, "snapshot_at": "2026-11-01"},
        "superseded": [{"field": "return_pct", "value": 0.14}, {"field": "closed_trades", "value": 12}],
    }
    (root / "claims.json").write_text(json.dumps(claims), encoding="utf-8")


def _write_readme(root: Path, body: str) -> None:
    (root / "README.md").write_text(body, encoding="utf-8")


def test_missing_claims_file_is_a_failure(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.returncode == 1
    assert "missing claims file" in result.stdout


def test_consistent_text_passes(tmp_path: Path) -> None:
    _write_claims(tmp_path)
    _write_readme(
        tmp_path,
        "tests-100-brightgreen\n\n100 tests collected, 100 run in CI.\n\n"
        "| Closed trades | 14 | — |\n| Analyses | 83 cycles |\n",
    )

    result = _run(tmp_path)

    assert result.returncode == 0
    assert "claims consistent" in result.stdout


def test_stale_value_is_reported(tmp_path: Path) -> None:
    _write_claims(tmp_path)
    _write_readme(
        tmp_path,
        "tests-100-brightgreen\n\n100 tests collected, 100 run in CI.\n\n"
        "| Closed trades | 14 | — |\n| Analyses | 83 cycles |\n\n"
        "Older text claimed +0.14% in the window.\n",
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "stale value" in result.stdout
    assert "+0.14%" in result.stdout


def test_badge_and_row_mismatch_are_reported(tmp_path: Path) -> None:
    _write_claims(tmp_path)
    _write_readme(
        tmp_path,
        "tests-999-brightgreen\n\n100 tests collected, 100 run in CI.\n\n"
        "| Closed trades | 12 | — |\n| Analyses | 83 cycles |\n",
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "badge says 999" in result.stdout
    assert "row 'Closed trades' says 12" in result.stdout


def test_site_project_json_is_checked(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    site_root = tmp_path / "site"
    (project_root).mkdir()
    (site_root / "public").mkdir(parents=True)
    (site_root / "src" / "pages").mkdir(parents=True)
    _write_claims(project_root)
    _write_readme(
        project_root,
        "tests-100-brightgreen\n\n100 tests collected, 100 run in CI.\n\n"
        "| Closed trades | 14 | — |\n| Analyses | 83 cycles |\n",
    )
    (site_root / "public" / "project.json").write_text(
        json.dumps({"project": {"commits": 1, "tests": {"collected": 999}}}), encoding="utf-8"
    )
    (site_root / "src" / "pages" / "index.astro").write_text("const TESTS_TOTAL = '999';", encoding="utf-8")

    result = _run(project_root, "--site-root", str(site_root))

    assert result.returncode == 1
    assert "site project.json: tests 999" in result.stdout
    assert "site index.astro: TESTS_TOTAL 999" in result.stdout
