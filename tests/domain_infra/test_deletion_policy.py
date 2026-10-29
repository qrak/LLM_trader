from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_agents_md_names_the_deletion_and_comment_gates() -> None:
    policy = _read("AGENTS.md")

    assert "net-negative" in policy
    assert "check_dead_symbols.py" in policy
    assert "check_net_diff.py" in policy
    assert "check_comment_policy.py" in policy
    assert "test-only" in policy


def test_ci_runs_the_deletion_and_comment_gates() -> None:
    workflow = _read(".github/workflows/tests.yml")

    assert "scripts/check_dead_symbols.py" in workflow
    assert "scripts/check_net_diff.py" in workflow
    assert "scripts/check_comment_policy.py" in workflow


def test_contributing_documents_the_policy() -> None:
    contributing = _read("CONTRIBUTING.md")

    assert "Deletion Discipline" in contributing
    assert "Comment and Docstring Policy" in contributing
    assert "check_dead_symbols.py" in contributing
    assert "check_comment_policy.py" in contributing


def test_the_pull_request_template_asks_for_deletion_evidence() -> None:
    template = _read(".github/PULL_REQUEST_TEMPLATE.md")

    assert "Net:" in template
    assert "check_dead_symbols.py" in template
    assert "check_net_diff.py" in template


def test_the_local_hook_runs_the_orphan_gate() -> None:
    hook = _read(".githooks/pre-commit")

    assert "check_dead_symbols.py" in hook
