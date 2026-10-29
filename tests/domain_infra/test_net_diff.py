from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_net_diff import DiffBudget, NetDiffChecker


def _checker() -> NetDiffChecker:
    return NetDiffChecker(root=Path("."), base="origin/master", head="HEAD")


def test_a_normal_numstat_line_is_parsed() -> None:
    budget = DiffBudget.from_numstat("10\t2\tsrc/pkg/a.py\n")

    assert budget.added == 10
    assert budget.deleted == 2
    assert budget.net == 8
    assert budget.files[0].path == "src/pkg/a.py"


def test_a_binary_line_counts_as_zero() -> None:
    budget = DiffBudget.from_numstat("-\t-\tsrc/dashboard/static/logo.png\n")

    assert budget.added == 0
    assert budget.deleted == 0
    assert budget.net == 0


def test_a_path_with_spaces_is_preserved() -> None:
    budget = DiffBudget.from_numstat("3\t1\tsrc/pkg/a b.py\n")

    assert budget.files[0].path == "src/pkg/a b.py"


def test_a_rename_keeps_the_raw_path_and_the_counts() -> None:
    budget = DiffBudget.from_numstat("5\t5\tsrc/old.py => src/new.py\n")

    assert budget.files[0].path == "src/old.py => src/new.py"
    assert budget.net == 0


def test_code_scope_ignores_docs_tests_and_fixtures() -> None:
    budget = DiffBudget.from_numstat(
        "10\t1\tsrc/pkg/a.py\n"
        "20\t1\ttests/test_a.py\n"
        "5\t0\tdocs/FEATURES.md\n"
        "1\t0\tREADME.md\n"
        "2\t0\ttests/fixtures/decisions/buy.json\n"
    )

    assert [file.path for file in budget.files] == ["src/pkg/a.py"]
    assert budget.net == 9


def test_total_scope_includes_tests_but_not_docs() -> None:
    budget = DiffBudget.from_numstat(
        "10\t1\tsrc/pkg/a.py\n"
        "20\t1\ttests/test_a.py\n"
        "5\t0\tdocs/FEATURES.md\n",
        scope="total",
    )

    assert [file.path for file in budget.files] == ["src/pkg/a.py", "tests/test_a.py"]
    assert budget.net == 28


def test_start_py_and_scripts_are_code() -> None:
    budget = DiffBudget.from_numstat("1\t2\tstart.py\n3\t4\tscripts/tool.py\n")

    assert budget.net == -2


def test_offenders_are_ordered_by_net_addition() -> None:
    budget = DiffBudget.from_numstat("10\t1\tsrc/a.py\n2\t0\tsrc/b.py\n1\t20\tsrc/c.py\n")

    assert [file.path for file in budget.offenders()] == ["src/a.py", "src/b.py"]


def test_violations_name_the_quota_and_the_top_additions() -> None:
    budget = DiffBudget.from_numstat("10\t1\tsrc/a.py\n")

    violations = budget.violations(-1)

    assert violations
    assert "quota -1" in violations[0]
    assert "adds +9: src/a.py" in violations[1]


def test_a_refactor_label_triggers_the_budget() -> None:
    assert NetDiffChecker.is_refactor(labels="refactor", title="")
    assert NetDiffChecker.is_refactor(labels="cleanup, urgend", title="")
    assert NetDiffChecker.is_refactor(labels="", title="refactor(auth): replace the pipeline")


def test_a_feature_title_does_not_trigger_the_budget() -> None:
    assert not NetDiffChecker.is_refactor(labels="feature", title="feat(auth): add a provider")


def test_an_additive_refactor_fails() -> None:
    code, output = _checker().check(labels="refactor", numstat="10\t2\tsrc/a.py\n")

    assert code == 1
    assert "FAIL" in output
    assert "+10 / -2 = +8" in output


def test_a_net_negative_refactor_passes() -> None:
    code, output = _checker().check(labels="refactor", numstat="2\t10\tsrc/a.py\n")

    assert code == 0
    assert "PASS" in output
    assert "-8" in output


def test_zero_net_is_not_negative() -> None:
    code, _ = _checker().check(labels="refactor", numstat="5\t5\tsrc/a.py\n")

    assert code == 1


def test_a_feature_pull_request_only_reports() -> None:
    code, output = _checker().check(labels="feature", numstat="10\t2\tsrc/a.py\n")

    assert code == 0
    assert "REPORT ONLY" in output


def test_require_negative_forces_the_budget_without_a_label() -> None:
    code, output = _checker().check(require_negative=True, numstat="10\t2\tsrc/a.py\n")

    assert code == 1
    assert "FAIL" in output


def test_a_custom_quota_is_enforced() -> None:
    code, _ = _checker().check(labels="cleanup", quota=-150, numstat="10\t110\tsrc/a.py\n")
    assert code == 1

    code, _ = _checker().check(labels="cleanup", quota=-150, numstat="10\t210\tsrc/a.py\n")
    assert code == 0


def test_the_quota_is_reported() -> None:
    _, output = _checker().check(labels="refactor", quota=-150, numstat="10\t210\tsrc/a.py\n")

    assert "quota -150" in output


def test_a_malformed_numstat_line_is_rejected() -> None:
    with pytest.raises(ValueError, match="numstat"):
        DiffBudget.from_numstat("not-a-numstat-line\n")


def test_worktree_mode_measures_the_working_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "10\t20\tsrc/a.py\n"
        stderr = ""

    def fake_run(command: list[str], **kwargs) -> Result:
        commands.append(command)
        return Result()

    monkeypatch.setattr("scripts.check_net_diff.subprocess.run", fake_run)
    budget = NetDiffChecker(root=Path("."), base="origin/master", worktree=True).budget()

    assert commands[0] == ["git", "diff", "--numstat", "origin/master"]
    assert budget.net == -10
