

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DASHBOARD_COPY = (
    "src/dashboard/static/index.html",
    "src/dashboard/static/landing.html",
)

PERSONAL_PHRASES = (
    r"Piotr",
    r"Kurnicki",
    r"\bBSH\b",
    r"Sprzęt Gospodarstwa",
    r"Supply Controller",
    r"open to offers",
    r"hobbyist",
    r"no computer-science degree",
    r"professional backend engineer",
    r"remote or on-site",
)

PERSONAL_PLACES = (r"Wrocław", r"Wroclaw")

FIRST_PERSON = (r"\bI'm\b", r"\bI've\b", r"\bI build\b", r"\bmy\b", r"\bMy\b")


def _markdown_files() -> list[Path]:

    root = sorted(REPO_ROOT.glob("*.md"))
    docs = sorted((REPO_ROOT / "docs").rglob("*.md"))
    agents = sorted((REPO_ROOT / "src").rglob("AGENTS.md"))
    return root + docs + agents


@pytest.mark.parametrize(
    "phrase",
    [pytest.param(phrase, id=phrase) for phrase in PERSONAL_PHRASES + PERSONAL_PLACES],
)
def test_markdown_has_no_personal_identifiers(phrase):
    offenders = []
    for path in _markdown_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(phrase, line, re.IGNORECASE):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not offenders, f"{phrase!r} names a person, employer or career framing: {offenders}"


@pytest.mark.parametrize("relative", DASHBOARD_COPY)
@pytest.mark.parametrize("phrase", [pytest.param(p, id=p) for p in PERSONAL_PHRASES + PERSONAL_PLACES])
def test_shipped_dashboard_copy_has_no_personal_identifiers(relative, phrase):
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert not re.search(phrase, text, re.IGNORECASE), (
        f"{relative} names a person, employer or career framing: {phrase!r}"
    )


@pytest.mark.parametrize("relative", DASHBOARD_COPY)
@pytest.mark.parametrize("phrase", [pytest.param(p, id=p) for p in FIRST_PERSON])
def test_shipped_dashboard_copy_is_not_first_person(relative, phrase):
    offenders = [
        number
        for number, line in enumerate((REPO_ROOT / relative).read_text(encoding="utf-8").splitlines(), 1)
        if re.search(phrase, line)
    ]
    assert not offenders, f"{relative} speaks in the first person at lines {offenders}: {phrase!r}"
