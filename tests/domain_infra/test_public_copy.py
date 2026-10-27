"""The public copy names the project, not the author.

`src/dashboard/static/` is served verbatim and every tracked `.md` file is read on GitHub, so a
personal phrase that lands in either is published, not merely stored. This gate fails when a
personal identifier, a career framing or a first-person voice returns to either surface.

The phrases that a reader needs — paper trading, profitability not established, the sample size,
the fees the ledger does not book — are facts about the system and stay. What this gate rejects is
material that belongs to a personal profile: names, employers, job titles, availability, degrees,
locations, and the "I built this after hours" register.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DASHBOARD_COPY = (
    "src/dashboard/static/index.html",
    "src/dashboard/static/landing.html",
)

# A person, an employer or a career framing: none of it describes this system.
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

# Geography only matters when it is the author's; the market is BTC/USDC, not a city.
PERSONAL_PLACES = (r"Wrocław", r"Wroclaw")

# First person in shipped dashboard copy, where there is no code for it to hide in. The bare
# pronoun "I" is left out on purpose: the dashboard quotes the bot asking itself "Have I seen this
# pattern before?", which is the product speaking, not the author.
FIRST_PERSON = (r"\bI'm\b", r"\bI've\b", r"\bI build\b", r"\bmy\b", r"\bMy\b")


def _markdown_files() -> list[Path]:
    """Every tracked-looking markdown file: root, docs/ and the per-package AGENTS files."""
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
