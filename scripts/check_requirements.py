"""Print requirement lines that the current environment does not satisfy.

The start scripts run this before pip install: a non-empty output means the
launcher runs `pip install -r requirements.txt -c constraints.txt`, so version
floors (>=, ==, <, ranges) are actually enforced instead of only checking that a
package with the right name exists.

constraints.txt holds the floors and caps of packages this project never imports
directly; pip only pulls them in as dependencies, so a constrained package that is
not installed is not reported here (matching pip, which does not install
constraints). The file is read from the second argument when given, otherwise from
`constraints.txt` next to the requirements file.
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

CONSTRAINTS_FILENAME = "constraints.txt"


def _unsatisfied_lines(path: Path, report_absent: bool) -> list[str]:
    """Return lines in one requirements-style file the environment does not satisfy."""
    missing: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            missing.append(line)
            continue
        try:
            installed_version = version(requirement.name)
        except PackageNotFoundError:
            if report_absent:
                missing.append(line)
            continue
        if requirement.specifier and not requirement.specifier.contains(installed_version, prereleases=True):
            missing.append(line)
    return missing


def find_missing(requirements_path: Path, constraints_path: Path | None = None) -> list[str]:
    """Return unsatisfied lines from requirements and (when given) constraints."""
    missing = _unsatisfied_lines(requirements_path, report_absent=True)
    if constraints_path is not None and constraints_path.is_file():
        missing += _unsatisfied_lines(constraints_path, report_absent=False)
    return missing


def resolve_constraints_path(requirements_path: Path, given: str | None = None) -> Path | None:
    """Return the constraints file to check: the given one, else constraints.txt beside requirements."""
    if given:
        return Path(given)
    candidate = requirements_path.parent / CONSTRAINTS_FILENAME
    return candidate if candidate.is_file() else None


def main() -> int:
    requirements_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("requirements.txt")
    given_constraints = sys.argv[2] if len(sys.argv) > 2 else None
    constraints_path = resolve_constraints_path(requirements_path, given_constraints)
    for line in find_missing(requirements_path, constraints_path):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
