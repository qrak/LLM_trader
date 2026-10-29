
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

CONSTRAINTS_FILENAME = "constraints.txt"


def _unsatisfied_lines(path: Path, report_absent: bool) -> list[str]:

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

    missing = _unsatisfied_lines(requirements_path, report_absent=True)
    if constraints_path is not None and constraints_path.is_file():
        missing += _unsatisfied_lines(constraints_path, report_absent=False)
    return missing


def resolve_constraints_path(requirements_path: Path, given: str | None = None) -> Path | None:

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
