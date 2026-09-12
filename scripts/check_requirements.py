"""Print requirements.txt lines that the current environment does not satisfy.

The start scripts run this before pip install: a non-empty output means the
launcher runs `pip install -r requirements.txt`, so version floors (>=, ==,
<, ranges) are actually enforced instead of only checking that a package
with the right name exists.
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement


def find_missing(requirements_path: Path) -> list[str]:
    """Return requirements.txt lines not satisfied by the current interpreter's environment."""
    missing: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
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
            missing.append(line)
            continue
        if requirement.specifier and not requirement.specifier.contains(installed_version, prereleases=True):
            missing.append(line)
    return missing


def main() -> int:
    requirements_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("requirements.txt")
    for line in find_missing(requirements_path):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
