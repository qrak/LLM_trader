# Contributing to LLM Trader

Thank you for helping improve the project.

## Quick Start

1. Fork the repository and clone your fork.
2. Create a short-lived feature branch from `master`.
3. Set up and activate the local virtual environment.
4. Implement your change with tests and documentation updates.
5. Open a Pull Request with a clear summary and validation steps.

## Local Development Setup (Windows PowerShell)

```powershell
python -m venv .venv
& ./.venv/Scripts/Activate.ps1
pip install --no-compile -r requirements.txt -c constraints.txt
pip install --no-compile -r requirements-dev.txt -c constraints.txt
```

Use the interpreter inside `.venv` for all development commands. `--no-compile` keeps the
environment ~300 MB smaller at rest; the imported modules are compiled on first use.

`constraints.txt` carries the version floors and compatibility caps for packages the code never imports directly (pip pulls them in as dependencies). The launcher scripts pass it to pip with `-c`, and `scripts/check_requirements.py` verifies both files before deciding whether anything needs installing.

## Dependency Changes

`requirements.txt` states a bound for every entry, never a bare floor: `==` for the SDKs and
clients whose next release changes behaviour, `~=` for libraries whose API is stable across a
minor line. A floor lets a fresh install resolve a newer major than the venv runs, which is how
CI once installed `torch` and `pydantic` a minor ahead of the bot.

- Bumping an existing pin: change the line, run `pip install -r requirements.txt -c
  constraints.txt`, then the full suite plus `ruff` and `pyright`. Keep the version the venv
  actually runs in the same change, and record why in `CHANGELOG.md`.
- Adding or removing a dependency, or changing the embedding path: ask first (root `AGENTS.md`,
  "Ask before").
- A package that reaches the environment only as a dependency of one of these belongs in
  `constraints.txt` as a floor or a cap, never in `requirements.txt`.
- A hold (a package deliberately kept behind the newest release) keeps its comment: name the
  conflict or the gate that blocks the bump, and the test that would have to change.

## Code Standards

- Follow the existing code style and architecture patterns used in `src/` and `tests/`.
- Keep architecture dependency-injected through the composition root (`start.py`), rather than constructing service dependencies inside service classes.
- Use explicit type hints on new or changed APIs.
- A class may carry one concise English docstring. Modules, functions and methods carry none; their names, signatures and types explain them. No `#` or `//` comments outside machine directives (`scripts/check_comment_policy.py` is the gate).
- Do not commit secrets, API keys, or local runtime data from `data/`.

## Deletion Discipline

A refactor replaces code; it does not park a new implementation beside the old one. The full contract lives in `AGENTS.md` §10 and the enforcement map in §12.1. The short version:

- Delete the superseded function, method, class, alias, guard, endpoint, CSS rule and DOM id in the same change that replaces them.
- Before deleting, run `python scripts/check_dead_symbols.py --symbol <name>`. Zero references means deletion is mandatory; a symbol referenced only by its own test or double is test-only and goes with the test.
- A refactor or retirement states its budget in the task and proves it with `python scripts/check_net_diff.py --base origin/master --head HEAD --require-negative --quota -N`. Before staging, use `--worktree --base origin/master`; after staging, `--staged --base origin/master`. The production scope is `src/`, `scripts/` and `start.py`; tests and docs may grow.
- No deprecation warnings, `_v2`/`legacy_*` aliases or compatibility shims for our own code. External payload migrations and cache cleanup are the only documented exceptions.
- `python scripts/check_dead_symbols.py` runs on every push and fails on a dead, test-only or deprecation-shim definition. Refactor pull requests also run the net-diff budget; add the `refactor` label so CI enforces it.

The optional local hook runs the same orphan gate before a commit:

```powershell
git config core.hooksPath .githooks
```

On Linux/macOS make it executable once (`chmod +x .githooks/pre-commit`). `git commit --no-verify`
bypasses it. CI remains the blocker.

## Comment and Docstring Policy

- No `#` comments in Python and no `//` comments in JS/CSS/HTML. Machine directives (`# noqa`, `# type: ignore`, `# pylint:`, `# pragma:`, `/* eslint-disable */`) are the only exceptions.
- A class may carry one docstring. A module, function or method carries none.
- `python scripts/check_comment_policy.py` fails the build otherwise.

## Testing Expectations

Run targeted tests for changed modules first, then a broader pass when practical.

```powershell
pytest tests/
```

If your change touches prompts, parsing, trading logic, dashboard routes, or news ingestion, include or update relevant regression tests.

## Documentation and Changelog Requirements

- Update `README.md` when user-facing behavior changes.
- Update docs under `docs/` when architecture or workflows change.
- Update `CHANGELOG.md` for material behavior, configuration, dependency, API, or workflow changes. Add entries at the top; when the file grows, move the entries below the previous release into `docs/changelog-archive-<year>.md` and link them from the `## Archive` section.
- **Write about the system, not the author.** The changelog, commit messages, release notes and the copy under `src/dashboard/static/` describe what the code, the data and the deployment do. No first person ("I built", "my strength"), no biography, no day job, no availability or job-seeking wording, no degree or employer names, no personal geography — the handle `@qrak` and `contact@qrak.org` are the whole identity a repository page needs. Self-assessment is not a change entry: write what the old code did, what the new code does and what pins it, not that the old code was "honest" or that it "aged badly". Caveats that a reader needs (paper trading, profitability not established, sample size, fees not booked) stay, phrased as facts. `tests/domain_infra/test_public_copy.py` fails the suite when a personal or first-person phrase returns to the tracked copy.

## Branches, Versions and Releases

`master` is the only long-lived branch: it is what the bot runs, and it must stay green. Work happens
on short-lived topic branches, merged through a pull request. There is no `develop` and no `nightly`
branch — with a single deployer and `git pull` as the deploy, a second long-lived branch would only
duplicate every merge.

Releases are automatic. After a green `tests` run on `master`, `.github/workflows/release.yml` creates
an annotated date tag (`build-YYYY.MM.DD-HHMM`, UTC) and publishes the "Latest build" release whose
body lists the commits since the previous build; older build entries are rolled off the page while
their tags stay, so `git checkout build-2026.10.26-1530` is the rollback point. Nothing in this
repository ships as a versioned artifact — there is no `__version__`, no packaged build and no download
URL — so running a specific version never requires a release.

Semantic-version tags (`v1.x`) are therefore reserved for milestones worth pointing a person at, not
for a cadence: no release PR per merge, and no renumbering of releases after the fact.

`master` carries a ruleset ("master: green before merge"): the branch cannot be deleted or
force-pushed, and merging a pull request needs `python checks` (ruff, pyright, unit tests),
`Analyze (python)` and `Analyze (javascript-typescript)`. Repository
administrators bypass it, so a direct fix push still works — the ruleset stops accidents, not you.
If you add a required job, add its exact job `name` to the ruleset in the same change; a required
context that never reports keeps the branch permanently unmergeable.

## Pull Request Checklist

- Describe what changed and why.
- Link related issues.
- Include test evidence (commands and results summary).
- Change a loader key? Ship `config/config.ini.example`, `docs/CONFIGURATION.md` and the contract test in the same PR; `python scripts/check_config_contract.py` must pass.
- Six repository gates run in CI on every push and must pass locally too:
  `python scripts/check_agents_md.py` (every path quoted in the AGENTS.md files and in the
  reader-facing docs exists), `python scripts/check_config_contract.py`,
  `python scripts/check_hardcoded_timeframes.py`, `python scripts/check_loose_helpers.py`,
  `python scripts/check_dead_symbols.py`, `python scripts/check_comment_policy.py`.
- A refactor or retirement pull request carries the `refactor` label, deletes every superseded
  symbol with its tests and doubles, and reports `Net: -N lines` from
  `python scripts/check_net_diff.py --base origin/master --head HEAD --require-negative`.
- Mention documentation updates included in the PR.
- Confirm no secrets were introduced.

## Issues and Feature Requests

Use GitHub Issues for bug reports and enhancement ideas. Include reproduction steps, expected behavior, actual behavior, and environment details.

## License and Contribution Terms

By contributing, you agree your contributions are licensed under the [MIT License](LICENSE.md).

## Contact

For questions, open an issue or join Discord: <https://discord.gg/ZC48aTTqR2>
