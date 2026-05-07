# Contributing to LLM Trader

Thank you for helping improve the project.

## Quick Start

1. Fork the repository and clone your fork.
2. Create a feature branch from `main`.
3. Set up and activate the local virtual environment.
4. Implement your change with tests and documentation updates.
5. Open a Pull Request with a clear summary and validation steps.

## Local Development Setup (Windows PowerShell)

```powershell
python -m venv .venv
& ./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Use the interpreter inside `.venv` for all development commands.

## Code Standards

- Follow the existing code style and architecture patterns used in `src/` and `tests/`.
- Keep architecture dependency-injected through the composition root (`start.py`), rather than constructing service dependencies inside service classes.
- Use explicit type hints on new or changed APIs.
- Use concise English docstrings for non-trivial classes and functions.
- Do not commit secrets, API keys, or local runtime data from `data/`.

## Testing Expectations

Run targeted tests for changed modules first, then a broader pass when practical.

```powershell
pytest tests/
```

If your change touches prompts, parsing, trading logic, dashboard routes, or news ingestion, include or update relevant regression tests.

## Documentation and Changelog Requirements

- Update `README.md` when user-facing behavior changes.
- Update docs under `docs/` when architecture or workflows change.
- Update `CHANGELOG.md` for material behavior, configuration, dependency, API, or workflow changes.

## Pull Request Checklist

- Describe what changed and why.
- Link related issues.
- Include test evidence (commands and results summary).
- Mention documentation updates included in the PR.
- Confirm no secrets were introduced.

## Issues and Feature Requests

Use GitHub Issues for bug reports and enhancement ideas. Include reproduction steps, expected behavior, actual behavior, and environment details.

## License and Contribution Terms

By contributing, you agree your contributions are licensed under the [MIT License](LICENSE.md).

## Contact

For questions, open an issue or join Discord: https://discord.gg/ZC48aTTqR2
