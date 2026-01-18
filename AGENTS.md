# AGENTS.md

This file helps AI coding agents (like Jules) understand and work effectively with this repository.

## Project Overview

**LLM_trader** is a Python-based automated trading bot that uses Large Language Models for market analysis and trading decisions.

- **Framework**: FastAPI for the dashboard
- **Frontend**: Vanilla HTML/CSS/JavaScript (no npm/pnpm/React/Vue)
- **Python Version**: 3.10+
- **Environment**: Windows with virtual environment

## Build & Test Commands

```powershell
# Activate virtual environment (REQUIRED before any Python command)
& ./.venv/Scripts/Activate.ps1

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_refactoring.py -v

# Lint check
ruff check src/

# Lint auto-fix
ruff check src/ --fix

# Type check (optional)
pyright src/
```

## Code Style Guidelines

### Language
- **English only** in all code, comments, variable names, and documentation
- Never use Polish even if requested in Polish

### Formatting
- Max 1 consecutive blank line
- No `# ===` separator comments
- Use docstrings instead of `#` comments when possible

### Type Hints (Mandatory)
- Use Python 3.10+ syntax: `list[str]`, `str | None`
- All class attributes and method signatures must have type hints

### Data Modeling
- Use **Pydantic** (`pydantic.BaseModel`) for all data schemas
- Use attribute access (`response.content`) not dict access (`response['content']`)
- Reference: `src/platforms/ai_providers/response_models.py`

### Architecture (DI & DRY)
- Check `src/utils/format_utils.py` and `src/parsing/unified_parser.py` before creating new utilities
- Pass dependencies via constructors (see `src/app.py`)
- No duplicate logic - refactor existing methods to be generic

### Pythonic Patterns
- No `hasattr()`/`getattr()` for known class properties
- No redundant `isinstance()` checks for typed arguments
- No delegation methods that just wrap member calls

## Testing Instructions

Before creating any PR:
1. Activate venv: `& ./.venv/Scripts/Activate.ps1`
2. Run tests: `pytest tests/ -v` (MUST pass 100%)
3. Run lint: `ruff check src/` (MUST have 0 errors)

## Security Considerations

- Never commit API keys or secrets
- Use `keys.env` for all sensitive configuration
- Never log sensitive data (API keys, passwords)
- Don't expose stack traces in error responses

## PR Guidelines

### Branch Naming
```
jules/{persona}/{short-description}
```
Examples:
- `jules/bolt/optimize-stochastic`
- `jules/sentinel/fix-api-key-leak`
- `jules/palette/add-aria-labels`

### PR Title Formats
- Performance: `⚡ Bolt: [optimization name]`
- Security: `🛡️ Sentinel: [severity] [issue name]`
- UX/A11y: `🎨 Palette: [improvement name]`

### PR Description Requirements
- What was changed
- Why it was needed
- How to verify the change
- Any breaking changes or concerns

## Extra Instructions

### Terminal Commands
- Use PowerShell syntax for all commands
- **File-First Protocol**: Never parse complex data from stdout
  - ❌ Bad: `gh pr list --json ...`
  - ✅ Good: `gh pr list --json ... > temp.json; Get-Content temp.json`

### GitHub Integration
- Prefer MCP tools (`mcp_github_*`) over `gh` CLI
- No Docker-based execution (Windows-native only)
