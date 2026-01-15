# AI Agent Guidelines (AGENTS.md)

This file provides architectural context and mandatory coding standards for AI agents (Jules, Gemini Code Assist, etc.) working on the LLM_trader project.

## Project Overview
LLM_trader is a Python-based trading bot utilizing Large Language Models for decision making. It follows a modular architecture with strict Dependency Injection.

## Mandatory Coding Rules

### 1. Style & Documentation
- **Language**: ALL code, comments, and documentation must be in **English**.
- **Docstrings**: Prioritize docstrings for documentation. Use standard `#` comments only when explaining complex logic.
- **Vertical Spacing**: Maximum **one** consecutive blank line. No extra spaces between methods.
- **No Hash Headers**: Never use comments like `#######` or `=======` as visual separators.

### 2. Architecture & Patterns
- **Dependency Injection (DI)**: Follow the pattern in `src/app.py`. Pass shared instances (UnifiedParser, Logger, Config) via constructors. Use the Composition Root.
- **DRY Principle**: Search for existing methods in `UnifiedParser` or `FormatUtils` before implementing new logic.
- **SRP (Single Responsibility Principle)**: Keep classes/methods focused. Refactor files approaching 1000 lines into specialized components.
- **Data Modeling**: Use **dataclasses** for structured data (see `src/trading/dataclasses.py`).

### 3. Pythonic Best Practices
- **Mandatory Typing**: Use the `typing` library (`Optional`, `List`, `Protocol`) for ALL attributes and signatures.
- **Attribute Access**: Access known properties directly. Avoid `hasattr()` or `getattr()` for attributes defined in the class contract.
- **Duck Typing**: Trust type hint contracts; avoid redundant `isinstance()` checks unless handling truly disparate types.

## Environment & Tools
- Always ensure the `.venv` is used for execution.
- Refer to `requirements.txt` for dependencies.
