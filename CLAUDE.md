# LLM Trader — Agent Instructions

> **This file is a pointer.** The canonical documentation is now modular — see the master architecture blueprint and per-agent docs linked below.

## Quick Navigation

| Document | Location |
|----------|----------|
| **🏗️ Master Architecture Blueprint** | [AGENTS.md](./AGENTS.md) — system overview, Mermaid flow diagram, agent table |
| **🧠 Brain Agent** (TradingBrainService) | [src/trading/AGENTS.md](./src/trading/AGENTS.md) — reflection loops, context injection, semantic rules |
| **🔬 Analysis Engine Agent** | [src/analyzer/AGENTS.md](./src/analyzer/AGENTS.md) — 40+ indicators, pattern engine, chart generator, AI signal |
| **📰 RAG Engine Agent** | [src/rag/AGENTS.md](./src/rag/AGENTS.md) — RSS ingestion, Crawl4AI, fundamentals, scoring |
| **⚙️ Risk Manager & Provider Orchestrator** | [src/managers/AGENTS.md](./src/managers/AGENTS.md) — SL/TP scaling, fallback chain |
| **🛡️ Governance Pipeline** | [src/trading/guards/AGENTS.md](./src/trading/guards/AGENTS.md) — pre-execution guard chain |
| **📊 Dashboard Agent** | [src/dashboard/AGENTS.md](./src/dashboard/AGENTS.md) — FastAPI + WebSocket UI |

## Tech Stack

- **Python:** 3.13, `.venv/`, `python start.py`
- **Entry point:** `python start.py`
- **Live dashboard:** [https://semanticsignal.qrak.org](https://semanticsignal.qrak.org)
- **Vector DB:** ChromaDB
- **AI:** Google Gemini 3.5 Flash (primary), LM Studio (local text fallback), OpenRouter (secondary provider with configurable base and fallback models)
