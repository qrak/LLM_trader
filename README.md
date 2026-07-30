# 🤖 SEMANTIC SIGNAL LLM (LLM Trader)

*An autonomous AI trading agent that reads charts, remembers outcomes, sharpens its strategy in real time, and can search its own codebase using vector semantics.*

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE.md)
[![GitHub Stars](https://img.shields.io/github/stars/qrak/LLM_trader?style=flat&logo=github)](https://github.com/qrak/LLM_trader)

📊 **[Live Dashboard](https://semanticsignal.qrak.org)** — Watch the neural trading brain in action  
📖 **[Read the Full Story (Medium)](https://medium.com/@donqrakko/i-built-a-trading-bot-that-doesnt-just-calculate-it-reasons-remembers-and-learns-from-its-749064869d73)**  
💬 **[Join the Discord](https://discord.gg/ZC48aTTqR2)**  

---

> 💡 **Paper trading by default.** A real exchange execution service ([llm_trader_executor](https://github.com/qrak/llm_trader_executor)) is currently in testing — it consumes this bot's decisions and places live CCXT orders. Coming soon. Stay tuned.

---

## Quick Start

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp keys.env.example keys.env  # add your API keys (Google Gemini free tier works)
python start.py               # dashboard at http://localhost:8000
```

<details>
<summary>Detailed setup for Windows, Linux, macOS →</summary>

**Platform-specific scripts** live in `scripts/`:

| Script | Purpose |
|--------|---------|
| `scripts/start_script_main.ps1` | Start the bot (Windows) |
| `scripts/start_script_main_linux.sh` | Start the bot (Linux) |
| `scripts/start_script_main_macos.sh` | Start the bot (macOS) |
| `scripts/run_all_tests.sh` | Run full test suite in `.venv` |
| `scripts/query_trade_history.py` | CLI utility to inspect SQLite trade history |
| `scripts/query_codebase.py` | **Semantic codebase search** via ChromaDB vector index |
| `scripts/rotate_journals.py` | Auto-rotate AI agent journal files (runs on startup) |
</details>

### Semantic Codebase Search

Query the entire codebase using natural language — the AST-level vector index finds relevant classes, functions, and modules:

```bash
# Search all indexed symbols
python scripts/query_codebase.py "how does the risk manager calculate position size"

# Filter by symbol type
python scripts/query_codebase.py --type function "RSI calculation"

# Force re-index before searching
python scripts/query_codebase.py --reindex "vector memory retrieval"

# Show index statistics
python scripts/query_codebase.py --stats
```

The index auto-updates on bot startup. Currently indexes **160 files → 2,306 semantic chunks** across all Python source and Markdown documentation.

### Runtime Controls

| Key | Action |
|-----|--------|
| `a` | Force analysis — run immediate market check |
| `d` | Toggle dashboard on/off |
| `h` | Help — show available commands |
| `q` | Quit — graceful shutdown with state preservation |

---

## System Requirements

| Component | Minimum | Recommended (RL Training) |
|-----------|---------|---------------------------|
| **Python** | 3.13+ | 3.14+ |
| **RAM** | 4 GB | 8+ GB |
| **Disk** | 2 GB | 5+ GB (model checkpoints) |
| **CPU** | 2 cores | 4+ cores (Ryzen 5700G+) |
| **GPU** | Not required | Optional for RL training |
| **OS** | Windows 10+, Linux, macOS | Linux (WSL2) |
| **Internet** | Required (API calls) | Required |

**RL Policy Training** (optional `[rl_training] enabled=true`):
- Model: Qwen3-0.6B-Instruct (~1.2 GB download, HuggingFace)
- Inference: CPU-viable at ~2-5 tokens/sec on modern desktop CPUs
- Training: PPO requires 8+ GB RAM; GPU strongly recommended for training speed
- First run auto-downloads the model from HuggingFace

## 📸 Screenshots

![Dashboard Overview](img/1.png)

![Decision Pathways](img/2.png)

---

## Features

- **🧠 Brain with Memory** — ChromaDB vector store retains trade experiences, semantic rules, system rejections, and confidence statistics. Past outcomes are retrieved by similarity to current market conditions and injected into every LLM prompt.

- **📈 Vision AI Chart Analysis** — Generates 4K PNG candlestick charts with indicators, sends them to a multimodal LLM (Gemini 3.5 Flash) for visual pattern recognition. Chart-pattern code was dropped because the AI reads charts better than hardcoded rules.

- **🔄 Reflection Engine** — After every `N` closed trades, the system synthesizes best-practice rules, anti-patterns, and AI-mistake rules with **surprise ratio** annotation — high-surprise outcomes are flagged so the LLM discounts lucky/unlucky noise. Rules persist in vector memory and influence future decisions. The bot learns from its own outcomes.

- **🧠 VectorMemoryRulesMixin** — Semantic rule lifecycle management with decay scoring, evidence-weighted ranking, contradiction tracking, and surprise-ratio annotation. Rules are soft-ranked by similarity, evidence quality, timeframe freshness, and contradiction count — no hard pruning on age alone.

- **🔍 AST Codebase Vector Index** — Parses every Python source file at the AST level into ChromaDB (`data/codebase_index/`). Enables natural-language queries against the entire codebase via `scripts/query_codebase.py`. Auto-updates on bot startup. Zero secrets or sensitive files indexed (verified by security audit).

- **✅ Claim Validation** — Every LLM response is cross-checked against computed indicators. Reported trend strength is compared against actual ADX; pattern quality is replaced by a deterministic scorer. No blind trust in AI numeric claims.

- **📰 RAG News Engine** — Aggregates crypto news from free RSS feeds (CoinDesk, CoinTelegraph, Decrypt, CryptoSlate) with optional Crawl4AI enrichment, plus fundamentals from DeFiLlama and CoinGecko.

- **📊 Live Dashboard** — FastAPI + WebSocket real-time SPA at `0.0.0.0:8000` (or [semanticsignal.qrak.org](https://semanticsignal.qrak.org)). Nine tabs with brain activity, last prompt/response, position state, performance stats, news, market data, and memory bank.

- **🛡️ Risk Pipeline** — Pre-execution guard chain (symbol whitelist, max position size, cooldown) + dynamic SL/TP scaling with minimum 1.5 R:R enforced. Soft exits at candle close, hard exits at configurable intervals against live ticker price.

- **🔄 Multi-Provider AI Routing** — Primary: Google Gemini 3.5 Flash (free tier). Fallback chain through OpenRouter and LM Studio. Chart vision support on every provider that allows it.

- **🧪 1,200+ Tests** — Fully mocked test suite covering LLM output corruption, async races, rate-limit backoff, vector-DB boundaries, friction-reporting, closed-loop feedback, AST code indexing, and positional market types (spot / perpetual futures).

- **🤖 Multi-Agent AI Development** — Five specialized AI agents (Bolt ⚡, Palette 🎨, Sentinel 🛡️, Refactor ✨, Bugfixer 🐛) coordinate via a Supervisor 🧠. Each agent writes journal entries to `.ai/` — the project's collective memory. Journals auto-rotate on startup.

---

## Architecture

```mermaid
flowchart TB
    subgraph Data["Data Sources"]
        EX["Exchanges (CCXT) → OHLCV + Order Book + Trade Flow"]
        NEWS["RSS Feeds + Crawl4AI"]
        FUND["CoinGecko + DeFiLlama + Alternative.me"]
    end
    subgraph Analysis["Analysis Engine"]
        TC["Technical Calculator<br/>40+ indicators"]
        PE["Pattern Engine<br/>Deterministic indicator patterns"]
        CG["Chart Generator<br/>4K PNG with SMA/RSI/Volume"]
        RAG["RAG Engine<br/>News relevance scoring"]
    end
    subgraph Brain["🧠 Brain Layer"]
        VM["Vector Memory<br/>ChromaDB (4 collections)<br/>Experiences + Rules +<br/>Blocked Trades + Codebase Index"]
        REFL["Reflection Engine<br/>Rules from closed trades"]
        CTX["Context Builder<br/>Similarity retrieval +<br/>surprise ratio + confidence calibration"]
    end
    subgraph Search["🔍 Codebase Index"]
        CVI["AST Codebase Indexer<br/>160 files · 2,306 chunks<br/>Types: class/function/method/docs"]
        QUERY["query_codebase.py CLI<br/>Semantic code search<br/>by type, re-index, stats"]
    end
    subgraph Execution["Paper Execution"]
        RP["Risk Manager<br/>SL/TP, sizing, R:R,<br/>friction tracking"]
        GP["Guard Pipeline<br/>Symbol → Size → Cooldown"]
        STRAT["Trading Strategy<br/>ExitMonitor +<br/>PositionStatusMonitor"]
    end
    Data --> Analysis
    Analysis --> Brain
    Brain --> AI["AI Provider<br/>(Gemini / OpenRouter / LM Studio)"]
    AI --> RP --> GP --> STRAT
    STRAT -.->|Closed trade feedback| Brain
    CVI -.->|Indexes source| Analysis
    CVI -.->|Indexes source| Brain
    QUERY --> CVI
```

### Key Files

| Path | Role |
|------|------|
| `start.py` | Entry point — 8-stage dependency injection, ChromaDB + CoinGecko cache + journal rotation + codebase index auto-update |
| `src/app.py` | `CryptoTradingBot` — main async loop, ticker fetch, analysis orchestration |
| `src/trading/brain.py` | `TradingBrainService` — context assembly, experience recording, reflection triggers |
| `src/trading/vector_memory.py` | ChromaDB interface — trade experiences, semantic rules, blocked trades, embedding cache |
| `src/trading/vector_memory_rules.py` | `VectorMemoryRulesMixin` — semantic rule lifecycle: decay scoring, evidence ranking, surprise ratio |
| `src/rag/code_vector_index.py` | AST codebase indexer — parses Python source → ChromaDB chunks via SentenceTransformer |
| `src/analyzer/analysis_engine.py` | Market analysis orchestration — indicators, chart, RAG, LLM call |
| `src/managers/provider_orchestrator.py` | AI provider fallback chain with retry logic |
| `src/managers/risk_manager.py` | Dynamic SL/TP, position sizing, friction tracking |
| `src/managers/post_mortem_repository.py` | AI-written post-mortem after every closed trade |
| `src/trading/trading_strategy.py` | Position lifecycle, guard enforcement, exit monitoring |
| `src/analyzer/prompts/template_manager.py` | System prompt construction with falsification-based invalidation step |
| `src/analyzer/trend_validator.py` | Cross-checks LLM-reported trend strength against computed ADX |
| `src/analyzer/pattern_quality_scorer.py` | Deterministic pattern quality scoring replacing LLM's self-reported score |
| `src/notifiers/notifier.py` | Discord notifications with message expiration tracking |
| `scripts/query_codebase.py` | CLI tool for natural-language semantic codebase search |
| `scripts/rotate_journals.py` | Auto-rotation of AI agent journal files |

---

## Testing

```bash
# Full suite (1,200+ tests)
pytest tests/ -q

# Focused
pytest tests/test_ticker_retry.py tests/test_brain_integration.py tests/test_code_vector_index.py -q

# Linting
ruff check src tests start.py
```

| Test area | Count | Notes |
|-----------|-------|-------|
| Core trading | ~500 | Signals, orders, exits, risk, post-mortem |
| Vector memory | ~180 | ChromaDB operations, rules, scoring, embedding cache |
| Dashboard / brain router | ~120 | Decision pathways, admin endpoints, WS streaming |
| AST codebase index | ~80 | Chunking, search, re-index, type filtering |
| RAG / news / fundamentals | ~160 | RSS, Crawl4AI, news database, market data |
| Provider orchestration | ~100 | Fallback chain, retries, model pricing |
| Executor bridge | ~60 | Decision forwarding, dead letters, HTTP client |

---

## Configuration

Key settings in `config/config.ini`:

| Setting | Default | Description |
|---------|---------|-------------|
| `crypto_pair` | BTC/USDC | Trading pair |
| `timeframe` | 4h | Analysis candle timeframe |
| `provider` | googleai | AI provider (googleai, openrouter, lmstudio) |
| `demo_quote_capital` | 10000 | Simulated capital |
| `max_position_size` | 0.10 | Max position as fraction of capital |
| `stop_loss_type` | hard | hard (interval check) or soft (candle close) |
| `stop_loss_interval_minutes` | 15 | Hard exit check interval |
| `codebase_index_enabled` | true | Auto-build AST codebase vector index on startup |
| `codebase_index_dir` | data/codebase_index | ChromaDB persistent directory for codebase index |

Required API keys in `keys.env`:

| Variable | Required | For |
|----------|----------|-----|
| `GOOGLE_STUDIO_API_KEY` | Yes | Primary AI provider (free tier) |
| `GOOGLE_STUDIO_PAID_API_KEY` | If used | Paid tier Google AI |
| `OPENROUTER_API_KEY` | If used | Secondary AI provider |
| `BOT_TOKEN_DISCORD` | If used | Discord notifications |
| `MAIN_CHANNEL_ID` | If used | Discord notification channel |
| `COINGECKO_API_KEY` | No | Market metrics (rate limit boost) |
| `HF_TOKEN` | No | HuggingFace model access |

---

## Multi-Agent AI Development

The codebase uses a **Supervisor + 5 specialized agents** pattern for AI-assisted development:

| Agent | Emoji | Scope | Journal |
|-------|-------|-------|---------|
| **Supervisor** | 🧠 | Orchestrator — reads all journals, delegates to the right specialist | `.ai/supervisor.md` |
| **Bolt** | ⚡ | Performance — caching, async patterns, I/O, numpy, hot paths | `.ai/journal.md` |
| **Palette** | 🎨 | UX & Accessibility — dashboard HTML/CSS/JS, ARIA, responsive design | `.ai/palette-journal.md` |
| **Sentinel** | 🛡️ | Security — auth, CSP, rate limiting, XSS, input validation | `.ai/sentinel-journal.md` |
| **Refactor** | ✨ | Clean Code — isinstance chains, DRY violations, DI enforcement | `.ai/refactor-journal.md` |
| **Bugfixer** | 🐛 | Bugs & Regressions — verifying changes, running full suite | `.ai/bugfixing-journal.md` |

Journals auto-rotate on startup via `scripts/rotate_journals.py`. The full architecture blueprint lives in [`AGENTS.md`](AGENTS.md).

---

## Roadmap
 🔄 **Live Trading** — Real exchange order execution via [llm_trader_executor](https://github.com/qrak/llm_trader_executor) — currently in testing
- ⏳ **Multiple Trading Agent Personalities** — Conservative, aggressive, contrarian, trend-following strategists *(aspirational)*
- ⏳ **Multi-Model Consensus** — "Council of Models" architecture for collective decision-making *(aspirational)*

---

## Disclaimer

**NOT FINANCIAL ADVICE.** This software is experimental and in BETA. A real exchange execution service is in testing — use with caution. No warranty provided. Use at your own risk.

## License

[MIT](LICENSE.md)
