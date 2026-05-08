# 🤖 SEMANTIC SIGNAL LLM (LLM Trader)

> **Status:** BETA / Research Edition
>
> **Note:** This project runs in demo-account and paper-trading mode. Real exchange order execution is not implemented in this public branch.
>
> **Autonomous, asyncio-first trading bot that turns market + news + chart context into structured BUY/SELL/HOLD decisions.**

🔗 **[Live Dashboard](https://semanticsignal.qrak.org)** — Real-time view of the neural trading brain

## Key Features

- **Vector-Only Trading Brain**: ChromaDB vector store for semantic trade retrieval and adaptive thresholds.
- **Outcome-Aware Memory System**: Timeframe-aware recency decay and active relevance windows keep prompt memory focused on fresh market regimes.
- **Semantic Rule Learning**: Reflection loops generate best-practice, anti-pattern, corrective, and AI-mistake rules with diagnostics such as win/loss split, expectancy, and dominant exit profile.
- **Hard Exit Monitoring**: Bot-side interval checks for stop-loss and take-profit against live ticker prices, independent of candle closes.
- **RAG Engine**: Aggregates news from free RSS feeds with optional Crawl4AI enrichment, plus fundamentals from DefiLlama.
- **AI & LLM Support**: Multi-provider support (Google Gemini, OpenRouter, BlockRun.AI, LM Studio) with fallback logic and vision-assisted trading.
- **Multi-Exchange Aggregation**: Fetches data via `ccxt` from Binance, KuCoin, Gate.io, MEXC, Hyperliquid.

![Semantic Signal LLM Dashboard - Overview](img/dashboard1.png)
![Semantic Signal LLM Dashboard - Brain Activity](img/dashboard2.png)
![Semantic Signal LLM Dashboard - Last Prompt](img/dashboard3.png)
![Semantic Signal LLM Dashboard - Last Response](img/dashboard4.png)
![Semantic Signal LLM Dashboard - Statistics](img/dashboard5.png)
![Semantic Signal LLM Dashboard - Latest News](img/dashboard6.png)
![Semantic Signal LLM Dashboard - Market Data](img/dashboard7.png)
![Semantic Signal LLM Dashboard - Memory Bank](img/dashboard8.png)

## Tech Stack

- **Language**: Python 3.13+
- **Database (Vector)**: ChromaDB
- **Dashboard Backend**: FastAPI, WebSockets
- **Dashboard Frontend**: HTML, Vanilla JS, Vis.js, ApexCharts
- **AI Integrations**: Google Gemini, OpenRouter, BlockRun.AI, LM Studio
- **Market Data**: CCXT, [CoinGecko](https://www.coingecko.com), Alternative.me, DefiLlama
- **Code Quality**: Ruff, Pylint, Mypy

## Prerequisites

- Python 3.13+
- [LM Studio](https://lmstudio.ai/) (Optional — for local offline inference)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/qrak/LLM_trader.git
cd LLM_trader
```

### 2. Setup Virtual Environment

```powershell
# Setup Virtual Environment
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
# Install required dependencies
pip install -r requirements.txt

# Optional but recommended for browser-grade news extraction
crawl4ai-setup

# For development (linting, testing tools)
pip install -r requirements-dev.txt
```

### 4. Environment Setup

Copy the example keys file:

```powershell
cp keys.env.example keys.env
```

Configure the following variables in `keys.env`:

| Variable | Description |
| --- | --- |
| `OPENROUTER_API_KEY` | (Required) OpenRouter API key if used as a provider. |
| `GOOGLE_STUDIO_API_KEY` | (Required) Google AI Studio API key (free tier). |
| `GOOGLE_STUDIO_PAID_API_KEY` | (Optional) Google AI Studio API key (paid tier). |
| `COINGECKO_API_KEY` | (Optional) Free demo key for market metrics. |
| `BLOCKRUN_WALLET_KEY` | (Optional) Private key for BlockRun.AI x402 micropayments. |
| `HF_TOKEN` | (Optional) Hugging Face token for improved model download/auth rate limits when embeddings/models are fetched. |

### 5. Bot Configuration

Copy the example config file:

```powershell
cp config/config.ini.example config/config.ini
```

Key sections to configure:

```ini
[ai_providers]
# Options: "local", "googleai", "openrouter", "blockrun", "all"
provider = googleai
google_studio_model = gemini-3-flash-preview
openrouter_base_model = google/gemini-3-flash-preview

[general]
crypto_pair = BTC/USDC
timeframe = 4h

[model_config]
google_temperature = 1.0
google_thinking_level = high

[dashboard]
host = 0.0.0.0
port = 8000

[demo_trading]
demo_quote_capital = 10000
transaction_fee_percent = 0.00075

[risk_management]
# soft = candle-close checks, hard = bot-side interval checks on live ticker
stop_loss_type = soft
stop_loss_check_interval = 1h
take_profit_type = soft
take_profit_check_interval = 1h
max_position_size = 0.10
position_size_fallback_low = 0.03
position_size_fallback_medium = 0.05
position_size_fallback_high = 0.07

[rag]
# Whitelist filter — only these source keys are enabled. Leave empty to enable all configured news_source_*_url entries.
news_sources = coindesk,cointelegraph,decrypt

# Use Crawl4AI for page enrichment
news_crawl4ai_enabled = true
```

Both exit check intervals must be less than or equal to `[general] timeframe`. Soft exits are evaluated only at candle close; hard exits are bot-side checks against live ticker price at the configured interval. Position sizing is capped by `max_position_size`, and fallback size tiers are used only when AI returns missing or invalid `position_size`.

### 6. Start the Bot

Run the bot directly via Python:

```powershell
python start.py
```

The dashboard will be available at `http://localhost:8000`.

### 7. Controls

| Key | Action |
| --- | --- |
| **`a`** | **Force Analysis**: Run immediate market check |
| **`d`** | **Toggle Dashboard**: Enable or disable the dashboard while the program is running |
| **`h`** | **Help**: Show available commands |
| **`q`** | **Quit**: Gracefully shutdown the bot |

## Architecture

At its core, the Crypto Trading Bot leverages LLMs along with a Retrieval-Augmented Generation (RAG) engine to ingest market news, evaluate technical indicators, pattern recognition, and internal trading history ("brain memory"). By combining statistical indicators with human-like textual evaluation, it formulates `BUY`, `SELL`, `HOLD`, or `CLOSE` decisions.

```mermaid
graph TD
    subgraph Data Sources
        Ex["Exchanges (CCXT)"] --> |OHLCV/Trades| DC(Market Data Collector)
        News[RSS Feeds + Crawl4AI] --> |Articles| RAG(RAG Engine)
        Sent[Alternative.me] --> |Fear & Greed| DC
        DeFi[DefiLlama] --> |TVL/Fundamentals| RAG
    end

    subgraph Analysis Core
        DC --> |Market Data| TC[Technical Calculator]
        DC --> |Price History| PA[Pattern Analyzer]
        DC --> |Candles| CG[Chart Generator]
        
        RAG --> |News Context| CB[Context Builder]
        
        %% Orchestration / Assembly
        TC --> |Indicators| PB[Prompt Builder]
        PA --> |Patterns| PB
        CB --> |RAG Context| PB
        CG --> |Chart Image| PB
        
        PB --> |System & User Prompt| MM{Model Manager}
    end

    subgraph AI Processing
        %% Provider Selection Logic (Sequential / Fallback)
        MM -.-> |Primary| Google["Google Gemini (Text + Vision)"]
        MM -.-> |Fallback| OR["OpenRouter (Text + Vision)"]
        MM -.-> |Pay-per-request| BR["BlockRun.AI"]
        MM -.-> |Local| Local["LM Studio"]
        
        Google --> |Response| ARP[Analysis Result Processor]
        OR --> |Response| ARP
        BR --> |Response| ARP
        Local --> |Response| ARP
    end

    subgraph Execution ["Execution (Paper Only)"]
        ARP --> |JSON Signal| TS[Trading Strategy]
        TS --> |Simulated Order| DP[Data Persistence]
        TS --> |Notification| DN["Notifier"]
    end
```

### Application Entry Points

- `start.py`
  - The true entry point implementing the **Composition Root** and Dependency Injection (DI) pattern.
  - Bootstraps API clients, memory layers, LLM managers, and the RAG engine concurrently.
  - Instantiates the `DashboardServer`.
- `src/app.py`
  - Contains the `CryptoTradingBot` class. Manages the continuous polling rhythm, trading lifecycle, and real-time Discord alerts.

### Directory Structure & Subsystems

```text
src/
├── analyzer/          # Turns mathematical bounds into descriptive semantic markers
│   ├── pattern_engine/# Validates topological shapes & regressions (Head & Shoulders, Trenlines)
│   ├── formatters/    # Converts array flows and objects into markdown strings
│   └── prompts/       # Dynamic composition of system/user blocks for LLM contexts
├── rag/               # Retrieval-Augmented Knowledge Engine
├── trading/           # State, positions, risk metrics & biological "Brain" tracking  
├── managers/          # Shared state persistence and AI model routing
├── platforms/         # External REST/GraphQL integrations (CCXT, Gemini, OpenRouter)
├── dashboard/         # Real-time Web UI telemetry (FastAPI, WebSockets)
├── indicators/        # Massive suite of NumPy/Numba powered array calculation files
├── parsing/           # Resilient LLM JSON output bounds checking
├── notifiers/         # Markdown-styled embedded notifications for Discord/Console
└── factories/         # Safe DI dependency construction masking internal logic
tests/                 # Extensive unit and integration validations with API knocking
docs/                  # Deep technical documentation and component plans
```

### Runtime Mechanics

#### 1. Scheduler and Loop Control

- `CryptoTradingBot` in `src/app.py` controls the main async loop.
- The loop wakes up on configured cadence (timeframe-aware), or immediately when forced by hotkey.
- A cycle can be skipped when guard conditions fail (for example, missing market data), which prevents low-quality prompts.

#### 2. Market and Context Assembly

- The market-data pipeline collects OHLCV and related market state through `ccxt` integrations.
- Technical calculators transform raw candles into structured indicator payloads and pattern signals.
- The RAG path fetches and normalizes crypto news from RSS sources, then optionally enriches article content through Crawl4AI.
- Fundamentals and sentiment inputs are merged into the same context window so the model sees both price structure and narrative pressure.

#### 3. Memory Retrieval and Similarity Weighting

- Vector memory is queried for similar historical setups using technical/context features from the current snapshot.
- Similarity is not the only ranking factor: recency decay is applied so fresh regimes have more influence than stale periods.
- Timeframe-aware windows constrain what is considered relevant (for example, a 4h profile uses tighter freshness than a higher timeframe profile).

#### 4. Prompt Building and Contract Enforcement

- Prompt builders combine market structure, indicators, patterns, news evidence, and memory snippets into a strict system/user prompt format.
- News and external snippets are treated as untrusted evidence in the prompt hierarchy, so they cannot override policy instructions.
- The expected response format is a compact, parser-safe JSON contract plus concise reasoning fields.

#### 5. Model Routing and Fallback Strategy

- `model_manager` selects the configured primary provider and can fall back across supported providers when needed.
- Text and optional chart-vision paths are coordinated so the response still lands in the same output contract.
- Provider differences are normalized before parsing, which keeps downstream trading logic provider-agnostic.

#### 6. Parsing, Validation, and Risk Normalization

- Raw model output is parsed through resilient JSON extraction and contract checks.
- Trading fields such as signal, confidence, SL/TP, and position size are normalized before strategy execution.
- Position sizing is hard-capped by `max_position_size`; fallback sizing tiers are used only when AI output is missing or invalid.

#### 7. Paper Execution and Exit Mechanics

- The strategy layer converts validated decisions into paper-trade actions, persistence updates, and notifier output.
- Soft exits are evaluated on candle-close strategy checks.
- Hard exits are evaluated by interval monitors against live ticker prices, independent of candle close.
- Dashboard WebSocket updates stream the current state, position metrics, and latest decision telemetry in near real time.

#### 8. Reflection and Continuous Learning

- Closed trades feed post-trade reflection in the brain/memory layer.
- The system synthesizes semantic rules from repeated outcomes: best-practice patterns, anti-patterns, corrective rules, and AI-mistake rules.
- These rules and similar-experience retrieval influence future prompts, forming a feedback loop between outcome quality and next-cycle decision context.

## Testing

The codebase contains a rigorous `tests/` directory covering integration logic, mocking, and unit testing validation. This minimizes regressions specifically in LLM parsing behavior.

```powershell
# Run the test suite using pytest (make sure pytest is installed)
pytest tests/
```

## Roadmap

- [x] **Local LLM Support** (LM Studio Integrated)
- [x] **Vision Analysis** (Chart Image Generation & Processing)
- [x] **RAG News Relevance Scoring**
- [x] **Vector Memory System** (ChromaDB + Semantic Search)
- [x] **Discord Integration** (Real-time signals, positions, and performance stats)
- [x] **Interactive CLI** (Hotkeys for manual control)
- [x] **Web Dashboard**: Real-time visualization of synaptic pathways and neural state.
- [x] **BlockRun.AI Integration**: Pay-per-request AI access via x402 micropayments.
- [x] **DefiLlama Fundamentals**: On-chain TVL context in the RAG pipeline.
- [ ] **Multiple Trading Agent Personalities**: Diverse strategist personalities (conservative, aggressive, contrarian, trend-following).
- [ ] **Multi-Model Consensus Decision-Making**: A "Council of Models" architecture.
- [ ] **Live Trading**: Execution Layer integration for verified order placement.
- [ ] **Static Documentation Site**: Transition docs into a browsable static site (e.g. `MkDocs` or `VitePress`).

## Community & Support
- **Discord**: [Join our community](https://discord.gg/ZC48aTTqR2) for live signals, development chat, and support.
- **GitHub Issues**: Report bugs or suggest new features.

## Disclaimer
**NOT FINANCIAL ADVICE.** This software is experimental and in **BETA**. It is configured for demo-account and paper-trading workflows, and real exchange order execution is not implemented in this public branch.

Use of this repository is at your own risk. You are solely responsible for:
- Verifying whether your intended use is permitted in your jurisdiction.
- Complying with local laws, regulations, and platform terms before any real-money deployment.
- Validating AI-generated signals independently before making trading decisions.

No warranty is provided, and the authors and contributors assume no liability for losses, misuse, or regulatory non-compliance. See [LICENSE.md](LICENSE.md) for legal terms.

## Contributors
- **Vicky (1bcMax)**: Implementation of BlockRun.AI provider and x402 payment integration.

## License
Licensed under the [MIT License](LICENSE.md).
