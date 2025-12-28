# 🤖 LLM_Trader v2

> **Status:** BETA / Paper Trading Only
>
> **Autonomous, asyncio-first trading bot that turns market + news + chart context into structured BUY/SELL/HOLD decisions.**

This is the **v2 evolution** of the original LLM_Trader project.
LLM_Trader v2 focuses on **continuous trading**, **risk-managed execution**, and **machine-readable (JSON) decisions** that can be reliably parsed and acted on.

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Data Sources
        Ex[Exchanges (CCXT)] --> |OHLCV/Trades| DC(Market Data Collector)
        News[CryptoCompare] --> |Articles| RAG(RAG Engine)
        Sent[Alternative.me] --> |Fear & Greed| DC
    end

    subgraph Analysis Core
        DC --> |Market Data| TC[Technical Calculator]
        DC --> |Price History| PA[Pattern Analyzer]
        DC --> |Candles| CG[Chart Generator]
        
        RAG --> |Context/Snippets| PB[Context Builder]
        TC --> |Indicators| PB
        PA --> |Patterns| PB
        CG --> |Chart Image| PB
        
        PB --> |System & User Prompt| MM{Model Manager}
    end

    subgraph AI Processing
        MM --> |Text/Image| P1[Google Gemini (Flash Latest)]
        MM --> |Text| P2[Claude 4.5 / 3.5 (OpenRouter)]
        MM --> |Text| P3[DeepSeek-R1 / Gemini 3.0]
        
        P1 --> |Response| ARP[Analysis Result Processor]
        P2 --> |Response| ARP
        P3 --> |Response| ARP
    end

    subgraph Execution (Paper Only)
        ARP --> |JSON Signal| TS[Trading Strategy]
        TS --> |Simulated Order| DP[Data Persistence]
        TS --> |Notification| DN[Discord Interface]
    end
```

## ✨ Verified Features

### 🧠 AI & LLM Support
- **Multi-Provider Support**: 
  - **Google Gemini (Flash Latest)**: Uses `gemini-flash-latest` for zero-maintenance updates.
  - **Claude 4.5 / Google Gemini 3 Pro **: Support for state-of-the-art reasoning models via OpenRouter.
  - **LM Studio**: Local LLM support verified via `lm_studio_base_url`.
- **Fallback Logic**: Automatically switches providers if primary fails (Google AI -> OpenRouter -> Local).
- **Vision-Assisted Trading**: Generates technical charts with indicators and sends them to vision-capable models (e.g., Gemini Flash) for visual pattern confirmation.


### 📢 RAG Engine (News & Context)
- **News Aggregator**: Requires a **CryptoCompare API Key**. The free tier typically offers ~150k lifetime requests, which is sufficient for continuous bot operation.
- **Smart Relevance Scoring**: Uses keyword density, category matching, and coin-specific heuristics (not just vector search) to filter noise.
- **Segmentation**: Uses `wtpsplit` for precise sentence segmentation to extract key facts/numbers.
- **Configurable Limits**: Adjustable token limits and article counts to manage context window.

### 🌍 Market Data & Exchanges
- **Multi-Exchange Aggregation**: Fetches data via `ccxt` from **5+ exchanges**:
  - Binance, KuCoin, Gate.io, MEXC, Hyperliquid
- **Comprehensive Data**:
  - OHLCV Candles (1m to 1w)
  - Order Book Depth & Spread Analysis
  - Recent Trade Flow (Buyer/Seller Pressure)
  - Funding Rates (for Perpetual Futures)

### ⚙️ Core Capabilities
- **Paper Trading Only**: Zero real-money risk. All orders are simulated (`create_order` is not connected to live exchange execution).
- **Continuity**: Tracks "Trading Brain" stats (confidence calibration, factor performance) to improve over time.

## 🗺️ Roadmap

- [x] **Local LLM Support** (LM Studio Integrated)
- [x] **Vision Analysis** (Chart Image Generation & Processing)
- [x] **RAG News Relevance Scoring**
- [ ] **Live Trading** (Execution Layer)
- [ ] **HuggingFace Local Embeddings** (Upgrade from keyword scoring)
- [ ] **Portfolio Management** (Multi-coin balancing)

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- [LM Studio](https://lmstudio.ai/) (Optional, for local inference)

### 2. Installation

```powershell
# Clone repo
git clone https://github.com/qrak/LLM_trader.git
cd LLM_trader

# Setup Virtual Environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```

### 3. Configuration

1. **Credentials**: Copy `keys.env.example` to `keys.env`.
   ```ini
   OPENROUTER_API_KEY=your_key_here
   GOOGLE_STUDIO_API_KEY=your_key_here
   # Optional but Recommended Keys
   # -----------------------------
   # CRYPTOCOMPARE_API_KEY: Increases rate limits and reliability.
   # Free Tier available: https://min-api.cryptocompare.com/
   CRYPTOCOMPARE_API_KEY=your_key_here
   # COINGECKO_API_KEY: Increases rate limits (~30 req/min vs ~10 req/min public).
   # Free Demo API Key available (header: x-cg-demo-api-key)
   COINGECKO_API_KEY=your_key_here
   ```

2. **Bot Config**: Edit `config/config.ini`.
   ```ini
   [ai_providers]
   provider = googleai  # or "local", "openrouter", "all"
   lm_studio_base_url = http://localhost:1234/v1
   
   [general]
   crypto_pair = BTC/USDT
   timeframe = 1h
   ```

## 🎮 Usage

Run the bot:
```powershell
python start.py              # Default from config
python start.py ETH/USDT     # Specific pair
```

### ⌨️ Controls
| Key | Action |
| :--- | :--- |
| **`a`** | **Force Analysis**: Run immediate market check |
| **`h`** | **Help**: Show available commands |
| **`q`** | **Quit**: Gracefully shutdown the bot |

## ⚠️ Disclaimer
**EDUCATIONAL USE ONLY.** This software is currently in **BETA** and configured for **PAPER TRADING**. No real financial transactions are executed. The authors are not responsible for any financial decisions made based on this software.
