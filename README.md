# Crypto Trading Bot

> **AI-powered cryptocurrency market analysis console application**

A Python-based tool that analyzes cryptocurrency markets using advanced AI models (Google AI, OpenRouter, LM Studio), technical indicators, pattern detection, and market sentiment analysis. Results are displayed directly in your terminal.

## 🚀 Features

- **Multi-Exchange Support**: Connects to Binance, KuCoin, and other major exchanges
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, Ichimoku, and more
- **Pattern Detection**: Identifies chart patterns (head & shoulders, triangles, wedges) and indicator patterns
- **AI-Powered Analysis**: Uses advanced AI models for market insights
- **News & Sentiment**: Integrates cryptocurrency news and market sentiment via RAG system
- **Console-First**: Simple, clean terminal output with no dependencies on external services

## 📋 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

```powershell
# Clone the repository
git clone <your-repo-url>
cd LLM_trader

# Create and activate virtual environment (Windows)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp keys.env.example keys.env
# Edit keys.env and add your API keys
```

### Configuration

1. **API Keys** (`keys.env`):
   - Add your AI provider keys (Google AI, OpenRouter, or LM Studio)
   - Add CoinGecko API key (optional)

2. **Settings** (`config/config.ini`):
   - Choose AI provider
   - Set default timeframe
   - Configure exchanges

### Usage

```powershell
# Analyze default symbol (BTC/USDT)
python start.py

# Analyze specific symbol
python start.py ETH/USDT

# Analyze with specific timeframe
python start.py SOL/USDT 1h
```

## 📊 Example Output

```
================================================================================
ANALYSIS RESULTS FOR BTC/USDT
================================================================================

--------------------------------------------------------------------------------
AI ANALYSIS:
--------------------------------------------------------------------------------
## Market Analysis for BTC/USDT

### Summary
Bitcoin shows strong bullish momentum with RSI at 67.3 indicating healthy buying 
pressure without being overbought. The MACD histogram has crossed positive, 
confirming trend strength...

[Detailed technical analysis with patterns, support/resistance levels, and recommendations]

--------------------------------------------------------------------------------
METADATA:
--------------------------------------------------------------------------------
Provider: googleai
Model: gemini-2.0-flash-exp
Timeframe: 4h
Language: English

--------------------------------------------------------------------------------
NEWS SOURCES:
--------------------------------------------------------------------------------
  general: https://example.com/crypto-news/general
  bitcoin: https://example.com/crypto-news/bitcoin
================================================================================
```

## 🏗️ Architecture

```
CryptoTradingBot
├── ExchangeManager          # Multi-exchange connectivity
├── RagEngine                # News & context enrichment
├── ModelManager             # AI provider orchestration
└── AnalysisEngine
    ├── DataCollector        # Market data fetching
    ├── TechnicalCalculator  # Indicator computation
    ├── PatternAnalyzer      # Pattern detection
    ├── PromptBuilder        # AI prompt generation
    └── ResultProcessor      # AI response parsing
```

## 🔧 Configuration Options

### Supported Timeframes
`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`

### AI Providers
- **Google AI**: Gemini models (fast, cost-effective)
- **OpenRouter**: Access to multiple AI models
- **LM Studio**: Local AI models (privacy-focused)

### Exchanges
- Binance
- KuCoin
- Gate.io
- And more via ccxt library

## 📚 Documentation

- **[USAGE.md](USAGE.md)**: Detailed usage guide
- **[AGENTS.md](AGENTS.md)**: Architecture and development guidelines
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: How to contribute

## 🛠️ Development

### Project Structure

```
LLM_trader/
├── src/
│   ├── analyzer/          # Analysis engine
│   ├── indicators/        # Technical indicators
│   ├── platforms/         # Exchange & AI provider integrations
│   ├── rag/              # News & context system
│   ├── models/           # AI model management
│   └── utils/            # Utilities
├── config/               # Configuration files
├── data/                 # Data storage
└── logs/                 # Application logs
```

### Running Tests

```powershell
pytest
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. Cryptocurrency trading involves substantial risk of loss. Always do your own research before making investment decisions.

## 🙏 Acknowledgments

- Built with [ccxt](https://github.com/ccxt/ccxt) for exchange connectivity
- Uses [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical indicators
- Powered by various AI providers (Google AI, OpenRouter, LM Studio)

---

**Made with ❤️ for crypto traders and developers**
