# LLM-Trader - AI-Powered Trading Analysis Tool

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![Numba](https://img.shields.io/badge/numba-0.61.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CCXT](https://img.shields.io/badge/ccxt-4.4.52-blue.svg)
![Aiohttp](https://img.shields.io/badge/aiohttp-3.10.11-blue.svg)
![Rich](https://img.shields.io/badge/rich-13.9.4-blue.svg)
![Numpy](https://img.shields.io/badge/numpy-2.1.3-blue.svg)
![Scipy](https://img.shields.io/badge/scipy-1.15.1-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-2.2.3-blue.svg)
![OpenAI](https://img.shields.io/badge/openai-1.61.0-blue.svg)

LLM-Trader is a demo of how transformer models can be leveraged to analyze crypto markets and make trading decisions. This advanced algorithmic trading analysis tool uses large language models (LLMs) to analyze market data and provide trading insights. The bot processes technical indicators, market sentiment, and historical data to generate well-reasoned trading decisions.

## Features

- **Real-time Market Analysis**: Fetch and analyze OHLCV (Open, High, Low, Close, Volume) data from various crypto exchanges. This feature ensures that the bot operates on the most current market information, providing timely insights for trading decisions.
- **Advanced Technical Analysis**: Calculate and interpret over 20 technical indicators, including volume, momentum, trend, volatility, and statistical measures. These indicators provide a comprehensive view of market dynamics, enabling informed decision-making.
- **Sentiment Analysis**: Incorporate Fear & Greed Index data into market analysis to gauge overall market sentiment. This helps in understanding the emotional factors driving market movements, which can be crucial for predicting potential shifts.
- **AI-Powered Reasoning**: Use LLMs (Large Language Models) to analyze market conditions with chain-of-thought reasoning. The LLM processes the analyzed data to generate well-reasoned trading decisions, offering a clear and logical basis for each recommendation.
- **Position Management**: Automated trade suggestions with stop-loss and take-profit levels to manage risk and secure profits. This feature helps users to define and adhere to their risk management strategies.
- **Stream Processing**: Efficiently handle streaming responses from AI models, ensuring low-latency analysis and quick adaptation to changing market conditions.
- **Fallback Mechanisms**: Automatic switching to backup models if the primary model fails, ensuring continuous operation and reliable analysis even in case of service disruptions.

## Architecture

LLM-Trader consists of several core components:

- **MarketAnalyzer**: Fetches market data from the configured exchange, calculates technical indicators, incorporates sentiment data, and coordinates the overall market analysis process. It serves as the central hub for data processing and analysis.
- **ModelManager**: Handles interactions with AI models, including sending prompts, receiving responses, and switching to fallback models if necessary. This component ensures seamless communication with the AI models.
- **DataFetcher**: Retrieves OHLCV data from the configured exchange and Fear & Greed Index data from the Alternative.me API. It abstracts the data retrieval process, making it easier to switch data sources if needed.
- **TechnicalIndicators**: Calculates various market indicators, such as VWAP, RSI, MACD, and Bollinger Bands. This component provides the mathematical foundation for understanding market trends and conditions.
- **DataPersistence**: Manages position tracking and trade history, storing and retrieving data from a local JSON file. This ensures that the bot can maintain context across sessions and make informed decisions based on past performance.

## Technical Indicators Used

The bot analyzes markets using multiple indicators:

- **Volume**:
    - **VWAP (Volume Weighted Average Price)**: Measures the average price weighted by volume.
    - **TWAP (Time Weighted Average Price)**: Measures the average price over a specified time period.
    - **MFI (Money Flow Index)**: An oscillator that uses price and volume to identify overbought or oversold conditions.
    - **OBV (On Balance Volume)**: A momentum indicator that uses volume flow to predict price changes.
    - **CMF (Chaikin Money Flow)**: Measures the amount of money flow over a given period.
    - **Force Index**: Uses price and volume to assess the strength of a price movement.
- **Momentum**:
    - **RSI (Relative Strength Index)**: Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
    - **MACD (Moving Average Convergence Divergence)**: A trend-following momentum indicator that shows the relationship between two moving averages of a price.
    - **Stochastics**: Compares a security's closing price to its price range over a given period.
    - **Williams %R**: A momentum indicator that measures overbought or oversold levels.
- **Trend**:
    - **ADX (Average Directional Index)**: Measures the strength of a trend.
    - **SuperTrend**: A trend-following indicator that uses ATR (Average True Range) to determine the trend direction.
    - **Parabolic SAR (Parabolic Stop and Reverse)**: Identifies potential reversal points in the price trend.
- **Volatility**:
    - **ATR (Average True Range)**: Measures market volatility.
    - **Bollinger Bands**: Bands placed above and below a moving average, indicating areas of potential support and resistance.
- **Statistical**:
    - **Hurst Exponent**: Measures the randomness of a time series.
    - **Kurtosis**: Measures the "tailedness" of the probability distribution of a real-valued random variable.
    - **Z-Score**: Measures how many standard deviations away from the mean a data point is.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/qrak/LLM_trader.git
cd LLM_trader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the AI model in `config/model_config.ini`:
```ini
[model]
name = your_model_name
base_url = your_model_api_url
api_key = your_api_key
```

4. Update `config/config.ini` with your trading settings.

## Usage

Run the bot with the following command:

```bash
python main.py
```

The bot will:
1. Connect to the exchange
2. Fetch market data for the configured symbol
3. Calculate technical indicators
4. Retrieve market sentiment data
5. Generate a trading analysis using the configured LLM
6. Display the analysis and trading recommendation

## Configuration

### Exchange Settings
Configure the exchange, symbol, and timeframes in `config/config.ini`:

```ini
[exchange]
name = binance
symbol = BTC/USDT
timeframe = 1h
limit = 500
```

### Model Configuration
Set up your LLM model details in `config/model_config.ini`:

```ini
[model]
name = deepseek/deepseek-r1:free
base_url = https://openrouter.ai/api/v1
api_key = your_api_key_here
```

## Model Configuration Options

### Using OpenRouter (Recommended for Testing)

[OpenRouter](https://openrouter.ai/) is an API gateway that provides access to various LLMs through a unified interface. It's recommended for testing this project because:

1. **Free Access Tier**: Get started without upfront costs
2. **Multiple Models Available**: Access to various powerful models
3. **Simple API**: Unified interface compatible with OpenAI's API format
4. **Reliable Performance**: Professional hosting with good uptime

For initial testing, we recommend using the **deepseek-r1** model via OpenRouter:

1. Create an account at [OpenRouter](https://openrouter.ai/)
2. Generate an API key from your dashboard
3. Configure your `model_config.ini` file:
```ini
[model]
name = deepseek-r1
base_url = https://openrouter.ai/api/v1
api_key = your_api_key_here
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
