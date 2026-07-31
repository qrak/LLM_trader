# 📖 Semantic Signal: The Complete 7-Month Engineering Journey & System Architecture

*By [@qrak](https://github.com/qrak) — Self-taught developer from Wrocław, Poland*

---

> 💡 **Abstract:** Most commercial and open-source "AI trading bots" are rigid strategy calculators wrapped in subscriptions. They execute simple moving average crossovers, flatten charts into text floats, have zero memory of past mistakes, and hallucinate trends.
> 
> **Semantic Signal** is an open-source, asyncio-first cognitive trading bot built over 7 months (December 2025 – July 2026) to treat LLM cognitive constraints as core software engineering problems.

---

## 🔗 Quick Navigation & Live Links

*   📊 **[Live Dashboard (24/7 Watch the Brain Think)](https://semanticsignal.qrak.org)**
*   📖 **[Interactive Web Story & Technology](https://semanticsignal.qrak.org/story)**
*   🌐 **[Public Showcase & Landing Page](https://semanticsignal.qrak.org/landing.html)**
*   🐙 **[GitHub Repository](https://github.com/qrak/LLM_trader)**
*   💬 **[Discord Development Community](https://discord.gg/ZC48aTTqR2)**
*   ☕ **[Support Development on BuyMeACoffee](https://www.buymeacoffee.com/qrak)**


---

## 1. Chronological Timeline & Bottleneck Evolution

```
Phase 1: Float Prompts  ──►  Phase 2: Chart Vision  ──►  Phase 3: ChromaDB Memory  ──►  Phase 4: EV & Falsification  ──►  Phase 5: 8-Agent Dev  ──►  Phase 6: Hardened Executor
(Dec 25 - Jan 26)            (Feb - Mar 26)              (Apr - May 26)              (Jun 26)                     (Jul 26)             (End of Jul 26)
```

### Phase 1 (Dec 2025 – Jan 2026): The "Stateless Float" Trap
Started in a Wrocław apartment after 8-hour warehouse shifts. Early versions calculated RSI/MACD indicators and dumped them as text into Claude/Gemini prompts. The bot was stateless, forgot past trades, hallucinated trends, and bled capital (-4.2%).

### Phase 2 (Feb – Mar 2026): Multimodal Chart Vision
Realizing numbers strip out visual chart geometry, 900 lines of hardcoded pattern-matching heuristics were deleted. Plotly now renders a 1080p chart image (SMA, RSI, Volume, CMF, OBV) fed directly to Gemini Flash. The visual model spots pattern geometry (support wick rejections, wedge breakouts) far better than programmatic rules.

### Phase 3 (Apr – May 2026): Stateful Vector Memory & SQLite
Integrated local SQLite database (`trade_history.db`) and ChromaDB (768D BAAI/bge-base-en-v1.5 embeddings). Every closed trade is embedded. The bot queries top-5 similar past setups before deciding. Introduced the **Surprise Ratio** metric to filter out market noise.

### Phase 4 (June 2026): Expected Value & Falsification Gates
Added deterministic Expected Value math (Kelly sizing, min 1.5 R:R threshold) and the **Falsification Gate** (forcing the LLM to write a strict price invalidation trigger before any trade is executed).

### Phase 5 (Mid-Jul 2026): 8-Agent Dev System
Built a local multi-agent system (`.ai/` directory). A Supervisor orchestrates 7 specialized developer agents (Bolt, Palette, Sentinel, Refactor, Concise, Bugfixer, Smoke Tests) to maintain the codebase, resulting in 130+ verified commits.

### Phase 6 (Late-Jul 2026): Hardened Executor Separation
Decoupled the engine into Semantic Signal (reasoning, vision, news RAG, vector memory) and `llm_trader_executor` (CCXT order placement, leverage, OCO stop-losses, dead-letter queue).

---

## 2. Core System Architecture

```
OHLCV Data (999 candles, 5 exchanges) → 50+ Indicators (Numba JIT)
→ 1080p Chart Image (Gemini visual analysis)
→ Reddit Sentiment → 5 RSS Articles (RAG)
→ Vector Memory Query (ChromaDB, top-5 similar trades)
→ Brain Context (semantic rules + confidence stats)
→ LLM Prompt (self-debate + falsification check)
→ Gemini 3.6 Flash Analysis
→ TrendValidator (cross-check ADX) → PatternQualityScorer (deterministic score)
→ EV Calculation → Risk Profile Selection
→ Guard Pipeline (symbol → size → cooldown → R:R)
→ Decision (BUY/SELL/HOLD with SL/TP/Size)
→ Post-Mortem (closed trades) → Experience Recording → Vector Embedding
→ Reflection Engine (every 5 trades) → Rule Update
```

---

## 3. Key Mathematical & Algorithmic Foundations

### A. Numba JIT Indicator Engine
Custom indicators written in NumPy and Numba execute in microseconds on CPU:
```python
from numba import njit
import numpy as np

@njit(cache=True)
def _ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.empty_like(prices)
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
    return result
```

### B. The Surprise Ratio Metric
To prevent the model from learning bad habits from lucky wins (market noise):
```
Surprise Ratio = |Realized P&L - Expected P&L| / |Expected P&L|
```
Trades with a ratio > 1.5 carry a `⚠️ high surprise` annotation so vector memory discounts them in future setup queries.

### C. Deterministic Expected Value (EV) Gate
```
EV = (Win Rate × Average Win) - ((1 - Win Rate) × Average Loss) - Fees
```
Calculates historical win rates of matching vector setups, enforces Kelly Criterion position sizing, and rejects signals with negative EV or < 1.5 Risk-to-Reward ratio.

---

## 4. Verification & Testing

The codebase features **1,300+ automated unit and integration tests** covering:
- LLM output corruption & malformed JSON
- Async concurrency race conditions
- Rate-limiting backoff & exponential retries
- Vector DB boundary conditions
- Chaos engineering fault injection

---

*Built with ☕ and no sleep in Wrocław, Poland. No degree. No bootcamp. Just Python, asyncio, and a Ryzen 5700G.*
