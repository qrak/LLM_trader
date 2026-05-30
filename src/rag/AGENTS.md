# 📰 RAG Engine Agent — News Aggregation & Market Fundamentals

> **Module path:** `src/rag/rag_engine.py` (orchestrator) + 15+ collaborators in `src/rag/`
> **Type:** Retrieval-Augmented Generation pipeline for news and market fundamentals
> **Sources:** CoinDesk, CoinTelegraph, Decrypt, CryptoSlate, RSS feeds + Crawl4AI enrichment + DefiLlama

---

## Agent Persona & Role

The RAG Engine is the **news and fundamentals intelligence subsystem**. It aggregates, enriches, categorizes, and indexes crypto market news and fundamentals data into a context window that the AnalysisEngine injects into every LLM decision prompt.

Unlike the Reading Agent, the RAG Engine does **not** send queries to an LLM — it performs deterministic information retrieval over a curated, time-decayed local index. The output is a text block of recent, relevant articles and fundamental metrics formatted for prompt injection.

### Key Collaborators

| Module | File | Responsibility |
|--------|------|----------------|
| `NewsManager` | `news_manager.py` | Fetch → deduplicate → persist news lifecycle |
| `NewsRepository` | `news_repository.py` | Read/write interface for news storage |
| `ContextBuilder` | `context_builder.py` | Keyword search + token-limited context formatting |
| `ScoringPolicy` | `scoring_policy.py` | 5-factor relevance scoring for news articles |
| `LocalTaxonomy` | `local_taxonomy.py` | Domain-specific crypto category hierarchy |
| `TickerManager` | `ticker_manager.py` | Coin/ticker ↔ name mapping for symbol detection |
| `ArticleProcessor` | `article_processor.py` | Normalization, body extraction, boilerplate stripping |
| `CategoryProcessor` | `category_processor.py` | News → taxonomy category classification |
| `IndexManager` | `index_manager.py` | 4 in-memory indices: category, tag, coin, keyword |
| `RSSProvider` | `news_ingestion/rss_provider.py` | RSS feed polling from configured sources |
| `Crawl4AIEnricher` | `news_ingestion/crawl4ai_enricher.py` | Web-page enrichment via Crawl4AI (optional) |
| `SchemaMapper` | `news_ingestion/schema_mapper.py` | Maps source-specific schemas to unified format |
| `MarketDataManager` | `market_data_manager.py` | Lifecycle management for market data |
| `MarketDataCache` | `market_components/market_data_cache.py` | Caching layer for market data fetches |
| `MarketDataFetcher` | `market_components/market_data_fetcher.py` | Fetches market data from CoinGecko/DeFiLlama |
| `MarketOverviewBuilder` | `market_components/market_overview_builder.py` | Builds market overview text from raw data |

---

## Inputs

### From RSS/Web
- RSS feed XML from configured crypto news sources
- Raw HTML pages (via Crawl4AI enrichment fallback chain)
- Per-source raw and normalized JSON artifacts (saved to `data/news_fetch_preview/`)

### From External APIs
- CoinGecko — market-wide stats (dominance, volume, BTC dominance)
- DeFiLlama — TVL data, protocol fundamentals

### From Configuration
- `config/config.ini`: news update interval (4h), max articles (5)
- `rag_priorities.json`: per-ticker/category priority weights
- `LocalTaxonomy`: predefined category hierarchy for news classification

### From Query (Consumer-side — AnalysisEngine)
- Keyword search terms (derived from active trading pair, trending coins)
- Token limit for context window truncation

---

## Outputs

### RAG Context Block (for LLM prompt injection)
- Formatted string: "--- Market News & Fundamentals ---\n[Article 1 summary]\n[Article 2 summary]\n..."
- Token-limited to fit within model context window
- Timestamp-stamped for freshness awareness

### Persisted State
- `data/news_cache/` — recent news JSON (per-source, categorized)
- `data/news_fetch_preview/` — pre-enrichment raw + normalized artifacts
- `data/backup/news_cache/` — cold storage for historical retrieval

### Indexed Artifacts (via IndexManager)
4 in-memory indices maintained in sync:
- **category_index** — taxonomy category → article indices
- **tag_index** — article tag → article indices
- **coin_index** — detected coin/ticker → article indices
- **keyword_index** — category and title keywords → article indices

---

## Pipeline: News Ingestion Flow

```
RSS Provider polls configured RSS sources (4 by default, every 4h)
    ↓
Crawl4AI Enricher (optional — degrades to aiohttp on failure)
    ↓
SchemaMapper → unified format
    ↓
ArticleProcessor (dedup, normalize, strip boilerplate)
    ↓
CategoryProcessor → LocalTaxonomy mapping
    ↓
ScoringPolicy (5-factor relevance scoring)
    ↓
CollisionResolver (handle cache collisions)
    ↓
NewsRepository → IndexManager (category/tag/coin/keyword)
    ↓
ContextBuilder → token-limited formatted context block
```

### Scoring Policy — 5 Factors
1. **Recency** — newer articles score higher
2. **Source authority** — configured per-source weights
3. **Ticker relevance** — coin/ticker mention count
4. **Category match** — alignment with trading pair category
5. **Body length / completeness** — penalizes truncated or thin articles

### Enrichment Fallback Chain
```
Crawl4AI (primary) → aiohttp direct fetch (degradation) → store raw RSS text
```

---

## Edge Cases & Guardrails

| Scenario | Handling |
|----------|----------|
| **Crawl4AI unavailable** | Degrades gracefully to aiohttp direct fetch |
| **RSS feed down** | Skips source, logs warning, continues with other sources |
| **Duplicate article detected** | `CollisionResolver` uses body-length-aware dedup (exact + fuzzy matching) |
| **Empty news cycle** | Returns empty context — AnalysisEngine continues without news |
| **Provider rate-limited** | Cache fallback — serves stale data from `news_cache/` |
| **Body too short (<100 chars)** | Treated as low-quality, penalized in scoring |
| **Token limit exceeded** | `ContextBuilder` truncates to token budget, keeps highest-scored articles |
| **Index corruption** | `file_handler.py` provides file-based fallback storage |
| **New ticker not in TickerManager** | Dynamic symbol detection with name→ticker resolution |
| **Market data fetch fails** | `MarketDataFetcher` falls back to cache, stale data flagged |
| **Boilerplate in article body** | `ArticleProcessor` strips common boilerplate patterns |
| **Schema mismatch per source** | `SchemaMapper` handles per-source field name variants |
| **Concurrent fetch cycles** | `NewsManager` serializes via lock on update interval |
