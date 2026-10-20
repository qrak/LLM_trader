# AGENTS.md — RAG Engine

Scope: `src/rag/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical.

Deterministic retrieval — no LLM call. `rag_engine.py` builds a time-decayed local news/fundamentals index and hands the AnalysisEngine a token-limited text block.

| Module | File | Role |
|---|---|---|
| `NewsManager` | `news_manager.py` | fetch → dedupe → persist lifecycle, lock per update interval |
| `NewsRepository` | `news_repository.py` | read/write news storage |
| `ContextBuilder` | `context_builder.py` | keyword search + token-limited block |
| `ArticleScoringPolicy` | `scoring_policy.py` | field-weighted relevance + recency |
| `LocalTaxonomy` | `local_taxonomy.py` | crypto category hierarchy |
| `TickerManager` | `ticker_manager.py` | coin ↔ ticker mapping, dynamic symbol detection |
| `ArticleProcessor` | `article_processor.py` | normalize, extract body, strip boilerplate |
| `CategoryProcessor` | `category_processor.py` | news → taxonomy category |
| `IndexManager` | `index_manager.py` | 4 in-memory indices |
| `CategoryCollisionResolver` | `collision_resolver.py` | category-word priority collisions |
| `RSSCrawl4AINewsProvider` | `news_ingestion/rss_provider.py` | RSS polling (4 feeds by default) |
| `Crawl4AIEnricher` | `news_ingestion/crawl4ai_enricher.py` | optional page enrichment |
| schema functions | `news_ingestion/schema_mapper.py` | `to_article_schema`, `strip_news_boilerplate`, `make_article_id` |
| `file_handler.py` | | file-based fallback store when indices break |
| `MarketDataManager` | `market_data_manager.py` | market-data lifecycle |
| `MarketDataCache` / `MarketDataFetcher` / `MarketOverviewBuilder` | `market_components/` | cache / CoinGecko+DeFiLlama fetch / overview text |

Pipeline: RSS poll → Crawl4AI enrich (optional) → schema mapper → `ArticleProcessor` → `CategoryProcessor` → scoring → collision resolver → `NewsRepository` → `IndexManager` (category, tag, coin, keyword) → `ContextBuilder`.

## Rules

- Dedupe is **URL-first** (`news_manager.py`, `rss_primitives.py`), no fuzzy matching; URL normalization stays local.
- RSS contract: validate with defusedxml before `feedparser`, keep the RSS-only contract and first creator, drop Dublin Core description aliases so CoinDesk's empty duplicate cannot overwrite the real summary.
- Scoring: `FIELD_WEIGHTS = {"title": 10, "body": 3, "categories": 5, "tags": 4}` plus a recency factor; no source-authority weighting.
- Enrichment chain: Crawl4AI HTML → trafilatura text → Crawl4AI markdown when short/unusable → HTML extractor → aiohttp direct → raw RSS text.
- Density penalty defaults to 300 (`loader.py`). Long contaminated bodies are retried; a URL merge accepts a clean body ≥400 chars over a contaminated cache even when shorter, keeps the stored ID, and never downgrades clean full text to a summary.
- Dependency bounds are deliberate: `constraints.txt` caps htmldate <1.11, `requirements.txt` pins trafilatura 2.1.0 (newer need charset-normalizer/urllib3 incompatible with the installed CCXT); launchers pass `-c constraints.txt`. Run `pip check` before touching them.
- Config: 4h news interval, 5 articles; `config/rag_priorities.json` carries `important_categories` + `generic_priorities` only, never per-ticker weights.
- State: `data/news_cache/` (per-source, categorized), `data/news_fetch_preview/` (raw + normalized pre-enrichment), `data/backup/news_cache/` (cold storage).

## Degradation

- Crawl4AI missing → aiohttp direct fetch; feed down → skip source, warn, continue; rate-limited → serve stale `news_cache/`; empty cycle → empty context, analysis continues.
- Token budget exceeded → `ContextBuilder` truncates, keeps highest-scored articles; new ticker → dynamic name→ticker resolution; schema variants → per-source mapping in `schema_mapper.py`.
- `MarketDataFetcher` failure → cache fallback with stale flag.
