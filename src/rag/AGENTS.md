# AGENTS.md — RAG Engine

Scope: `src/rag/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical.

Deterministic retrieval — no LLM call. `rag_engine.py` builds a time-decayed local news/fundamentals index and hands the AnalysisEngine a token-limited text block.

| Module | File | Role |
|---|---|---|
| `NewsManager` | `news_manager.py` | fetch → dedupe → persist lifecycle, lock per update interval |
| `ContextBuilder` | `context_builder.py` | keyword search + token-limited block |
| `ArticleScoringPolicy` | `scoring_policy.py` | field-weighted relevance + recency |
| `LocalTaxonomyProvider` | `local_taxonomy.py` | crypto category hierarchy |
| `TickerManager` | `ticker_manager.py` | coin ↔ ticker mapping, dynamic symbol detection |
| `ArticleProcessor` | `article_processor.py` | normalize, extract body, strip boilerplate |
| `CategoryProcessor` | `category_processor.py` | news → taxonomy category |
| `IndexManager` | `index_manager.py` | one `coin_index` (coin → article offsets) |
| `CategoryCollisionResolver` | `collision_resolver.py` | category-word priority collisions |
| `RSSNewsProvider` | `news_ingestion/rss_provider.py` | RSS polling (4 feeds by default) |
| `ArticleBodyEnricher` | `news_ingestion/article_body_enricher.py` | page-body enrichment over plain HTTP |
| schema functions | `news_ingestion/schema_mapper.py` | `to_article_schema`, `strip_news_boilerplate`, `make_article_id` |
| `RagFileHandler` | `file_handler.py` | persisted news cache, fallback article loading, ticker map and priorities |
| `MarketDataManager` | `market_data_manager.py` | market-data lifecycle |
| `MarketDataCache` / `MarketDataFetcher` / `MarketOverviewBuilder` | `market_components/` | cache / CoinGecko+DeFiLlama fetch / overview text |

Pipeline: RSS poll → article-body enrichment → schema mapper → `ArticleProcessor` → `CategoryProcessor` → scoring → collision resolver → `RagFileHandler` → `IndexManager` (`coin_index`) → `ContextBuilder`.

## Rules

- Dedupe is **URL-first** at ingestion (`news_manager.py`, `rss_primitives.py`); at retrieval `RagEngine._dedupe_by_story` collapses one story told by several feeds (published within 2 h, ≥4 shared title words, Jaccard ≥0.4) to its best-scored article.
- `RSSNewsProvider` satisfies the implicit `news_client` protocol `NewsManager` expects: `async fetch_news(session)` and `filter_by_age(articles, max_age_hours)`.
- `NewsBoilerplateCleaner` strips publisher chrome, bylines, share widgets and footer/newsletter blocks; a body that loses all its content comes back empty so the caller drops the article.
- RSS contract: validate with defusedxml before `feedparser`, keep the RSS-only contract and first creator, drop Dublin Core description aliases so CoinDesk's empty duplicate cannot overwrite the real summary.
- Scoring: `FIELD_WEIGHTS = {"title": 10, "body": 3, "categories": 5, "tags": 4}`; keyword score scaled by `min(1, sqrt(1000 / len(body)))`; hard 24 h recency window; per-source weight from `[rag] news_source_weights`; the query appends a market-state phrase from the ticker's 24 h change. `calculate_article_relevance` returns an `ArticleScore` with every component, so the ranking can be explained, not only sorted.
- With `logger_debug` on, `retrieve_context` logs one `RAG ranking` line (`skip_live_stream`, so `Bot.log` only): stage counts (deduped, pool, shortlist, rendered) and up to 25 articles — ten rendered in order, ten best-scored that did not fit, five at zero — each with its stage status (`rendered`, `shortlisted_not_rendered`, `not_shortlisted`, `outside_pool`, `story_duplicate`, `backfill`, `nonpositive`), id, title, publication time, rank and score components. No bodies, no raw URLs, at most one line, and nothing collected when DEBUG is off. A missing story is then a stage, not a guess.
- The coin-index backfill only adds articles the scorer never ranked and that are younger than 24 h.
- `symbol_name_map` (`data/known_tickers.json`) carries the **full coin name** used to match headlines: `CoinGeckoAPI.coin_name` reads it from `coins/list` by the canonical id in `CANONICAL_IDS`, because one symbol maps to many coins (12 entries for `BTC`, 49 for `USDC`). A ticker whose stored name equals the ticker is a placeholder and gets replaced on the next `save_known_tickers`; a hand-written name is kept. An ambiguous symbol without a canonical id keeps the ticker.
- Enrichment chain: trafilatura text on the fetched HTML → paragraph extractor → raw RSS text.
- Density penalty defaults to 300 (`src/config/loader.py`). Long contaminated bodies are retried; a URL merge accepts a clean body ≥400 chars over a contaminated cache even when shorter, keeps the stored ID, and never downgrades clean full text to a summary.
- Dependency bounds are deliberate: `constraints.txt` caps htmldate <1.11, `requirements.txt` pins trafilatura 2.1.0 (newer need charset-normalizer/urllib3 incompatible with the installed CCXT); launchers pass `-c constraints.txt`. Run `pip check` before touching them.
- Config: 4h news interval, 5 articles; `config/rag_priorities.json` carries `important_categories` + `generic_priorities` only, never per-ticker weights.
- State: `data/news_cache/` (per-source, categorized), `data/news_fetch_preview/` (raw + normalized pre-enrichment), `data/backup/news_cache/` (cold storage).

## Degradation

- Page unreachable or unreadable → keep the feed text; feed down → skip source, warn, continue; rate-limited → serve stale `news_cache/`; empty cycle → empty context, analysis continues.
- Token budget exceeded → `ContextBuilder` truncates, keeps highest-scored articles; new ticker → dynamic name→ticker resolution; schema variants → per-source mapping in `schema_mapper.py`.
- `MarketDataFetcher` failure → cache fallback with stale flag.
