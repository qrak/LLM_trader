# AGENTS.md — src/

Root [`AGENTS.md`](../AGENTS.md) §13 is canonical. Contracts below cover `src/` and the packages without their own file: `parsing/`, `config/`, `utils/`, `notifiers/`, `platforms/`, `logger/`.

- `analyzer/` candle fetch returns `(np.ndarray, float)`; signatures frozen.
- `parsing/` `UnifiedParser` keeps codeblock → `raw_decode` → error fallback, signal disambiguation, `_convert_value` tuples, `ValueError` on a bad datetime.
- `config/` `ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))` plus `value.strip()` before conversion.
- `rag/` `enrich_items()` with crawl4ai on or off; `fetch_market_overview`; non-finite timestamps rejected.
- `utils/` all helper/decorator signatures (`@retry_async`, `@retry_api_call`, format/data utils).
- Retries cover missing/null/empty `choices` within the bounded budget; explicit provider errors keep their class; a valid post-mortem is accepted once, never read as an absent decision.
- SDK conversion keeps error detail and classifiable codes, redacts keys — never an empty completion.
- `notifiers/` identical arguments across Discord, console, file.
- `dashboard/` DOM refs in sync, `?v=` bumped, WS reconnect, admin login intact.
- `trading/statistics_calculator.py` rejects `entry_price <= 0`.
