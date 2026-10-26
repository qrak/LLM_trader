# AGENTS.md — src/

Root [`AGENTS.md`](../AGENTS.md) §13 is canonical. Contracts below cover `src/` and the packages without their own file: `parsing/`, `config/`, `utils/`, `notifiers/`, `platforms/`, `logger/`.

- `analyzer/` candle fetch returns `(np.ndarray, float)`; signatures frozen.
- `parsing/` `UnifiedParser` keeps codeblock → `raw_decode` → error fallback, signal disambiguation, `_convert_value` tuples, `ValueError` on a bad datetime.
- `config/` `ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))` plus `value.strip()` before conversion.
- `rag/` `enrich_items()` with crawl4ai on or off; `fetch_market_overview`; non-finite timestamps rejected.
- `platforms/` DefiLlama bodies are decoded as JSON whatever Content-Type they declare (`/v2/chains` serves JSON under `text/html`); the container shape is checked once in `_fetch_json(url, expected_type)`, and a non-200, non-JSON or wrong-shape answer is logged with its URL and never raised into the cycle.
- `utils/` all helper/decorator signatures (`@retry_async`, `@retry_api_call`, format/data utils); `time_utils` is the only place that decides which clock a rendered timestamp uses — `local_stamp()`/`day_stamp()` for log lines and their folders, `utc_stamp()` for UTC-only text, `utc_and_local()` for a paired label. `local_timezone()` reads `[general] timezone` and falls back to `TZ`/OS, with the zone cached. `CostStorage.snapshot()` is the read-only cumulative cost/token total across providers; `record_usage()`, `save()` and `_load_or_create()` keep their existing semantics.
- `logger/` file lines carry the display zone **with its UTC offset** (`25.10.2026 02:04:15 +01:00`) and the day folder (`logs/Bot/YYYY_MM_DD`) uses that same zone, so the folder always matches the lines inside it; crash lines from `_write_fallback_crash` use the identical stamp.
- Retries cover missing/null/empty `choices` within the bounded budget; explicit provider errors keep their class; a valid post-mortem is accepted once, never read as an absent decision.
- SDK conversion keeps error detail and classifiable codes, redacts keys — never an empty completion.
- `notifiers/` identical arguments across Discord, console, file.
- `dashboard/` DOM refs in sync, `?v=` bumped, WS reconnect, admin login intact.
- `trading/statistics_calculator.py` rejects `entry_price <= 0`.
