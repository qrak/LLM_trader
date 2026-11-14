# AGENTS.md — src/

Root [`AGENTS.md`](../AGENTS.md) §13 is canonical. Contracts below cover `src/` and the packages without their own file: `parsing/`, `config/`, `utils/`, `notifiers/`, `platforms/`, `logger/`, `composition/`, `mcp_tools/`.

- `analyzer/` candle fetch returns `(np.ndarray, float)`; signatures frozen.
- `parsing/` `UnifiedParser` keeps codeblock → `raw_decode` → error fallback, signal disambiguation, `_convert_value` tuples, `ValueError` on a bad datetime.
- `config/` `ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))` plus `value.strip()` before conversion. `Config(config_path=...)` validates a candidate INI without touching the live file; `SettingsEditor` preserves comments/order/line endings and writes atomically.
- `rag/` `enrich_items()` over plain HTTP; `fetch_market_overview`; non-finite timestamps rejected.
- `platforms/` DefiLlama bodies are decoded as JSON whatever Content-Type they declare (`/v2/chains` serves JSON under `text/html`); the container shape is checked once in `_fetch_json(url, expected_type)`, and a non-200, non-JSON or wrong-shape answer is logged with its URL and never raised into the cycle.
- `utils/` all helper/decorator signatures (`@retry_async`, `@retry_api_call`, format/data utils); `time_utils` is the only place that decides which clock a rendered timestamp uses — `local_stamp()`/`day_stamp()` for log lines and their folders, `utc_stamp()` for UTC-only text, `utc_and_local()` for a paired label. `local_timezone()` reads `[general] timezone` and falls back to `TZ`/OS, with the zone cached. `CostStorage.snapshot()` is the read-only cumulative cost/token total across providers; `record_usage()`, `save()` and `_load_or_create()` keep their existing semantics.
- Private helpers live in the class that calls them — `python scripts/check_loose_helpers.py` is the gate — and a family of helpers is a class: `NewsBoilerplateCleaner` owns the publisher-chrome cleaners, `ArticleBodyTextExtractor` the HTML body readers, `ExitRecords` the exit-record verification and `TradeHistoryReader` the read-only history queries.
- `logger/` file lines carry the display zone **with its UTC offset** (`25.10.2026 02:04:15 +01:00`) and the day folder (`logs/Bot/YYYY_MM_DD`) uses that same zone, so the folder always matches the lines inside it; crash lines from `_write_fallback_crash` use the identical stamp.
- Retries cover missing/null/empty `choices` within the bounded budget; explicit provider errors keep their class; a valid post-mortem is accepted once, never read as an absent decision.
- A rejected JSON-mode reply names its reason (`reply carried no choices`, `empty reply`, `prose without a JSON object`, `object without a payload key`, `truncated at the output cap`, `reply stopped by the content filter`, `malformed JSON (...)`) and logs its full text; `finish_reason` is the discriminator between a reply cut by the output cap and one the model ended with broken JSON, and a recovered attempt leaves a single INFO line.
- SDK conversion keeps error detail and classifiable codes, redacts keys — never an empty completion.
- `notifiers/` identical arguments across Discord, console, file.
- `dashboard/` DOM refs in sync, `?v=` bumped, WS reconnect, `/admin` is the local settings desk (loopback config/keys, public docs).
- `trading/statistics_calculator.py` rejects `entry_price <= 0`.
- `utils/` `SerializableMixin` converts a field only when its declared annotation **is** `datetime`; a `datetime | None` annotation skips conversion — keep the bare annotation on a field that must round-trip.
- `composition/` `ProvisioningMixin` is mixed into `CompositionRoot`; each `_provision_*` stage builds one slice, and nothing outside that file constructs a collaborator.
- `mcp_tools/` `TradeHistoryReader` opens its own connection per query with `mode=ro` and `PRAGMA query_only`, so a write fails inside SQLite instead of being trusted to the caller.
