# AGENTS.md — src/

Scope: `src/`. Root [`AGENTS.md`](../AGENTS.md) is canonical for system-wide policy and never overridden here.

Package detail: [`trading/`](trading/AGENTS.md), [`trading/guards/`](trading/guards/AGENTS.md), [`analyzer/`](analyzer/AGENTS.md), [`indicators/`](indicators/AGENTS.md), [`managers/`](managers/AGENTS.md), [`rag/`](rag/AGENTS.md), [`dashboard/`](dashboard/AGENTS.md). This file carries the contracts that span `src/` and the packages that have no file of their own (`parsing/`, `utils/`, `notifiers/`, `platforms/`, `logger/`).

## Module Regression Contracts

- `analyzer/` candle fetching returns `(np.ndarray, float)`, not `(list, float)`, and the public signatures stay unchanged.
- `parsing/` (UnifiedParser) keeps deserializing AI responses: `ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))` with `value.strip()` before type conversion, regex signal disambiguation, `_convert_value` handling plain tuples and raising `ValueError` on an invalid datetime.
- `rag/` `enrich_items()` works with crawl4ai enabled and disabled; `fetch_market_overview` keeps its contract; the market data cache rejects a non-finite timestamp.
- `utils/` keeps every helper and decorator signature (`@retry_async`, `@retry_api_call`, format and data utilities).
- JSON-provider retries include missing/null/empty `choices`, with the existing bounded retry budget and an unchanged request. Explicit provider errors retain their own classification; valid post-mortem objects are accepted once, not mistaken for absent trading decisions.
- SDK response conversion preserves top-level provider error details with API-key redaction. Keep serialized error codes classifiable after conversion; never reduce an error envelope to an unexplained empty completion.
- `notifiers/` keeps the same method arguments across the Discord, console and file notifiers.
- `dashboard/` keeps DOM references in sync, the `?v=` cache-busting version bumped on asset change, WebSocket reconnection working and admin login intact.
- `statistics_calculator.py` rejects `entry_price <= 0`.
