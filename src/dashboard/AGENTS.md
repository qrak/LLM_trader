# AGENTS.md — Dashboard

Scope: `src/dashboard/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical for system-wide policy.

`server.py` (`DashboardServer`, FastAPI + WebSocket) is composed in `start.py` with `brain_service, vector_memory, analysis_engine, config, logger, unified_parser, persistence, exchange_manager`; shared state is the `dashboard_state.py` singleton. 7 routers: brain `/api/brain/*` · monitor `/api/monitor/*` · performance `/api/performance/*` · visuals `/api/visuals/*` · admin `/api/admin/*` · console · ws `/ws/*`.

`monitor` also serves `/api/monitor/github_activity`. `github_activity.py` makes the one network call (the account's public events) and `build_feed` turns events into rows in GitHub's wording; the route caches the page 300 s, serves the last good page as `stale` for a day when GitHub fails, and answers with `error` only when nothing is cached. `GITHUB_USER` selects the account, `GITHUB_TOKEN` is optional (60 → 5000 requests/hour); the panel is `static/modules/github_activity_panel.js` in the "GitHub Activity" tab, and a new event type must be skipped, not rendered as a blank row.

`/admin`: HMAC-SHA256 cookie sessions, credentials from `keys.env` (`ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH`, `ADMIN_SIGNING_KEY`). `WritableConfig` (`src/config/writable_config.py`) does atomic INI writes (`os.replace()`) and edits the FILE only: `Config` is read once at import, so every key is `restart`-scoped and the admin form must not promise `hot`/`cycle` application (`POST /api/admin/config/reload` refreshes the form, not the bot). The provider enum offers every value `VALID_PROVIDERS` accepts; a value missing from `options` renders as the first option and the row's Save writes that first option back. Controls `POST /api/admin/system/trigger-analysis` (fires the force-analysis event) and two API-only state endpoints, `/toggle-feed` and `/human-input`: they store a value that no consumer reads yet (the feed flag is reported by `/system/status`; the operator note is stored and returned by the same endpoint, and is NOT injected into the prompt), so the admin UI does not present them as working controls. `LogStreamHandler` feeds subscriber `asyncio.Queue`s at `/api/admin/logs/stream?token=…`; the console WS `/api/admin/console?token=…` accepts `force_analysis`, `toggle_feed`, `human_input`, `get_status`. UI at `static/admin/index.html`.

Runtime: `0.0.0.0:8000`, GZip on, `static/` mounted (`index.html`, `main.js`, `css/`, `modules/`), lifespan startup/shutdown; CORS off unless `dashboard.enable_cors = true`.

## Persistence

History reaches the UI only through the injected `PersistenceManager` — routes never open `trade_history.json` or the DB. `PerformanceRouter` gets `persistence` from `DashboardServer`; brain/vector endpoints call `persistence.load_trade_history()`. Empty history renders an empty state, not an error.

`GET /api/performance/cycles` includes `inference_cost {analyses, total_usd, per_analysis_usd, coverage}` from the same `summarize()` call; existing keys stay unchanged. Journal lines written before cost tracking simply lower `coverage` instead of failing the endpoint.

## Cache (Cloudflare edge)

- Bypass `/api/brain/refresh-price` and `/api/brain/vectors?query=*`; cache `GET /api/visuals/charts/latest` 300s keeping its revision query (never strip it at the edge — rotate after an analysis or tab return, reject late responses from an older revision); cache shells and versioned assets; never a broad `/api/*` rule — positions and status stay fresh.
- `/api/brain/refresh-price` coalesces callers into one exchange attempt behind an asyncio lock and reuses the result 10s (monotonic clock, failures included); query strings, reload and browser no-cache cannot bypass it, HTTP stays `no-store`, trading-loop fetches and WS cadence unchanged.
- `?preview=true` on `/api/monitor/news` returns the same 30 cards with 300-char excerpts, no bodies or internal search fields; the default response and RAG cache stay full. `/api/status/countdown` is cacheable only as a short-lived fallback (live updates use WS). `DashboardServer` hands `analysis_engine.rag_engine` to `MonitorRouter` (the brain does not own news RAG); verify startup plus the route with a real `TradingBrainService`, including no-brain/no-RAG and disk fallback.
- Vector factor stats read the closed-trade recorder's metadata: S/R uses `support_resistance_strength_score`, Trend groups use the entry snapshot not exit-time `market_regime`. Keep API factor names stable.

## UI conventions

- UI work never changes backend logic: one micro-improvement at a time, under ~50 lines, never a redesign. Animation stays 0.2s ease / 0.3s cubic-bezier and honours `@media (prefers-reduced-motion: reduce)`.
- Reuse the CSS variables and classes (`.toolbar-btn`, `.panel`, `.btn-primary`, `.spinner`, `.empty-state`). No hardcoded colour, no inline style, no `onclick=""` — wire with `addEventListener`.
- Accessibility: `aria-label`/`role` on interactive elements, full keyboard operability (Tab, Enter/Space, Esc, focus trap in modals and lightbox), WCAG 2.1 AA contrast, `aria-live="polite"` for dynamic regions, decorative emoji `aria-hidden="true"`, inputs labelled with `aria-required`/`aria-invalid`.
- Every async action shows state (disabled + spinner, "Copied!", inline errors): no dead click, no silent failure.
- One number has one source: the Overview "Active Rules" KPI reads `rule_count` from `/api/brain/status` (the store-wide `semantic_rule_count`, same value the Memory tab prints). `/api/brain/rules` is capped at 20 and only drives the empty-state hint, so it must never feed the KPI.
- No build step: no npm/yarn/pnpm, no `package.json`, no new CDN scripts; bump `?v=` on every JS/CSS change (one-year immutable cache). A change inside a module also needs `main.js?v=` bumped in `index.html`: `main.js` carries the module URLs and is itself immutable for a year, so a module bump alone never reaches a returning browser.
- Web layer, beyond the root security rules: strict CSP with CDN allowlist, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy` limiting geolocation/microphone/camera, HSTS over HTTPS, per-visitor rate limiting (300 reads / 30 writes per minute keyed by `resolve_client_ip()`, so tunnel traffic is not one bucket), 50-request cap with 503/504 shedding, WebSocket origin validation and connection limits.

Degradation: WS disconnect → reconnect and resubscribe; no data or no SQLite rows → empty state; in-process restart → `DashboardState` keeps last-known values; large vector queries → truncated/paginated.
