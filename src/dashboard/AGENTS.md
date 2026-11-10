# AGENTS.md — Dashboard

Scope: `src/dashboard/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical for system-wide policy.

`server.py` (`DashboardServer`, FastAPI + WebSocket) is composed in `start.py` with `brain_service, vector_memory, analysis_engine, config, logger, unified_parser, persistence, exchange_manager`; shared state is the `dashboard_state.py` singleton. 7 routers: brain `/api/brain/*` · monitor `/api/monitor/*` · performance `/api/performance/*` · visuals `/api/visuals/*` · settings `/api/settings/*` · console · ws `/ws/*`.

`/admin/` is the local settings desk (`static/admin/index.html`, `admin.css`, `admin.js`). `GET /api/settings/status` and `GET /api/settings/docs/{slug}` are public; config/keys reads and every write are loopback-only, checked against the client IP, `Host` and `Origin`, with `CF-Connecting-IP` and public `X-Forwarded-For` rejected. The keys API never returns a value; config/keys writes are atomic; INI edits preserve comments/order/line endings and are validated with `Config(config_path=temp)`; the desk is `restart`-scoped. `docs/assets` is mounted at `/docs-assets`; `/api/settings/*` is `no-store`. The retired console — server-side sessions, `WritableConfig` INI writes, live log streaming, the force-analysis trigger, the feed/human-input flags and the human-approval gate — lives in git history only.


`GET /api/status/state` returns the bot lifecycle snapshot (`available:false` when the dashboard has no provider); `/api/brain/position` embeds the same object as `lifecycle`. The Overview strip (`#state-strip`, `static/modules/lifecycle_panel.js`) reads it from the position payload and renders the API's `label`/`severity` verbatim — the UI must not re-map state values. The position card adds the execution-proof line and an `entering`/`closing`/`divergence` notice. `docs/STATE_MACHINES.md` is the vocabulary map.

Runtime: `0.0.0.0:8000`, GZip on, `static/` mounted (`index.html`, `main.js`, `css/`, `modules/`), lifespan startup/shutdown; CORS off unless `dashboard.enable_cors = true`.

## Persistence

History reaches the UI only through the injected `PersistenceManager` — routes never open `trade_history.json` or the DB. `PerformanceRouter` gets `persistence` from `DashboardServer`; brain/vector endpoints call `persistence.load_trade_history()`. Empty history renders an empty state, not an error.

`GET /api/performance/cycles` includes `inference_cost {analyses, total_usd, per_analysis_usd, coverage}` from the same `summarize()` call; existing keys stay unchanged. Journal lines written before cost tracking simply lower `coverage` instead of failing the endpoint.

## Cache (Cloudflare edge)

- Bypass `/api/brain/refresh-price` and `/api/brain/vectors?query=*`; cache `GET /api/visuals/charts/latest` 300s keeping its revision query (never strip it at the edge — rotate after an analysis or tab return, reject late responses from an older revision); cache shells and versioned assets; never a broad `/api/*` rule — positions and status stay fresh.
- `/api/brain/refresh-price` coalesces callers into one exchange attempt behind an asyncio lock and reuses the result 10s (monotonic clock, failures included); query strings, reload and browser no-cache cannot bypass it, HTTP stays `no-store`, trading-loop fetches and WS cadence unchanged.
- `?preview=true` on `/api/monitor/news` returns the same 30 cards with 300-char excerpts, no bodies or internal search fields; the default response and RAG cache stay full. `/api/status/countdown` is cacheable only as a short-lived fallback (live updates use WS). `DashboardServer` hands `analysis_engine.rag_engine` to `MonitorRouter` (the brain does not own news RAG); verify startup plus the route with a real `TradingBrainService`, including no-brain/no-RAG and disk fallback.
- Vector factor stats read the closed-trade recorder's metadata: S/R uses `support_resistance_strength_score`, Trend groups use the entry snapshot not exit-time `market_regime`. Keep API factor names stable.
- `/api/settings/*` is always `no-store`; `/docs-assets/*` is a `StaticFiles` mount of `docs/assets` (path traversal and cache headers handled there). The settings page and docs are public; only the config/keys API is loopback.

## UI conventions

- UI work never changes backend logic: one micro-improvement at a time, under ~50 lines, never a redesign. Animation stays 0.2s ease / 0.3s cubic-bezier and honours `@media (prefers-reduced-motion: reduce)`.
- Reuse the CSS variables and classes (`.toolbar-btn`, `.panel`, `.btn-primary`, `.spinner`, `.empty-state`). No hardcoded colour, no inline style, no `onclick=""` — wire with `addEventListener`.
- Accessibility: `aria-label`/`role` on interactive elements, full keyboard operability (Tab, Enter/Space, Esc, focus trap in modals and lightbox), WCAG 2.1 AA contrast, `aria-live="polite"` for dynamic regions, decorative emoji `aria-hidden="true"`, inputs labelled with `aria-required`/`aria-invalid`.
- Every async action shows state (disabled + spinner, "Copied!", inline errors): no dead click, no silent failure.
- One number has one source: the Overview "Active Rules" KPI reads `rule_count` from `/api/brain/status` (the store-wide `semantic_rule_count`, same value the Memory tab prints). `/api/brain/rules` is capped at 20 and only drives the empty-state hint, so it must never feed the KPI.
- No build step: no npm/yarn/pnpm, no `package.json`, no new CDN scripts; bump `?v=` on every JS/CSS change (one-year immutable cache). A change inside a module also needs `main.js?v=` bumped in `index.html`: `main.js` carries the module URLs and is itself immutable for a year, so a module bump alone never reaches a returning browser.
- Web layer, beyond the root security rules: strict CSP with CDN allowlist, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy` limiting geolocation/microphone/camera, HSTS over HTTPS, per-visitor rate limiting (300 reads / 30 writes per minute keyed by `resolve_client_ip()`, so tunnel traffic is not one bucket), 50-request cap with 503/504 shedding, WebSocket origin validation and connection limits.

Degradation: WS disconnect → reconnect and resubscribe; no data or no SQLite rows → empty state; in-process restart → `DashboardState` keeps last-known values; large vector queries → truncated/paginated.
