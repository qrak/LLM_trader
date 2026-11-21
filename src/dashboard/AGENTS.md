# AGENTS.md — Dashboard

Scope: `src/dashboard/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical for system-wide policy.

`server.py` (`DashboardServer`, FastAPI + WebSocket) is composed in `start.py` with `brain_service, vector_memory, analysis_engine, config, logger, unified_parser, persistence, exchange_manager`; shared state is the `dashboard_state.py` singleton. 7 routers: brain `/api/brain/*` · monitor `/api/monitor/*` · performance `/api/performance/*` · visuals `/api/visuals/*` · settings `/api/settings/*` · console · ws `/ws/*`.

`/admin/` is the local settings desk (`static/admin/index.html`, `admin.css`, `admin.js`). `GET /api/settings/status` and `GET /api/settings/docs/{slug}` are public; config/keys reads and every write are loopback-only, checked against the client IP, `Host` and `Origin`, with `CF-Connecting-IP` and public `X-Forwarded-For` rejected. The keys API never returns a value; config/keys writes are atomic; INI edits preserve comments/order/line endings and are validated with `Config(config_path=temp)`; the desk is `restart`-scoped. `docs/assets` is mounted at `/docs-assets`; `/api/settings/*` is `no-store`. The retired console — server-side sessions, `WritableConfig` INI writes, live log streaming, the force-analysis trigger, the feed/human-input flags and the human-approval gate — lives in git history only.


`GET /api/status/state` returns the bot lifecycle snapshot (`available:false` when the dashboard has no provider); `/api/brain/position` embeds the same object as `lifecycle`. The Overview strip (`#state-strip`, `static/modules/lifecycle_panel.js`) reads it from the position payload and renders the API's `label`/`severity` verbatim — the UI must not re-map state values. The position card adds the execution-proof line and an `entering`/`closing`/`divergence` notice. It labels the strategy level `Bot SL` and the saved entry ratio `Entry R:R`. The separate `venue_stop_price` is the executor-reported trigger from a recent matching position/protection reconciliation, never a price computed from the requested SL and never a promised fill. Unverified or superseded evidence is unavailable. `docs/STATE_MACHINES.md` is the vocabulary map.

Runtime: `0.0.0.0:8000`, GZip on, `static/` mounted (`index.html`, `main.js`, `css/`, `modules/`), lifespan startup/shutdown; CORS off unless `dashboard.enable_cors = true`.

- `LogStreamHandler` gives every subscriber its own bounded `asyncio.Queue` (500) and drops that subscriber's oldest entry instead of blocking the logger; `ConsoleBuffer` keeps the same lines in a day-grouped ring buffer (7 days, 10 000 lines each) addressed by offset, `0` = today.
- `DocsLibrary` renders only its allowlisted `DOCS` slugs to safe HTML; an unknown slug yields nothing rather than a path.

## Persistence

History reaches the UI only through the injected `PersistenceManager` — routes never open `trade_history.json` or the DB. `PerformanceRouter` gets `persistence` from `DashboardServer`; brain/vector endpoints call `persistence.load_trade_history()`. Empty history renders an empty state, not an error.

`GET /api/performance/cycles` includes `inference_cost {analyses, total_usd, per_analysis_usd, coverage}` from the same `summarize()` call; existing keys stay unchanged. Journal lines written before cost tracking simply lower `coverage` instead of failing the endpoint.

## Cache (Cloudflare edge)

- Bypass `/api/brain/refresh-price` and `/api/brain/vectors?query=*`; cache `GET /api/visuals/charts/latest` 300s keeping its revision query (never strip it at the edge — rotate after an analysis or tab return, reject late responses from an older revision); cache shells and versioned assets; never a broad `/api/*` rule — positions and status stay fresh.
- `/api/brain/refresh-price` coalesces callers into one exchange attempt behind an asyncio lock and reuses the result 10s (monotonic clock, failures included); query strings, reload and browser no-cache cannot bypass it, HTTP stays `no-store`, trading-loop fetches and WS cadence unchanged.
- `?preview=true` on `/api/monitor/news` returns the same 30 cards with 301-char excerpts, no bodies or internal search fields; the default response and RAG cache stay full. `/api/status/countdown` is cacheable only as a short-lived fallback (live updates use WS). `DashboardServer` hands `analysis_engine.rag_engine` to `MonitorRouter` (the brain does not own news RAG); verify startup plus the route with a real `TradingBrainService`, including no-brain/no-RAG and disk fallback.
- Vector factor stats read the closed-trade recorder's metadata: S/R uses `support_resistance_strength_score`, Trend groups use the entry snapshot not exit-time `market_regime`. Keep API factor names stable.
- `/api/settings/*` is always `no-store`; `/docs-assets/*` is a `StaticFiles` mount of `docs/assets` (path traversal and cache headers handled there). The settings page and docs are public; only the config/keys API is loopback.

## UI conventions

- The dashboard is a presentation layer only: UI work never changes backend logic or an API payload.
  A redesign (moving panels between tabs, replacing a visualisation) is allowed when the user asks
  for one and has approved a written plan; ordinary polish stays small and single-purpose. Motion
  stays 0.2s ease / 0.3s cubic-bezier and honours `@media (prefers-reduced-motion: reduce)`.
- Colour lives in `static/css/base.css` as custom properties and nowhere else: `--accent-primary`
  for interactive elements and the one primary number, `--accent-success/-warning/-danger` for
  meaning, neutrals for surfaces and text, `--brand-bmc` for the one branded button. No glow
  (`text-shadow`, coloured `box-shadow`, gradient hairlines, animated logo/status dot). No hardcoded
  hue in a stylesheet or a JS string; JS reads a token with
  `getComputedStyle(document.documentElement).getPropertyValue('--name')`.
- Headings and panel titles are sentence case; `text-transform: uppercase` is reserved for values at
  or below 12 px (KPI labels, table headers, chips).
- Reuse the CSS variables and classes (`.toolbar-btn`, `.panel`, `.btn`, `.spinner`, `.empty-state`,
  `.chain-num`). No inline style, no `onclick=""` — wire with `addEventListener`.
- One module per panel in `static/modules/`, imported by `main.js`: `decision_panel.js` renders the
  last decision and its market context, `vector_panel.js` the Memory panel (recalled rows by
  default, the whole store on demand), `lessons_panel.js` the post-mortem archive and its FTS
  search, `rules_panel.js` the learned rules and the scope they were learned in.
- The Brain tab owns every decision input — market context, recalled trades, post-mortem lessons,
  learned rules, blocked attempts — because they all feed the same question. Secondary payloads
  stay behind a native `<details>` (the win-rate breakdown) or a five-row default with an explicit
  "show all" button, so the tab opens light. `fullscreen.js` maps a panel only when it has a
  content element; a panel without an entry gets no button.
- Accessibility: `aria-label`/`role` on interactive elements, full keyboard operability (Tab, Enter/
  Space, Esc, focus trap in modals and lightbox), WCAG 2.1 AA contrast, `aria-live="polite"` for
  dynamic regions, decorative emoji `aria-hidden="true"`, inputs labelled with
  `aria-required`/`aria-invalid`. Drill-down uses native `<details>`, not a scripted pane.
- Every async action shows state (disabled + spinner, "Copied!", inline errors): no dead click, no
  silent failure.
- One number has one source: the Overview "Active Rules" KPI reads `rule_count` from
  `/api/brain/status` (`semantic_rule_count`, which counts only `active: true` rules — the same value
  the Brain tab's rules panel counts); Memory no longer prints that count. `/api/brain/rules` is
  capped at 20 and feeds the Brain rules panel and the empty-state hint only, so it must never feed
  the KPI.
- No build step: no npm/yarn/pnpm, no `package.json`, no new CDN scripts; bump `?v=` on every JS/CSS change (one-year immutable cache). A change inside a module also needs `main.js?v=` bumped in `index.html`: `main.js` carries the module URLs and is itself immutable for a year, so a module bump alone never reaches a returning browser. A CDN script is pinned to an exact version — `apexcharts@latest` silently crossed v3→v8 under this repo.
- Phones (≤768 px) are a first-class target: KPI grid 2×2, `.panel-header` wraps, `.stat-strip` is a 2×2 grid, console controls are at least 36 px, and the experience table becomes one card per row through `td[data-label]` rather than a sideways scroll — a cell added without `data-label` is unlabelled on a phone, so add the attribute with the cell.
- The performance chart plots percent change since the first recorded point, not dollars (±0.5% on a $10k account is unreadable on a currency axis). `updatePerformanceData()` must only call `chart.updateSeries` when the fetched points actually changed; the 30 s poll otherwise resets zoom and hand-set ranges, which is exactly the bug it was fixed for. `chart.updateOptions` on `yaxis` needs an **array** (`[{...}]`) or Apex resets the axis labels.
- Web layer, beyond the root security rules: strict CSP with CDN allowlist, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy` limiting geolocation/microphone/camera, HSTS over HTTPS, per-visitor rate limiting (300 reads / 30 writes per minute keyed by `resolve_client_ip()`, so tunnel traffic is not one bucket), 50-request cap with 503/504 shedding, WebSocket origin validation and connection limits.

Degradation: WS disconnect → reconnect and resubscribe; no data or no SQLite rows → empty state; in-process restart → `DashboardState` keeps last-known values; large vector queries → truncated/paginated.
