# 📊 Dashboard Agent — Real-Time Web UI

> **Module path:** `src/dashboard/server.py` (FastAPI) + 5 routers in `src/dashboard/`
> **Type:** FastAPI + WebSocket streaming dashboard
> **Live URL:** [https://semanticsignal.qrak.org](https://semanticsignal.qrak.org)

---

## Agent Persona & Role

The Dashboard is the **real-time monitoring and visualization interface** for the LLM Trader system. It provides:

- **Live decision streaming** — WebSocket push of each analysis cycle result
- **Performance analytics** — SQLite-backed trade history, P&L curves, win rates
- **Brain state inspection** — vector memory contents, learned rules, confidence stats
- **System health monitoring** — provider status, token costs, cycle timing

---

## Architecture

### Server (`server.py`)
Injected via DI from `start.py`:
```
DashboardServer(
  brain_service, vector_memory, analysis_engine,
  config, logger, unified_parser, persistence, exchange_manager
)
```

### 5 API Routers

| Router | File | Endpoints |
|--------|------|-----------|
| **Brain** | `routers/brain.py` | `/api/brain/*` — decisions, signals, context |
| **Monitor** | `routers/monitor.py` | `/api/monitor/*` — system health, provider status |
| **Performance** | `routers/performance.py` | `/api/performance/*` — P&L, statistics, trade history |
| **Visuals** | `routers/visuals.py` | `/api/visuals/*` — chart data, indicator plots |
| **WebSocket** | `routers/ws_router.py` | `/ws/*` — real-time streaming |

### Static Frontend
- `src/dashboard/static/` — HTML, Vanilla JS, Vis.js, ApexCharts
- Dark theme, auto-refreshing WebSocket data
- Real-time candlestick charts + indicator overlays

---

## Key Behaviors

- **Startup:** Bound to `0.0.0.0:8000` (configurable)
- **CORS:** Enabled for dashboard frontend access
- **GZip:** Compression enabled for API responses
- **Static files:** Mounted from `static/` directory
- **Lifecycle:** Managed via FastAPI lifespan context manager
- **State:** Shared via `dashboard_state.py` singleton
- **Trade history:** Performance and brain endpoints read via injected `PersistenceManager`; direct `trade_history.json` reads are forbidden

## Persistence Contract

- `PerformanceRouter` receives `persistence` from `DashboardServer` and calls persistence-backed history APIs.
- Brain/vector endpoints use `persistence.load_trade_history()` when they need trade-history context.
- Dashboard routes must not open runtime files directly for trade history; SQLite access stays behind `PersistenceManager` / `SQLiteTradeHistory`.
- Empty SQLite history should render empty dashboard state gracefully, not as an error.

---

## Cache Strategy (Cloudflare)

| Rule | Target | Reason |
|------|--------|--------|
| **Bypass** | `/api/brain/refresh-price` | Volatile price endpoint |
| **Bypass** | `/api/brain/vectors?query=*` | High-cardinality search |
| **Cache** | `/api/status/countdown` | Static countdown data |
| **Cache** | `/api/*` | Safe GET traffic |
| **Cache** | HTML shell pages | Static shell |
| **Cache** | Static assets | Versioned assets |

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| **WebSocket disconnect** | Client auto-reconnects, resends subscription |
| **No data yet** | Returns empty state gracefully |
| **No SQLite trade history rows** | Performance and brain routes return empty history/state gracefully |
| **Backend restart** | DashboardState retains last-known values across restarts |
| **Large vector memory queries** | Truncated/paginated API responses |
| **Concurrent dashboard access** | FastAPI async handles concurrent requests |
