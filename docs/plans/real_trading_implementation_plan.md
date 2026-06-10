# Real SPOT + FUTURES Trading — Implementation Plan

> **Target:** Convert the bot from paper simulation to real **Binance** order execution (spot + futures), gated behind config toggles. Default behavior stays paper.
> **Strategy:** Binance first → testnet first → native exchange SL/TP → configurable order type → real-balance capital.
> **Audience:** An implementer (including a less-capable model) who needs exact files, signatures, and step order. Do **NOT** skip phases. Each phase ends with a concrete *Verify* gate.
> **Repo:** LLM_trader — Python 3.13, `.venv/`, run with `python start.py`, tests with `python -m pytest tests/ -q`.

---

## 0. CRITICAL WARNINGS — RESOLUTIONS

These were raised during planning and are now **resolved** with the decisions baked into this plan:

1. **Secrets exposure — RESOLVED.** [`keys.env`](../../keys.env) is already gitignored ([`.gitignore`](../../.gitignore) line 53). No further action beyond adding the new exchange key slots (Phase 0). Still create exchange API keys as **trade-only, withdrawals DISABLED, IP allow-listed**.

2. **EU / futures availability — RESOLVED via runtime jurisdiction guard.** Crypto perpetual futures are restricted in many countries (MiCA/ESMA in Europe, CFTC/FCA bans, AML/KYC limits elsewhere) and Binance geo-blocks them. The bot must **detect the user's country/jurisdiction at startup when possible** (see Phase 1b — `JurisdictionService`) and check it against a concrete blocked-country list (see §4 Phase 1b for the August-2025 list). Behavior:
   - If detection succeeds and the country is on the **blocked list**, and `market_type = futures` is configured → **fail loudly at startup** with a clear, actionable error naming the country.
   - If detection **is not possible** (network/API failure) and `market_type = futures` → also **fail loudly** ("futures availability could not be verified for your jurisdiction; set `market_type = spot` or `allow_unverified_futures = true` to override at your own legal risk").
   - The startup banner must always **tell the user whether futures is enabled/available for their country**, so they know their status. This is **not legal advice**; it is an availability guard. The blocked list is point-in-time data (August 2025) and must be operator-maintainable — see Phase 1b step 1 + §10 item 10.

3. **Native SL/TP on exchange — RESOLVED.** Use native exchange stop/TP orders that survive bot downtime, with the bot-side monitor as a backstop. (Phase 3.)

4. **Paper mode must stay byte-for-byte unchanged — RESOLVED.** Every real-mode branch is additive and guarded by `execution_mode == 'real'`. Default config stays `paper`.

> A **Future Additions** section (§10) lists follow-on work to implement **after** this plan lands, so the codebase is ready for multi-exchange and advanced futures analytics.

---

## 1. Confirmed Decisions

| Topic | Decision |
|---|---|
| First exchange | **Binance** only (others later via the same interface) |
| Environment | **Testnet first**, mainnet only after validation |
| Market scope | **Spot AND futures**, behind `market_type` toggle |
| SL/TP placement | **Native exchange orders** (survive bot downtime) + bot monitoring backstop |
| Order type | **Configurable**: `market` or `limit` (limit uses AI `entry_price`) |
| Capital source | **Real balance from exchange**, capped by `MAX_POSITION_SIZE` % |
| EU futures | **Runtime jurisdiction guard** + loud startup error + user-facing status banner |
| Docs | Update root + module `AGENTS.md` files |

---

## 2. Current State (verified anchors)

Everything below is **paper-only today**. There is **zero order-execution code**; CCXT is used for market data only.

- **Exchange client:** [`src/platforms/exchange_manager.py`](../../src/platforms/exchange_manager.py) — `exchange_config` is hardcoded to `{'enableRateLimit': True, 'options': {'defaultType': 'spot'}}` with **no credentials and no sandbox flag**.
- **Strategy entry:** [`src/trading/trading_strategy.py`](../../src/trading/trading_strategy.py)
  - `_open_new_position()` (line ~439): builds `OrderIntent` → `guard_pipeline.evaluate()` → `risk_manager.calculate_entry_parameters()` → creates `Position` → `intent.transition_to(OrderLifecycle.EXECUTED)`. **Entry price/quantity/fee come from `risk_assessment` (simulated), no exchange call.**
  - `close_position()` (line ~186): computes `pnl` and `closing_fee` locally, persists a `CLOSE_*` decision — **no exchange order.**
  - `check_position()` (line ~138): soft/hard SL/TP via `Position.is_stop_hit()` / `is_target_hit()` price comparison.
  - `TradingStrategy.__init__` (line ~39) has **no exchange handle**. It is constructed in `start.py _provision_trading_layer` (line ~618).
- **Lifecycle:** [`src/trading/order_lifecycle.py`](../../src/trading/order_lifecycle.py) — enum `INTENT, READY_FOR_REVIEW, REJECTED, EXECUTED`; `OrderIntent` pydantic model; `_ALLOWED_TRANSITIONS` map.
- **Risk:** [`src/managers/risk_manager.py`](../../src/managers/risk_manager.py) — `validate_signal()` (line ~24) allows `BUY, SELL, CLOSE, CLOSE_LONG, CLOSE_SHORT`; `calculate_entry_parameters()` (line ~82) maps `direction = "LONG" if signal == "BUY" else "SHORT"`. Returns `RiskAssessment`.
- **Data models:** [`src/trading/data_models.py`](../../src/trading/data_models.py) — `Position` (line ~12), `RiskAssessment` (line ~331), `TradeDecision` (line ~102), `MarketConditions` (line ~309). P&L math is already direction-aware (LONG/SHORT).
- **Persistence:** [`src/managers/sqlite_trade_history.py`](../../src/managers/sqlite_trade_history.py) — `SCHEMA_SQL` and `_INSERT_COLS` (no exchange/leverage columns).
- **Config:** [`src/config/loader.py`](../../src/config/loader.py) `@property` + `get_config(section, key, default)` pattern; sections `[exchanges]`, `[demo_trading]`, `[risk_management]` in [`config/config.ini`](../../config/config.ini).
- **Data fetch:** [`src/analyzer/data_fetcher.py`](../../src/analyzer/data_fetcher.py) — `fetch_funding_rate()` (line ~558) exists; spot symbol `BTC/USDC`, perpetual symbol `BTC/USDT:USDT`.
- **Schema (LLM output):** [`src/platforms/ai_providers/response_models.py`](../../src/platforms/ai_providers/response_models.py) — `TradingAnalysisModel` (no leverage / no spot-vs-futures field).
- **Prompt:** [`src/analyzer/prompts/template_manager.py`](../../src/analyzer/prompts/template_manager.py) — system prompt, `build_response_template()`, `build_analysis_steps()` (step 5.5 invalidation). Spot-framed; no leverage/funding/liquidation guidance; never states paper vs real.

---

## 3. New Files to Create

| New file | Purpose | Key interface |
|---|---|---|
| `src/platforms/trading_exchange_client.py` | Authenticated CCXT wrapper for orders | class `TradingExchangeClient` (see §4 Phase 1) |
| `src/platforms/jurisdiction_service.py` | Detect country/jurisdiction + futures availability | class `JurisdictionService` (see §4 Phase 1b) |
| `src/trading/order_executor.py` | Bridges strategy ↔ exchange client | class `OrderExecutor` (see §4 Phase 3) |
| `src/trading/execution/order_result.py` *(or extend `data_models.py`)* | `OrderResult`, `CloseOrderResult` dataclasses | see §4 Phase 2 |
| `src/trading/guards/exchange_balance.py` | Guard: real balance/margin sufficiency | `ExchangeBalanceGuard.check(...)` matching existing guard interface |
| `tests/test_order_executor.py` | Unit tests with mocked CCXT | see §9 |
| `tests/test_trading_mode_config.py` | Config toggle tests | see §9 |
| `tests/test_spot_short_rejection.py` | Spot cannot short | see §9 |
| `tests/test_jurisdiction_guard.py` | Futures jurisdiction gating | see §9 |

> Conventions (repo-wide): **behavior lives in classes**, **dependencies injected from the composition root** ([`start.py`](../../start.py)), **no service constructs its own dependencies**, avoid defensive `hasattr`/`getattr` when contracts are known.

---

## 4. Phased Implementation

### Phase 0 — Secrets + config scaffolding
*(no behavior change; default stays paper)*

1. **[`keys.env.example`](../../keys.env.example)** — append (empty values):
   ```env
   BINANCE_API_KEY=
   BINANCE_API_SECRET=
   BINANCE_TESTNET_API_KEY=
   BINANCE_TESTNET_API_SECRET=
   ```
   Add the same keys to [`keys.env`](../../keys.env) (real testnet values; leave mainnet empty until ready).

2. **[`src/config/loader.py`](../../src/config/loader.py)** — add `@property` accessors reading from **environment** (same mechanism AI keys use): `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`. **Never read secrets from config.ini.**

3. **[`config/config.ini`](../../config/config.ini)** + [`config/config.ini.example`](../../config/config.ini.example) — add:
   ```ini
   [trading_mode]
   execution_mode = paper          # paper | real
   market_type = spot              # spot | futures
   testnet = true                  # use exchange sandbox endpoint
   order_type = market             # market | limit
   leverage = 1                    # futures only; 1 = no leverage
   margin_mode = isolated          # futures only: isolated | cross
   trade_exchange = binance        # which exchange to route real orders to
   max_notional_per_trade = 0      # hard cap in quote ccy; 0 = disabled (kill-switch)
   allow_unverified_futures = false # override when jurisdiction cannot be verified (legal risk)
   ```

4. **[`src/config/loader.py`](../../src/config/loader.py)** — add a `@property` for each `[trading_mode]` key (mirror the `MAX_POSITION_SIZE` pattern with safe defaults). Add convenience `IS_REAL_TRADING` = `execution_mode == 'real'` and `IS_FUTURES` = `market_type == 'futures'`.

**Verify:** app starts with defaults; `config.EXECUTION_MODE == 'paper'`; full test suite green.

---

### Phase 1 — Authenticated exchange + trading client *(testnet)*
*(parallel with Phase 2 and Phase 5)*

1. **[`src/platforms/exchange_manager.py`](../../src/platforms/exchange_manager.py)** — add a method `get_trading_exchange()` that builds an **authenticated** CCXT instance **separate** from the read-only market exchanges. Do **not** put credentials on the shared market `exchange_config`. The authenticated instance must set:
   - `apiKey`/`secret` from config (testnet vs mainnet selected by `config.TRADING_TESTNET`),
   - `options.defaultType` = `'spot'` or `'future'` from `config.MARKET_TYPE` (use `binanceusdm`/`defaultType=future` for perpetuals),
   - `enableRateLimit: True`,
   - `exchange.set_sandbox_mode(True)` when `testnet` is true.

2. **Create `src/platforms/trading_exchange_client.py`** → class `TradingExchangeClient` (ctor: `logger`, authenticated ccxt exchange, `config`). Async methods:
   - `fetch_balance(quote_asset: str) -> float`
   - `fetch_trading_fees(symbol: str) -> dict` (maker/taker)
   - `create_entry_order(symbol, side, amount, order_type, price=None, params=None) -> dict` — `side` `'buy'`/`'sell'`; limit passes `price`.
   - `create_native_stop_loss(symbol, side, amount, stop_price, params) -> dict` — CCXT `STOP_MARKET` (futures) or stop-limit/OCO (spot). `side` is the **closing** side.
   - `create_native_take_profit(symbol, side, amount, tp_price, params) -> dict`
   - `cancel_order(order_id, symbol) -> dict`
   - `fetch_order(order_id, symbol) -> dict`
   - `fetch_open_position(symbol) -> dict | None` (futures)
   - `close_position_market(symbol, side, amount, params) -> dict`
   - `set_leverage(symbol, leverage) -> None`, `set_margin_mode(symbol, mode) -> None` (futures only; no-op for spot)
   - `amount_to_precision(symbol, amount)`, `price_to_precision(symbol, price)`, `check_min_notional(symbol, amount, price) -> bool` — **must** round to exchange precision and enforce `minNotional` before sending.
   - Always pass a deterministic `clientOrderId` derived from the strategy `order_id` (idempotency).

**Verify:** unit test with mocked ccxt confirms each method calls the right ccxt function with precision-rounded args.

---

### Phase 1b — Jurisdiction / futures-availability guard
*(resolves Critical Warning #2; can run parallel with Phase 1)*

1. **Create `src/platforms/jurisdiction_service.py`** → class `JurisdictionService` (ctor: `logger`, `config`, aiohttp session). Responsibilities:
   - `async detect_jurisdiction() -> JurisdictionInfo` — best-effort country/region detection. Detection sources (try in order, fail-soft to next):
     1. Explicit override: a `country_code` config key if the user sets one.
     2. Exchange-reported restrictions: CCXT `exchange.fetch_status()` / Binance `sapi` location endpoints where available.
     3. IP geolocation via a lightweight public endpoint (e.g., country-only IP lookup) using the existing aiohttp session.
   - `is_futures_available(jurisdiction) -> bool` — return False when the detected country is on the **blocked-country list**; keep the list in a `config/` data file (e.g. `config/futures_blocked_regions.json`) so it's editable without code changes (see §10 item 10).
   - Return a `JurisdictionInfo` dataclass: `country_code: str | None`, `is_eea: bool`, `futures_available: bool`, `source: str`, `verified: bool`.

   **Blocked-country list (Binance Futures, as of August 2025 — store as ISO-3166 alpha-2 codes in the config data file):**
   - **North America:** United States (US), Canada (CA)
   - **Europe:** Austria (AT), Bulgaria (BG), Cyprus (CY), Czech Republic (CZ), Denmark (DK), Estonia (EE), Finland (FI), Greece (GR), Hungary (HU), Iceland (IS), Ireland (IE), Latvia (LV), Liechtenstein (LI), Lithuania (LT), Luxembourg (LU), Malta (MT), Norway (NO), Poland (PL), Portugal (PT), Romania (RO), Slovakia (SK), Slovenia (SI), Germany (DE), Italy (IT), Netherlands (NL), United Kingdom (GB)
   - **Asia & Pacific:** Australia (AU), Malaysia (MY), Singapore (SG), Japan (JP), China (CN — excluding Hong Kong HK and Taiwan TW), India (IN), Indonesia (ID), Thailand (TH), Vietnam (VN), Bangladesh (BD), Kazakhstan (KZ), New Zealand (NZ)
   - **Middle East & Africa:** Algeria (DZ), Armenia (AM), Bahrain (BH), Democratic Republic of Congo (CD), Egypt (EG), Israel (IL), Jordan (JO), Lebanon (LB), Morocco (MA), Rwanda (RW), Saudi Arabia (SA), Uganda (UG), United Arab Emirates (AE)
   - **South America:** Bolivia (BO), Colombia (CO), Ecuador (EC), Guyana (GY)
   - **Other:** Iran (IR), Myanmar (MM), Bosnia and Herzegovina (BA)

   > This list is point-in-time and exchange-specific (Binance). When other exchanges are added (§10 item 1), maintain a per-exchange list. The `JurisdictionInfo.is_eea` flag stays useful for MiCA-driven messaging, but the **gating decision uses the explicit country list**, not `is_eea` alone (the list includes non-EEA bans like US, Japan, Singapore).

2. **Startup enforcement** — in [`start.py`](../../start.py) during provisioning, when `config.IS_FUTURES`:
   - Call `JurisdictionService.detect_jurisdiction()`.
   - If `futures_available` is False (verified — country on the blocked list) → **raise a fatal startup error**: clear message naming the detected country and that Binance blocks retail futures there; tell the user to set `market_type = spot`.
   - If `verified` is False (detection failed) and `config.ALLOW_UNVERIFIED_FUTURES` is False → **raise a fatal startup error** instructing the user to either set `market_type = spot` or set `allow_unverified_futures = true` (explicit legal-risk override).
   - If allowed → proceed.
   - **Always** log a user-facing banner stating: detected country (or "unknown"), whether **futures is ENABLED/AVAILABLE** for them, and the detection source. Spot mode also logs the banner (informational, never blocks).

3. **Not legal advice:** the banner/error text must state this is an availability guard, not legal advice.

**Verify:** `tests/test_jurisdiction_guard.py` — (a) blocked country (e.g. DE/US/JP) + futures → raises; (b) detection failure + futures + no override → raises; (c) detection failure + override → proceeds with warning; (d) spot mode → never raises, banner logged; (e) a non-blocked country (e.g. allowed jurisdiction) + futures → proceeds, banner shows ENABLED.

---

### Phase 2 — Mode toggles + capital sourcing
*(parallel with Phase 1)*

1. Add `OrderResult` and `CloseOrderResult` dataclasses (in `data_models.py` or new `src/trading/execution/order_result.py`):
   - `OrderResult`: `order_id, client_order_id, status, filled_quantity, average_price, fee, fee_currency, raw, sl_order_id=None, tp_order_id=None, leverage=1, timestamp`.
   - `CloseOrderResult`: `order_id, status, filled_quantity, average_price, fee, realized_pnl=None, raw, timestamp`.

2. **Capital sourcing helper** used by the strategy: in **real** mode capital = `await trading_client.fetch_balance(quote_asset)`; in **paper** mode keep `statistics_service.get_current_capital(self.config.DEMO_QUOTE_CAPITAL)`. The `MAX_POSITION_SIZE` cap continues to apply in `RiskManager._resolve_position_size_pct()`.

**Verify:** config toggles resolve correct capital source and client type; paper path untouched.

---

### Phase 3 — Strategy + lifecycle integration
*(depends on Phases 1–2)*

1. **[`src/trading/order_lifecycle.py`](../../src/trading/order_lifecycle.py)**:
   - Add `PENDING_EXECUTION` to `OrderLifecycle`.
   - Update `_ALLOWED_TRANSITIONS`: `READY_FOR_REVIEW → (PENDING_EXECUTION, EXECUTED, REJECTED)`, `PENDING_EXECUTION → (EXECUTED, REJECTED)`.
   - Add fields to `OrderIntent`: `exchange_order_id: str | None = None`, `sl_order_id: str | None = None`, `tp_order_id: str | None = None`, `leverage: int = 1`.

2. **Create `src/trading/order_executor.py`** → class `OrderExecutor` (ctor: `logger`, `TradingExchangeClient`, `config`). Async methods:
   - `execute_entry_order(intent, risk_assessment) -> OrderResult`: (futures) `set_leverage`/`set_margin_mode`; enforce `max_notional_per_trade` kill-switch; round amount; place entry order (`order_type` from config); poll `fetch_order` until filled/partially filled; then place native SL and TP (closing side opposite to entry); return `OrderResult` with real fill data + sl/tp ids.
   - `close_position(position) -> CloseOrderResult`: cancel resting SL/TP orders (`position.sl_order_id`, `position.tp_order_id`); send market close; return realized pnl/fee from exchange.
   - `sync_position_status(position) -> str | None`: poll exchange; return `"stop_loss"`/`"take_profit"`/`None` based on which native order filled or whether the position closed.

3. **[`src/trading/trading_strategy.py`](../../src/trading/trading_strategy.py)**:
   - Inject `order_executor: OrderExecutor | None = None`; read `self._is_real = config.IS_REAL_TRADING` in `__init__`.
   - `_open_new_position()` (line ~439): after `intent.transition_to(READY_FOR_REVIEW)` and after `risk_assessment` + R/R check pass, **branch**:
     - **paper:** existing code (create `Position` from `risk_assessment`, transition `EXECUTED`).
     - **real:** `intent.transition_to(PENDING_EXECUTION)` → `result = await self.order_executor.execute_entry_order(intent, risk_assessment)` → build `Position` from `result.average_price`, `result.filled_quantity`, `result.fee`, store `result.order_id/sl_order_id/tp_order_id/leverage` on the `Position`. On failure: `transition_to(REJECTED)`, audit, return HOLD decision.
   - `close_position()` (line ~186): in real mode call `await self.order_executor.close_position(closed_position)` and use exchange-reported pnl/fee instead of `calculate_pnl`/`calculate_closing_fee`.
   - `check_position()` (line ~138): in real mode replace price-compare with `await self.order_executor.sync_position_status(self.current_position)`; keep price-compare as backstop only.

4. **[`src/trading/data_models.py`](../../src/trading/data_models.py)** `Position` — add fields: `exchange_order_id: str | None = None`, `sl_order_id: str | None = None`, `tp_order_id: str | None = None`, `leverage: int = 1`, `market_type: str = "spot"`, `funding_paid: float = 0.0`. (All defaulted → paper serialization unaffected.)

5. **`src/trading/guards/exchange_balance.py`** — `ExchangeBalanceGuard` matching the existing guard `check(intent, *, capital, config)` signature; in real mode verify exchange balance/margin covers the intended notional. Register it in the [`start.py`](../../start.py) guard pipeline (line ~611) **only when** real mode is active.

6. **Composition root `start.py _provision_trading_layer`** (line ~551) — when `config.IS_REAL_TRADING`: build `TradingExchangeClient` from `exchange_manager.get_trading_exchange()` and an `OrderExecutor`, then pass `order_executor=...` into `TradingStrategy(...)` (line ~618). `_provision_trading_layer` currently receives only `utils`; also pass `infra` (holds `exchange_manager`).

**Verify:** real-mode unit tests (mocked executor) exercise entry → native SL/TP → close; paper tests unchanged.

---

### Phase 4 — Persistence + statistics reconciliation
*(depends on Phase 3)*

1. **[`src/managers/sqlite_trade_history.py`](../../src/managers/sqlite_trade_history.py)**:
   - Extend `SCHEMA_SQL` with new nullable columns: `exchange_order_id TEXT`, `exchange_status TEXT`, `actual_quantity REAL`, `actual_fee REAL`, `exchange_pnl REAL`, `leverage REAL`, `funding_paid REAL`, `market_type TEXT`.
   - Add a **migration on init**: read `PRAGMA table_info(trade_history)`; `ALTER TABLE ... ADD COLUMN` for any missing column (idempotent, additive — never drops). Keeps existing DBs working.
   - Extend `_INSERT_COLS` and `_coerce_col` for the new columns.

2. **[`src/managers/persistence_manager.py`](../../src/managers/persistence_manager.py)** — pass new fields through when serializing a `TradeDecision`/close.

3. **[`src/trading/statistics.py`](../../src/trading/statistics.py)** — in real mode prefer `exchange_pnl` when present; `get_current_capital()` reconciles against the exchange balance periodically (log a warning on drift).

**Verify:** old DB opens and migrates; new inserts persist exchange fields; paper inserts leave them null.

---

### Phase 5 — Prompt + schema (spot/futures aware)
*(parallel with Phases 3–4)*

1. **[`src/analyzer/prompts/template_manager.py`](../../src/analyzer/prompts/template_manager.py)** `build_system_prompt()`:
   - State the active **execution mode** (paper/real) and **market type**.
   - **Spot:** instruct the model it can only go **long** (BUY to open, CLOSE/sell to exit); `SELL`-to-open-short is **invalid**.
   - **Futures:** allow long and short; add guidance on **funding cost on holds**, **liquidation distance vs leverage**, and **margin mode**. Inject the configured `leverage`.

2. **[`src/platforms/ai_providers/response_models.py`](../../src/platforms/ai_providers/response_models.py)** `TradingAnalysisModel` — add optional `leverage: int | None` (futures only; validate ≤ configured cap). Document that in spot mode `SELL` means "close", not "open short".

3. **[`src/managers/risk_manager.py`](../../src/managers/risk_manager.py)** `validate_signal()` / `calculate_entry_parameters()` — when `market_type == 'spot'`, **reject** a `SELL`-to-open as an invalid short (return rejection/HOLD path); pass `leverage` through to `RiskAssessment` (add `leverage` field to the dataclass).

4. **[`src/parsing/`](../../src/parsing)** / `PositionExtractor` — ensure spot-short rejection happens before order placement.

5. **[`src/analyzer/data_fetcher.py`](../../src/analyzer/data_fetcher.py)** — when `market_type == 'futures'`, use the perpetual symbol form (`BTC/USDT:USDT`) and invoke `fetch_funding_rate()`. *(Optional: `fetch_open_interest` + mark price for richer context.)*

**Verify:** prompt snapshot test shows spot long-only vs futures long/short framing; schema accepts/validates `leverage`; spot-short rejection test passes.

---

### Phase 6 — Tests + validation

1. **Unit (mocked ccxt):** `OrderExecutor` entry fill, native SL/TP placement, partial fill, precision/minNotional rounding, `max_notional_per_trade` kill-switch, close + cancel-resting-orders.
2. **Unit:** config toggles (`paper↔real`, `spot↔futures`) resolve correct client/symbol/capital source.
3. **Unit:** spot mode rejects `SELL`-to-open; futures allows short + leverage passthrough; schema leverage validation.
4. **Unit:** jurisdiction guard matrix (Phase 1b *Verify*).
5. **Integration (Binance testnet):** one full cycle — entry → native SL/TP visible on exchange → close — asserting persisted `exchange_order_id` and reconciled pnl. Gate behind an env flag so CI mocks it.
6. **Regression:** with `execution_mode = paper`, the entire existing suite stays green and behavior is identical.

Run: `python -m pytest tests/ -q` (trust raw pytest output only, per repo terminal rules).

---

### Phase 7 — AGENTS.md updates *(mandatory)*

- **Root [`AGENTS.md`](../../AGENTS.md):** update §3 lifecycle (real-execution branch + jurisdiction guard), §5 config (new `[trading_mode]`), §7 platform integrations (authenticated trading client + jurisdiction service), §8 safety (replace "paper only" absolute; document real-mode guardrails + EU futures jurisdiction gate), AGENTS-only governance note.
- **[`src/trading/AGENTS.md`](../../src/trading/AGENTS.md):** `OrderExecutor`, lifecycle `PENDING_EXECUTION`, native SL/TP, real vs paper branch.
- **[`src/managers/AGENTS.md`](../../src/managers/AGENTS.md):** real-balance capital, fee sourcing, schema migration.
- **[`src/trading/guards/AGENTS.md`](../../src/trading/guards/AGENTS.md):** `ExchangeBalanceGuard`.
- **[`src/analyzer/AGENTS.md`](../../src/analyzer/AGENTS.md):** spot/futures prompt framing, leverage field, funding/liquidation guidance.
- **New `src/platforms/AGENTS.md`** (or extend existing platform docs): `TradingExchangeClient`, `JurisdictionService`, testnet/sandbox, credential handling.

---

## 5. Direct Answers to the Original Questions

1. **Prompt change — yes (Phase 5).** Today it's spot-framed, never states paper/real, no leverage/funding/liquidation guidance, and `SELL` wrongly maps to "open short" unconditionally.
2. **keys.env — yes (Phase 0).** No exchange keys exist; add Binance testnet+mainnet slots and update the example.
3. **Prompt/data futures-ready? Partially.** Funding rate exists and P&L math is direction-aware; missing explicit futures framing, a `leverage` field, open interest, mark price, liquidation handling.
4. **EU compliance — handled at runtime (Phase 1b).** Jurisdiction guard detects country, fails loudly when futures is unavailable/unverified, and shows a user-facing availability banner. Not legal advice.
5. **config.ini — moderate additions.** New `[trading_mode]` section + Binance key env vars. **Capital comes from the real exchange balance** in real mode (capped by `MAX_POSITION_SIZE`); `demo_quote_capital` stays for paper.
6. **Also account for:** rotate any exposed secrets; trade-only API keys (no withdrawal) + IP allow-list; idempotent `clientOrderId`; partial fills; exchange precision + `minNotional` rounding; rate limits; crash/restart recovery (reconcile open exchange position on startup); `max_notional_per_trade` kill-switch; funding-cost tracking; liquidation monitoring (or cap leverage to 1–2x for the first futures phase).

---

## 6. Scope Boundaries

**In scope:** Binance spot+futures real execution, testnet-first, native SL/TP, configurable order type, real-balance capital, jurisdiction guard, schema migration, prompt/schema futures-awareness, tests, AGENTS docs.
**Out of scope (this plan):** Other exchanges, advanced futures analytics — see §10.

---

## 7. Recommended Rollout Order

`Phase 0 → (Phase 1 ∥ Phase 1b ∥ Phase 2 ∥ Phase 5) → Phase 3 → Phase 4 → Phase 6 → Phase 7`, then **testnet soak**, then a single small mainnet trade with `max_notional_per_trade` set low, then gradual scale-up.

---

## 8. Implementation Checklist (tick as you go)

- [ ] Phase 0: env key slots + `[trading_mode]` config + loader properties
- [ ] Phase 1: `get_trading_exchange()` + `TradingExchangeClient`
- [ ] Phase 1b: `JurisdictionService` + startup enforcement + banner
- [ ] Phase 2: `OrderResult`/`CloseOrderResult` + capital sourcing helper
- [ ] Phase 3: lifecycle `PENDING_EXECUTION` + `OrderExecutor` + strategy branch + `Position` fields + `ExchangeBalanceGuard` + DI wiring
- [ ] Phase 4: SQLite migration + persistence passthrough + stats reconciliation
- [ ] Phase 5: prompt spot/futures framing + schema `leverage` + spot-short rejection + futures symbol/funding
- [ ] Phase 6: all tests green incl. testnet smoke
- [ ] Phase 7: all AGENTS.md updates

---

## 9. Test File Map

| Test file | Asserts |
|---|---|
| `tests/test_order_executor.py` | entry fill, native SL/TP, partial fill, precision/minNotional, kill-switch, close+cancel (mocked ccxt) |
| `tests/test_trading_mode_config.py` | toggles resolve correct mode/client/capital/symbol |
| `tests/test_spot_short_rejection.py` | spot rejects SELL-to-open; futures allows short + leverage |
| `tests/test_jurisdiction_guard.py` | blocked country (DE/US/JP) raises; unverified+no-override raises; override proceeds; spot never raises; allowed country proceeds (banner ENABLED) |
| existing suite | unchanged under `execution_mode = paper` |

---

## 10. Future Additions (implement AFTER this plan lands)

Designed-for but intentionally deferred. The `TradingExchangeClient` interface and `[trading_mode]` config are the seams that make these incremental.

1. **Multi-exchange real trading.** Implement `TradingExchangeClient` subclasses/adapters for **KuCoin** (requires API passphrase → add `KUCOIN_API_PASSPHRASE`), **Gate.io**, **MEXC**, and **Hyperliquid** (wallet private key auth, not key/secret → separate credential path). Route via `trade_exchange` config. Add per-exchange symbol-format and fee quirks.
2. **Advanced futures analytics in the prompt.** Add `fetch_open_interest`, mark price, basis (futures−spot), and a computed **liquidation price** to `DataFetcher` and inject into the prompt/formatters. Feed liquidation distance into `RiskManager` so leverage is auto-capped to keep liquidation beyond SL.
3. **Funding-cost-aware holding logic.** Accumulate `funding_paid` per open position; surface cumulative funding in the dashboard and let the brain learn funding drag on long holds.
4. **WebSocket order/position streams.** Replace polling in `OrderExecutor.sync_position_status` with CCXT Pro / exchange user-data streams for instant fill/SL/TP notifications.
5. **Startup reconciliation service.** On boot in real mode, fetch open positions/orders from the exchange and reconcile against persisted `Position` state (handle restarts mid-trade, orphaned SL/TP orders).
6. **Portfolio / multi-pair support.** Generalize `ConfiguredSymbolGuard` and capital allocation to manage several pairs concurrently with per-pair risk budgets.
7. **Risk circuit breakers for real funds.** Daily max-loss kill-switch, max-open-positions cap, consecutive-loss cooldown, and an emergency "flatten all" command.
8. **Dashboard real-mode panels.** Show live exchange balance, open exchange orders, native SL/TP order ids, leverage/margin mode, and realized-vs-calculated P&L drift.
9. **Maker/limit execution refinement.** Post-only limit entries with timeout-to-market fallback to reduce taker fees; track realized fee savings.
10. **Configurable jurisdiction policy file.** Keep the blocked-country list in a versioned `config/futures_blocked_regions.json` (seeded from the August-2025 Binance list in Phase 1b) with a refresh/update mechanism and a `last_reviewed` date, per-exchange sub-lists, and extend the guard to other restricted products/regions. The list will drift over time — surface its age in the startup banner.

> **Not legal advice.** Real trading and crypto derivatives are regulated differently per jurisdiction. The jurisdiction guard is an availability convenience, not compliance certification; the operator remains responsible for legal use.
