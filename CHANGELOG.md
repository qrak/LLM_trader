# Changelog

## 2026-10-22 — The real price runs the stop; the exchange leg becomes a backstop

### Changed
- With the executor live the bot used to skip its own bracket entirely (`manages_exits() = False`), so the only stop was the protective leg on the testnet book — a book thin enough to print prices the real market never made. The bot now evaluates the position's SL/TP against the **real** market it already reads (`[exchanges] sandbox = false`) on its own fast cadence, `[risk_management] reference_exit_check_seconds` (default `10`, `0` disables, otherwise bounded to 2–120 s). A hit goes through the same `_close_on_exit` intent as every other local exit condition: an INTENT, never a booking.
- The monitor hands that CLOSE to the executor **immediately** (`PositionStatusMonitor.check_reference_exits` → `CryptoTradingBot.forward_local_exit_request`) instead of waiting for the next analysis cycle, which could be hours. The forward is retried every 60 s with the same `order_id` (the executor's `SafetyGuard` dedups on it), and the operator alert is sent once per (position, reason) so a position held at its stop does not page every ten seconds.
- `TradingStrategy.check_reference_exit()` is deliberately side-effect free: it does not update live MAE/MFE or persist the position, so a check every few seconds never becomes a write every few seconds. Nothing is booked — the trade still closes only on the executor's fill evidence, and `divergent`/`unavailable` verification keeps its quarantine.
- An unavailable real price skips the check and says so once per 5 minutes: the position then rests on the exchange backstop alone, and the startup banner warns when `sandbox = true` makes the "real" check measure the testnet book.
- The executor side of the same change (see its changelog) places the leg at `2.5x` the stop distance with a `1.5%` floor, so venue noise cannot pre-empt the monitor. The pair is deliberate: turning this check off (`0`) belongs with `PROTECTION_BACKSTOP_MULTIPLIER=1.0` / `PROTECTION_BACKSTOP_MIN_PCT=0` in the executor.
- `docs/adr/0005-real-price-exit-ownership.md` records the decision, the alternatives (no exchange leg, both legs widened, re-pricing/trailing, heartbeat-driven widening) and why they lost.

### Verified
- 33 new tests: the 2026-10-22 scenario (real 84,037 against an 83,100 stop) closes nothing while a real price at the stop requests a CLOSE and forwards it, short-side symmetry, no executor → no evaluation, missing price → one paced warning, forward retry without alert spam, a new position reporting again, a failing forward reported instead of swallowed, the side-effect-free strategy check, the cadence bounds and the public forwarding hook.
- Full suite: **2428 passed, 34 deselected** (`pytest -m "not integration"`); `ruff check src scripts` clean.
- No bot/executor process was started, restarted or signalled: both still run the previous code until the user restarts them.

## 2026-10-22 — Fill check reads a window, not a single second; "no sample" stops reading as a divergence

### Changed
- The executor's fill-vs-market check now reads a window reaching back from the fill (`FILL_REFERENCE_WINDOW_SECONDS`, default 10 s, bounded at 120 s) and falls back to a deeper quote twin (`BTC/USDC` → `BTC/USDT`) when the traded pair printed nothing. `market_verification_is_verified` (`src/trading/data_models.py`) validates the reference print against the window the executor reports instead of a hard one-second rule, and refuses a window beyond the bound — a widened window may never pass a print from minutes ago as "the price at the fill". A payload that states no window keeps the old one-second rule.
- The two findings are now named apart: `divergent` (the venue's price is not a price the real market made) and `unavailable` (nothing was measured). A close refused for a divergent fill logs `LESSON_SKIPPED_FILL_DIVERGENCE` and keeps the `EXIT FILL DIVERGENCE` marker; a close refused for a missing sample keeps `LESSON_SKIPPED_NO_EVIDENCE`, carries `EXECUTION UNVERIFIED`, and the refusal now reads "no public print on the real market in the 10s reference window before the fill — the execution price could not be market-checked" instead of the vague "execution price was not market-verified (unavailable)". The entry-side refusal names the same reason, in `trading_strategy.py` and `brain.py` alike.
- `MARKET_VERIFICATION_FIELDS` carries the new executor fields (`reference_window_seconds`, `reference_symbol`, `verification_reason`) into the exit evidence, so the window the fill was measured in is visible to the operator and the model.

### Verified
- Measured on the live public API: the 2026-10-18 and 2026-10-19 fills that the one-second window discarded now verify at `+0.000%` and `-0.005%` deviation; the 2026-10-22 venue artifact stays `divergent` at `-1.214%` with the same reference price (84,037.19).
- 10 new tests: window bounds and legacy payloads, future print refused, twin fallback, gap wording, and an `unavailable` close keeping the no-evidence headline (the divergent-close test now pins the divergence headline instead of "no exit evidence").
- Full suite: **2395 passed, 34 deselected** (`pytest -m "not integration"`); `ruff check src scripts` clean.

## 2026-10-21 — Dashboard sidebar fits at 100% zoom; decision graph legible

### Fixed
- Left sidebar no longer loses its bottom at 100% browser zoom: spacing is height-aware (`clamp()` on `vh`), the sidebar scrolls as one column instead of clipping its footer, and the fixed supporter banner reserves exactly its own height (`--banner-height`). On a 600 px-tall viewport the whole column — ten tabs, status, Next Analysis countdown, donation block, contacts and CoinGecko note — is visible; below that it scrolls with nothing hidden behind the banner.
- The mobile "close menu" X was rendered on desktop because `.mobile-close-btn` was hidden only inside the `max-width: 768px` block; it was stealing header width and wrapping the brand into three lines. It is now desktop-hidden and drawer-only.
- `Brain Activity` decision graph was unreadable: `network.fit()` shrank it to 0.41× at 1920 px and 0.27× at 1366 px (14 px labels rendered at 4–6 px). Fit now stops at a 0.75 zoom floor, node labels are compact and shorter per type, hierarchical spacing is tighter, and the canvas re-fits after fonts load, on tab show and on resize. The `⟲` button still fits the whole map.
- `Memory Bank` semantic-rule cards render in 320–420 px columns instead of one 280 px column beside empty space; rule text, detail blocks and metadata move from 11 px to 12–13.5 px. `Trade Journal` close reasons and lessons likewise move to 12–13.5 px.

### Verified
- Playwright at 2560×1440, 1920×1200, 1920×910, 1366×657, 1536×600 and 390×844: sidebar `scrollHeight === clientHeight` at every height ≥ 600 px, footer bottom above the banner, no horizontal overflow, brand on one line.
- Full suite: **2385 passed, 34 deselected** (`pytest -m "not integration"`).

## 2026-10-20 — Suite proven with mutants; dead suites and agent-doc rot removed

### Changed
- Agent documentation trimmed from 117.6 KB to 70.0 KB across nine files (root `AGENTS.md` plus eight package files): duplicated policy removed, dead references dropped (`scripts/install_agent_terminal_guard.ps1`, `position_state.json`, `executed_state.json`, `refresh_semantic_rules_if_stale`, `contract-validation-warnings.md`), and false statements corrected — the files claimed paper-only execution while `[executor_api] enabled = true`, quoted a `1.50` R:R gate against `min_rr_entry = 0.0`, and named Gemini as the active provider while `provider = deepseek`.
- The README and the site no longer carry a personal name; the author is `@qrak`. The executor paragraph now states only that decisions may be forwarded to a separate, private service.
- Always-skipped suites deleted: `tests/test_shutdown_real.py` (one function, zero assertions, could not fail) and `tests/test_admin_live.py` (required the production admin password, duplicated the Playwright suite). Their durable CSP/HSTS coverage moved to `tests/domain_infra/test_dashboard_brain.py`.

### Fixed
- Four untested paths found by mutation testing now have assertions: the narrative R/R tolerance (`2.35` claimed against a verified `1.953`), BUY bracket ordering (`stop < entry < target`), `position_size > 1`, and a fill divergence above 1% (previously a 50% tolerance passed all 182 tests in that file).
- `tests/domain_infra/test_dashboard_brain.py` rate-limit test no longer depends on an internal call path; it seeds `app.state.request_counts` directly.

### Verified
- Mutation probe: 9 realistic mutants in `src/`; **5 killed** before the new assertions, **9 killed** after.
- Full suite: **2419 passed, 0 skipped, 0 failed** in ~160 s. CI gate `pytest -m "not integration"`: 2385 passed, 34 deselected. `ruff check src start.py scripts` clean.

## 2026-10-20 — Dashboard limits follow the real visitor; heavy endpoints shed load

### Fixed
- `/api/*` rate limiting, the WebSocket connection cap and the admin login limiter resolve the visitor through `resolve_client_ip()`, so traffic arriving through the Cloudflare tunnel is limited per visitor instead of sharing the tunnel's address. The header is trusted only on a loopback peer, so a direct connection cannot spoof it.
- Writes have their own budget instead of spending the read budget, and 429 responses carry `Retry-After`.
- Heavy endpoints (`/api/brain*`, `/api/visuals*`, `/api/console*`, `/api/monitor/news`, `/api/monitor/last_*`) are capped in concurrency: excess work is refused immediately and a slow GET is cancelled, so a flood cannot queue unbounded work in the event loop that also runs the analysis cycle.
- `tests/domain_infra/test_cycle_metrics.py::test_dashboard_endpoint_summarizes_the_journal` stamped its journal record with a fixed 2026-10-18 time while the endpoint reads the real clock, so it started failing two days later. The fixture now writes the record with the current time; the endpoint's day-window behaviour is unchanged and still covered by `test_summarize_counts_and_window`.

### Verified
- Dashboard suite: **70 passed**, 7 new tests — header trust, per-visitor isolation end to end through `TestClient`, WebSocket keying, the write budget, load shedding, and timeout release.
- Full suite: **2415 passed, 17 skipped, 0 failed**. Ruff clean on `src start.py scripts`; Pyright 0 errors / 0 warnings.

## 2026-10-18 — Historical fill verification and learning quarantine

### Fixed
- Executor fill checks use execution-time public Binance trades through CCXT, not the detection-time ticker; reference evidence is persisted with command receipts for restart replay. Invalid thresholds are refused.
- Entry quality survives position/history round-trips and governs the later close. Missing, disabled, legacy or inconsistent verification cannot teach the brain or count as strategy performance. Actual executions are still accounted for.
- Structured history eligibility drives statistics, short-term prompt totals and the dashboard equity curve, independently of free-text close reasons. Excluded post-mortems do not advertise profit.
- Offline historical repair is fingerprint-gated and backed up; sandbox-only closes are quarantined without changing their execution prices. Untraceable semantic-rule validation feedback is reset explicitly rather than guessed.


## 2026-10-16 — Pattern states: durations, honest labels, canonical ATR, prompt legend

### Changed
- **Prompt legend**: `## Key Terminology` in `src/analyzer/prompts/template_manager.py` now names the calculator's fixed windows (RSI 14 Wilder smoothing, MACD 12/26/9 EMA, Stochastic 14/3/3, Bollinger 20 with 2 population standard deviations, ATR and ATR% 20 Wilder, Keltner EMA(20) of close ± 1.5 × the same Wilder ATR(20), ADX/±DI 14, Choppiness 14, CCI 14, ROC 14, PPO 12/26, CMF 20, MFI 14, Force Index 20, VWAP/TWAP 20, Donchian 20, Chandelier 20 × 3 ATR) and every detector threshold (RSI 30/70; Stochastic 20/80 with crosses counted inside 30/70; volume spike ≥2.5× and dry-up ≤0.5× the 20-bar average; climax volume ≥3× the 50-bar average; ATR spike ≥1.5× the previous 14-bar average; BB squeeze = band width inside the lowest 20% of 120 bars; TTM squeeze = the bands inside a Keltner channel; MACD histogram trend = majority of its last 3 changes; volatility trend = 6 of the last 9 ATR changes; accumulation/distribution = up-bar share above 60% / below 40% over 10 bars with `strength = |share - 0.5| × 2`; volume-price divergence ±2%/volume below -15%; divergence spacing 5 bars), plus the `now` / `N bars ago` / `for N bars` convention and the 25-line cap. The legend grows 1852 → 3574 characters (≈ +430 prompt tokens on a ~100k-token budget), and the printed Wilder ATR is now named instead of leaving the model to assume the KC's smoothing.
- **State durations**: `macd_histogram_trend_streak_numba`, `volatility_trend_streak_numba`, `bb_squeeze_streak_numba` and `accumulation_distribution_streak_numba` count the consecutive bars the same label has held — each re-evaluates its own detector's rule per bar through a shared `_*_at` helper, so the streak cannot drift from the label — and the engine renders `for N bars` (`IndicatorPatternEngine._bar_label` supplies the singular). Live BTC/USDC 4h output: `MACD histogram increasing for 4 bars now`, `Volatility decreasing for 1 bar now`, `Distribution for 3 bars (strength: 0.25) over last 10 periods now`.
- **Labels**: `bb_squeeze` states its own measurement (`for N bars (band width in the lowest 20% of the last 120 bars)`) instead of "low volatility, breakout imminent"; `MACD histogram … (momentum shift)` is gone; Stochastic cross descriptions no longer add "(strong signal)"; `RSI W-Bottom` / `M-Top` describe the shape (two troughs in oversold / two peaks in overbought) instead of claiming "bullish/bearish reversal confirmation". `BB_SQUEEZE_PERCENTILE` / `BB_SQUEEZE_LOOKBACK` live in `volatility_patterns.py` and the engine quotes them, so prompt text cannot drift from the detector.
- **ATR**: `keltner_channels_numba` accepts `atr_mamode` (default `"ema"`, keeping the existing two-ATR pinning test valid) and `TechnicalCalculator` now asks for `multiplier=1.5, atr_mamode="rma"`, so the Keltner channel uses the same Wilder ATR as the printed `ATR:` line while its middle stays EMA(20).
- `src/analyzer/AGENTS.md` documents the canonical pairing, the measured per-cycle state frequencies (MACD histogram direction 100%, Stochastic cross 96%, 20/50 SMA cross 85%, MACD cross 77%, volatility trend 64%, accumulation/distribution 62%, TTM squeeze 41%, BB squeeze 34% of 608 logged cycles), the streak helpers and the legend's role.

### Verification
- Real pipeline probe (`TechnicalCalculator` → `IndicatorPatternEngine` → `filter_recent_patterns`) on live ccxt BTC/USDC 4h (999 closed bars): `kc_upper - kc_middle` equals `1.5 × ATR` to 7.3e-12 (same smoothing), and the emitted volatility/momentum/volume lines carry their durations.
- `tests/domain_rag_indicators/test_indicators_and_patterns.py` + `tests/domain_brain/test_prompt_template.py`: **411 passed**, 8 new tests (four streak pins, BB-squeeze constants vs detector defaults, engine duration/claim assertions, Stochastic wording, singular `1 bar`). Full suite **2317 passed, 17 skipped** (155 s). Ruff clean on `src start.py scripts`; Pyright 0 errors / 0 warnings; `import start` smoke OK. No paid model calls, bot restarts or orders.

## 2026-10-16 — TTM Squeeze: canonical channel width, duration and release reporting

### Changed
- `TechnicalCalculator._calculate_volatility_indicators` builds the Keltner Channel at **1.5 × ATR** (was 2.0), so the squeeze condition matches John Carter's published TTM Squeeze — BB(20, 2σ) inside KC(20, 1.5 × ATR). 2.0 × ATR is the loose Squeeze Pro tier, not extreme compression. One channel definition feeds both the pattern engine and the prompt's `KC: U/M/L` line, so that line narrows with it (no second, contradictory Keltner channel).
- `detect_keltner_squeeze_numba` returns `(in_squeeze, bars_in_squeeze, bars_since_release)` instead of a bare bool. The engine now describes the state with its length (`TTM Squeeze (in compression for 3 bars) now`) instead of the fixed "extreme low volatility" claim, and emits a separate `ttm_squeeze_fired` event (`TTM Squeeze fired (compression ended) 7 bars ago`) once the bands leave the channel. The release reaches the prompt only while it fits the existing volatility recency window (5 bars = 20 h on 4h), so it does not linger as a stale signal.
- `src/analyzer/AGENTS.md` documents the canonical pairing, the measured 42% vs 15% firing rate and the state-plus-release contract.

### Verification
- Real pipeline probe on live ccxt BTC/USDC 4h (999 closed bars, 2026-05-02 → 2026-10-15) through `TechnicalCalculator` → `IndicatorPatternEngine` → `filter_recent_patterns`: the squeeze now reads ON on **15%** of bars (42% at 2.0 × ATR), longest run 14 bars. Today's closed bar reports `ttm_squeeze_fired` 7 bars ago — outside the 5-bar window, so the prompt carries **no** volatility line, where the old threshold claimed "squeeze on" (BB 80651.91/84337.45 vs KC 1.5 80745.76/83938.74; KC 2.0 was 80213.60/84470.91).
- `tests/domain_rag_indicators/test_indicators_and_patterns.py`: **292 passed**, 6 new tests (run length, bars since the bands left, never-inside, engine duration line, engine release event, and a `KC = EMA(20) ± 1.5 × ATR` pin driven through the real `TechnicalCalculator`). Full suite **2309 passed, 17 skipped**; Ruff clean on `src start.py scripts`; Pyright 0 errors / 0 warnings; `import start` smoke OK. No paid model calls, bot restarts or orders.

## 2026-10-16 — Crawl4AI Cloud promo banner silenced in the bot console

### Changed
- Importing `src/rag/news_ingestion/crawl4ai_enricher.py` (the bot's only Crawl4AI consumer) now exports `CRAWL4AI_NO_CLOUD_NOTICE=1`, so Crawl4AI's hosted-cloud promo notice no longer prints into the bot console — neither `show_once` at crawler start nor `show_blocked` after an anti-bot block. The library reads the flag at every call site, so no start-up ordering is involved, and nothing about crawling behaviour changes.

### Verification
- Real A/B probe through the Windows venv with an isolated notice-marker directory: without the module, `show_once()` printed the 436-character banner; after importing the module, `show_once()` returned `False` with 0 characters written, `show_blocked()` returned `False` with 0 characters, and `crawl4ai.cloud_notice.is_off()` returned `True`.
- `tests/domain_rag_indicators/test_rag_news.py`: **130 passed in 4.17 s**, including the new `test_crawl4ai_enricher_module_import_disables_the_cloud_notice`, which imports the module in a clean child interpreter with the variable removed from the environment. Ruff clean on `src start.py scripts`; Pyright 0 errors / 0 warnings. No paid model calls, bot restarts or orders.

## 2026-10-15 — Full research reports and token-based verbosity

### Changed
- Add `model_verbosity = full` as the loader and configuration-template default. Low/medium/high retain compact 3/5/13-line formats; full requests developed Markdown research sections inspired by DiscordCryptoAnalyzer, without changing trading JSON, execution rules or Python R/R validation labels.
- Verbosity now specifies approximate **new narrative output-token targets**: 400 / 1000 / 2000 / 8000. These exclude decision JSON and hidden reasoning; hard provider caps remain separate and replies are never cut after generation.
- Remove the separate previous-reasoning length cap entirely. Keep instruction/schema sanitization and the structured-thesis preference. Existing whole-request PromptBudgetPolicy handles provider/model context capacity and output/image/safety reserves, including smaller fallback models; protected Python R/R feedback is never silently removed. Remove the template-only token-counter dependency made unnecessary by this change.
- Move general settings, provider/model configuration and risk/executor settings above RAG internals, paths and debug controls in `config.ini.example`. Document full in the admin dropdown, README and analyzer contract.
- Check tiktoken 0.11.0 against upstream 0.14.0: updates add OpenAI model mappings/performance, not native Gemini/DeepSeek tokenizers or rate cards. Leave the dependency unchanged; provider API usage remains the source for cost estimates. Cache-hit discounts remain outside the current rate-table calculation.

### Verification
- Full isolated Windows suite: **2302 passed, 17 skipped** (this follow-up baseline: 2293 passed, 17 skipped). Ruff clean; Pyright: 0 errors / 0 warnings. Import/compile smoke confirms the full default, all four provider envelopes and uncapped prior-reasoning continuity protected by the whole-request budget. A 20,014-estimated-token prior thesis now survives intact (previously cut to 3,000, losing its final invalidation trigger). Regression coverage includes spot/futures, chart/no-chart guidance, admin persistence, Unicode, API-usage precedence, long Markdown replies through the real parser/processor, smaller-model fallback reduction and refusal to drop protected Python R/R feedback. No paid model calls, bot restarts or orders.

## 2026-10-15 — Site freshness pass: claims re-measured, redundant article removed

### Changed
- `src/dashboard/static/landing.html` (served at `semanticsignal.qrak.org/landing.html`):
  the test figure moves from "1,380+" to **2,265**, "50+" indicators becomes **79
  Numba-compiled indicator functions**, the memory card says **three** ChromaDB collections
  and **three** nearest past trades (was "four" and "top-5"), and the risk copy no longer
  claims a fixed "minimum 1.5 R:R" — the floor is learned from expectancy with
  `min_rr_entry = 0.0`. "zero lint errors, CI-ready" becomes "ruff clean on `src/` and
  `scripts/`": a repo-wide `ruff check .` still reports 18 pre-existing findings in three
  browser test files.
- `website/src/pages/story.astro`: the provenance constants at the top carry the measured run
  (**2,265 collected, 2,247 passing, 17 skipped, 1 failing** — the browser test that drives a
  live dashboard), the September timeline entry keeps its dated 1,549, the Numba count is 79 in
  `src/indicators/` (116 across `src/`), the header date moves to 15 October 2026, Quick Start
  installs with `-c constraints.txt`, and the errata list records both corrections.
- `website/src/pages/index.astro`, `about.astro` and `articles/chart-vision.astro`: the `TESTS_*`
  constants gained `TESTS_FAILED`, the "top-5 most similar past setups" claim became three, the
  Numba figures moved to 79 / 116, and the chart-vision article no longer claims a deterministic
  expected-value gate ("if EV comes out negative, the signal is rejected") or that a missed +EV
  setup is scored as a loss — EV lives in the prompt; the deterministic checks are the symbol
  whitelist, the position cap and validated stop/target levels.
- About copy (landing + `/about`): the "Open to Work" badge and the "Open to work" section are gone.
  He works in a backend/AI role and the site is a portfolio, so the copy now states that instead of
  framing the project as a job application. "Currently working in warehouse logistics" is history
  (the December 2025 origin story), not the current situation.
- `website/public/llms.txt` and `project.json`: commits 444 → **478**, tests 1,549 → **2,265**
  (with `failing: 1` and the reason), 97 → **79** indicator functions, the intro's "deterministic
  expected-value gate" corrected to the real pre-execution checks, BlockRun added to the provider
  list, and the generated/measured dates moved to 2026-10-15.
- `articles/architecture_and_journey.md` deleted: it duplicated the /story page, was linked from
  neither the site navigation nor the sitemap, and carried stale figures (1,380+ tests, Kelly
  sizing, a 1.5 R/R gate). The README no longer links it.

### Verification
- Every number re-measured on commit `c094bdb`: `python -m pytest tests -q`,
  `grep -rhoE "@njit" src/indicators | wc -l`, `git rev-list --count HEAD` (478), the collection
  constants in `src/trading/vector_memory.py`, and the retrieval limits in
  `src/trading/brain_context.py` (`k=3`, rules `n_results=3`, stats `k=20`).
- A grep after the edit finds no stale figure left in either file except the two deliberately
  dated mentions in the timeline and the errata list.
- Live checks use `?cb=` probes against `semanticsignal.qrak.org/landing.html` and `/story`.
- No bot restart, no model request, no order.

## 2026-10-15 — Move non-imported dependency pins into constraints.txt

### Changed
- Add `constraints.txt` holding the eight entries nothing in the tree imports:
  security floors (`accelerate>=1.15.0`, `jinja2>=3.1.6`, `Pillow==12.3.0`,
  `requests>=2.34.2`, `transformers>=5.14.1`, `urllib3>=2.7.0`) and compatibility
  caps (`chardet<6`, `htmldate>=1.10,<1.11`). Pip still resolves against them, but
  requirements.txt no longer reads as if the code imported them.
- requirements.txt now lists what the code imports plus the feature-gated entries
  (crawl4ai, beautifulsoup4, blockrun-llm) and points at the constraints file.
- All six launchers (`start_script_develop.ps1`, `start_script_futures.ps1`,
  `start_script_main.ps1`, `start_script_main_linux.sh`,
  `start_script_main_macos.sh`, `run_all_tests.sh`) pass `-c constraints.txt`
  when the file exists.
- `scripts/check_requirements.py` verifies both files: constraints come from the
  second argument or from `constraints.txt` beside the requirements file, and a
  constrained package that is not installed is not reported — pip does not
  install constraints, so it cannot be the reason to run pip.
- CONTRIBUTING install commands, the AGENTS.md tree line and the
  src/rag/AGENTS.md pin note now name constraints.txt.

### Verification
- Real runs of `scripts/check_requirements.py`: empty output for the repository
  files (every floor and cap satisfied by the venv); a probe file with
  `pytest>=9999` plus a non-installed constrained package reports only the unmet
  floor, while the same package listed in requirements is still reported.
- `pip install --dry-run -r requirements.txt`: whole graph already satisfied,
  nothing to install; `pip check` clean.
- `bash -n` passes on the three shell scripts; the PowerShell parser reports
  PARSE-OK for the three `.ps1` launchers.
- `pytest tests/domain_infra/test_platform_utils.py` → **27 passed** (2 new cases).
- No bot restart, no model request, no order submission.

## 2026-10-15 — Refresh README against the current tree

### Changed
- README cut from 300 to 278 lines: duplicated paragraphs merged, the stale
  "searches its own codebase using vector semantics" claim and its dangling
  `CVI`/`QUERY` Mermaid nodes (feature removed earlier) dropped, and the executor
  section collapsed into one status callout.
- Every number re-derived in this session: **2,265 collected tests** (was
  "1,380+"), a **1920×1080** chart PNG (was "4K"), **79 Numba-compiled indicator
  functions across 14 modules** (was "50+ indicators"), **ten** dashboard tabs
  (was "nine"), **nine** provisioning stages in `start.py` (was "8-stage"), a
  ≈2.3 GB environment (disk guidance raised) and 477 commits since 2025-12-21.
- Configuration table corrected against `config/config.ini`: `provider = deepseek`
  (was googleai), the provider list including blockrun, `stop_loss_check_interval
  = 15m` (was the non-existent `stop_loss_interval_minutes`), and `min_rr_entry =
  0.0` replacing the "minimum 1.5 R:R enforced" claim. The testing recipes now
  name test files that exist.
- The `llm_trader_executor` repository link is gone (private for now): the README
  describes the executor capability and its testnet status without linking it.
- Quick Start installs with `-c constraints.txt`.

### Verification
- Figures re-measured on this revision: `pytest tests/ --collect-only`,
  `git rev-list --count HEAD`, `grep -c @njit src/indicators`, the tab list in
  `src/dashboard/static/index.html`, the `_provision_*` calls in `start.py`,
  `du -sm .venv`.
- Documentation only — no source file changed by this entry.

## 2026-10-15 — Drop three dependency entries that no code imports

### Changed
- Remove `aiosqlite==0.21.0`, `scipy==1.16.1` and `bitsandbytes>=0.45.0` from
  `requirements.txt`. Nothing in the tree imports or names them: an AST scan of
  236 Python files plus a text search over configs, scripts, workflows and docs
  finds no reference, and the committed tree at `654d92f` is the same. The SQLite
  layer uses the standard `sqlite3` (`src/managers/sqlite_trade_history.py`), and
  no code path quantises a model (`load_in_8bit`/`bnb_config` do not exist).
- Each removed package still reaches the environment through a dependency that is
  used (`aiohttp-client-cache` → `aiosqlite`, `sentence-transformers` →
  `scipy`/`scikit-learn`, `accelerate` → `bitsandbytes`), so the installed
  environment and the imported API surface are unchanged. `blockrun-llm` stays:
  the provider integration is kept even though `config.ini` selects
  `deepseek-flash` today.
- No other pin moves. Imported packages (`torch`, `chromadb`, `python-dotenv`,
  `pydantic`) keep their floors, as do the non-imported security and
  compatibility pins (`requests`, `urllib3`, `Pillow`, `jinja2`, `transformers`,
  `accelerate`, `chardet<6`, `htmldate<1.11`).

### Verification
- `pip check` reports no broken requirements. `pip install --dry-run -r
  requirements.txt` resolves the full graph with every requirement already
  satisfied and nothing to install.
- `scripts/check_requirements.py` prints nothing, so the launcher's dependency
  check stays satisfied and no `pip install` is triggered.
- `pytest tests/domain_infra/test_platform_utils.py` → **25 passed**.
- No bot restart, no model request, no order submission.

## 2026-10-15 — Recency-weight SL-tightening outcome evidence

### Changed
- Weight the existing paired UPDATE/close outcome heuristic exponentially by age
  (4h: 14-day half-life, 56-day maximum age), with fractional-day precision.
  Keep one observation per closed position, including UPDATE aliases, and age
  it from the selected UPDATE rather than a potentially delayed close write.
- Require valid chronological timestamps and finite non-boolean progress/P&L.
  For each candidate, use its eligible sample count, weighted expectancy and
  effective sample size. The policy requires both raw and effective evidence
  to meet the configured minimum; otherwise it uses the timeframe configuration.
- Expire cached brain statistics after five monotonic minutes on the next request,
  even without new experiences. Do not mutate cached nested SL evidence when
  adding the effective policy/display fields.
- This is the recency-only stage: it retains the paired-outcome heuristic and
  does not establish the incremental benefit of tightening. No intrabar replay,
  alternative-exit comparison, cost optimization or out-of-sample validation.
  Retrieval decay, risk configuration and live memories are unchanged.

### Verification
- Add 27 regression cases; 21 fail against the original implementation. Existing
  assertions and expected outcomes are retained; relevant fixtures now supply
  timestamps/effective sample counts required by the new contract.
- Real isolated Chroma round-trips for LONG/SHORT verify activation, time-only
  expiry, configuration fallback and the updated prompt with unchanged row count.
- Full suite: **2245 passed, 17 skipped, 1 failed**. Baseline: **2218 passed,
  17 skipped, 1 failed**. The same pre-existing live-dashboard security-header
  test fails in both runs; it is not changed or suppressed. Ruff and full Pyright
  pass with **0 errors, 0 warnings**; compilation and `import start` pass.
- Read-only metadata snapshot of the real brain: eight updates, three positions,
  effective sample count about 2.985; no learned override, configuration remains
  15% progress for 4h. No model calls, live-data migration or bot restart.

## 2026-10-15 — Preserve portfolio return in trading-memory snapshots

### Fixed
- Pass the already-extracted `_portfolio_pnl_pct` into `MarketConditions` instead
  of silently leaving the dataclass default at zero. Preserve signed portfolio
  return separately from each trade's P&L through position serialization, vector
  storage and historical-experience prompt rendering.
- Keep the existing zero default when the input is absent. Do not migrate old
  memories or substitute today's equity for an unknown historical snapshot.

### Verification
- Six new regression cases cover small/large profits, a loss, explicit zero,
  missing input and numeric-string conversion. Four fail against the original
  extractor; all six pass with the one-line fix, including real isolated Chroma
  persistence and prompt retrieval. Existing test definitions are unchanged.
- Full suite: **2219 passed, 17 skipped**, zero failures (baseline: 2213 passed,
  17 skipped). Targeted brain/strategy tests: **287 passed**. Ruff clean; full
  Pyright: **0 errors, 0 warnings**; compilation and `import start` pass.
- Tests run in an isolated copy with dummy credentials. No paid model call,
  live-memory write, order operation or bot restart; operator restart is needed.

## 2026-10-15 — Retry empty DeepSeek completions and preserve provider errors

### Fixed
- Treat missing, null or empty `choices` as a retryable empty reply for providers
  that opt into JSON-response retries. Previously this shape bypassed retries,
  losing a closed-trade post-mortem at 04:40 and the 06:00 analysis after both
  chart and text requests failed. Keep the existing three-retry budget and send
  the identical request; valid post-mortem objects still complete without retries.
- Preserve top-level SDK error details with API-key redaction instead of dropping
  them into `choices=[] error=None`. Retain retryable server-error codes across
  conversion; explicit permanent errors are not retried just because choices
  are absent. Do not change provider settings, trading rules or live state.

### Verification
- Reproduce 14 failures through the actual OpenAI SDK using controlled HTTP
  responses before the fix. Add 22 regression cases for text/chart recovery,
  analysis/post-mortem payloads, retry exhaustion, opt-out, error classification
  and credential redaction. Existing test definitions and assertions are unchanged.
- Full suite: **2213 passed, 17 skipped**, zero failures (baseline: 2191 passed,
  17 skipped). Ruff clean; full Pyright: **0 errors, 0 warnings**; compilation
  and `import start` pass.
- Real loopback HTTP probe: an empty completion is retried with the same body,
  then the returned fixture post-mortem is stored once and read back through a
  fresh SQLite connection and FTS. All tests run in an isolated copy with dummy
  credentials; no paid model call, exchange operation or bot restart.
- The original provider-side failure is not recoverable from the old log, which
  omitted raw error details. This repair does not recreate the missed historical
  lesson or guarantee that an unavailable provider recovers within the retry budget.

## 2026-10-15 — Replace cached news navigation with real article bodies

### Fixed
- Allow a clean body of at least 400 characters to replace detected ticker/menu
  clutter for the same URL, even when shorter. Keep the stored article ID and
  normalized fields; never replace a clean full article with a shorter summary.
- Recognize compact price rows such as `$81,615.00-2.05%`; strip the ticker before
  prompt formatting and reject unrecoverable navigation rather than treating its
  length as article content. Ordinary article price prose is preserved.
- Retry long contaminated RSS bodies. If Crawl4AI cleaned HTML yields navigation,
  try its original rendered HTML before markdown/HTTP fallback; reject contaminated
  fallback text as well.

### Verification
- Re-fetch all 20 detected contaminated URLs through Crawl4AI: all yield real
  article text. Persist 19 still-current repairs; the existing 24-hour retention
  policy expires four old records, including the twentieth candidate. The other
  51 retained articles stay unchanged; IDs, URLs, dates and categories are preserved.
- Full suite: **2191 passed, 17 skipped**, zero failures; targeted news suite:
  **129 passed**. Ruff clean; full Pyright: **0 errors, 0 warnings**.
- Existing test functions and assertions are unchanged. No model calls, order
  operations or bot restart; the running process needs an operator restart to load
  the repaired cache and new replacement policy.

## 2026-10-14 — Keep browser retrieval, simplify news parsing and remove unused indicators

### Changed
- Parse RSS fields with feedparser while preserving the raw-item schema, URL
  normalization, dates, item limits and strict malformed/non-RSS rejection.
  Keep defusedxml at the input boundary and prevent empty Dublin Core descriptions
  from overwriting CoinDesk summaries; preserve the first creator.
- Extract article text with trafilatura from Crawl4AI's browser-rendered HTML.
  Too-short, unusable or rejected extraction falls back to the existing Crawl4AI
  markdown, HTML extractor and HTTP/RSS degradation paths. Browser retrieval stays enabled.
- Remove 19 unused indicator functions, their reexports and the three source files
  in the unused EOT helper package. Quantile and all live indicator implementations
  remain unchanged; AST comparison verifies all 44 surviving functions in the edited
  calculation modules. Indicator source shrinks by 547 lines net.
- Install and verify Crawl4AI 0.9.5, feedparser 6.0.14 and trafilatura 2.1.0.
  Bound htmldate below 1.11 to preserve compatibility with current CCXT's exact
  charset-normalizer/urllib3 pins; trafilatura 2.3.1 conflicts with those pins.

### Verification
- Compare every raw field on 100 live RSS entries across all four sources: no
  differences except the intentionally ignored fetch timestamp. Retrieve eight
  real article URLs with Crawl4AI 0.9.5: all retain usable extracted article text.
- Existing test functions and assertions are preserved; new coverage pins real
  extraction, markdown/HTML fallback, duplicate DC fields, dates and entity rejection.
- Final suite: **2178 passed, 17 skipped**, zero failures. Ruff and full Pyright
  (**0 errors, 0 warnings**) pass; `pip check`, compilation of changed Python
  files and `import start` pass. No bot restart, model request or order submission.

## 2026-10-14 — Share the provider client lifecycle, the chart block and the atomic JSON write

### Changed
- Move the OpenAI-SDK client lifecycle (build, lazy ensure, close), the chart
  multimodal content block and the default provider exception handler into
  `BaseAIClient`. `DeepSeekClient`, `OpenRouterClient` and `LMStudioClient` now
  declare only how their SDK client is created; `BlockRunClient` keeps its own
  async ensure path (`_ensure_sdk_client`) and redacting error handler.
- Replace nine copies of the same `@retry_api_call(...)` arguments and six copies of
  `@retry_async(...)` with the `retry_api_call_default` / `retry_async_default` presets.
- Add `atomic_write_json` to `src/utils/data_utils.py` (temp file next to the target,
  `os.replace`, temp cleanup on failure, optional `fsync`) and use it in the CoinGecko
  cache, the DeFiLlama cache, the RAG file handler and the token-cost store.
  `PersistenceManager` keeps its local write: a pinned test patches that module's
  `os.replace` failure path, so that module stays untouched.
- `requirements.txt`: `crawl4ai>=0.9.5` (was `>=0.9.2`; the dependency set is unchanged,
  so no browser re-download) and `pydantic>=2.12` declared explicitly — it is imported
  by `response_models.py` and was only transitive before.

### Verification
- Offline suite: **2166 passed, 17 skipped**. Ruff clean; full Pyright **0 errors, 0 warnings**.
  Every pre-existing test function and assertion is unchanged.
- Crawl4AI 0.9.5 compatibility checked against the `v0.9.5` tag: every
  `BrowserConfig`/`CrawlerRunConfig` field the enricher sets, `AsyncWebCrawler.arun_many`
  and `DefaultMarkdownGenerator` are present, and `PruningContentFilter` still exists
  (deprecated in favour of the faster lxml pruning).

## 2026-10-14 — Treat an unavailable executor as an availability warning

### Fixed
- Report executor position-query connection failures and transport timeouts as a
  concise warning instead of `CRITICAL` with a traceback. Keep unexpected failures
  on the error path; do not assume a disconnected executor means a closed position.
- Preserve the unverified state, local position and existing UPDATE/CLOSE safety
  gates. The next successful query resumes normal verification without a reset.

### Regression coverage
- Connection failure, connect/read timeout, reset and server disconnect; recovery
  after an outage; unexpected failures still retain their traceback.
- A real unavailable loopback endpoint exercises startup reconciliation and proves
  no position/history write or invented exit occurs.
- Offline suite: **2157 passed, 1 skipped** (live public-dashboard tests excluded).
  Ruff, full Pyright (**0 errors, 0 warnings**), compilation and startup import pass;
  all pre-existing test functions and assertions remain unchanged.

## 2026-10-13 — Settle entry accounting from executor fill receipts

### Fixed
- Replace planned entry price, quantity, allocation and known quote-currency commission
  with the exact correlated executor fill, in both the position and its existing
  SQLite history row. Refresh short-term decision memory without adding another trade.
- Recover late receipts and previously confirmed-but-unsettled entries during independent
  reconciliation, including after restart. Preserve entry identity, current SL/TP,
  genuine unknown fees and partial-exit remainders; reject unrelated or malformed fills.
- Persist exact entry-history correlation for new commands. Refuse missing or ambiguous
  historical matches rather than selecting another trade. Failed writes remain replayable.

### Verification
- Original full baseline: **2130 passed, 17 skipped**. Final offline suite:
  **2150 passed, 1 skipped**; Ruff and full Pyright pass. Existing test functions and
  assertions are unchanged. Public-dashboard tests require the running service and
  are excluded from the offline gate; the stopped-service run failed its live header check.
- Replay of copied live stores settles only entry history row **37**, preserves all
  **27** history rows and current protection levels, survives restart and is idempotent.
  Live store hashes are unchanged; no exchange/model calls, order changes or process starts.

## 2026-10-13 — Shared estimated-token budgets and model-aware fallback guards

### Changed
- Replace the brain's character cutoff with a configurable **50,000-estimated-token**
  emergency budget, counted by the existing shared `TokenCounter`. Preserve complete
  lessons, rejection feedback and risk guidance; remove optional experiences/rules
  as whole owned ranges instead of slicing the tail. Refuse oversized protected
  context rather than sending a partial instruction set.
- Add a configurable **100,000-estimated-token** whole-request ceiling with output,
  framing, chart and model-window safety reserves. Check every actual target,
  including provider/model overrides, OpenRouter's internal fallback and local
  streaming. Unknown model windows and invalid output reserves fail before API calls.
- Drop complete optional news/history/previous-analysis/experience/rule sections
  without mutating the original fallback input. Ambiguous or forged headers cannot
  cause protected position, risk, response schema or lesson sections to be removed.
- Log the actual prepared request at the provider boundary. Replace the hardcoded
  20,000-token preflight warning with the configurable whole-request ceiling.
- Expose restart-scoped budget settings in config/admin metadata. Keep exact model
  ID case in `[model_context_windows]` while retaining case-insensitive standard INI
  options and duplicate-option rejection. Treat special-token marker strings as
  ordinary text when counting, not as tiktoken errors.

### Scope
- `cl100k_base` counts are estimates for other providers; chart reserves are not
  exact vision accounting. This guards size/cost, not general prompt injection.
- Cloud default capacities were checked against official DeepSeek/Google docs and
  the public OpenRouter catalog. Local/other targets require an explicit configured
  capacity; do not assume a million-token window for a smaller fallback.

### Verification
- Isolated baseline: **2090 passed, 17 skipped**. Final suite: **2130 passed,
  17 skipped**; no existing test functions removed. Budget-specific expectations
  use estimated tokens and explicit refusal instead of clipping protected feedback.
- Ruff, full Pyright and startup/AST/compile gates passed. Vulture at 90% and 80%
  reported no candidates; no code removed based on the heuristic.
- Read-only copied SQLite/captured-vector replay preserves all five real lessons,
  stays identical over 200 brain calls and leaves the offline rebuilt DeepSeek request
  unreduced at **20,409 estimated text tokens**. A smaller configured request budget
  reduces it to **14,811** without losing those lessons. No LLM/exchange calls or
  live stores/config modifications; activation still requires a bot restart.

## 2026-10-13 — Restore longer brain context and bound repeated failed cycles

### Fixed
- Raise the brain context's character budget from **12,000 to 64,000**. Keep the
  emergency cutoff and its existing in-prompt marker, but also log an explicit
  warning with original/rendered character counts whenever a cutoff occurs.
  Retrieval remains three experiences, three rules and five recent lessons; this
  does not insert the whole growing history or change the model's context window.
- Clear per-analysis RAG snippets in `AnalysisEngine.analyze_market`'s `finally`
  block, not only after success. Previously repeated errors or cancellations after
  news injection retained old snapshots and accumulated them in later prompts.
  Preserve failure results and cancellation propagation.

### Verified
- **2090 passed, 17 skipped**, versus **2081 passed, 17 skipped** before this change;
  targeted tests: **163 passed**. Ruff, full Pyright and startup import/compilation
  pass. Existing assertions are retained; the legacy cutoff test sets its original
  12,000-character limit locally and new tests exercise the larger default.
- Real isolated provisioning, SQLite and system-prompt assembly preserve the latest
  five lessons with 18,000/48,000 characters of vector context. A >64,000-character
  stress input remains bounded and emits the warning. Rebuilding a large brain
  context 200 times does not append previous contexts.
- Thirty sequential cycles per success/error/cancellation path carry only the
  current news snapshot. The new error/cancellation regressions failed before the
  fix. Replay using a read-only backup of the real trade database and captured
  experience metadata includes all five recent real lessons in the rebuilt prompt;
  this is offline verification, not a fresh live LLM call.

## 2026-10-13 — Complete runtime wiring and historical MAE evidence

### Fixed
- Wire the dashboard news router to `AnalysisEngine.rag_engine`, not the trading
  brain's nonexistent RAG attribute. Preserve full/preview responses, caching and
  the existing disk fallback.
- Read recorded `support_resistance_strength_score` in factor statistics, with
  unchanged API bucket names and support for older persisted S/R metadata.
- Persist entry-time trend separately from exit-time market regime so the dashboard
  can group new closed experiences by the actual entry snapshot. Historical missing
  entry trends stay unknown; no live memory migration is performed.
- Register `ExecutorHandler.close()` for graceful shutdown and in-place reload.

### Added
- Compute `safe_mae_pct` from the observed maximum adverse move of at least five
  valid winning-trade observations by default. Convert signed percentage points to
  fraction units; skip missing/invalid/non-finite/boolean/positive values and
  non-winning entries. Pass the sample count through brain thresholds to the prompt.
  Explicit zero baselines are recognized. The prompt states that the historical
  maximum is not a guaranteed safe limit or an SL floor and must not alone widen stops.

### Verified
- **2081 passed, 17 skipped**, versus **2044 passed, 17 skipped** before this change;
  targeted tests: **404 passed**. Existing assertions are unchanged; the existing
  news test's two mock-source references now point at the actual analysis RAG owner.
- Regressions exercise real temporary ChromaDB storage, LONG/SHORT close learning,
  dashboard HTTP statistics and real prompt assembly, plus startup news wiring and
  registered shutdown execution. No live model/exchange calls or service restarts.
- Full Ruff passes; Pyright reports **0 errors, 0 warnings**; startup import and
  compilation pass. Vulture finds **0 candidates at 100% confidence** and the same
  **106 lower-confidence candidates** at 60% as the baseline, with none introduced.

## 2026-10-13 — Connect saved post-mortem lessons to the trading brain

### Fixed
- Inject the existing shared `PostMortemRepository` into `TradingBrainService`
  during trading-layer provisioning. Saved lessons no longer remain available
  only to post-mortem storage and dashboard views while brain context has no
  repository to read.

### Verified
- Added real provisioning/SQLite/system-prompt regressions for empty history,
  the five-most-recent lesson limit and a lesson added after the first context.
  Existing test bodies and assertions are unchanged. Full isolated Windows suite:
  **2044 passed, 17 skipped** (baseline: **2042 passed, 17 skipped**).
  Targeted tests: **175 passed**; full Pyright: **0 errors, 0 warnings**; Ruff passes.
- An offline probe using a copied live trade-history database included five saved
  lessons in the system prompt, including the latest profitable short's lesson.
  No live data, model calls, exchange orders or service restarts were involved.

## 2026-10-12 — Persist Python R/R corrections for both response contracts

### Fixed
- Extract the narrative preceding a fenced JSON decision before R/R validation.
  Render the validated narrative and analysis for both JSON-object and fenced replies,
  so a blocked entry is saved as HOLD and the next prompt receives Python feedback.
- Use explicit source equality in the admin CSP test instead of the membership form
  flagged by CodeQL. Accept the bare/HTTPS Tailwind host and reject lookalike sources.
  Alert closure still requires a CodeQL scan of the published correction.

### Verified
- Added seven response-shape/persistence regressions and six offline CSP cases.
  Existing R/R tests and assertions are unchanged. Full isolated Windows suite:
  **2042 passed, 17 skipped**; full Pyright: **0 errors, 0 warnings**; Ruff passes.
- Historical dashboard cache-busting omission is already fixed on HEAD; no static
  assets, website, CI configuration, live state or provider settings were changed.

## 2026-10-12 — Late UPDATE confirmations and next-analysis context

### Fixed
- Allow 30 seconds for UPDATE/CLOSE queue delay plus exchange execution, instead of
  assuming one 10-second executor poll covers both. Missing evidence still stays UNKNOWN.
- Revisit pending/unknown UPDATE receipts during independent reconciliation and before
  analysis. Settle the exact command once, persist its corrected SL/TP mirror, and retain
  genuine refusal alerts. Never resend an order or alter another trade or newer UPDATE.
- Scope the next prompt's outcome to the current trade and order commands by creation,
  not delayed receipt arrival. Confirmed outcomes explicitly supersede the earlier
  missing-confirmation alert and keep the original command time.

### Verified
- Added 14 regression cases; trading tests: **625 passed**. Full Windows suite:
  **2029 passed, 17 skipped**. Full Pyright: **0 errors, 0 warnings**; Ruff passes.
- Replayed the actual affected intent and executor receipt on copied data: UNKNOWN becomes
  CONFIRMED, SL **84350**, TP **82000**, with no stale alarm in the next prompt context.
  Live journals and position state remain untouched. Activation requires the user's bot
  restart; no executor change, service restart, commit or push is needed.

## 2026-10-11 — Clean full type check and safe post-await position handling

### Fixed
- Removed the remaining 15 Pyright diagnostics without new ignores, casts,
  blanket Any declarations or weaker checker settings. Known asynchronous
  strategy/notifier methods are called directly.
- Declared the mixin's executor-query and intent-registration contracts. Moved
  the shared static commission formatter into the position-management mixin
  without changing its signature, body or inherited callers.
- Recheck the nullable local position after executor verification and after an
  awaited parameter update. If another task closed it, skip the stale command
  instead of dereferencing None or recording a command for a nonexistent position.
- Removed four obsolete position-access type ignores.

### Verified
- Full `pyright src start.py`: **0 errors, 0 warnings**. Ruff and startup import pass.
- Full Windows suite: **2011 passed, 17 skipped**. Every existing test file and
  assertion remains byte-identical to the source snapshot before this correction.
- Separate isolated runtime probes reproduced AttributeError in the old CLOSE,
  UPDATE and save-in-progress paths; corrected paths emit no stale decision,
  history row, intent or learning event. Protection-version and one-shot alert
  probes pass. No trading services or external orders were started by the probes.

## 2026-10-11 — Direct configuration access in position-monitor wiring

### Fixed
- Read the reconciliation interval through the known Config.get_config method
  directly, removing dynamic method discovery and its incorrect object inference
  in the composition root. Keep the existing default for missing/malformed values.
- Added six composition-root wiring cases covering the default, numeric values,
  numeric strings and invalid configuration values.

### Verified
- Full Windows suite: **2011 passed, 17 skipped**. Ruff passes.
- Full Pyright: **15 errors**, down from 16; start.py has no diagnostics.
  The remaining diagnostics in app.py, position_management.py and
  position_status_monitor.py are unchanged and still unresolved.

## 2026-10-11 — Evidence-based Reddit context in the existing trading analysis

### Changed
- Replaced bullish/bearish title-keyword scoring with a bounded sample interpreted
  by the trading LLM in the same request. No new model, vector index or API call.
- Select up to 15 distinct titles published within 48 hours; exclude organizational
  threads, missing/future dates and duplicates. The complete section, including
  instructions and UTC dates, is capped at 600 locally estimated tokens. Titles
  are kept whole rather than truncated to fit.
- The model returns BULLISH, BEARISH, MIXED or INSUFFICIENT_EVIDENCE with cited
  title IDs and a short explanation. Missing/malformed assessments and references
  outside the supplied sample are discarded; no data means null, not neutral.
- Validated model outlook reaches the entry-time market snapshot and learning.
  Removed the old scoring/formatting methods, fake vote/comment metrics, keyword
  topics and scraper-derived result label; no compatibility aliases or fallback.

### Verified
- Full Windows-venv suite: **2005 passed, 17 skipped**, compared with **1984 passed,
  17 skipped** before changes. Ruff passes; full Pyright retains the same 16
  pre-existing diagnostics, with no new errors.
- Real Reddit fetch returned 40 titles without errors; 14 selected titles used
  **596 tokens**, 404 more than the previous section. A text-only replay through
  the configured DeepSeek model returned a valid INSUFFICIENT_EVIDENCE assessment,
  distinguishing jokes/wallet-security concerns from price expectations. No chart
  was supplied and no trading services or orders were started by the probe.
- HTTP retry/partial-failure, recency/deduplication/budget, response-evidence,
  application wiring and fresh position-load regressions pass. Bot activation
  requires the user's normal restart; no restart, commit or push was performed.

## 2026-10-10 — Shared dashboard price refresh cooldown

### Fixed
- Dashboard `/api/brain/refresh-price` shares one exchange attempt across concurrent
  visitors and keeps the result in process memory for 10 seconds using a monotonic clock.
  Errors are reused too, preventing repeated retry traffic when the exchange fails.
- Browser reloads, no-cache headers and changing query strings cannot bypass the
  cooldown. Response format, HTTP no-store, trading-loop fetches and polling stay unchanged.

### Verified
- Full Windows suite: **1984 passed, 17 skipped**; dashboard subset: **151 passed**.
  Ruff passes; changed router has zero Pyright diagnostics. Full Pyright retains
  the 16 documented errors in unchanged files. Composition-root import succeeds.
- Isolated HTTP test against the real Binance public ticker: 41 dashboard requests
  produced one exchange request; a request after 10.1 seconds produced the second.
  Concurrent caller, expiry-boundary, result-copy, query/IP and failure regressions pass.
- Live dashboard stays connected with its 1920x1080 chart and 30 news cards;
  live-state fingerprints are unchanged by tests. Backend changes require the
  user's normal restart; no bot/executor restart, commit or push was performed.
- Additional Cloudflare rate limiting was rejected by the Free plan: its one slot
  is occupied by the leaked-credential protection. That existing rule was preserved;
  no additional edge rate limit is claimed or active.

## 2026-10-10 — Fresh chart revisions and lightweight dashboard news

### Fixed
- Preserve one chart revision URL between polls and rotate it after analysis or return
  to a visible tab. Late HTTP responses and JSON decoding cannot overwrite newer charts.
- Disabled the Cloudflare query-stripping rewrite that defeated immediate chart refresh;
  narrowed the 300-second edge rule to GET `/api/visuals/charts/latest`. New revisions
  produce MISS; repeated requests to the same revision produce HIT.

### Changed
- The news panel requests `?preview=true`: the same 30 rendered cards and excerpts,
  without full article bodies or internal search fields. Full API responses and RAG
  contents remain unchanged; fast/slow polling intervals remain 10/30 seconds.
- Bumped modified static imports and purged only the dashboard HTML shell URLs.

### Verified
- Full Windows-venv suite: **1977 passed, 17 skipped**; dashboard subset: **144 passed**.
  Ruff passes; changed backend has zero Pyright diagnostics. Full Pyright reports
  16 errors in unchanged files, matching the previously documented baseline.
- Real news payload: 610,664 → 20,197 bytes; gzip comparison: 157,797 → 7,022 bytes.
  Both payloads render identical HTML for all 30 actual news cards.
- Six frontend checks pass: stable revision, forced revision persistence, hidden-tab
  suppression, HTTP/JSON race protection and news rendering parity. Live browser
  loads the 1920×1080 chart, opens WebSocket, renders 30 news cards and immediately
  requests a fresh revision on the simulated client-side analysis-complete event.
- Public position/status/price remain DYNAMIC; admin remains 403 and unauthenticated
  Hermes sessions remain 401. Live-state fingerprints are unchanged by the suite.
- Cloudflare/frontend changes are live. The news backend awaits the user's normal bot
  restart; the running older backend safely ignores the new query. No bot/executor
  restart, order, commit or push was performed.

## 2026-10-10 — Prompt deduplication and unsupported ticker isolation

### Fixed
- Prompt v1.4 omits retrieved entry insights already present verbatim in recent trade
  history and retains distinct insights and trade outcomes. A structured previous-decision
  thesis replaces the additional narrative replay; Python R/R feedback remains available.
- Brain context renders relevant learned rules once, instead of also appending the global
  rule block to retrieved experiences. Learned statistics start on a separate paragraph.
- Aligned CLOSE quantity guidance with the current position quantity, clarified that the
  advisory sizing reference is never a minimum or permission to exceed the active cap,
  and reinforced current-period high/low attribution and OBV delta interpretation.
- Explicit ticker batches skip symbols absent from the exchange's loaded markets. An
  unsupported asset no longer discards supported prices; empty/unsupported-only lists
  return no data, while `None` still requests all tickers.

### Verified
- Full Windows-venv suite: **1970 passed, 17 skipped**. Ruff passes; touched production
  files have zero Pyright diagnostics. Full Pyright still reports the same 16 previously
  documented errors in unchanged files. The composition-root import passes.
- Reconstructed the latest audited prompt using HEAD and updated templates: locally
  estimated text tokens fell from **20,607 to 18,748** (1,859 fewer, approximately 9%).
  These are tokenizer estimates, not measured DeepSeek API billing or a new model reply.
- Public Binance markets exclude `FIGR_HELOC/USDT`. The original mixed batch reproduced
  the reported BadSymbol error; the patched batch returned a real BTC/USDT price.
- Live position paths and bot intent / executor verdict / exit journals were unchanged
  by the full suite. DeepSeek V4.1 Flash, ADX computation/correction, execution logic
  and trading gates were not changed. No model API call, order, restart or commit was made.

## v1.1.4 — 2026-10-10

### Changed
- Removed the minimum interval between position UPDATE commands completely, including
  the timeframe multiplier and last-update timestamp. Valid consecutive updates can now
  adjust LONG or SHORT protection immediately. Existing SL tightening/widening rules,
  executor-state checks, execution receipts and brain update tracking remain unchanged.
- Secrets may come from the OS/service environment (recommended) or an optional `keys.env`.
  Process variables override file values, including explicitly empty values. A missing
  file no longer prevents startup. Credential strings retain their original contents;
  only Discord channel IDs and the admin ID list receive numeric conversion.
- Replaced fake credentials in `keys.env.example` with empty optional entries and documented
  provider-specific requirements, environment/file precedence, restart requirements and
  security limitations. DeepSeek and OpenRouter do not need direct Google credentials;
  an OpenRouter-hosted vendor model needs only the OpenRouter key.
- Clarified that `analysis.trend.strength_4h` / `strength_daily` must copy computed ADX
  readings, rounded to integers, rather than subjective 0–100 strength scores. The existing
  deterministic ADX correction stays in place and does not veto the trading signal.

### Removed
- `.trivyignore`: accepted ChromaDB server-mode risks left over from the retired scanner.
  No tracked workflow or script invokes Trivy; CodeQL configuration is unchanged. Future
  Trivy scans will report these findings instead of silently suppressing them.

### Verified
- Full Windows-venv suite: **1948 passed, 17 skipped**. New regressions cover consecutive
  LONG/SHORT updates at 15m/1h/4h/1d, environment-only and file-only credentials for each
  supported provider mode, precedence, empty overrides and preservation of numeric keys.
- Ruff passes for `src`, `start.py`, `scripts` and all changed tests; compilation and
  `import start` pass. Pyright reports the same **16 pre-existing errors** on both the
  pre-change HEAD snapshot and the updated tree, with no new diagnostics.
- Re-fetched public Binance BTC/USDC candles at the two reported analysis boundaries and
  independently implemented Wilder ADX(14). Production/reference differences are below
  `7e-14`: 2026-10-09 20:00 UTC → 4h **33.93202572**, daily **40.24306105**;
  2026-10-10 04:00 UTC → 4h **32.51545513**, daily **37.91793615**. The model's
  corresponding 68/80 and 65/76 values were approximately twice the actual ADX readings.
- Live bot position/intents and executor verdict/exit journals were unchanged by the full
  suite. No bot/executor restart, exchange order or historical accounting change was made.

## 2026-10-10 — ESLint can parse the website's ES modules

### Fixed
- Set ECMAScript 2022 / module parsing at the root of `.eslintrc.json`. Previously those
  options applied only to dashboard JavaScript, so Codacy's ESLint step failed on both
  `website/astro.config.mjs` and `website/scripts/build-llms-full.mjs` with `import is reserved`.
- Reproduced both parser errors with ESLint 8.57.1 and confirmed the same invocation passes
  after this configuration change. The hosted Codacy result requires a new analysis after push;
  no Codacy account settings or GitHub integration permissions were changed.

## 2026-10-10 — Landing page build migrated to Tailwind CSS 4

### Changed
- Updated `website` to Tailwind CSS 4.3.3 with the official `@tailwindcss/vite` plugin;
  removed the unused Autoprefixer dependency and regenerated the lockfile.
- Imported Tailwind's theme and utilities from the shared Astro layout, with source detection
  scoped to `website/src`. Preflight is deliberately omitted: the site already has its own
  reset and component styles, which remain unchanged. Tailwind 3 was installed but not wired
  into the previous build; this migration does not rewrite the site's templates.
- Removed the vulnerable transitive `postcss-selector-parser` dependency behind Dependabot
  alert #82 / PR #17. The prior exposure was limited to building trusted local CSS.

### Verified
- Production build generates all 11 routes and the same `llms-full.txt` content.
- Chromium comparison of all routes at 375, 768 and 1280 px: identical sampled computed styles,
  element geometry and visible text; cookie decline persists after reload; no JavaScript errors.
- `npm audit`: zero known vulnerabilities in the updated dependency tree.

## 2026-10-10 — Dashboard chart endpoint stops eating the upload (1.3 GB/day → kB/day)

### Changed
- **`/api/visuals/charts/latest` carries an ETag derived from the chart bytes** (`routers/visuals.py`)
  instead of letting the middleware hash a body that contains a fresh timestamp — the old ETag changed
  on every call, so every poll re-downloaded the whole image. A stable ETag means revalidations answer
  `304` with no body.
- **`/api/visuals/*` gets its own cache policy** (`server.py::_api_cache_policies`): browsers 60 s,
  Cloudflare edge 300 s. The endpoint hands out ~690 kB of base64 per call and only changes when an
  analysis produces a new chart.
- **The dashboard stopped defeating its own cache** (`static/modules/visuals.js`, `static/main.js`):
  the chart poll appended a random `?t=<Date.now()>` to every request, so no browser or edge cache could
  ever serve it. Now the plain URL is polled (and the chart is force-refreshed on `analysis-complete`),
  and a hidden tab no longer polls at all — it catches up once when it comes back into view.
- Cloudflare cache rule added for `semanticsignal.qrak.org/api/visuals/*`: cache eligible, edge TTL 300 s
  (verified: `cf-cache-status: HIT` with `age` past 300 s, then `EXPIRED`).

### Why
Cloudflare analytics showed 1.46 GB/day leaving this box, dominated by ~1,900 calls/day to that one
endpoint from viewers in three countries. Measured after the change: the edge serves the polls, the
origin is fetched at most once per 300 s, and an unchanged chart answers 304.

## 2026-10-10 — Dependency security bumps (four of five Dependabot alerts cleared)

### Changed
- `website/package.json` overrides: `sharp` `^0.35.4` → `^0.35.5`, `smol-toml` `^1.7.1` → `^1.9.1`,
  plus new pins `source-map-js` `^1.2.2` and `http-cache-semantics` `^4.3.0`; the lockfile was
  regenerated once (`npm install --package-lock-only`) so all four land in a single commit instead of
  four Dependabot lockfile PRs that would conflict with each other.
- Cleared alerts: `sharp` (high, librsvg CVE-2026-96889), `source-map-js` (high, CVE-2026-93749),
  `http-cache-semantics` (high, CVE-2026-93748), `smol-toml` (medium, quadratic `parse()`).
- Deliberately NOT cleared: `postcss-selector-parser < 7.1.6` (medium) — reachable only through
  `tailwindcss@3` and `postcss-nested`, whose declared ranges stop at `^6.x`. The fix requires the
  Tailwind 3 → 4 migration (that is what Dependabot PR #17 proposes; it rewrites the whole PostCSS
  plugin chain) and is a separate, build-verified change.

## 2026-09-26 — DeepSeek thinking effort `high` → `low` (measured, not guessed)

### Changed
- **`deepseek_reasoning_effort = low`** in `config.ini` + `config.ini.example` (comment now lists every
  level the API accepts: `minimal | low | medium | high | max`, and carries the reasoning below).
- Picked by replaying **18 real production prompts** (4h analysis prompts from June, August and
  September logs) through the repo's own DeepSeek client at every effort — 108 calls total — and
  judging each raw reply with the real `UnifiedParser`, the `TradingAnalysisResponseModel` contract,
  `TrendValidator`, and ground truth from ccxt 4h BTC/USDC candles:
  - **Contract/reliability**: `minimal` 27/27 clean, `high` 27/27 (one unclosed JSON object, recovered
    by the parser's lenient path), `low` 26/27 (one `strength_4h: 18.56` float where the model wants an
    int), `medium` **lost a whole cycle** (answered with prose, not JSON).
  - **Numeric grounding** (share of cited numbers that exist in the prompt): `low` 0.90 / `minimal` 0.90
    / `high` 0.87 / `medium` 0.86 on calm prompts — and 0.90 / 0.88 / 0.70 / 0.73 on the volatile
    June–August prompts, i.e. the top of the scale drifts into inventing figures exactly when data
    quality matters most.
  - **Speed and cost per call**: `low` 15.7 s / $0.0060, `minimal` 20.1 s / $0.0064, `high` 28.8 s /
    $0.0081, `medium` 31.5 s / $0.0082 (p90 up to 47 s).
  - **Decision stability** (3 prompts × 4 repeats): `low` and `minimal` never flipped the signal;
    `medium` and `high` each flipped once.
  - Thinking volume is NOT monotonic: `medium` burned 16 426 thinking tokens on a prompt where `high`
    stopped at 2 682.
  - Only 4 of 108 replies were actionable BUY/SELL (the rest HOLD), so the ranking above is about
    reliability, grounding, latency and cost — not proof that one level *trades* better.

## 2026-09-26 — Peak/off-peak rates are configurable (`config/peak_rates.json`)

### Added
- **`config/peak_rates.json` (optional, with `config/peak_rates.example.json`) + `src/utils/peak_rates.py`.**
  Providers that bill peak/off-peak (DeepSeek: half price off-peak) are now reported at the price
  actually charged instead of a flat peak rate. Per-token rates stay in `config/model_pricing.json`
  and are treated as the BASE (peak) rates; the new file only declares WHEN a window applies
  (`peak_windows_utc`: day names or ranges such as `["mon-fri"]`, UTC clock values, start-inclusive
  and end-exclusive) and how it scales them (`peak_multiplier` / `off_peak_multiplier`). A provider
  entry is merged over its built-in default, so only the changed keys need to be listed; `_default`
  covers anything unlisted (multiplier 1.0 = flat). Without the file the built-in defaults in
  `src/utils/peak_rates.py` apply, so a checkout works unchanged. The file is read at startup.
- `ModelPricing` now applies that window in `get_cost()` and exposes `cost_note()`; the log line
  reads `Request cost: $0.007000 (off-peak x0.5)` so the applied window is visible.
- README section **Billing windows (`config/peak_rates.json`)** with the field table and an example.

### Verified
- New tests pin: built-in defaults without a file (DeepSeek peak vs off-peak, weekend),
  unlisted provider/model staying flat, the file merging over the built-in entry, a custom
  `_default`, a malformed file falling back to defaults, `get_cost` scaling, and the log note.

## 2026-09-26 — DeepSeek: empty replies are re-sent, thinking effort lowered to `high`

### Changed
- **`reasoning_effort = high` for DeepSeek** (`config.ini`, `config.ini.example`). Measured on the
  production 4h prompt, `max` spent 14.9k–24.2k tokens of thinking per call (72–110 s, $0.013–0.017)
  where `high` spent 1.8k–2.1k (15 s, $0.0024–0.0063), and both returned the same decision:
- **An empty DeepSeek reply is re-sent instead of quietly becoming a HOLD.** `retry_api_call` gained
  `retry_on_empty`, which treats a reply carrying no JSON object (empty content, or prose with no `{`)
  as retryable; the DeepSeek client opts in on both request paths (text and chart). Up to 3 retries
  with the existing backoff → 4 attempts at most, after which the last reply flows into the normal
  fallback. The request body is rebuilt from the same kwargs: no prompt repair, no conversation
  replay, just the same request again (the removed `send_contract_repair` paid for a second full
  generation instead).
- `_log_usage` reads `completion_tokens_details` defensively: a reply without that field used to raise
  inside the request method, and the swallowed exception became a `None` that no retry could see.

### Verified
- `pytest tests/` → **1887 passed, 17 skipped**. New tests pin the opt-in classification (blank/prose
  retryable, real JSON never), the re-send, and the 3-retry ceiling.
- Live DeepSeek calls, same prompt and same day (weekend = off-peak, cache warm after the first call):

| run | contract | signal / confidence | sections | analysis keys | grounded values | label values | thinking tokens | wall time | cost |
|---|---|---|---|---|---|---|---|---|---|
| high ×3 | 3/3 one JSON object | HOLD / 72, 72, 70 | 13 | 16 | 103/113, 106/109, 118/118 | 9/9, 8/8, 9/9 | 1 751–2 124 | 14.6–15.2 s | $0.0063 (cold), $0.0026, $0.0024 |
| max ×2 | 2/2 one JSON object | HOLD / 73, 73 | 13 | 16 | 124/131, 144/155 | 9/9, 8/8 | 14 949–24 166 | 72.0–109.7 s | $0.0131, $0.0175 |

- Live retry proof against the real API (request engineered to answer with nothing): the client sent
  the request **4 times** (1 + 3 retries), logging `Provider returned an empty reply` before each
  re-send, then passed the last empty reply on instead of crashing.


## 2026-09-26 — DeepSeek answers with enforced JSON instead of a copied fence

### Changed
- **DeepSeek runs in JSON-output mode.** Every DeepSeek request now carries `response_format: {"type": "json_object"}` (added to the DeepSeek model config in the loader, so it rides the existing `_execute_with_param_retry` path for text and chart calls alike), and while `provider = deepseek` the rendered prompt asks for **one JSON object** — `{"narrative": ..., "analysis": {...}}` — instead of narrative prose plus a fenced ```json block. Verified live against `api.deepseek.com` with the real production system+user prompt: one object back, `analysis` contract `valid`, 48.5 s, 18.5k prompt / 11.2k completion tokens.
- **`deepseek_max_tokens` raised 32768 → 65536** (`config.ini` + `config.ini.example`). The API allows up to 384k and thinking tokens count toward the cap; at `reasoning_effort = max` a single reply already spent 22k tokens on thinking, and the decision block used to sit at the very END of the reply — so a hit cap silently removed the decision. The cap costs nothing unless the tokens are actually generated.
- **`AnalysisResultProcessor._render_response_text`**: a JSON-mode reply is written back into the canonical "narrative + trailing fenced block" text for `raw_response`, so notifiers, dashboard history and the next cycle's previous-response context keep working unchanged.

### Removed
- **The contract-repair round trip** (`_repair_missing_json_block` in the processor, `ModelManager.send_contract_repair` and its tests). It used to fire a second, conversation-replaying request whenever a reply lacked the fenced block — extra latency and cost for a format the API itself can now guarantee. A reply that still carries no parseable JSON is logged and falls back to the HOLD default with a single API call, same as before the patch existed.
- The now-unused `json_block` test helper.

## 2026-09-19 — Legacy Page Rule deleted; /ads.txt now served directly

### Fixed
- **Root cause of the `/ads.txt` redirect confirmed and removed.** With Page Rule API access the zone showed exactly one active Page Rule: `qrak.org/ads.txt` → 301 `https://semanticsignal.qrak.org/ads.txt`. Page Rule patterns match the URL *including* its query string, which is why the query-less URL redirected while `?anything` reached Pages. The rule was deleted and the URL Rewrite workaround from the entry below was removed with it — `/ads.txt` now serves the file straight from Pages (`200 text/plain`, AdSense record `pub-0077039593558808`, verified with a Googlebot user agent too). One disabled legacy Page Rule remains (`qrak.org/*` → `.../landing`, status `disabled`); it has no effect and was left alone.
- Note for future sessions: the account-owned `CLOUDFLARE_API_TOKEN` cannot touch Page Rules (`error 1011`), and a *user* token with the right permissions still answers `9109`/empty zone lists until a zone is assigned under **Zone Resources**. Page Rules were reached with the Global API Key sent as `X-Auth-Key` + `X-Auth-Email`; tooling lives in the `cloudflare-pages-deploy` skill (`scripts/cf-page-rules.py`, now tries each credential until one can see the zone).

### Notes
- The repository's default branch is **`master`**, so Dependabot alert 78 (`devalue` 5.8.1 → 5.9.2, already fixed on `develop` and pushed) stays open until `develop` is merged into `master`.

## 2026-09-19 — Machine-readable site for AI tools, and /ads.txt served again

### Added
- **`website/public/llms.txt`** — llmstxt.org-style index of the site for LLM tools: verified facts (tests, indicators, retrieval sizes, modes), page list, repositories, and an explicit note telling model authors not to summarise this as a profitable or production system.
- **`website/public/project.json` + `project.schema.json`** — the same facts as structured JSON, with a `status` block (paper capital, testnet executor, unproven profitability, reconciliation as the known weak spot) and a `not_claims` list so downstream summaries carry the caveats.
- **`website/scripts/build-llms-full.mjs`** — generates `dist/llms-full.txt` (every page as one plain-text document, currently 10 pages / 37 KB) from the built HTML, wired into `npm run build` (`build:site` runs Astro alone). Generated, never hand-edited, so it cannot drift from the deployed pages.
- `robots.txt`: explicit `Allow: /` for the AI/assistant crawlers (GPTBot, OAI-SearchBot, ChatGPT-User, ClaudeBot, Claude-User, Claude-SearchBot, anthropic-ai, PerplexityBot, Perplexity-User, Google-Extended, Applebot-Extended, CCBot, meta-externalagent, Amazonbot, DuckAssistBot, cohere-ai, YouBot, Bytespider, PetalBot) plus a Cloudflare **Content-Signal** line (`search=yes, ai-input=yes, ai-train=yes`).
- `Layout.astro`: JSON-LD extended to a `@graph` with `WebSite` + `SoftwareSourceCode` (code repository, runtime, licence, keywords), `rel="alternate"` links to `/llms.txt` and `/project.json`, and a footer line listing the machine-readable files.

### Fixed
- **`/ads.txt` returned a 301 to `semanticsignal.qrak.org` while the same path with any query string served the file** — a legacy Page Rule whose pattern has no query string, so it only matched the query-less URL. Account-scoped API tokens cannot read or edit Page Rules (`error 1011`), and the zone has no other redirect source (the `semantic` dynamic-redirect rule is disabled, no Bulk Redirects, no Worker routes). Fixed with a zone **URL Rewrite Rule** (`http_request_transform`, description says it is a workaround): a URL Rewrite takes precedence over Page Rules, so `/ads.txt` with an empty query is rewritten to `/ads.txt?static-file=1` and Pages serves the real file — `https://qrak.org/ads.txt` is now `200 text/plain` with the correct `pub-0077039593558808` record. **Delete the Page Rule in Rules → Page Rules and this rule can go**; it exists only to route around it.

### Notes
- The repository's default branch is **`master`**, so Dependabot alert 78 (`devalue` 5.8.1 → 5.9.2, already fixed on `develop` and pushed) stays open until `develop` is merged into `master`.

## 2026-09-19 — Website copy pass: every public claim aligned with the code

### Changed
- **Landing page (`website/src/pages/index.astro`) rewritten.** The scoreboard now includes the rows where this project loses — no real-money trading, no production history, not hosted for you — plus `?` wherever a competitor's documentation could not be verified, instead of a wall of ✘. A new "where every number comes from" section lists the repo paths behind each claim.
- **`/story` rebuilt from `git log`.** Timeline dates are now the real ones: `chart_generator.py` 2025-12-22 (`git log --diff-filter=A`), `brain_experience` / `brain_context` / `brain_reflection` 2026-05-16, `.ai/` 2026-07-26, executor integration 2026-08-14, provider consolidation 2026-09-12. Added an "Honest status" section (paper capital, testnet executor last exercised 2026-08-15, profitability unproven) and an errata section listing the claims this site previously got wrong.
- **Claims removed:** "Kelly Criterion position sizing" (no Kelly code exists in `src/`; sizing is the model's proposal capped at 10% of capital with 1/2/3% fallbacks), "hard 1.5 R/R floor rejects the signal" (`min_rr_entry = 0.0`; EV is the only hard gate), "how the system earns real money" (executor runs with `ENABLE_TESTNET=true`), "zero lint errors" (`ruff check .` reports 22 findings), and the "1,270+ / 1,380+ automated tests" numbers.
- **Test count is now generated from a run**, not typed: `1,549` collected / `1,532` passed / `17` skipped — `python -m pytest tests -q` on the Windows venv, 55s.
- **Articles:** the `_ema_numba` sample was replaced with the real `supertrend_numba` from `src/indicators/trend/trend_indicators.py`; retrieval described as it works (`k=3` experiences, 20-trade stats, 5 blocked-trade snippets, 3 rules); the executor article rewritten around the seven `SafetyGuard` checks and the verdict journal, with the reconciliation caveat stated instead of a dead-letter queue that does not exist.
- **Privacy policy** states that no ad script is loaded today; **disclaimer** now says testnet and names the changelog-documented losing-streak reset.

### Added
- `website/src/pages/404.astro` — until now Cloudflare Pages answered every unknown URL with `200` and a copy of the homepage.
- `website/public/{favicon.svg,og-image.png,ads.txt}` — browsers were being served the HTML shell in place of a favicon, there was no social card, and no `ads.txt` for AdSense (`pub-0077039593558808`).
- Canonical URLs, `og:image` / Twitter card, `theme-color`, `rel="noopener"` on external links, and a footer line naming the stack with a link to this site's own source.

### Fixed
- `website/astro.config.mjs` had `site: 'https://semanticsignal.qrak.org'` (the dashboard host) — canonical URLs for a `qrak.org` deployment pointed at the wrong origin.
- `website/public/sitemap.xml`: trailing slashes (what Pages actually serves) and refreshed `lastmod`.

### Security
- `devalue` 5.8.1 → **5.9.2** (`website/package.json` + lockfile) clearing GHSA-9rgm-9g3h-6x36 (moderate DoS via malformed input); `npm audit` → 0 vulnerabilities, `npm run build` → 11 pages.
- Note: `/ads.txt` still returned a cached `301` to `semanticsignal.qrak.org` on the bare URL after deploy while the same path with any query string served the file correctly — consistent with a legacy Page Rule / cached redirect that the account-scoped API token cannot inspect (`Page Rules endpoint does not support account owned tokens`, error 1011). Left for the user to check in the dashboard.

## 2026-09-19 — R/R gate: `min_rr_entry = 0` now really means "no floor"

### Changed
- **One shared floor policy.** `src/trading/rr_policy.py` (new) owns the entry floor: `max(config.MIN_RR_ENTRY, brain rr_borderline_min)`, plus normalization (`0` preserved, non-finite/negative values fall back safely). The prompt renderer and the executor both call `resolve_entry_rr_floor()`, so the model can no longer be shown a floor the executor does not enforce. Before this, `Config.MIN_RR_ENTRY` defaulted to `1.0` and an untrained brain always reported `rr_borderline_min = 0.5`, so a configured `0` never actually disabled the gate.
- **`min_rr_entry` defaults to `0.0`** (`config/config.ini.example`, `Config.MIN_RR_ENTRY`, `BrainContextProvider`). With no learned floor the gate is inactive and R/R stays an EV input instead of a veto. The prompt now renders "No hard R/R floor is active" rather than the meaningless `R/R < 0.0: REJECTED` hard block.
- **The brain can only add `0.5` or `1.0`, and only from evidence.** It needs at least 10 closed trades below 1.0 R/R (with at least one loss) before it looks at that bucket's expectancy: below −0.05R raises the floor to `1.0`, between −0.05R and `0` raises it to `0.5`, positive expectancy leaves it at `0.0`. The old ladder that jumped straight to `1.3`/`1.5`/`1.8` after 3 sub-threshold trades is gone, along with the choppiness hack that silently lowered the floor.
- **Prompt honesty pass** (`template_manager.py`, `ev_formatter.py`): the prompt no longer calls R/R the only hard gate, documents the executor's SL/TP normalization (SL expanded to ≥1%, clamped to ≤10%, TP clamped to ≤50%, R/R recomputed from the corrected levels), labels the historical winning-average R/R and SL figures as guidance rather than caps, and points the PRE-FLIGHT CHECKLIST at Decision Rules instead of a stale section. The ADX trend threshold is read from the brain instead of a hardcoded 25.

### Tests
- R/R floor assertions updated in `tests/domain_brain/test_vector_memory.py`, `test_brain_learning.py`, `test_prompt_template.py` and `tests/domain_trading/test_risk_management.py`, plus new coverage that a profitable low-R/R bucket keeps the floor at `0.0` and that the prompt drops the hard block when no floor is active.
- Full suite green (`499 passed` in the R/R-affected modules; `ruff check src start.py` clean; `pyright src start.py` 0 errors / 0 warnings).

## 2026-09-16 — Log levels: a transient provider overload is a warning, not an error

### Fixed
- `src/platforms/ai_providers/base.py` and `google.py`: an `overloaded` / `503 UNAVAILABLE` response is now logged at **warning**. The condition is already handled downstream — `ProviderOrchestrator` retries on the paid key (logged as a warning) and reports a genuine failure at error level when the paid client also fails — so the provider-side `error` level only added two benign lines to `errors.log` on every cycle and buried real failures. `rate_limit`/quota, `authentication`, `timeout`, `connection` and unexpected errors stay at **error**. A healthy `errors.log` is now empty on a normal day; the overload still shows in `Bot.log` at warning level.

## 2026-09-16 — Refactor pass: 1000-line cap, god-class splits, comment/docstring sweep

### Changed
- **Every module and class is now under 1000 lines.** Before: `start.py` 1397, `src/dashboard/routers/brain.py` 1238, `src/trading/trading_strategy.py` 1209 (class `TradingStrategy` 1156). All splits are behaviour-preserving moves into mixins the existing classes inherit, so every call site and test seam keeps working:
  - `src/composition/provisioners.py` (new, `ProvisioningMixin`): the 9 provisioning stages plus the directory/maintenance/startup-summary helpers; `start.py` 1397 → 502. `src/composition/startup_support.py` (new): env + GPU probe, optional Tk error dialog, console banner, startup summary table.
  - `src/trading/executor_reconciliation.py` (new, `ExecutorReconciliationMixin`): executor position verification, verdict-journal correlation, phantom rollback. `src/trading/position_management.py` (new, `PositionManagementMixin`): entry, SL/TP updates, existing-position decisions. `trading_strategy.py` 1209 → 480.
  - `src/dashboard/decision_presenter.py` (new): the pure view-model builders (decision graph + synopsis, market status, current market context) moved out of `routers/brain.py` (1238 → 702).
- `_open_new_position` (235 lines) decomposed into `_reject_intent`, `_store_risk_frictions`, `_check_entry_thresholds` plus the entry body; the three exit-check entry points now share `_close_on_exit`.

### Fixed
- **Admin config path after the composition split**: `_provision_dashboard_layer` derived `config/config.ini` from `Path(__file__).parent`, which resolved to `src/composition/` once the method moved. Now anchored on a `PROJECT_ROOT` constant.

### Removed
- All prose `#` comments (1 980 full-line + 339 trailing), keeping only machine directives (`# noqa`, `# type: ignore`, `# pylint:`, `# pragma:`) — one `# noqa: SIM103` that shared a line with prose was preserved.
- Docstring `Args:` / `Raises:` boilerplate that restated signatures, across 75 files (2 136 lines); `Returns:` sections and dict-key documentation kept.

### Tests
- `tests/test_trading_strategy_branches.py`: the 27 `ENTRY_CONFIRM_*` patch targets repointed to `src.trading.executor_reconciliation` — patching the old module would have patched a dead name and silently stopped intercepting.
- Two test files import the relocated dashboard helpers from `src.dashboard.decision_presenter`.
- Full suite green; `ruff check src start.py` clean; `pyright src start.py` 0 errors / 0 warnings.

## 2026-09-15 — In-place reload: SHIFT+R restarts the bot with no manual restart

### Added
- **In-place reload (SHIFT+R)**: in the bot console, SHIFT+R now performs a full graceful shutdown and exits with code `42`; the launcher scripts (re)start the bot in the same window, so source/config edits apply with no manual restart. The shutdown runs to completion (all callbacks, awaited like the Ctrl+C path) before the exit code is returned (`start.py` `RELOAD_EXIT_CODE`). Guard: the command refuses politely unless the launcher exports `LLM_TRADER_RELOAD_SUPPORTED=1`.
- **Launcher relaunch loops**: `scripts/start_script_main.ps1`, `start_script_develop.ps1`, `start_script_futures.ps1`, `start_script_main_linux.sh`, `start_script_main_macos.sh` restart the bot on exit code 42 (plus a banner hint).
- `tests/test_reload_command.py`: reload flag semantics, command guards (env + already-shutting-down), SHIFT+R key matching.

### Fixed
- **Linux/macOS launchers**: empty-array expansion under `set -u` (stock macOS bash 3.2 would abort on a default no-argument start); the requirements-check failure path no longer prints the `__CHECK_FAILED__` sentinel as a package name (honest message + exit code, pip install extracted into one function); `-t` with no value now reports cleanly.

## 2026-09-15 — EV framework: fees scale with position, not portfolio

### Fixed
- **EV fee basis** (`src/analyzer/formatters/ev_formatter.py`): the round-trip fee was computed from the whole portfolio (0.075% of $10k = $7.50 per trade) although positions are capped at 8% ($800) — ~6× the real ~$1.20 round trip. The inflated fee alone flipped gross-positive candidates negative (e.g. gross +$2.02 → EV −$5.48) and the $11.25 entry threshold inherited the same basis, silently suppressing entries in the flat market. The fee now scales with the position notional (0.150% round trip) and the worked example/threshold anchor on the standard NEUTRAL 8% cap — $1.20 fee → $1.80 threshold at $10k — pinned by test to `RegimeRiskProfileSelector.get_position_size_cap(NEUTRAL)`.

### Removed
- Dead `EVFrameworkFormatter.build_ev_quick_section` (no production caller, never present in any prompt dump) and its test.

### Tests
- `tests/test_ev_formatter.py`: round-trip fee scales with notional; position-based worked example; regression guard keeping the old capital-based $7.50 fee out.

## 2026-09-12 — Website dependency security (9 Dependabot alerts cleared)

### Security
- npm tree in `website/` refreshed to clear all 9 open Dependabot alerts: astro 7.1.3 → 7.3.2 (CRITICAL GHSA-26w7-cxv4-gfx2 + MEDIUM GHSA-376h-93r7-7g6f), plus overrides for transitive deps — sharp ^0.35.4, svgo ^4.1.0, browserslist ^4.28.7, js-yaml 4.3.2, smol-toml ^1.7.1, baseline-browser-mapping ^2.11.0. `npm audit` after refresh: 0 vulnerabilities; `npm run build` OK (10 pages).

## 2026-09-12 — Startup fixes: ticker validation, provider name, launcher dependency check

### Fixed
- **Ticker validation vs lazy exchanges**: the strict validator ran at startup before any exchange was loaded (`ExchangeManager` loads venues lazily), so it always saw zero symbols — "Exchange symbol data unavailable" on every start and no ticker was ever validated. `TickerManager` now calls `ExchangeManager.ensure_symbols_loaded()` (loads the first reachable supported exchange — ~1.4s on binance, cached and reused by the trading path) before validating; the warning remains only for a genuinely unreachable venue.
- **Startup log / summary provider name**: `start.py` read a non-existent `AI_PROVIDER` key, so the fallback-chain log line and the summary table printed "googleai" regardless of `provider` in config.ini — now reads `config.PROVIDER`.
- **Launcher dependency check**: the pre-launch check in `scripts/start_script_*.ps1`/`.sh` only verified that a package *name* was installed, so version floors (`>=`, ranges) above the installed version were treated as satisfied and pip never ran. It now evaluates real version specifiers via `scripts/check_requirements.py` (`packaging`) and runs `pip install -r requirements.txt` only when a constraint is actually unmet; `packaging>=24.0` declared in requirements.txt.

## 2026-09-12 — Provider transport consolidation + startup banner v1.1

### Changed
- **OpenRouter on the raw `openai` SDK**: `OpenRouterClient` now uses `AsyncOpenAI` (`base_url = https://openrouter.ai/api/v1`); `reasoning` effort travels via `extra_body`; generation cost/stats come from the REST endpoint `GET /api/v1/generation?id=…` (httpx) with the same mapping (`total_cost`, `native_prompt_tokens`, `native_completion_tokens`, …), same signature/`retry_delay` semantics; the SDK `server_url` fallback and `__aexit__`/`__exit__` cleanup hacks are gone. Wire parity verified live (captured request kwargs, old vs new, for text + chart calls).
- **LM Studio on the raw `openai` SDK**: `AsyncOpenAI` against the server's OpenAI-compatible `/v1` endpoint; model auto-select via `GET /v1/models` (cached); streaming keeps the per-chunk `callback` and partial-output semantics and now records token `usage` when the server provides it; the "System: …" user-prefix rewrite and the GPU-crash error mapping are unchanged. Verified with unit mocks + a stub OpenAI-compatible HTTP server.
- **Startup banner** (`start.py`): redrawn, complete "LLM TRADER" wordmark (unicode block art) with a "v1.1" version stamp; provisioning steps unchanged (Stage 1/9 … 9/9).

### Removed
- **Dependencies**: `openrouter>=1.1.10` and `lmstudio==1.5.0` removed from `requirements.txt` — DeepSeek, OpenRouter and LM Studio now share the `openai` transport.

### Fixed
- **OpenRouter reasoning effort**: now sent on every request. Previously the first call consumed `openrouter_reasoning_effort` from the shared model config (in-place `pop`), so all later calls silently dropped it — found during the live transport-parity capture and fixed by popping from a per-call copy (regression test added).

## 2026-09-12 — DeepSeek official API provider + config keys

### Added
- **DeepSeek provider** (`src/platforms/ai_providers/deepseek.py`): `DeepSeekClient` speaking the official OpenAI-compatible `api.deepseek.com` API via the `openai` SDK — text + vision (chart) support. `deepseek-flash` (DeepSeek-V4.1-Flash) handles images natively (verified live) — one model for text + charts, no vision-model split.
- **Config**: `[ai_providers]` → `deepseek_base_url`, `deepseek_model` (+ optional `deepseek_vision_model` override); `[model_config]` → `deepseek_reasoning_effort` (low|high|max), `deepseek_max_tokens`; `DEEPSEEK_API_KEY` in `keys.env`; `provider = deepseek` is now a valid selection.
- **Cost tracking**: `deepseek` bucket in `data/trading/api_costs.json`, dashboard `/api/monitor/costs`, and peak/off-peak reference rates in `config/model_pricing.json`.
- **Tests** (`tests/test_deepseek_provider.py`): SDK wiring, reasoning-effort forwarding, multimodal shape, orchestrator text/vision routing, fallback-chain membership.

### Changed
- `deepseek` joins the `all` provider fallback chains (text + chart).
- `BaseAIClient._extract_user_text_from_messages` shared by the OpenRouter and DeepSeek clients (DRY).
- **Dependencies**: `openai>=2.49.0` promoted to a direct dependency in `requirements.txt` (DeepSeek client transport; previously only transitive via crawl4ai/litellm).
- **Model defaults refreshed (2026-09)**: OpenRouter fallback → `deepseek/deepseek-v4.1-flash` (the previous `deepseek/deepseek-r1:free` no longer exists on OpenRouter — verified against the live model list); loader defaults aligned with the live config (`google_studio_model` → `gemini-3.8-flash`, OpenRouter base → `google/gemini-3-flash-preview`).
- **ai_providers cleanup**: `_prepare_multimodal_messages` deduplicated into `BaseAIClient` (OpenRouter/BlockRun copies removed); dead `unwrap_response` parameter dropped from `convert_pydantic_response`; Sep-2026 model docstrings/defaults refreshed; `requirements.txt` floors raised (`google-genai>=2.14.0`).
- **Copy refresh**: dashboard landing + public site now say "multimodal AI" (no vendor branding); test-count claims updated to "1,380+"; "7-month" journey claims refreshed to "9-month"; quick-start key hints no longer claim a dead Google free tier.

### Fixed
- **DeepSeek model id**: switched to the canonical `deepseek-flash` (DeepSeek-V4.1-Flash, released 2026-09-10, native multimodal). The legacy `deepseek-v4-flash` string still resolves but self-reports as `deepseek-flash`; `deepseek-v4.1-flash` is rejected by the API. Updated `config.ini`, `config.ini.example`, loader default, `model_pricing.json`, test stubs, and the CI workflow.
- **Single-model DeepSeek**: removed the `deepseek_vision_model` override and the `ProviderMetadata.chart_model` mechanism — the DeepSeek provider always uses `deepseek_model` (the app requires vision-capable models end-to-end).

## 2026-07-31 — v1.1.1 — Showcase Website, Codacy Security CI, SARIF Remediation & Regime Risk Profile

### Added
- **Project Showcase Website** (`website/`): Built Astro static site showcase with interactive landing and project story pages (`src/pages/index.astro`, `src/pages/story.astro`, `src/layouts/Layout.astro`).
- **Codacy Security Scan CI Workflow** (`.github/workflows/codacy.yml`): Configured automated security scanning and SARIF report upload using `upload-sarif@v4`.
- **SARIF Remediation Engine & Test Suite** (`scripts/fix_sarif.py`, `tests/test_fix_sarif.py`): Resolved GitHub Code Scanning upload failures by fixing null `tool.driver.rules` arrays and normalizing non-standard result level enums to SARIF standard enums (`"none"`, `"note"`, `"warning"`, `"error"`).
- **Architecture & Journey Article** (`articles/architecture_and_journey.md`): Detailed technical journey and architectural overview.

### Changed
- **Regime Risk Profile Refactor**: Replaced `risk_profile_selector.py` with `regime_risk_profile.py` for regime-aware profile selection integrated into `RiskManager`, `BrainContext`, and `MarketConditions`.
- **WebSocket DI Compliance**: Enforced dependency injection for `ConnectionManager` instantiated at composition root in `start.py` and passed down to `DashboardServer`.
- **Dependencies**: Updated `beautifulsoup4` (`>=4.15.0`) and `crawl4ai` (`>=0.9.2`) in `requirements.txt`.

### Fixed
- **ATR 0.0% Wiring Bug**: Fixed zero-value ATR calculation in `brain.py` and `analysis_engine.py`.
- **Dead Code & Lint Cleanup**: Removed 25 confirmed-dead items across core modules and resolved all ruff linting errors.

### Removed
- **Nitter Sentiment Analyst**: Removed deprecated and unreliable `NitterSentimentAnalyst` module and associated tests (`test_nitter_sentiment.py`).
- Cleaned up Nitter references across initialization (`start.py`, `app.py`), configuration loader, and vector memory context.

## 2026-07-30 — v1.1.0 — 768D Vector Memory Upgrade + EV Framework + Social Sentiment + Risk Profiles

### Breaking Changes
- **768D Vector Memory Model Upgrade**: Switched ChromaDB vector embedding engine to `BAAI/bge-base-en-v1.5` (768-dimensional embeddings). Legacy 384D ChromaDB collections require re-indexing/re-creation on upgrade.

### Added
- **Expected Value (EV) Framework** (`src/analyzer/formatters/ev_formatter.py`) — calculates win rates, Expected Value ratios, and R:R thresholds for prompt context synthesis
- **Social Sentiment Analysis Modules**:
  - `src/analyzer/nitter_sentiment.py` — decentralized Twitter/X sentiment scraper and processor
  - `src/analyzer/sentiment_analyst.py` — multi-source social sentiment scoring engine (Reddit & Nitter)
- **Risk Profile Selector & RL Policy**:
  - `src/trading/risk_profile_selector.py` — dynamic regime-aware risk profile selection
  - `src/trading/rl_policy.py` — reinforcement learning post-mortem feedback loop for position size adjustment
- **Public Showcase Landing Page** (`src/dashboard/static/landing.html`, `layout.css`) — interactive landing page for Semantic Signal LLM Trader dashboard

### Changed
- `rag_engine.py` — updated RAG engine initialization logic to integrate social sentiment streams
- `template_manager.py` & `prompt_builder.py` — injected EV metrics and social sentiment indicators into primary analysis prompts

## Archive

Entries up to and including `v1.0.4` (2025-12-21 - 2026-07-30) were moved out to keep this
file readable. Nothing was deleted.

- [docs/changelog-archive-2026.md](docs/changelog-archive-2026.md)
- [docs/changelog-archive-2025.md](docs/changelog-archive-2025.md)
