# Changelog

## 2026-11-20 — Ground price levels and probability estimates in the prompt

The 17:00 CET analysis misstated a distance and returned a support level without
citing its source. The prompt now supplies signed distances from the live price to
named technical levels in quote units and ATR. It separates sourced levels from
projections, and Brain's past win rate from the model's setup-specific estimate.
The response schema, risk checks and learning policy are unchanged.

## 2026-11-20 — Give the model the computed pattern-quality score

The 13:00 CET cycle returned 60 against the bot's 93: the prompt requested a
share of supporting formations, not the system's weighted score. The chart
and text prompts now supply the same system-computed number for the model to
copy. Post-response validation and the decision schema are unchanged.

## 2026-11-20 — Report pattern-quality discrepancies from the decision JSON

The 05:00 CET cycle returned `confluence_factors.pattern_quality = 66`; the bot
computed 93 but missed the 27-point warning because it read a top-level field.
The validator now compares the nested value before overwriting it. The computed
score and decision are unchanged.

## 2026-11-19 — Align the compact action line with the decision JSON

The 21:00 CET cycle returned a confirmed `UPDATE`, which the compact `ACTION` line
omitted from its examples, and added a fourth paragraph for counter-evidence. The
compact prompt now asks for the same signal as `analysis.signal` and keeps material
counter-evidence within the three narrative lines. The decision schema and execution
rules are unchanged.

## 2026-11-19 — Keep DEBUG diagnostics in the local log

`logger_debug` adds filterable severity and source locations to the local log files;
the terminal and dashboard log stream stay at INFO and above. Successful prompt-budget
checks log estimated tokens versus the effective limit, without logging another prompt.
Generic model send/receive notices and duplicate provider error dumps are gone; empty
provider replies show their finish reason. Full audit dumps and trading behavior are unchanged.

## 2026-11-19 — Keep the compact model response to three lines

The 17:00 CET cycle returned a fourth `RR_CHECK: N/A` line even though its compact prompt
asked for three narrative lines plus JSON. The shared R/R rules now reserve hypothetical
checks and their N/A placeholder for formats with a dedicated R/R section. Compact mode
explicitly disallows a separate check line. Decision JSON and Python R/R validation are unchanged.

## 2026-11-19 — The position rail survives a stop that trails past the entry

`12c6112` gated the position rail on the entry sitting strictly between SL and TP, so the panel
printed four values as a list instead of drawing the rail as soon as the stop trailed onto or past
the entry. The live SHORT crossed that line on 2026-11-18: entry 65 309.13, stop moved 65 900 →
65 250 (16:01 UTC) → 64 400 (20:00 UTC), per `data/trading/bot_position_intents.jsonl`. The tab
that holds the card was also still called "Market Data".

- **`static/modules/position_panel.js`**: `hasGaugeRange()` and `calculateGaugePosition()` become
  one `gaugeScale()`. The three levels are ranked by price for the direction into the three slots
  (0/50/100), so a trailed stop is drawn between the entry and the target instead of cancelling the
  rail; the current price interpolates inside the two segments and clamps at the outer levels, and
  left is still the stop side. The dots carry their slot inline because the slot no longer equals
  the marker. The plain value list is now only for a level that is missing or non-numeric, or for
  all three being equal.
- **`static/index.html`**: "Market Data" → "Current Position" (the panel already said "Active
  Market Position"), `main.js?v=9.17`, `position_panel.js?v=4.17` and `lifecycle_panel.js?v=1.5`
  in `main.js`, and the import inside `lifecycle_panel.js`.
- The rail keeps equal steps per level: it is a three-slot scale, not a price axis, so the gaps
  between SL, entry and TP say nothing about their size.
- Verified on the live SHORT that exposed the bug (stop under the entry, Current dot at 71 % of
  the rail): rail drawn with `Entry 65 309.13`, `SL 64 400.00`, `Current 63 059.44`, `TP 61 200.00`
  and no value list, at 1500 px and 430 px with no horizontal overflow; `pytest tests -q` → **2912
  passed, 0 skipped** in 131 s; `ruff`, `pyright src start.py`, `check_comment_policy`,
  `check_dead_symbols`, `check_loose_helpers`, `check_agents_md`, `check_config_contract`,
  `check_hardcoded_timeframes` and `check_claims --pytest` green.

## 2026-11-19 — One cost ledger, one chart fallback, and two labels that lied

The second pass over the same provider and sentiment work: the code that survived the first review
still carried a second cost ledger nobody read, a duplicated chart-to-text branch, two copies of the
entry-to-target arithmetic, and two places where the settings desk and the post-mortem prompt
described something other than what they held.

- **A second cost ledger nobody read** (`utils/token_counter.py`, `trading/data_models.py`): the
  `SessionCosts` model and `record_api_usage()` accumulated per-provider session totals that no
  production code ever read — `CostStorage` is the only durable source. The model, the increments and
  the resets are gone; the token counters, the cost log line and its billing label stay.
- **Two chart-to-text branches** (`analyzer/analysis_result_processor.py`): a failed chart request
  repeated the text call inline. The chart attempt now only fills the response; one text call follows
  when it is still empty, and `chart_analysis` still reports the transport that produced the answer.
- **The entry-to-target arithmetic twice** (`trading/stop_loss_tightening_policy.py`): the automatic
  breakeven floor and the tightening gate computed the same progress. One private helper feeds both;
  the floor still ignores the learned threshold and the gate still applies it.
- **The fraction validator twice** (`config/loader.py`): `sl_breakeven_progress` was validated once
  while parsing the INI and again in the property. One `_parse_fraction` serves both, so `0` and `1`
  are still read as fractions rather than booleans.
- **The position rail built by mutation** (`static/modules/position_panel.js`): the current price was
  pushed into the marker list and then found and filtered out again, and the LONG/SHORT arithmetic
  was written twice. Three fixed levels plus one optional marker, and one signed interpolation.
- **Two Reddit leftovers** (`analyzer/sentiment_analyst.py`, `parsing/unified_parser.py`): a warning
  for an empty success the single StockTwits stream cannot return, a `Partial fetch:` line that only
  made sense with several subreddits, and a one-element retired-key list. The retired-key diagnostic
  itself stays.
- **The settings desk turned `off` into `false`** (`config/settings_editor.py`): the type was guessed
  from the current value, so `commandcode_reasoning_effort = off` rendered as a checkbox and could
  never be set back to `high` or cleared. That key is now typed as text; a real flag such as
  `enable_cors` stays boolean.
- **The post-mortem called the close-cycle snapshot the entry conditions**
  (`trading/post_mortem.py`, `trading_strategy.py`): the prompt received the market conditions of the
  analysis that closed the trade and labelled them "at entry". It now reads
  `closed_position.conditions_at_entry`, and the parameter is gone.
- Verified on the final tree: `pytest tests -q` → **2912 passed, 0 skipped**; `ruff`,
  `pyright src start.py`, `check_comment_policy`, `check_agents_md`, `check_config_contract`,
  `check_dead_symbols`, `check_loose_helpers`, `check_hardcoded_timeframes` and
  `check_claims --pytest` green. Production diff of the refactor alone: **−80 lines**.

## 2026-11-19 — The protection mirror follows its own command, and the dashboard stops inventing numbers

The defects found while reviewing the unpushed provider and sentiment work, each pinned by a
regression test, plus the deletions that came out of the same pass.

- **Stale recommendation after the fresh ticker read** (`src/app.py`): the position snapshot was
  compared before `_fresh_decision_price()` awaited the ticker, so a position that closed or was
  replaced during that read still forwarded the old UPDATE. The snapshot is now taken after the
  await.
- **Discarded UPDATE rolled back the wrong intent** (`position_management.py`): the guard reverted
  whichever UPDATE was newest and compared levels by value, so a newer command could be refused and
  a new position could inherit the old levels. `revert_unforwarded_update(order_id=...)` refuses
  exactly the discarded intent and restores its previous levels only while the mirror still holds
  that intent's levels on the same position.
- **A brain write that stored nothing was reported as a rebuild** (`trading_strategy.py`): the
  `False` return of `update_from_closed_trade` was dropped, the exit event stayed marked as learned
  and the dashboard showed success. `False` now behaves like a raised error: the event stays
  retryable and the rebuild is reported as failed.
- **`chart_analysis` described the render, not the request** (`analysis_result_processor.py`,
  `analysis_engine.py`): a chart transport failure or a text-only contract retry still reported
  chart analysis. The processor records the transport that produced the final answer and the engine
  reports that value.
- **Post-mortem prompt claimed venue protection** (`post_mortem.py`): the local mirror levels were
  labelled "in force at exit" and the entry snapshot "market conditions at exit". Both labels now
  say what they hold.
- **`commandcode_reasoning_effort = off` was dropped** (`config/loader.py`): the generic bool
  coercion turned `off` into `False`, which the model config then removed. The key is read as text.
- **`/api/monitor/costs` hid Command Code** (`dashboard/routers/monitor.py`): the endpoint summed
  four hardcoded providers, so a Command Code-only session rendered `Free`. It now sums
  `CostStorage.PROVIDERS`.
- **Position card printed a false percentage** (`static/modules/position_panel.js`, `panels.css`):
  the duplicated SL/TP line paired the live price with the entry-time distance, e.g. `SL $100
  (-10.0%)` at breakeven. The rail already names every level and price, so the duplicate line, its
  `formatPercent` helper and its CSS are gone.
- **One OHLCV mapping** (`indicators/base/indicator_base.py`): a list is converted to float64 and
  handed to the array path; `_handle_list` is deleted.
- **One `all` chain** (`managers/provider_orchestrator.py`): the text, chart and `supports_chart`
  paths read the same ordered constant instead of three copies.
- Verified on the final tree: `pytest tests -q` → **2909 passed, 0 skipped** in 126 s; `ruff`,
  `pyright src start.py`, `check_comment_policy`, `check_agents_md`, `check_config_contract`,
  `check_dead_symbols`, `check_loose_helpers`, `check_hardcoded_timeframes` and
  `check_claims --pytest` green.

## 2026-11-19 — Command Code joins the provider layer, and the live deployment moves onto it

The GOAT subscription was already paying for the same `deepseek-v4.1-flash` the bot ran on OpenCode
Go, at the same off-peak/peak rates, so the provider layer gained a second gateway instead of a
second code path: `CommandCodeClient` is the `OpenCodeClient` shape minus the OpenCode-only session
header, and everything else (gateway routing, per-model prompt budget, fallback model, cost table)
reuses the machinery that already existed.

- **`src/platforms/ai_providers/commandcode.py`**: `AsyncOpenAI` against
  `https://api.commandcode.ai/provider/v1`, `User-Agent: llm-trader/1.0` only. Model ids are
  vendor-namespaced (`deepseek/deepseek-v4.1-flash`), so the id carries its own slash.
- **`src/config/loader.py`**: `commandcode` in `VALID_PROVIDERS`, `COMMANDCODE_API_KEY` /
  `_BASE_URL` / `_MODEL` / `_FALLBACK_MODEL`, its own `max_tokens` + `reasoning_effort` pair, and a
  fail-fast when the provider is selected without a key.
- **`provider_orchestrator.py`**: metadata entry, `GATEWAY_PROVIDERS`, both `all` chains (after
  OpenCode Go) and the missing-key guidance.
- **Cost**: `commandcode` section in `model_pricing.json` (off-peak $0.15/$0.60, peak $0.30/$1.20,
  cache read $0.003 — the GOAT page's own numbers), a `peak_rates.py` entry with the same two UTC
  windows, and the provider added to `SessionCosts`, `TokenCounter` and `CostStorage`.
- **Config**: `commandcode_*` keys in `config.ini.example`, `COMMANDCODE_API_KEY` in
  `keys.env.example`, and the context window `commandcode/deepseek/deepseek-v4.1-flash = 1000000`.
- **Live deployment**: `config/config.ini` now runs `provider = commandcode`; OpenCode Go stays
  configured, so the rollback is that one line plus a restart.
- Verified against the real endpoint before wiring: `/provider/v1/models` lists the model with a
  1M window, a 96×96 PNG answered `Red`, `reasoning_effort` accepted `off/low/high/max`, and
  `max_tokens: 131072` answered 200. The response carries no `usage.cost`, so the dollar figure is
  the rate table, not the provider's own number.

## 2026-11-19 — Probing a known type is a defect, not a guard

The rule lived only in a Hermes-era line and in the changelog history, so it kept coming back.
`AGENTS.md` §10 now states it, `CONTRIBUTING.md` repeats it, and the places where the annotation
already names the type stop probing it.

- **`AGENTS.md` §10 + `CONTRIBUTING.md`**: `getattr` / `hasattr` / `isinstance` on a value whose
  type the signature, dataclass or protocol states is a defect; a guard survives only at a genuine
  boundary — raw LLM/API/JSON payloads, third-party SDK objects typed `Any`, exception
  classification, a declared union, and `sys.frozen`. Tests and `scripts/` are exempt.
- **Eight probes removed**: `analysis_result_processor` (context timeframe), `decision_gate`
  (five context fields), `post_mortem` (typed `Position`/`TradeDecision` parameters instead of
  `Any` + `getattr`), `brain_context` (prompt-budget token counter), `logger` (handler stream),
  `template_manager` (timeframe validator), `provider_orchestrator` (`ChatResponseModel.error`),
  `model_manager` (`UsageModel.reported_cost`, and the `isinstance` that followed it).
- **`isinstance` sweep**: seven probes on values whose annotation already states the type are gone
  (`app` data dir, `ev_formatter` inference cost, `indicator_base` list element, `cycle_state`
  state, `unified_parser` response twice, `post_mortem` validator value). The remaining 236
  `isinstance` calls in `src/` are trust boundaries; the 55 `getattr`/`hasattr` stay as reviewed
  boundaries.
- **Regression tests**: the logger's day switch (including a closed stream) and the template
  manager's fail-fast on a validator without `validate_and_normalize`.
- **`mcp` extra**: `requirements-dev.txt` installs it, so `test_mcp_history.py` runs instead of
  skipping; the lazy import no longer needs a type ignore.
- Verified on the final tree: `pytest tests -q -rs` → **2882 passed, 0 skipped** in 133 s;
  `ruff`, `pyright src start.py`, `check_comment_policy`, `check_agents_md`,
  `check_config_contract`, `check_dead_symbols`, `check_loose_helpers`,
  `check_hardcoded_timeframes` and `check_claims --pytest` green.

## 2026-11-19 — Chart claims follow the real attachment, and a lesson keeps the entry plan apart from the exit protection

An earlier audit rollout landed a wide patch; this is the trimmed remainder. Three defects keep their
proof in the bot's own logs, and two further safeguards stay because they fix a real one-way path.

- **`analysis_engine.py`**: a failed chart render returned `(None, True)`, so the prompt demanded
  `CHART VALIDATION` and panel reads while no image was attached (17 XI 01:00: three Kaleido errors,
  text-only path, model still citing P1–P4). The flag is now the buffer, not the provider's
  capability. A text fallback and a contract retry also get a system prompt built with the chart
  flag off, so neither repeats instructions about an image the model never received.
- **`post_mortem.py`**: the prompt printed `closed_position.stop_loss/take_profit` — the levels in
  force at close — under `## Entry Data`. On 15 XI that turned SL 66 500 into 65 400 and produced a
  lesson claiming the plan was tightened by execution. The prompt now prints the entry plan from the
  entry decision, a separate `## Protection at Exit` block with the levels in force at close, and the
  fill labelled `Exit Fill Price`.
- **`brain_experience.py` + `brain.py`**: a failed `store_experience` still returned a P&L and ran
  rule validation feedback, so the brain reported a rebuild from an experience it never stored. A
  stored `False` now ends the learning path before any feedback and reports the failure upward.
- Verified on the final tree: `pytest tests -q -rs` → **2882 passed, 0 skipped** in 133 s;
  `ruff`, `pyright src start.py` clean; six regressions were reproduced red on `12c6112` before the
  fix and are green after it.

## 2026-11-18 — Position rail names every marker and prints its price

The Market tab's rail drew four dots and only two words: the rightmost (take-profit) dot carried
`Entry`, the grey entry dot had no label at all, the moving dot had none either, and every price
lived in a `title` tooltip, so the rail was unreadable on a phone and wrong at a glance.

- **`static/modules/position_panel.js`**: one `gaugeMarkers` list now feeds both the dots and
  their labels, so `SL`, `Entry`, `TP` and `Current` each get their own name and USD price under
  their dot; `Current` takes a second row with a tick because it moves and would collide. Prices
  need no hover, and a price outside the SL/TP range is pinned to the edge but still printed as
  its real value. One `formatUsd` formatter replaces the eight inline `Intl.NumberFormat` calls.
- **`static/css/panels.css`**: `.gauge-dot`/`.gauge-label` replace the two-label flex row; the
  current dot is blue with a ring instead of profit/loss green/red, so it can no longer be
  mistaken for SL or TP. Below 360 px the labels stack name over price.
- A level set that does not bracket the entry (a trailing stop moved onto or past it) drops the
  scale and prints the four values as a list instead of drawing a misleading rail.
- Verified in Chrome against the live SHORT and against injected LONG, current = entry, current
  beyond TP, missing price and trailing-stop-on-entry payloads, at 1500, 360 and 320 px;
  `?v=` bumped on `main.js`, `style.css`, `panels.css`, `position_panel.js` and
  `lifecycle_panel.js`. `pytest tests -q -m "not integration"` → 2874 passed, 1 skipped
  (`test_mcp_history.py`: the optional `mcp` package is not installed).

## 2026-11-18 — Social sentiment reads StockTwits through curl

Reddit closed every anonymous door: `/.rss` answers **404**, `.json` answers **403 Blocked**, the HTML
rewrites itself to a login redirect, and app creation now needs manual approval. The bot's social
section had been printing the same four fetch errors every refresh because the source was gone, not
because the code was broken.

The replacement is **StockTwits**, which needs no key and no account:
`GET https://api.stocktwits.com/api/2/streams/symbol/BTC.X.json?limit=30` serves the symbol stream,
with `created_at` in ISO-8601 and HTML entities in the bodies.

- **`src/analyzer/sentiment_analyst.py`**: `SocialSentimentAnalyst` reads one symbol stream derived
  from `crypto_pair` (`BTC/USDC` → `BTC.X`), maps `messages[].body` and `created_at` to the existing
  post contract, drops bodies under 20 characters, and deletes the subreddit loop, the 20 s pacing,
  the organizational-title regex and the whole OAuth client.
- **Transport**: Cloudflare answers `403 "Just a moment..."` to Python HTTP clients (`aiohttp`,
  `httpx` and `urllib` alike, measured 2026-11-18) while `curl.exe` answers 200, so the fetch shells
  out to curl in a worker thread. The binary is resolved once in `start.py` (one error at startup
  when it is missing), and the response parser reads the **final** header block, so a proxy's
  `200 Connection established` cannot hide the real status. A 403 is retried once after 5 s, a 429
  after `Retry-After` (seconds or an HTTP date, capped at 120 s), and the retry budget never sleeps
  after its last attempt.
- **Log noise**: a repeated identical failure is logged once at WARNING and then at DEBUG until the
  error changes; recovery is logged at INFO.
- **Names and data follow the source**: `analysis.social_sentiment` and
  `MarketConditions.social_sentiment` replace `reddit_sentiment` / `social_sentiment_reddit`.
  Nothing is migrated: entries written before this cutover read empty on that axis, a position saved
  by an older build loads with an empty entry sentiment, and a response that still carries the
  retired `reddit_sentiment` key is reported as `retired_field` instead of silently passing as valid.
- **Tests**: the local-server test serves a StockTwits stream through curl and asserts the `limit`
  parameter, the user agent, the 429 retry, the message contract, the final-header-block parse, the
  HTTP-date `Retry-After`, the 403 retry budget, the last reported status, the missing-curl error and
  the retired key's telemetry.

## 2026-11-18 — The stop reaches the entry on its own

The SHORT opened on 15.11 at 65,309.13 had its stop tightened three times by the model
(67,350 → 66,400 → 66,300 → 65,900) and still sat 0.9% **above** the entry after price covered 31.9%
of the way to the target, i.e. a hit still lost $4.09. Every `UPDATE` reached the executor; the gap
was that the level was purely the model's structural choice — the policy only gated *when* a tighten
is allowed, never *how far*, and the executor's OCO needs both legs.

### Changed
- **`src/trading/stop_loss_tightening_policy.py`**: `sl_breakeven_progress` (config, `0` disables the
  automatic move, default `0.5`) plus `breakeven_floor()`, which returns the entry once price covers
  that fraction of the entry-to-TP distance.
- **`src/trading/position_management.py`**: the floor is applied whichever decision the model sends —
  a worse proposal is clamped to it, and a proposal tighter than the floor still goes through the
  tightening policy but a rejection leaves the floor in force, so no learned threshold can keep the
  stop on the losing side. The floor is evaluated only during a successful analysis cycle and only
  against the ticker price fetched immediately before the decision; a missing fresh price leaves the
  active stop alone. A stop already at or beyond the entry is never widened back
  (`REJECTED SL widening`, recorded as a refused intent, and the bot's own floor UPDATE is named in
  the refusal when it is sent instead). The UPDATE decision always carries the effective SL **and** TP
  so the executor's spot OCO can place both legs.
- **Local authority over the mirror**: a protection/version guard that discards an UPDATE calls
  `revert_unforwarded_update()`, which restores the previous levels and refuses the still-pending
  intent (and does nothing when a newer change already moved the levels). Reconciliation reads the
  newest **forwarded** UPDATE for supersession — a local `bot_policy` refusal is not a newer command —
  and applies an executed or refused receipt only when the mirror still holds the levels that receipt
  describes, so a late receipt cannot overwrite an in-flight newer stop.
- **`config/config.ini.example`**, **`src/config/loader.py`**: the key, its documentation and a
  `0..1` numeric parse that keeps `0` and `1` numbers instead of the INI loader's booleans, so `0`
  really disables the floor and an invalid candidate stops the loader (the settings desk validates
  with the same constructor).
- **`src/trading/brain.py`**, **`src/analyzer/prompts/template_manager.py`**,
  **`src/trading/trading_strategy.py`**: the prompt states the floor at its real precision (a
  `0.004` floor renders `0.4%`, not `0`), says a tighter proposal still has to pass the tightening
  policy, states that `0` disables only the automatic move while the widening ban stays, and the live
  position context says whether the floor is active.
- **`src/trading/AGENTS.md`**, **`docs/FEATURES.md`**, **`docs/CONFIGURATION.md`**: the rule, the
  fresh-price requirement, the mirror/receipt authority and the payload contract.
- **Tests**: the policy cases (disabled, no price, empty target, LONG/SHORT at and below the
  threshold) and the strategy cases (fires on HOLD, silent below the threshold, clamps a worse
  proposal, keeps a tighter one, the floor survives an equal proposal and a rejected tighter one, no
  floor without a fresh price, refuses a widening after the floor, SHORT mirror, a learned gate
  cannot veto it, the payload carries both legs, a discarded UPDATE rolls back and refuses its
  intent, and it does not clobber newer levels). Reconciliation covers a refused receipt that must
  not undo an in-flight floor, a local refusal that must not supersede a pending receipt, and an
  executed receipt that must not overwrite a newer mirror.

### Verified
- `python -m pytest tests -q -m "not integration" --timeout=120` → **2,874 passed, 1 skipped** in 133 s.
- `ruff check src scripts start.py` clean; `pyright src start.py` clean; `check_config_contract`,
  `check_comment_policy`, `check_dead_symbols`, `check_loose_helpers`, `check_hardcoded_timeframes`,
  `check_agents_md` and `check_claims --pytest` all exit 0.

## 2026-11-18 — Price action names its side of VWAP

The prompt printed a bare VWAP level, so the model had to infer whether price traded above or below
it. `## Price Action` now reads `VWAP:98.00 (price +2.04% above)`, or `at VWAP` when the price equals
the level, and stays a bare level when the price is missing or non-finite. The level and the relation
come from one read of the production `ndarray` instead of two lookups that disagreed on the last
finite value.

## 2026-11-17 — The chart stops depending on a broken Kaleido pipe

Kaleido 1.5.0 could not drive Chrome 157 on the rebuilt box: every export failed in ~2.3 s with
"the browser seemed to close immediately", and the Playwright fallback never ran in the bot.
`start.py` set `WindowsSelectorEventLoopPolicy` process-wide, so the loop Playwright's sync API
created in its worker thread could not spawn the driver (`NotImplementedError`), and the fresh
Windows profile had no `%LOCALAPPDATA%\ms-playwright` browser after the disk move. Three retries
ended with no chart image at 01:00, while the 22:50 cycle after the previous restart still had one.

### Changed
- **`src/analyzer/pattern_engine/chart_generator.py`**: Kaleido and the fallback layer are gone;
  `_render_with_playwright` is the only renderer (30 s timeout, 3 retries, daemon thread), and the
  public `create_chart_image` contract is unchanged.
- **`start.py`**: the process default is Proactor and the main trading loop is an explicit selector
  loop, so aiodns keeps its loop and Playwright can start its driver from a worker thread.
- **`requirements.txt` / `requirements-dev.txt`**: `kaleido` removed; the duplicate Playwright pin
  and the stale admin-test comment removed. `playwright` stays a production dependency.
- **`scripts/start_script_main.ps1`**: installs the Playwright Chromium (idempotent) even under
  `-SkipInstall`, so a fresh profile cannot lose the chart browser again.
- **`docs/SETUP.md`**, **`src/analyzer/AGENTS.md`**: the Quick start and the chart paragraph describe
  the Playwright renderer and its browser install.
- **Tests**: the chart-export tests stub `_render_with_playwright`; the Kaleido/fallback/non-PNG
  cases and the `kaleido.stop_sync_server` shutdown patch are deleted.

### Verified
- `python -m playwright install chromium` → Chrome for Testing 149.0.7827.55 (183.6 MiB) in the
  fresh profile.
- A real 125-candle `ChartGenerator.create_chart_image` under the production loop arrangement
  (Proactor process default + selector main loop) → **1920×1080 PNG, 237,350 B in 9.6 s**; the
  candles, SMA/RSI/volume/CMF+OBV panels and max/min annotations render.
- `pytest tests/domain_rag_indicators/test_indicators_and_patterns.py tests/domain_infra/test_app_runtime.py -q`
  → **365 passed**; `ruff check src start.py scripts` and `pyright src start.py` clean.
- `scripts/results_chart.py --data-dir results/2026-11-12 --out …` → PNG, `bot +0.05% vs
  buy-and-hold -9.27%`.

## 2026-11-16 — The reader-facing trio says only what the code does

`claims.json` had been frozen since 2026-11-15 because the nightly publisher exited 1 on 2026-11-15
and 2026-11-16, and it logs only the last 3000 characters of the pytest output — which under `-q`
are the summary list and the counts, never a traceback. So the README advertised 2,781 tests and
~135 s while HEAD collected 2,807 and ran them in 123 s, and `docs/FEATURES.md` still claimed a
"2026-11-12, ~119 s" run.

The same pass audited the three surfaces a reader reaches from the README's "Start here" table
(Showcase site, Full feature list, Agent rules) against the tree. Removed: "the model gets 65
Numba-compiled indicator functions as text" (the 65 are kernels; the prompt carries their values),
the site's "browser suite … after `playwright install chromium`" (no test opens a browser), a printed
`supertrend_numba` excerpt that carried a docstring the file does not have, DeepSeek as the live
provider (`provider = opencode`, `deepseek-v4.1-flash`), the site's attribution to the engine's
repository, and a dangling `ADR 0005` citation. The surprise ratio now shows the denominator the code
uses, `max(|expected|, 0.01)`.

### Changed
- **`claims.json`**: 2,781 → **2,807** collected tests, 135 → **123** s, 615 → **631** commits,
  measured 2026-11-16; the ledger block is untouched and still names `results/2026-11-12`.
- **`scripts/claims_report.py`**, **`scripts/ledger_facts.py`**: `--snapshot-at <date>`, so
  re-publishing from an existing `results/<date>` keeps `ledger.snapshot_at` pointing at that
  directory. Without it the documented manual path wrote reproduce links to a snapshot that does not
  exist.
- **`scripts/publish_public_numbers.py`**: a red suite quotes the first 80 output lines before the
  tail, so the next failure carries its first traceback instead of only counts and a list of ids.
- **`docs/FEATURES.md`**: the regression-tests section takes its count and duration from
  `claims.json`, states that no test carries the `integration` marker, and names the one skip (the
  MCP history test without the optional `mcp` package); the chart-vision sentence; the surprise-ratio
  denominator.
- **`docs/PUBLIC-NUMBERS.md`**: `--suite-seconds` and `--snapshot-at` in the manual path.
- **`README.md`**: badge, Tests row and quick start at 2,807 / ~123 s; the provider line now reads
  `provider = opencode`.
- **`AGENTS.md`**: surprise-ratio denominator; `PromptBudgetPolicy` added to the utilities stage.
- **`src/trading/AGENTS.md`**: calibration is computed at prompt time, not stored; the LIMITED DATA
  marker and CoT Step 6 described the way the code renders them; the reflection cadence names the
  fourth call (`retire_contradicted_rules`); `confidence_threshold` keeps its 60 default between the
  bands instead of "else 70"; the ADR citation is gone.
- **`src/trading/guards/AGENTS.md`**: a guard failure is logged and the intent moves to `REJECTED`;
  no guard writes an audit record or a blocked-trade friction.
- **`tests/domain_infra/test_ledger_facts.py`** (new), **`tests/domain_infra/test_publish_public_numbers.py`**:
  three tests pinning the frozen snapshot date and the red-suite message.
- **`semantic-signal-site`** (separate repository): test figures render from `claims.json` again, the
  provider / repository / browser-suite / docstring corrections above, a "Corrections, 16 November
  2026" entry on the story page, and the results, measurement and replay pages marked for what they
  are.

### Verified
- `python -m pytest tests -q`: **2,806 passed, 1 skipped** in 123 s (the skip is the MCP history test
  without `requirements-mcp.txt`).
- `python -m pyright src start.py`: 0 errors, 0 warnings; `ruff check src scripts start.py`: clean.
- `python scripts/check_claims.py --pytest --site-root <site>`: consistent at 2,807 tests and
  snapshot 2026-11-12; `scripts/check_agents_md.py` (288 path references),
  `check_config_contract`, `check_hardcoded_timeframes`, `check_loose_helpers`, `check_dead_symbols`
  and `check_comment_policy` all exit 0.
- Site: `npm run build` green, including its own `check-claims.mjs` and `llms-full.txt` (13 pages).

## 2026-11-16 — The trade card says what it plots, in the book's own words

Two numbers on the Performance Metrics hover card had no subject. `+0.31%` on an open position is
the account's gain since the **first** trade — the chart's own Y value, not the position's result —
and `Since previous` was the step of that gain, hidden below a threshold until a close moved the
account. Neither label said what it measured, and entries spoke spot (`SELL`) while every
bot-booked close already spoke positions (`CLOSE_SHORT`).

The card now names both numbers and uses one vocabulary. Entries read `OPEN_LONG`/`OPEN_SHORT`
(mapped from whichever of `BUY`/`LONG`/`SELL`/`SHORT` the model sent — the bot books every entry
as a LONG or SHORT position), so an `OPEN_SHORT` matches the `CLOSE_SHORT` that closes it. The
stored action is untouched: `OPEN_*` is a display name, not a new signal.

The value line is labelled `Account gain since first trade`, and `Since previous` became
`Account change`, computed as the true percent step of capital at that point rather than the
difference of two gains — the same `+0.24%` today, but now literally what the label says. Every
close shows the row (`0%` included), so a grey dot explains itself instead of losing its number to
a threshold.

### Changed
- **`static/modules/performance_chart.js`**: `SERIES_NAME` = `Account gain since first trade`,
  `ENTRY_LABELS` for the four entry signals, `buildPoints` keeps `previousValue` and emits `delta`
  as the percent change of capital, and the tooltip prints the label line plus `Account change` on
  closes only (`?v=5.6`).
- **`static/css/panels.css`**: `.chart-tip-label` (11 px, `--text-dim`); `panels.css` 7.31.
- **`static/main.js`**, **`static/index.html`**, **`static/css/style.css`**: module 5.6,
  `main.js` 9.13, `style.css` 7.33.
- **`docs/FEATURES.md`**: the hover card's labels and the `0%` grey dot.

### Verified
- `pytest tests -q`: **2,803 passed, 1 skipped** (unchanged — no Python file changed); `pyright src
  start.py` 0 errors; `ruff check src/ scripts/ start.py`, `check_comment_policy`,
  `check_dead_symbols`, `check_loose_helpers`, `check_agents_md`, `check_config_contract` and
  `check_hardcoded_timeframes` exit 0.
- Live dashboard, 1600×1000, 51 points: the entry at 13 Nov 20:07 renders `+0.31% / Account gain
  since first trade / OPEN_SHORT 68 956,59 USD`; the close at 14 Nov 16:19 renders `+0.56% /
  CLOSE_SHORT 65 450,30 USD / Account change +0.24%`; a flat close renders `Account change 0%`.
  `Since previous` no longer appears in the assets.
- 390 px: the widest card is 207 px (right edge 238), `scrollWidth == innerWidth == 390`, no
  console errors.
- Net: **+34 / −19 = +15 lines** across the code and docs (changelog excluded), +77 / −19 over all
  7 files.

## 2026-11-15 — A rule says where it applies, and the chart shows where a trade starts

Three follow-ups to the learning-scope commit and the Brain merge.

**Rules.** `b6c7451` made the prompt rank a learned rule by the conditions it was learned in and
prefix every other rule with `scope mismatch`, but the dashboard dropped those fields
(`RULE_METADATA_FIELDS` had no `direction`/`regime`/`adx_bucket`) and every card looked equally
applicable. The rule card now prints `SHORT · BEARISH · HIGH ADX` (or `global`), and the payload
carries `scope_matches_now` computed from the live snapshot, so a rule that does not fit today's
regime or ADX bucket wears the same `scope mismatch` chip the prompt uses. The ADX threshold and the
comparison moved into `brain_patterns.adx_bucket_for_value` / `rule_scope_matches`, called by the
prompt builder and the dashboard, so the two cannot drift.

The rule count was wrong the other way: `semantic_rule_count` counted the collection (4) while
`get_active_rules` filters `active: true` (1), so the Overview "Active Rules" KPI and the
`Brain recall: rules_active=` log line both claimed four rules. The property counts active rules only,
which is what the KPI label and the log field always meant — 4 → 1 today.

**Memory.** The store view stopped showing a dead column: Similarity was `--` in every row because
the store scope reads the collection without an embedding, so it is now rendered only in the
recalled view, as a fixed `▼` ranking that cannot be clicked into an arbitrary order. The store also
never asks the API to sort by a key that is zero for every row.

**Chart.** Performance Metrics drew 25 dots for 25 closes and nothing for the 26 opens, so today's
open SELL at 13:01 had no marker at all. Entries (`BUY`/`LONG`/`SELL`/`SHORT`) are hollow circles now,
closes stay filled green/red/grey, and one caption line under the chart says which is which. The
eight grey closes are the ones that did not move the account — the same closes the statistics drop,
which is why the KPI reads "Trades 20" beside 25 dots.

### Changed
- **`src/trading/brain_patterns.py`**: module-level `adx_bucket_for_value()` and
  `rule_scope_matches()` (returns `None` for a rule that declares no scope);
  `TradePatternAnalyzer.adx_bucket_for_meta` delegates, so one function owns the 25/20 thresholds.
- **`src/trading/brain_context.py`**: `_rule_scope_matches` delegates to the shared comparison; the
  prompt's `scope mismatch` output and ranking are unchanged.
- **`src/trading/vector_memory_rules.py`**: `semantic_rule_count` counts `where={"active": True}`,
  the same filter `get_active_rules` uses.
- **`src/dashboard/decision_presenter.py`**: `RULE_METADATA_FIELDS` gains `direction`, `regime`,
  `adx_bucket`.
- **`src/dashboard/routers/brain.py`**: every rule carries `scope_matches_now`
  (`True`/`False`/`None`) from the cached brain status; `None` when no ADX snapshot exists or the
  rule declares no regime/bucket.
- **`static/modules/rules_panel.js`**, **`css/components.css`**: one `.rule-scope` chip in the rule
  meta row, plus `.is-mismatch` in the warning colour.
- **`static/modules/vector_panel.js`**: the Similarity header and cells render only in the recalled
  view (fixed `aria-sort="descending"`, not sortable), the detail row spans 7 columns in the store and
  8 in recall, and `sort_by` falls back to `date` for the store.
- **`static/modules/performance_chart.js`**, **`static/index.html`**: hollow entry markers, one
  caption legend under the chart; `main.js` 9.12, `style.css` 7.32, `components.css` 7.30, and the
  module versions `vector_panel` 5.6, `performance_chart` 5.5, `rules_panel` 1.2.
- **tests**: `test_get_active_rules_reports_scope_and_marks_the_live_mismatch` and
  `test_get_active_rules_leaves_the_scope_unknown_without_a_live_snapshot`; the retire test also
  pins `semantic_rule_count` at 1 before and 0 after a deactivation.
- **`docs/FEATURES.md`**, **`src/dashboard/AGENTS.md`**: the rule scope, the chart dots and the
  active-only rule count.

### Verified
- `pytest tests -q`: **2,803 passed, 1 skipped** (2,801 before, +2 tests); `pyright src start.py` 0
  errors; `ruff check src`, `check_dead_symbols`, `check_comment_policy`, `check_loose_helpers`,
  `check_agents_md` and `check_claims` exit 0. `claims.json` still records 2,781 — its ledger is a
  separate refresh.
- Performance Metrics at 1600×1000 on the live dashboard: **51 markers — 26 hollow entries and 25
  filled closes (9 red, 8 green, 8 grey)**, the open SELL at 13:01 included, with the legend line
  under the chart; the stat strip still reads `Trades 20` (statistics count only the closes it can
  score).
- Memory: the recalled view renders 8 columns with a static `Similarity ▼` header and four sortable
  headers; "Show all 17 stored" renders 7 columns (no Similarity), `colspan=7` detail rows, 17 rows
  and `17 stored · 17 shown`; back to recalled → 5 rows. 390 px: `scrollWidth == innerWidth`, no
  console errors.
- Rules: the card markup was exercised with a stubbed `/api/brain/rules` payload (`SHORT · BEARISH ·
  HIGH ADX`, `LONG · BULLISH`, `global`, one `scope_matches_now: false`) and rendered the scope chips
  with the mismatch only on the failing rule. The running dashboard process predates this change, so
  the three scope fields, `scope_matches_now` and the corrected `rule_count` appear on the live page
  after a restart; until then the KPI still reads 4 rules and the cards show `global`.
- Net: **+166 / −47 = +119 lines** across the code, docs and tests (changelog excluded), +225 / −47
  over all 17 files.

## 2026-11-15 — One tab for what the brain remembers

The closed-trade memory was rendered three times. The Brain tab's decision chain printed five
recalled experiences and five recent lessons, the **Trade Journal** tab printed the same post-mortems
from the same repository and the same FTS index, and the **Memory Bank** tab printed the same
experiences again as a sortable table — ten sidebar entries for one question, with the newest five
lessons visibly duplicated between the chain and the journal. There is now **one tab and one
rendering per record**: the sidebar is down to eight entries, and the Brain tab reads top to bottom
as the decision itself.

`Last decision` keeps the verdict line and the market context chips (with ADX/RSI) but loses the
chain: context, memory and lessons are no longer steps of a timeline, because memory and lessons now
have panels of their own. `Memory` lists the closed trades ranked by similarity to the market the
last decision saw — five rows, the same five the prompt recalled, with the whole store behind
"Show all 17 stored" (sortable columns, expandable stored context) and the win-rate breakdown by
confidence, ADX, sentiment and volatility behind one native disclosure. `Lessons` carries the
post-mortem archive: five newest by default, the newest 100 behind "Show all lessons", an FTS5
search box that overrides both, and the full analysis per card inside `<details>`. `Active rules`,
`Blocked attempts` and the `Analysis chart` stay where they were.

The merge also fixed the search that never survived a poll: the 30 s slow lane called
`updatePostMortemData()` with no argument, so every typed query was wiped within half a minute. The
lessons panel now owns `activeQuery` and `showAll` as module state, and the memory panel keeps its
recalled/all mode the same way. `fullscreen.js` now maps only panels that have a content element —
`Blocked attempts` carried a fullscreen button that logged `Content element not found` on every
click, and it is gone.

The first cut of the store view read it through the recall path, so "Show all 17 stored" was a
56-day, 50-row similarity sample labelled as everything. The API now takes `scope=store`, which reads
the collection without an embedding and sorts before truncating, and the panel asks for it. The
recalled view stopped painting a "When ▼" arrow over similarity-ordered rows, the lessons toggle is
disabled while a search is active instead of silently doing nothing, an invalid FTS query reaches the
panel as an error instead of "no matches", the tab no longer prints the context chips twice, the
memory toggle keeps keyboard focus across its own re-render, and its meta line no longer announces
itself to screen readers every half minute.

### Changed
- **`static/index.html`**: sidebar 10 → 8 entries (`Memory Bank`, `Trade Journal` deleted); the Brain
  tab rebuilt with `#panel-decision`, `#decision-context`, `#panel-memory`/`#vector-content`,
  `#panel-lessons`/`#lessons-body`/`#post-mortem-content`, search ids and a `#lessons-view-toggle`;
  `main.js` 9.11, `style.css` 7.31.
- **`src/dashboard/routers/brain.py`**: `GET /api/brain/vectors` takes `scope=recall|store` (unknown
  values fall back to `recall`); `store` skips the market-context embedding, lists the collection and
  sorts before truncating, and the 30 s cache key carries the scope.
- **`src/managers/post_mortem_repository.py`**: an FTS5 syntax error re-raises as
  `ValueError("Invalid search query: …")` instead of returning an empty page, so the panel can tell
  a broken query from an empty archive; `GET /api/brain/post-mortems` already returns it as `error`.
- **`static/modules/decision_panel.js`** (244 → 160 lines): chain, `lessonContent()` and the
  `Journal`/`Memory`/`Rules` counters deleted; the meta line is the payload's `updated` time, and the
  context renders as chips plus one muted ADX/RSI line.
- **`static/modules/post_mortem_panel.js`** → **`static/modules/lessons_panel.js`**: `initLessonsPanel`
  / `updateLessonsPanel`, `activeQuery` + `showAll` state, a 100-row archive, `5 newest` / `N shown` /
  `N matching` meta, the toggle disabled while a search is active, explicit HTTP-error, API-error and
  empty states, `aria-label` on the search field.
- **`static/modules/vector_panel.js`**: two modes — `limit=5&sort_by=similarity` (what the decision
  recalled) and `limit=200&sort_by=date&scope=store` (the store) — with the toggle rendered in the
  freshness row, `#memory-meta` showing `17 stored · 5 recalled · updated HH:MM:SS`, no sort mark on
  the recalled view, the context line cut to one sentence, the win-rate cards inside
  `<details class="memory-stats">`, focus restored after the toggle re-renders, and a sort-header
  click switching to the store (sorting a recalled top-five by P&L is not a recall).
- **`static/modules/fullscreen.js`**: `panel-memory` → `vector-content`, `panel-lessons` →
  `lessons-body`; `panel-decision` and `panel-vectors` entries gone; a panel without a content id no
  longer gets a button.
- **`static/modules/ui.js`**: entering the Brain tab refreshes the lessons and memory panels too.
- **`css/components.css`**: the `.chain*` block deleted (178 lines out, 12 kept: `.chain-num` and
  `.chain-empty`, which the blocked-attempts table still uses); `.memory-stats` and
  `.decision-context .context-metrics` added.
- **`css/panels.css`**: `#panel-vectors.full-height-panel` gone, `.pm-toolbar` is a wrapping row so
  the search field and the view toggle share it.
- **`tests/domain_infra/test_dashboard_brain.py`**: the binding tests moved to the new ids and now
  assert that `decision-chain`, `nav-memory`, `tab-memory`, `nav-journal`, `tab-journal`,
  `panel-vectors` and `Memory Bank`/`Trade Journal` are absent, and that `post_mortem_panel.js` and
  `updatePostMortemData` exist nowhere; two tests added for `scope=store` (reads the store without
  recall, truncates after sorting) and for an unknown scope falling back to recall.
- **`tests/domain_brain/test_post_mortem.py`**: an invalid FTS query now asserts
  `ValueError("Invalid search query: …")` instead of an empty page.
- **`docs/FEATURES.md`**, **`src/dashboard/AGENTS.md`**, **`static/landing.html`**: eight tabs, the
  Brain layout as it now stands, and the stale "decision graphs" phrase dropped.
- **`docs/assets`**: `dashboard-overview.png`, `dashboard-overview.webp` and `brain-activity.webp`
  recaptured at 1600×1000 — the old shots showed the two deleted tabs in the sidebar.

### Verified
- `pytest tests -q`: **2,801 passed, 1 skipped**. HEAD's `b6c7451` (learning scopes) took the suite to
  2,799 and this change adds two test functions. `claims.json` still records 2,781 from its own
  snapshot, so the README badge needs a ledger refresh that neither change performed.
- Live instance `http://127.0.0.1:8000` at 1600×1000 after the fixes: the recalled view shows five
  rows, `17 stored · 5 recalled`, the similarity column as the only sort mark and one context
  sentence (the seven chips now appear once, under `Last decision`); "Show all 17 stored" → 17 rows,
  `17 stored · 17 shown`, the date mark, and focus still on the toggle; "Show all lessons" → 18
  cards and `18 shown`; FTS `breakout` → 4 cards, 11 `<mark>` hits and a disabled view toggle, which
  re-enables when the query is cleared; `document.documentElement.scrollWidth` equals `innerWidth`,
  the memory table renders as labelled cards on the phone, and the browser console reports no page
  errors.
- `scope=store` is covered by
  `test_get_vector_details_store_scope_reads_the_store_without_recall` (120 trades stored, `limit=5`:
  no recall call, `get_all_experiences(limit=120)`, rows returned newest-first) and the unknown-scope
  fallback by `test_get_vector_details_ignores_an_unknown_scope`. The running dashboard process
  predates the route change, so the live store view still answers from the recall path until the app
  restarts.
- Net for this change, counted over the dashboard, docs and its tests only (changelog and the three
  binary assets excluded): **+452 / −498 = −46 lines**, the new 166-line lessons module included.
  `scripts/check_net_diff.py --worktree` reports +365 / −492 = −127 over 19 files, but that command
  diffs against `origin/master`, so its range also carries the unpushed `b6c7451`.
- Known caps, both truncating after the sort: the lessons archive asks for 100 post-mortems (the
  endpoint maximum), and the memory store view asks for 200 of `trade_count` (the endpoint accepts
  500); past those the meta line shows fewer `shown` than `stored`.

## 2026-11-15 — A learned rule now says where it was learned

The prompt has told the model for months to apply a learned rule "only when that declared condition
matches the current snapshot". No rule declared one. All four stored rules in
`data/trading/brain_BTC_USDC_4h` carried their scope in `source_pattern`
(`SHORT|BEARISH|HIGH_ADX|sl_hard_15m|tp_hard_15m`, `LONG|BULLISH|stop_loss|…`, a legacy
`sideways_failure|MEDIUM|LONG|BULLISH|…` and `SIDEWAYS_FAILURE|MEDIUM`) but not in metadata, so the
last cycle's prompt printed a bare `- SHORT trades perform well in BEARISH market with High ADX…`
(`logs/Bot/2026_11_15/Bot.log:3162`) and `update_rule_validation_feedback` matched **every** active
rule: empty declared conditions compare equal, so a closed SHORT/BEARISH trade also fed the counters of
the LONG/BULLISH rules that the ranking's `final_score` is built from.

`declared_conditions()` recovers `direction`, `regime` and `adx_bucket` from the stored pattern at read
time, anchored on the first `LONG`/`SHORT` token — that parses all four layouts above, including the
one that puts the mistake type first, and leaves `SIDEWAYS_FAILURE|MEDIUM` global. Explicit metadata is
never overwritten, nothing is migrated and nothing is re-embedded. The feedback pass matches on the
enriched metadata, the prompt prints the scope in front of the rule text, and a rule whose regime or
ADX bucket does not match the snapshot is ranked last and prefixed with `scope mismatch` instead of
being dropped: the direction of the next decision is not known when the prompt is built.

Retirement was the other half of the same contract. `AGENTS.md` has said that active rules are
"deactivated by evidence, not age" while `deactivate_semantic_rule` had no production caller — three of
the four rules were retired by an older one-off repair (`tautological exit-reason key`,
`outcome-conditioned mislabel`) and one stayed active. `retire_contradicted_rules()` now runs on the
reflection interval: three contradictions, at least half the rule's stored support and more
contradictions than validations deactivate the rule with an `auto: …` reason. The half-support gate is
what stops a rule built on 13 wins from dying after three losses. A retired id keeps its retirement
state when it is re-synthesised, so `python scripts/brain_rules.py reactivate --id <rule>` re-stores the
same text with `active=true` — without it the automatic write had no undo.

Each prompt build logs one `Brain recall:` line. The injected block is 10,531 characters ≈ 2,600 tokens
against `brain_max_tokens = 50000`, so the context is not being truncated; what was missing was knowing
what actually reached the model.

### Changed
- **`src/trading/vector_memory_rules.py`**: `declared_conditions()` reads the scope from
  `source_pattern`, `_metadata_with_rule_defaults` fills `direction`/`regime`/`adx_bucket` without
  overwriting explicit values, and `update_rule_validation_feedback` matches on the enriched metadata.
- **`src/trading/brain_reflection.py`**: `retire_contradicted_rules()` with
  `RETIRE_MIN_CONTRADICTIONS = 3` and a half-the-stored-support floor, a warning (not an exception) on
  failure, one INFO line per retired rule.
- **`src/trading/brain.py`**: retirement fires on the reflection interval next to the three reflection
  passes, exposed as `retire_contradicted_rules()`, and the context provider is handed the
  `pattern_analyzer` so both sides bucket ADX with one function.
- **`src/trading/brain_context.py`**: scope-aware rule block (`_rule_scope_matches`,
  `_current_adx_bucket`), matching rules first, a `scope mismatch` prefix for the others, the
  `Brain recall:` telemetry line, and `rules_scope_matched` counting only rules that declare a scope.
- **`scripts/brain_rules.py`**: `list` prints the derived `scope=[…]` per rule (`scope=global` when it
  declares none) and the new `reactivate --id …` reverses an automatic retirement.
- **`tests/domain_brain/test_vector_memory.py`**, **`test_brain_learning.py`**: 19 tests — scope
  parsing on all four live layouts, no override of explicit metadata, ADX-vocabulary round trip,
  scoped feedback, matching-first rendering with the `scope mismatch` prefix, retirement thresholds
  (mocked and on a real Chroma store), retirement on the interval, reactivation through an explicit
  `active=true`, telemetry payload.

### Verified
- Full suite **2,799 passed, 1 skipped** (2,780 before, +19); `pyright src start.py` 0 errors;
  `ruff check src/` clean; `check_loose_helpers`, `check_config_contract`, `check_hardcoded_timeframes`,
  `check_dead_symbols`, `check_comment_policy` and `check_agents_md` all clean.
- `python scripts/brain_rules.py list` against the live store prints `scope=[SHORT / BEARISH / HIGH_ADX]`
  for `rule_best_short_bearish_high_adx…`, `scope=[LONG / BULLISH]` for the two legacy LONG rules and
  `scope=[global]` for `SIDEWAYS_FAILURE|MEDIUM`; `reactivate --dry-run` answers
  `would reactivate <rule>`; the store is still 1,609,728 bytes.
- The running bot has not been restarted, so the first live `Brain recall:` line is not yet observed;
  confirm with `Select-String "Brain recall" logs/Bot/<date>/Bot.log` after the next cycle.

## 2026-11-15 — A dashboard that fits a phone

Three things were broken, and only one of them was cosmetic. The console's day buttons
(`▶ Live`, `11-14`, …) rendered **25 px wide on every screen** because `.toolbar-btn` hardcodes
`width: 1.75rem` and `.console-page-btn` only overrode the padding — "▶ Live" and "11-14" sat on top
of each other inside a 53 px nav. The performance chart's zoom worked until the next poll: clicking
`+` twice set `minX/maxX` to a 14-day window, and 36 s later (one `updateSlowLane` cycle calling
`chart.updateSeries`) the range was back to full. That same chart loaded ApexCharts from an
**unversioned** CDN URL: the code was written against v3/v4 and the URL now resolves to v8.8.2.
And on a 390 px phone the experience table was 743 px wide inside a 390 px viewport — Outcome, P&L,
Confidence and Similarity were clipped and unreachable — while the four KPI cards stacked one per row
for ~470 px of scroll.

The chart now plots **the gain since the first recorded point in percent**, not the account value in
dollars: at this account's ±0.5% range a dollar axis rendered a flat line with a spike, and the
auto-scaled Y axis flattered it. Closed trades are small dots coloured by whether that close moved
the account up or down; the toolbar icons are replaced by a **7D / 30D / All** selector that works
with a thumb, and the polling no longer resets the range (it only re-renders when the data actually
changed). The panel lost its fixed 600 px height, so ~250 px of empty space under the plot is gone.
On mobile the KPI grid is 2×2, the statistics cards are ~140 px instead of ~230 px, panel headers
wrap, the memory table becomes one card per experience with labelled fields, and the console
controls get 36 px tap targets. The unused vis-network leftovers are gone: `.network-container`,
`.verdict-caption`, `.btn-primary`, the `panel-thoughts` fullscreen mapping, and
**`build_decision_graph`** — 213 lines that built a node graph nothing has drawn since the Brain tab
was rebuilt, still shipped in every `/api/brain/decision-summary` response.

### Changed
- **`static/modules/performance_chart.js`**: percent series, straight stroke, close markers coloured
  by account movement, `7D/30D/All` range buttons, axis tick formatter, data-signature guard against
  the polling reset, `.chart-tip` tooltip instead of inline styles, height 320 px (260 px on phones).
- **`static/index.html`**: `apexcharts@8.8.2` pinned, range control in the panel header, stats moved
  out of the header into a `.stat-strip` row, `#panel-performance` no longer `.full-height-panel`.
- **`css/panels.css`**: `.console-page-btn { width: auto }`, `.stat-strip*`, `.chart-range`,
  `.range-btn`, `.chart-tip*`, and a mobile console block (scrolling day strip, 36 px controls).
- **`css/tables.css`**: the experience table becomes a card list at ≤768 px via `data-label`
  attributes, so all eight fields are readable without sideways scrolling.
- **`css/responsive.css`**: KPI 2×2, compact stat cards, stacked context indicator, wrapping panel
  headers, and the dead `.network-container` selector dropped.
- **`static/modules/vector_panel.js`**: `data-label` on each cell; **`fullscreen.js`**: no more
  `panel-thoughts` entry, chart size restored through `resetPerformanceChartSize()`.
- **`decision_presenter.py`**, **`routers/brain.py`**: `build_decision_graph` and `short_label`
  deleted, `"graph"` removed from the payload; the three tests that asserted the graph now assert the
  counts the dashboard header actually prints.

### Verified
- Full suite **2,780 passed, 1 skipped**; `check_net_diff.py --worktree` reports **+969 / −1457 =
  −488** for the code scope (the two new modules, +340, are not in the index yet, so the dashboard
  totals −148).
- 1600×1000 and 390×844: no overlapping controls, `scrollWidth == innerWidth` on every tab, the
  memory table shows all eight fields, the 7D range still holds after 60 s and two poll cycles.
- Contrast and palette untouched (the same 13 tokens); `apexcharts@8.8.2` verified by the CDN body's
  own version banner.

## 2026-11-15 — A dashboard you can read

The public dashboard shipped a sci-fi skin rather than information: the shipped dashboard assets
carried **46 distinct hex colours and 107 `rgba()` calls** (25 files, `landing.html` and `admin/`
excluded), neon cyan glow on the logo, KPI cards and status dot, gradient hairlines on every panel,
and ALL-CAPS headings. Two panels were actively misleading. **"Visual
Cortex Input"** was the OHLCV-plus-indicators image the model was handed, named after a metaphor
nobody outside the repo can decode. **Decision Pathways** drew that same decision as a vis-network
star: one root, six level-one nodes and fifteen leaves, with `READABLE_MIN_ZOOM = 0.75` forbidding
the zoom level that would have fitted them (measured on the live instance: labels clipped on both
edges, `FLAT` overlapping `High ADX + ME…`, `plan followe…` cut off). **Semantic Rules** and **Trade
Friction** were filed under Memory Bank even though both are decision inputs, and the experience
table printed the whole `close_reason` sentence inside the Outcome cell, making every row eight
lines tall.

The colour system is now one place. `base.css` holds the whole palette — four surface tokens, one
accent, three semantic colours plus their aliases, four text tokens, the borders and the one branded
button colour, 13 distinct hex values in all — and nothing outside that block names a hue except the
JS fallbacks that mirror it. Glows, gradient hairlines, the noise overlay, the logo and
status-dot animations and the `Exo 2` display font are gone; headings are sentence case and the
all-caps label survives only on values at 12 px or below. The Brain tab is a decision chain built from
one payload: a verdict line, then **market context** (classifier chips plus ADX/RSI), **recalled
experiences** (outcome, P&L, similarity, `<details>` for the stored context) and **recent lessons**
(verdict plus lesson). **Active rules** and **Blocked attempts** moved to Brain where they belong, and
the chart panel is now titled **Analysis chart** with a caption saying what it shows. Drill-down uses
native `<details>` and a chevron button with `aria-expanded`/`aria-controls`, so the table is
keyboard-operable for the first time.

### Changed
- **`src/dashboard/static/css/base.css`**: the single palette, `[hidden]`, a real `.sr-only`, and
  the focus ring retuned to the accent.
- **`css/layout.css`**, **`css/components.css`**, **`css/panels.css`**, **`css/tables.css`**,
  **`css/animations.css`**: glows and gradients removed, sentence-case headings, table headers muted,
  the whole `.decision-legend`/`.zoom-btn`/`.decision-graph`/`.decision-detail`/`.synopsis-*` block
  replaced by `.chain-*`, and the dead `.stack-layout`/`.large-panel`/`.meta-info`/`.trade-friction-block`
  rules deleted.
- **`static/index.html`**: `Visual Cortex Input` → **Analysis chart** (plus caption and aria labels),
  `Neural Network State` → **Brain Activity**, `Vector Experience Database` → **Memory Bank**,
  decision panels restructured, `vis-network` script dropped, `Exo 2` dropped, inline styles replaced
  by `.tab-meta`/`.mono-meta`, every `?v=` bumped (CSS 7.26, `main.js` 9.5).
- **`static/modules/decision_pathways_panel.js`** → **`decision_panel.js`**: 468 lines of vis-network
  layout, colour table, zoom controls and node-detail pane replaced by a 244-line renderer for the
  verdict, the three-step chain and the blocked-attempt table, all from the existing
  `/api/brain/decision-summary` payload (no backend change).
- **`static/modules/rules_panel.js`** (new): the learned-rule cards moved out of `vector_panel.js`,
  fed by `/api/brain/rules`; type is now a bordered label instead of a coloured card fill.
- **`static/modules/vector_panel.js`**: the rules, friction and blocked-trades fetches and their
  renderers removed (523 lines → 353); the table loses the Chroma-id column and the `direction` sort,
  gains an expander, moves `close_reason` into the expanded row, and shows the context indicators on
  one muted line.
- **`static/modules/fullscreen.js`**: network-resize branches deleted, mapping retargeted to
  `decision-chain` and `rules-content`.
- **`static/modules/ui.js`**: the 300 ms canvas delay on tab entry removed; both Brain panels refresh
  on entry.
- **`static/modules/performance_chart.js`**, **`visuals.js`**, **`log_viewer.js`**: colours read from
  the CSS tokens, the 70%-opaque area fill dropped, BUY/SELL/CLOSE markers reduced to
  green/red/neutral.
- **`tests/domain_infra/test_dashboard_brain.py`**: the two binding tests moved to the new ids and
  modules, and now assert that `vis-network`, `decision_pathways_panel.js` and
  `updateDecisionPathways` are gone rather than present.
- **`README.md`**, **`docs/assets/`**: new `dashboard-overview.png`/`.webp` and `brain-activity.webp`
  at 1600×1000 (the PNG drops from 313 KB to 119 KB); the 887 KB `dashboard-tour.gif` of the old UI is
  deleted together with its README line.
- **`docs/FEATURES.md`**, **`src/dashboard/AGENTS.md`**: the Brain/Memory split, the palette rule and
  the sentence-case rule recorded; the "never a redesign" line replaced with what makes a redesign
  acceptable.

### Verified
- Full Windows-venv suite: **2,779 passed, 1 skipped, 2,780 collected in 136.80 s**;
  `scripts/check_comment_policy.py`, `check_dead_symbols.py`, `check_loose_helpers.py`,
  `check_config_contract.py`, `check_hardcoded_timeframes.py` and `check_agents_md.py` (286 path
  references) all exit 0.
- Live instance `http://127.0.0.1:8000` screenshotted on all ten tabs: no canvas, no horizontal
  clipping in the Brain tab, experience rows down from eight lines to three, and `#panel-friction`
  hiding itself only when the store has no blocked attempt at all.
- Colour count across the shipped dashboard assets (`landing.html` and `admin/` excluded, same file
  filter before and after): **46 distinct hex values and 107 `rgba()` calls → 14 and 45**, where 13
  of the 14 are the token block in `base.css` and the fourteenth is the `&#039;` entity in
  `html_utils.js`.

## 2026-11-15 — A settled exit does not warn that it was never booked

At 10:01:42 the monitor printed `CRITICAL LOCAL EXIT CONDITION stop_loss @ 65451.11 NOT BOOKED
(state=pending, forwarded=yes)` and pushed the matching Discord alert — twelve seconds after the
same trade had been booked from the executor's fill receipt (exchange order 7348889 @ 65443.89).
The forward *is* the booking path (`_apply_confirmed_intent` → `book_executor_side_exit` →
`close_position`, post-mortem and statistics included), so when `check_reference_exits` published
its pre-forward `LocalExitRequest` snapshot the local position was already gone and the message
("local position kept, no statistics entry") was false. Nothing was lost: history row 79 and the
19 → 20 trade count were correct.

`_publish_local_exit_request` now returns before the alert when the local position no longer exists;
a forward that settles the close emits neither the CRITICAL nor the push, while a forward that
leaves the position open alerts exactly as before. The guard also covers the soft/hard exit callers,
which would have made the same false claim.

### Changed
- **`src/trading/position_status_monitor.py`**: `_publish_local_exit_request` skips the alert when
  `current_position` is None (the close was settled during the awaited forward).
- **`tests/domain_trading/test_guards_and_exits.py`**: a forward that settles the position emits no
  CRITICAL and no Discord message.
- **`src/trading/AGENTS.md`**: the reference-exit alert only fires while the local position is open.

### Verified
- Full Windows-venv suite: **2,780 passed, 1 skipped, 2,781 collected in 134.63 s**; targeted
  `tests/domain_trading/test_guards_and_exits.py` **101 passed**.
- `ruff check src/` clean, `pyright src start.py` **0 errors**, and the loose-helper, config-contract,
  hardcoded-timeframe, dead-symbol and comment-policy gates clean.
- Incident replay: `logs/Bot/2026_11_15/Bot.log` shows the executor fill BEFORE the monitor line
  (`09:01:30` booking vs `09:01:42` alert); the new test reproduces that ordering with a forward
  that books and asserts silence.

## 2026-11-14 — The live console is not a second prompt viewer

Each model call prints the full prompt and the full response: in `logs/Bot/2026_11_14/Bot.log` that
was 374,145 + 24,815 of 429,637 characters (93%), a prompt dump being 60–63 KB. The terminal is
where those dumps are read and `Bot.log` is the audit trail, but the same records also streamed into
the dashboard's Live Console and its seven-day pages, where the Last Prompt / Last Response panels
already show the exchange from `analysis_engine` state.

Both records now carry `skip_live_stream`; `LogStreamHandler.emit` drops a tagged record before it
reaches the WebSocket, the replay buffer or `ConsoleBuffer`. The Rich console handler and the
rotating file handler are untouched, so the terminal and `Bot.log` still print every dump in full.

### Changed
- **`src/dashboard/log_stream.py`**: `emit` returns early for a record marked `skip_live_stream`.
- **`src/managers/provider_orchestrator.py`**: `Full prompt content` carries the tag.
- **`src/managers/model_manager.py`**: `Full response content` carries the tag.
- **`src/analyzer/AGENTS.md`**, **`src/managers/AGENTS.md`**: the dump is described as terminal and
  `Bot.log`, with the live console skipping it.

### Verified
- Full Windows-venv suite: **2,779 passed, 1 skipped, 2,780 collected in 123.85 s**; `ruff check
  src/ scripts/ start.py` clean; `pyright src start.py` **0 errors**; `scripts/check_comment_policy.py`,
  `check_loose_helpers.py`, `check_config_contract.py`, `check_hardcoded_timeframes.py`,
  `check_dead_symbols.py` and `check_agents_md.py` clean.
- Three new tests: a tagged record never reaches a subscriber, the replay buffer or a late joiner;
  `prepare_request_messages` and `_process_result` assert `extra={"skip_live_stream": True}`, so a
  future dump without the tag cannot slip back into the live console.

## 2026-11-14 — A stop adjustment is three log lines, not six

The 05:01 cycle wrote six INFO lines for one SL tightening — `Tightening Stop Loss`, `Stored
experience`, `Position intent … state=pending`, `Position updated`, `Executor queued`, `Position
intent … -> confirmed` — and the intent lines carried a full ISO position id with a microsecond
order id, which a narrow console wrapped into ten rows. The same facts also printed twice per model
call (prompt/response/total tokens plus `Request cost`), the chart provider was named by two layers,
and `Skipping unsupported ticker symbols` repeated every cycle.

Nothing was double-executed. The pending and confirmed lines are the two states of one intent
(`update-20261114040123229472`): `data/trading/executor_verdicts.jsonl` holds exactly one verdict
for it (`UPDATE_PROTECTION`, `protection_version 3`), `data/trading/metrics/cycles.jsonl` exactly
one 05:00 cycle record with `inference_cost_usd 0.003337` and 19,403 + 4,420 tokens, and `Request
cost` vs `Cycle inference cost` showed the same dollar figure because both read that one request.

Now one update prints `UPDATE BTC/USDC: SL 69,600.00 → 68,750.00, TP 65,500.00 → 65,500.00 (SL
tightened at 64.5% progress) — awaiting executor confirmation` — both the value in force and the new
one, for SL and TP — then `Stored experience: UPDATE +3.34%`, then `UPDATE BTC/USDC confirmed by
executor`; one model call prints `Model call: 19,403 in + 4,420 out = 23,823 tokens, $0.003337
(off-peak x1 @ 04:01 UTC, from provider)`.

### Changed
- **`src/trading/position_management.py`**: the accepted tightening and the update share one line,
  which now carries the old and the new SL and TP plus the tightening's price progress;
  `Tightening Stop Loss` and `Position updated` are gone.
- **`src/trading/executor_reconciliation.py`**: a new intent is DEBUG; a resolved intent prints one
  line with its action and symbol (`UPDATE BTC/USDC confirmed by executor`), WARNING for `unknown`
  with the detail, instead of the raw key and evidence.
- **`src/trading/executor_handler.py`**: `Executor queued` is DEBUG — the update line already says
  the command awaits confirmation.
- **`src/trading/vector_memory.py`**: the stored outcome is one line; the trade id moved to DEBUG.
- **`src/utils/token_counter.py`**: prompt, completion, total and cost are one `Model call` line,
  and the estimated fallback is one line too.
- **`src/analyzer/analysis_result_processor.py`**: the duplicate `Using chart image analysis via …`
  line is deleted; `ProviderOrchestrator` already names the provider and model.
- **`src/analyzer/data_fetcher.py`**: the repeated unsupported-ticker skip is DEBUG.
- **`docs/CONFIGURATION.md`**: the billing example follows the `Model call` line.

### Verified
- Full Windows-venv suite: **2,776 passed, 1 skipped, 2,777 collected in 153 s**; `ruff check src/`
  clean; `pyright src start.py` **0 errors**; `scripts/check_comment_policy.py`,
  `check_loose_helpers.py`, `check_config_contract.py`, `check_hardcoded_timeframes.py` and
  `check_dead_symbols.py` clean.
- The update line's test asserts the old and the new stop on the logged call
  (`83,850.00 → 85,987.62`); the `Model call` text, the forward matrix's `debug` level, the ticker
  skip at `debug` and the intent-reconciliation tests stay green.
- `Full prompt content` / `Full response content` stay on the console and the dashboard live
  stream as before — the dashboard's Last Prompt / Last Response panels read
  `analysis_engine.last_generated_prompt` / `last_llm_response`, not the log stream.

## 2026-11-14 — A three-second-old market overview is no longer "a day old"

A cycle that starts after a bot restart rebuilt the market overview twice: the knowledge refresh
logged `Skipping unsupported ticker symbols on binance: FIGR_HELOC/USDT` at 01:00:05 and the
analysis enrichment logged the same line at 01:00:08, each one running `load_markets` plus
`fetch_tickers` against Binance for the same ten pairs. `MarketDataCache.is_overview_stale` read
`published_on`, and `MarketOverviewBuilder._finalize_overview` stamps that from CoinGecko's
`data_timestamp` — when the source data was *fetched*, not when the overview was *built*. The
CoinGecko client's own file cache lives 24 h, so the three-second-old overview carried a one-day-old
stamp and the analysis's one-hour gate discarded it.

Measured on the on-disk snapshot (`data/market_data/coingecko_global.json`, fetched
2026-11-13 00:03 UTC): the overview that used to answer `stale@1h: True` now answers `False`, while
an overview whose build stamp is two hours old still answers `True` — both through the raw path and
through `FormatUtils.parse_timestamp`.

### Changed
- **`src/rag/market_components/market_data_cache.py`**: freshness keys on the overview's build
  time (`overview["timestamp"]`), falling back to `published_on` for a structure that carries no
  build stamp; the branch without a timestamp parser now reads ISO build stamps instead of
  declaring the cache stale. The thresholds keep their meaning — the knowledge refresh still
  rebuilds a 24 h old overview, the analysis still rebuilds a 1 h old one — but neither judges the
  age of the CoinGecko snapshot, which `source_updated_at` and the prompt's "Market data source
  updated" line still report.
- **`tests/domain_rag_indicators/test_rag_market.py`**: one test proves both directions — a fresh
  build with a day-old source is not stale, a two-hour-old build with a fresh source is.

### Verified
- Full Windows-venv suite: **2776 passed, 1 skipped, 2777 collected in 136 s**; `ruff check src
  scripts start.py` clean; `pyright src start.py` **0 errors**; `scripts/check_claims.py --pytest`
  reports `claims consistent: 2,777 tests`.
- `README.md` badge, table and command comment plus `claims.json` moved to 2,777 collected / 136 s.
- No trading, execution, prompt or wire-format change: the diff is the cache predicate, its test
  and the bookkeeping. The next cycle after a restart fetches the overview once, not twice.

## 2026-11-13 — The code carries no prose: every docstring is gone


A class docstring is prose an agent pays for on every read and a reader skips: 294 of them held
40,545 characters, and 222 were a paraphrase of the class name (`# Client for DefiLlama API.`).
They are all deleted, together with the two bare string statements that stood in for attribute
documentation (`PHANTOM_ENTRY_GRACE_SECONDS`, `VectorMemoryAnalyticsMixin`). What the code cannot
show by name, signature and types moved into the package `AGENTS.md` as one-line contracts — about
30 lines total — and nothing else survived.

The gate changed with the rule: `scripts/check_comment_policy.py` now reports `DOCSTRING` for a
module, class, function or method docstring (the class exception is gone) and `STRING` for any bare
string statement, so a reintroduction fails the build instead of a review. `CONTRIBUTING.md`, root
`AGENTS.md` §10/§12.1 and `.pylintrc` (missing-*-docstring disabled) state the same rule.

The sweep also audited the nine `AGENTS.md` files against the code, because a docstring that lied
and an `AGENTS.md` that lied cost the same tokens. Fixed: `SafetyGuard` (no such class — duplicate
suppression is the cycle checkpoint plus the intent ledger's `decision_hash`, and the executor dedups
on `order_id`), `_parse_finite_float` (the helper is `_parse_finite_number`), `LocalTaxonomy` (the
class is `LocalTaxonomyProvider`), "4 in-memory indices" (`IndexManager` has one `coin_index`), a
`loader.py` path that only exists as `src/config/loader.py`, 300-char preview excerpts (the code
slices 301), `provider = deepseek` (the operator config runs `opencode`; the tracked example defaults
to `googleai`), "DeepSeek (active provider)", the missing `src/mcp_tools/` in the §6 tree, the
missing `deepseek` row in the provider table, and the retry backoff attributed to the wrong function.
`noul` is a real Decisions field (`edge_now.noul`) and stayed, with its origin named.

### Changed
- **294 class docstrings and 2 bare strings** removed across `src/`, `tests/` and `scripts/`
  (156 files); no module, function or method docstring existed. Four classes whose body was only a
  docstring now carry `pass`.
- **`scripts/check_comment_policy.py`**: no docstring of any kind and no bare string statement; only
  machine directives survive in comments. `tests/domain_infra/test_comment_policy.py` proves both
  new verdicts and keeps the 14-test count, so `claims.json` and the README badge stay unchanged.
- **`AGENTS.md` (root and 8 package files)**: the contracts that were worth keeping — exit evidence,
  intent-ledger replay, phantom grace, checkpoint atomicity, fee-as-evidence, protection version,
  `PostMortemResult`, log-stream backpressure, read-only MCP history — are one line each in the file
  that owns the code.

### Verified
- `python scripts/check_comment_policy.py`: `Comments: none outside machine directives.
  Docstrings: none.`; `check_agents_md.py` 286 paths in 19 files; `check_dead_symbols.py`,
  `check_loose_helpers.py`, `check_hardcoded_timeframes.py`, `check_config_contract.py` all green.
- `ruff check src scripts start.py` clean, `pyright src start.py` 0 errors, full suite unchanged
  (2 776 collected) with no new failure or skip.
- No runtime, wire-format, config, SQLite or Chroma behaviour changed: the diff touches docstring
  and string-statement lines plus the policy text.

## 2026-11-13 — The Reddit sample stops being the longest part of a cycle

A cycle recorded `latency_ms: 288203.6` in `data/trading/metrics/cycles.jsonl`. Its own log lines
split that 288 s into ~5 s of preparation and news refresh, **~154 s of `RedditSentimentAnalyst
.fetch_posts`**, ~8 s of candles plus indicators plus chart plus RAG, 109 s for the one model call
and 11 s of strategy, executor and notifier work. The Reddit fetch walks four subreddits strictly
one after another with a hard `asyncio.sleep(20)` between them and a 30 s backoff per HTTP 429, so
a sample that returns 40 titles in one live run took **153.78 s** — all of it waiting, none of it
work. Every other source in the cycle was measured under a second: four RSS feeds 0.06–0.19 s,
CoinGecko 0.05–0.21 s, DeFiLlama 0.07 s, `alternative.me` 0.38 s, and every ccxt binance call
(`fetch_ohlcv` 4h/1000, 1d/366, 1w/301, order book, trades, ticker) 0.26–0.32 s.

The pacing is right and stays — Reddit throttles that hard. What changed is where it happens: a
background task now refreshes the sample every 30 minutes, and the cycle only reads the last one,
so the section keeps its content and contract while costing nothing on the analysis path.

### Changed
- **`src/app.py`**: `_start_reddit_refresh()` starts one `Reddit-Sentiment-Refresh` task from
  `start_trading` (only with `[social_sentiment] enabled` and a sentiment analyst), registered in
  `_active_tasks` so the existing shutdown cancels it. `_reddit_refresh_loop()` keeps the last good
  sample when a refresh fails or comes back empty, logs `Reddit sample refreshed: N posts in X.Xs`
  at **debug** level (2026-11-14: it printed every 30 minutes on the console and in `Bot.log` at
  info, which is spam nobody reads), and sleeps `REDDIT_REFRESH_SECONDS` (1800) through
  `_interruptible_sleep(..., respect_force_analysis=False)`
  so a forced analysis cannot turn it into a retry loop. `_reddit_sample()` returns the cached
  sample, falls back to a direct fetch when no refresh task exists (tests, direct calls), and
  otherwise waits at most `REDDIT_FIRST_SAMPLE_TIMEOUT` (180 s) for the first one — bounded where
  the old call was unbounded. `_build_analysis_context` reads that sample instead of fetching.
- **`src/app.py`**: every cycle now logs `Cycle timings: prepare …, knowledge …, context …,
  analysis …, settle …, total …` and writes the same `stages_seconds` into the cycle journal, so the
  next "why is it slow" is one grep instead of an investigation.
- **`src/analyzer/analysis_engine.py`**: `Model call completed in X.Xs` separates the model call
  from the data collection that shares the `analysis` stage.
- **`config/config.ini`**: `[social_sentiment]` documents the background refresh instead of
  "requests are spaced ~20s apart", which described the old blocking behaviour.

### Verified
- Full Windows-venv suite: **2775 passed, 1 skipped, 2776 collected in 128 s**; `ruff check src
  scripts start.py` clean; `pyright src start.py` **0 errors**; `scripts/check_claims.py --pytest`
  reports the claims consistent.
- Six new tests in `tests/domain_rag_indicators/test_reddit_context.py`: the cached sample is used
  without touching Reddit, the first cycle waits for the background sample, a first-sample timeout
  sends the empty section and still completes the cycle, a failed refresh keeps the last good
  sample, no task is started without social sentiment, and the journal carries `stages_seconds`.
  The existing fetch-wiring and disabled/failed-fetch tests pass unchanged.
- Live measurement before the change: `fetch_posts` 153.78 s / 40 posts with no error, and four
  immediate raw requests answered HTTP 429. After the change the same work moves off the cycle;
  the expected cycle is ~130–140 s. A restart that lands right on a candle boundary can still make
  its first cycle wait up to the 180 s sample cap, because `start_trading` waits for the next candle
  while the refresh runs. No bot restart, commit or push was performed; the running process keeps
  the old code until the operator restarts it.

## 2026-11-13 — A dead Discord gateway no longer freezes startup

discord.py turns a failed gateway handshake (a 503 from `wss://gateway.discord.gg`, seen at 19:35)
into `AttributeError: 'NoneType' object has no attribute 'sequence'`, `DiscordNotifier.start()` logs
it and returns without setting the ready event, and `_provision_notifiers` then awaited that event
with no timeout. Startup blocked before the first cycle, so the bot never traded and the console
fallback written for exactly this case could not run. The wait is now bounded and a notifier that
never becomes ready reaches that fallback.

### Changed
- **`src/composition/provisioners.py`**: `_DISCORD_READY_TIMEOUT` (90 s, above discord.py's own 60 s
  connect timeout) and `asyncio.wait_for` around `wait_until_ready()`, so a failed Discord start is
  one warning plus `ConsoleNotifier` instead of a startup that hangs forever.

## 2026-11-13 — A second model may shrink the size and nothing else

The bot asks one model for a decision and three Python validators check it. The gate adds an
independent answer to a different question — is this entry's location good — and spends it only on
`position_size`. It cannot veto an entry: the live sample is 16 closed trades, so a filter that
removes entries would be a guess with a cost, while a multiplier can only reduce the damage of a
trade the bot was going to take anyway.

`mode = off` ships as the default and builds no gate at all. `shadow` makes the call every cycle and
writes `data/trading/decisions_gate.jsonl` while the decision stays byte-identical, so the mode that
moves size has to be justified from that file first.

### Added
- **`src/analyzer/decision_gate.py`**: `DecisionsClient` posts one record of three questions — a
  `noul` expectancy bit, a `choice` for the next three candles and a `score` for the entry location —
  to OpenRouter's alpha Decisions path (`/api/alpha/decisions`, `openai/gpt-6-luna-decisions`) and
  reads the answers plus the API's own `usage.cost`, so the bot never prices a token itself.
  `DecisionGate` builds a compact state (~1k tokens against the 19.5k-token analysis prompt): the
  decision without its `confidence` (the model must not anchor on its own number), the entry
  indicators, the last 20 candles and the `direction / regime / ADX bucket` triple the brain's rules
  now declare. Score 0–1 halves the size, 2 takes three quarters, 3 leaves it alone, a disagreeing
  direction multiplies by 0.75 again, the floor is 0.5 and the ceiling is 1.0 — the gate never grows
  a position. A refusal, timeout or malformed body is one warning and an untouched decision.
- **`[decisions_gate]`** in `config/config.ini.example` and the live `config.ini`: `mode`, `url`,
  `model`, `min_confidence_pct`, `timeout_seconds`; `src/config/settings_editor.py` exposes the mode
  as a picker, so `off → shadow → size` needs no file edit.
- **`scripts/decision_gate_report.py`**: `replay` scores the closed trades the bot already took
  (entry decision through `PersistenceManager.get_entry_decision_for_position`, candles through the
  ledger's own klines helper) and `journal` summarises `decisions_gate.jsonl` against the realised
  P&L, per confidence and ADX band.

### Changed
- **`src/analyzer/analysis_result_processor.py`**: an optional `decision_gate` argument; the call sits
  between `_log_analysis_result` and `_format_analysis_response`, so the gate sees the ADX and
  pattern quality the Python validators already overwrote, and its line lands above the JSON block in
  `raw_response` — which is what the next cycle reads as `## PREVIOUS ANALYSIS CONTEXT`.
- **`src/composition/provisioners.py`**: one construction site; `mode = off` returns `None`, and a
  non-off mode without `OPENROUTER_API_KEY` raises at startup instead of failing every cycle.

### Measured

`scripts/decision_gate_report.py replay` on the 14 closed trades that carry a P&L and an entry
decision, with the state rebuilt from the entry snapshot and the candles the bot actually saw:
`spearman(quality, next 1 / 3 / 6 candles)` = **0.59 / 0.60 / 0.61**, the same inside each side
(BUY 0.54, SELL 0.64), while `spearman(quality, realised P&L)` = **0.17** and the `noul` bit
correlates with nothing (0.05 / 0.03 / −0.16). The gate is a one-to-six-candle forecaster, not a
judge of a trade the bot holds for days — so `size` stays off until the shadow journal has enough
live entries, and the multiplier question has to be asked about the horizon the gate can actually
answer. The replay is deterministic for a fixed state but sensitive to it: keeping the in-progress
candle in the state moved one trade's quality from 1.84 to 1.14, which is why the replay drops it
the way the data fetcher does.

## 2026-11-13 — An AI-mistake rule must be a claim the entry snapshot contradicted

The live store carried `rule_ai_mistake_sideways_failure_medium` as its second-strongest rule (85.9 next to the
75%-win-rate best practice at 87.0). Its "7 mistake(s)" were exactly the 7 losses of the 13 MEDIUM-confidence
trades, and the card quoted that whole cohort — 6W/7L, +0.55% average P&L, PF 1.92 — under a label that told the
model to lower confidence. The same data could not become an anti-pattern rule: a 53.8% loss rate sits below the
60% gate and its 46.2% win rate is above the 40% baseline, so the AI-mistake path was the only way a profitable
cluster could be called a repeated error.

### Changed
- **`src/trading/brain_patterns.py`**: `classify_ai_mistake()` no longer reads `pnl_pct` or `outcome`. A mistake is
  a `regime_contradiction` — a continuation thesis in the stored reasoning (breakout/breakdown/trend/momentum) or
  HIGH confidence against a non-trending snapshot (regime NEUTRAL/SIDEWAYS/RANGING, `0 < adx_at_entry < 20`, or
  `choppiness_at_entry > 62`). The four outcome-shaped types and `is_sideways_failure()` are gone, so a winning
  trade that claimed continuation in chop is now the error and a losing trade inside a trend is not.
  `derive_ai_assumption()` also ranks a reversal/mean-reversion thesis above the bare word `trend`: the one trade
  the new rule still flagged on live data was a mean-reversion long whose text said "no clean trend", and reading
  that as a continuation claim would have been the same lie in a smaller font.
- **`src/trading/brain_reflection.py`**: `trigger_ai_mistake_reflection()` needs five flagged trades in the group
  (was two) and stores the rule only when their loss rate exceeds the unflagged complement's and
  `is_significantly_worse()` (same one-sided exact binomial, α = 0.05, ≥10 closed baseline) rejects luck. The text
  and metadata carry the flagged group's wins/losses/win rate and that baseline instead of the confidence cohort
  that merely contained the cluster; best-practice rules now also record their `direction`, `regime` and
  `adx_bucket`.
- **`src/trading/brain_context.py`**: the rule line shows the declared `direction / regime / ADX bucket` instead of
  an embedding `% match`, and Apply-Insights asks the model to apply a rule only when that declared condition fits
  the current snapshot. The old "matches (>50% similarity)" test was always true: the project's own
  `bge-base-en-v1.5` scores a rule against four different regime queries at 0.66–0.73, and the two live rules at
  0.82 against each other.
- **`src/trading/vector_memory_rules.py`**: `update_rule_validation_feedback()` matches a closed trade to rules by
  their declared `direction`/`regime`/`adx_bucket`/`entry_confidence` instead of an embedding at
  `min_similarity=0.5` — the entry context matched the live rules at 0.75 and 0.86, so every closed trade used to
  validate or contradict every rule. `src/trading/brain_experience.py` passes those conditions.
- **`AGENTS.md` / `src/trading/AGENTS.md`**: the rule-creation, rendering and validation contracts updated to the
  gates above.
- **Live store**: `rule_ai_mistake_sideways_failure_medium` retired through `scripts/brain_rules.py deactivate`
  after a full copy of the brain directory in `data/trading/backups/rule_retirement_20261113_065459/`.

### Verification
- `pytest tests -q`: 2,741 passed, 1 skipped, 2,742 collected (~125 s).
- `pytest tests/domain_brain -q`: 582 passed, including the new gates (five-sample floor, flagged group not worse,
  flagged-group metrics, a winning continuation claim in chop, a mean-reversion thesis that mentions the trend).
- Replaying the 15 closed trades from the live brain through the new classifier flags 0; the old one flagged 7.
- `python scripts/check_claims.py --pytest`: consistent at 2,742 tests; `ruff check src scripts start.py` and
  `pyright src start.py`: clean.
- `scripts/brain_rules.py check` on the live store prints an empty AVOID / AI-MISTAKE block, and
  `GET /api/brain/rules` returns only `rule_best_short_bearish_high_adx_sl_hard_15m_tp_hard_15m`.

## 2026-11-12 — A superseded number that rounds to the current one is not stale

### Fixed
- **`check_claims.py` no longer fails when the reformatted value collides with a superseded one.** The
  ledger moved from $7.1156 to $7.1222 of inference cost, and both print as `$7.12`: the gate read its own
  current number as a stale one and blocked the publication. The token builder is now shared
  (`entry_tokens`), and `check_texts` skips a superseded token that the current ledger also produces.

### Verification
- `pytest tests/domain_infra/test_claims_check.py -q`: 6 passed, including the new
  `test_a_superseded_value_that_rounds_to_the_current_one_is_not_stale`.
- `python scripts/check_claims.py` on the real tree: consistent, exit 0, with claims at 7.1222 and the
  superseded 7.1156 still in the file.
## 2026-11-12 — The published numbers refresh themselves once a day

### Changed
- **`scripts/publish_public_numbers.py` publishes without a human.** One run refuses to start on a
  dirty worktree, runs the full suite (a red suite publishes nothing), snapshots the ledger, re-renders
  the chart, regenerates `claims.json`, rewrites the block between `<!-- numbers:start -->` and
  `<!-- numbers:end -->` in the README together with the badge and the test count, then writes the
  site's `src/data/ledger.json`, `public/claims.json` and `public/project.json` — and only after
  `check_claims --pytest`, `check_claims --site-root`, `check_agents_md` and ruff pass does it commit,
  push, build and deploy. `--dry-run` prints the plan, `--force` redoes today's publication, and a
  second run on the same day exits 0 without touching anything: one day is one publication, marked by
  the existence of `results/<date>/`.
- **The volatile numbers left the hand-written pages.** The site's `index.astro`, `story.astro` and
  `about.astro` import `src/data/claims.json` instead of hardcoding the test count; `public/llms.txt`
  points at `/project.json` for the test and commit counts; `docs/FEATURES.md` points at `claims.json`.
  What remains hand-written is what a script cannot judge: prose and the caveats.
- **`scripts/publish_public_numbers.ps1`** loads the Cloudflare credentials from the DPAPI store, runs
  the publisher and appends to `logs/publish_public_numbers.log`. Task Scheduler job
  `LLM_trader publish numbers` runs it daily at 04:30, hidden through the shared `wsl_hidden.vbs`
  runner, with `IgnoreNew` so a slow run is never doubled.

### Verification
- `pytest tests/domain_infra/test_publish_public_numbers.py -q`: **7 passed** — the block carries every
  number the claims gate looks for, a loss renders as a signed amount, the README rewrite touches only
  the marked block, a README without markers is refused, a second run the same day is skipped, a dirty
  tree aborts before writing anything, and `--dry-run` writes nothing.
- `python scripts/publish_public_numbers.py --dry-run` → the plan only, tree untouched.
- A forced publication through the wrapper, end to end: `published 2026-11-12: 2736 passed, 16 closed
  trades, 147 cycles, return +0.05%` — the bot committed `81291b4` and pushed, `qrak.org` was
  rebuilt and redeployed, the site committed `892b695` and pushed.
- The scheduled task itself: first invocation failed on git's `detected dubious ownership` (the task
  runs non-elevated while both repositories are owned by `BUILTIN\\Administratorzy`), fixed with
  `git config --global --add safe.directory` for the two paths; the next invocation exited 0 with
  "nothing to do: results/2026-11-12/ already holds today's publication", which is the one-per-day guard.

## 2026-11-12 — CI was silently collecting 31 fewer tests than the repository claims

### Fixed
- **`aiosqlite` is declared in `requirements.txt`.** `src/platforms/coingecko.py` imports
  `aiohttp_client_cache.backends.sqlite`, which imports `aiosqlite` lazily; no file declared it, so a
  venv that had it (this one) worked and the CI runner did not. Two modules —
  `tests/domain_infra/test_platform_utils.py` and `tests/domain_infra/test_coingecko_cache_freshness.py` —
  therefore failed at collection on Linux with `ModuleNotFoundError: No module named 'aiosqlite'`, and
  their 31 tests never ran there. Found by running the claims gate on CI: it reported `2,698 tests
  collected` against the claimed 2,729, and a probe run on a branch (since deleted) reproduced the
  exact traceback and the two module names.
- Test count and suite duration re-measured after the fix: **2,729 collected, 2,728 passed, 1 skipped**
  in 119.42 s. The one skip is `test_build_server_exposes_only_the_two_read_tools`, which needs the
  optional `mcp` package.

### Verification
- `python -m pytest tests -q`: 2,728 passed, 1 skipped, 0 failed in 119.42 s (local Windows venv).
- `python -m pytest tests --collect-only -q`: 2,729 collected — the same number the CI runner collects
  once `aiosqlite` is installed, which is what the claims gate compares.
- `python scripts/check_requirements.py` passes with the new bound.

## 2026-11-12 — A newcomer can see the CI state and has somewhere to report

### Changed
- **`README.md` carries the dynamic `tests` and `CodeQL` workflow badges** next to the static ones, so
  the CI state cannot drift from the text; the static test count stays, because `check_claims.py` gates it.
- **`SECURITY.md` states what a vulnerability is in this repository** — the settings desk ceasing to be
  loopback-only, a path to the exchange that skips the guards, a secret in the open, the read-only MCP
  server writing, remote code execution through config, prompts, RSS or chart rendering — and what is
  not: a losing paper trade, a provider outage, the published ledger, the missing fees. Private
  reporting goes through a GitHub security advisory.
- **`.github/ISSUE_TEMPLATE/`** asks a bug report for the failing gate, the cycle evidence and the
  provider/timeframe, and a feature request for the dependency, config key, public number or risk-pipeline
  impact it would carry.
- **`.github/dependabot.yml` updates GitHub Actions weekly and deliberately not pip**, because
  `requirements.txt` is bounded, `constraints.txt` caps the transitive set and
  `scripts/check_requirements.py` verifies both; a bot rewriting them would fight the constraints.
- The repository topics gain `mcp`, `paper-trading`, `ai-agent`, `ccxt`, `technical-indicators` and
  `multimodal` (12 → 18 of 20), so the agent, MCP and paper-trading surfaces are findable.

### Verification
- `check_claims.py` clean with the two new badges: its `tests-<n>-brightgreen` pattern only reads the
  static badge, and the workflow badge URL does not match it.
- `check_agents_md.py`: 278 path references, all exist; ruff clean.
- Badge URLs resolve against the live repository: `actions/workflows/tests.yml` and
  `actions/workflows/codeql.yml` both exist and have run on `master`.
- `gh api repos/qrak/LLM_trader --jq .topics` shows the 18 topics after the update.

## 2026-11-12 — The read-only MCP server is documented and tested

### Changed
- **`docs/MCP.md` documents a server that no public text mentioned.** `src/mcp_tools/history_server.py`
  has existed with `requirements-mcp.txt` and reader tests, but the README, `docs/` and the site never
  named it. The page now records both tools (`recent`, `stats`), the optional install, the stdio
  command, a client configuration block, and why the server cannot damage the book: `mode=ro`,
  `PRAGMA query_only`, `limit` capped at 200, and no write, order or config tool by design.
- `README.md` gains the feature row and the documentation-map row, `docs/README.md` indexes the page,
  and `docs/FEATURES.md` gains the section naming the implementing files.
- `tests/domain_infra/test_mcp_history.py` now covers `build_server` behind
  `pytest.importorskip("mcp")` and asserts that exactly `recent` and `stats` are exposed.

### Verification
- `pytest tests/domain_infra/test_mcp_history.py -q`: 8 passed, 1 skipped — the skip is the `mcp`
  package, which is optional and absent from `.venv`, exactly as the page says.
- In a throwaway venv with `mcp 1.30.0`, `build_server` listed `['recent', 'stats']`, and against
  `results/2026-11-12/trade_history.db` `recent(limit=2)` returned the two newest rows (2026-11-10
  SELL BTC/USDC at 73 213.07, 2026-11-09 CLOSE) while `stats(days=3650)` returned coverage plus the
  per-action counts.
- ruff, `check_agents_md` (278 references) and `check_claims` clean.

## 2026-11-12 — The public numbers come from a frozen ledger, and the site is gated against it

### Changed
- **`results/2026-11-12/` is the first frozen ledger snapshot.** `scripts/snapshot_results.py` copies
  `statistics.json`, `api_costs.json`, `trade_history.db` (through SQLite's backup API, not a file copy)
  and `metrics/cycles.jsonl`, and records a `sha256` manifest that `--verify` re-checks. Why: `data/` is
  gitignored, so `scripts/results_report.py` in a clone printed zeros while the README claimed the table
  was reproducible with it.
- **`claims.json` is generated from the snapshot, not the live ledger** (`claims_report.py --data-dir
  results/<date>`), so the numbers a public text quotes stop moving under that text. The buy-and-hold for
  the same window is **-9.27%**, not the -4.64% the README carried: that figure belonged to the
  20 Sep – 31 Oct window the site still showed, while the README already quoted 20 Sep – 10 Nov.
- **`scripts/results_report.py --json PATH`** emits the same numbers as JSON plus the per-trade rows.
- **`check_claims.py --site-root` now fails when the site carries a stale `claims.json` or a ledger that
  disagrees with it.** Before this, the site's results page was outside every gate; its four remaining
  problems (test and commit counts) were found by running the gate that already existed.
- `scripts/claims_report.py` claimed "79 Numba-compiled indicator functions"; the measured count is 65 in
  `src/indicators` and 107 across `src/`, which is what the README already said.

### Verification
- `pytest tests -q`: **2,728 passed** in 134.97 s, 0 failed, 0 skipped. Five are new in
  `tests/domain_infra/test_snapshot_results.py`: manifest round-trip, database rows copied through the
  backup API, a tampered and a stray file reported by `--verify`, a missing ledger file refused, exit codes.
- `python scripts/snapshot_results.py --verify results/2026-11-12` → ok;
  `python scripts/results_report.py --data-dir results/2026-11-12` reprints the README block.
- `check_claims.py --pytest` and `check_claims.py --site-root` → 0 problems; ruff clean, pyright 0 errors,
  and `check_agents_md`, `check_config_contract`, `check_hardcoded_timeframes`, `check_loose_helpers`,
  `check_dead_symbols`, `check_comment_policy` all pass.
- Site (`semantic-signal-site`): `npm run build` green through its own `check-claims.mjs` gate; the page
  reads `src/data/ledger.json` and renders 16 (14 counted, 2 excluded), 37.5%, +0.05% against -9.27%,
  146 cycles, $7.12 — with no figure left over from the 31 October window.

## 2026-11-12 — The cost line names its billing window, Go reports the amount, the price table is current

### Changed
- **The peak/off-peak label carries the UTC clock of the instant it evaluated.** `PeakRates.tier_label`
  now returns `off-peak x1 @ 00:04 UTC`, so a log line proves its own window instead of asking the
  reader to convert a local timestamp. Why it mattered: the windows were already correct (112 historical
  `Request cost` lines, 0 mismatches against 01:00-04:00 and 06:00-10:00 UTC), but the bot's 05:04 local
  cycle is 04:04 UTC — four minutes past the first window — and nothing in the line said so.
- **OpenCode Go is reported, not estimated.** `convert_pydantic_response` reads `usage.opencode.allowance`
  (falling back to a positive `usage.cost`) into `UsageModel.reported_cost`, and `ModelManager._track_cost`
  prefers it over the rate-table estimate, tagging the line `, from provider`. A zero `usage.cost` on Go
  means "nothing from balance", so it never wins. Peak pricing and cache reads are the provider's
  arithmetic now, not ours.
- **`config/model_pricing.json` is sourced and current.** The DeepSeek Flash rows were 47% too high
  (0.44/1.32 → 0.30/1.20 peak, off-peak 0.15/0.60, cache-hit 0.006/0.003, per api-docs.deepseek.com).
  The table now holds 69 priced models, each naming its source and date: all 28 models the Go
  subscription covers with their monthly allowance, the models at the top of OpenRouter's October 2026
  token ranking plus the frontier names, and Gemini 4 Argon, Carbon and Omni Flash.
- Docs drift handled in the same change: `docs/CONFIGURATION.md` (window clock, provider-reported cost,
  table coverage), `README.md` and `docs/FEATURES.md` (current ledger and test counts), `claims.json`
  regenerated.

### Verification
- `pytest tests -q`: **2,723 passed** in 156.36 s, 0 failed, 0 skipped. 32 of them are new: window
  boundaries (00:59/01:00/03:59/04:00/05:59/06:00/09:59/10:00, Friday after, Saturday, Sunday night), the
  same 05:04 wall clock landing in peak under CEST and off-peak under CET, the label clock within a
  minute of now, the Go allowance beating the estimate and a zero balance cost not beating it, table
  hygiene (every row has positive rates and a source) and 12 popular models resolving to a price.
- `ruff check src/`, `py_compile` of the four touched modules, `import start`, `check_loose_helpers`,
  `check_config_contract`, `check_hardcoded_timeframes`, `check_dead_symbols`, `check_comment_policy`:
  clean. `python scripts/check_claims.py --pytest`: consistent at 2,723 tests.
- The windows themselves are unchanged — they already matched OpenCode's and DeepSeek's published
  schedules; only the reporting around them moved.

## 2026-11-11 — Live Console find bar

### Changed
- **The Live Console search is a find bar, not a filter.** `console_panel.js` highlights every match
  in place, shows the current position and total (`3 / 57`), steps through matches with
  Enter/Shift+Enter, F3/Shift+F3 or the two toolbar arrows, and focuses the field with Ctrl+F;
  Escape clears the query. The all/error/warning/info level filter still hides lines, the find bar
  never does.
- **The live tail no longer fights the search.** While a query is active, new WebSocket lines are
  still appended and highlighted where they match, but auto-scroll waits; clearing the query
  returns to the tail. The counter counts only matches on visible lines.
- Highlighting uses the CSS Custom Highlight API (`::highlight`), so matching never rewrites line
  markup or escapes log text; the registry is repainted on filter and page changes. Browsers
  without the API still get the counter and match navigation, without colour.
- Cache-busters: `style.css?v=7.23`, `panels.css?v=7.23`, `main.js?v=9.3`,
  `console_panel.js?v=1.4`. Docs: `docs/FEATURES.md`.

### Verification
- Playwright on the running dashboard: 39/39 checks — every line stayed visible, 5 of 5 matches
  highlighted, `1 / 5` → Enter → `2 / 5` → Shift+Enter → `1 / 5`, wrap to `5 / 5`, F3/Shift+F3,
  Ctrl+F focus, Escape clear, `No results` for a miss, the regex metacharacter `zz[` harmless,
  ERROR filter counted 3 of 3 visible matches, a synthetic WebSocket line was highlighted without
  moving `scrollTop`, Escape returned to the tail, and a mocked archive page counted 3 matches.
- 10,000 synthetic lines with 10,000 matches: the `input` handler took **19.7 ms** (budget 100 ms).
- `python scripts/check_comment_policy.py`: clean; `python scripts/check_claims.py --pytest`:
  2,691 tests consistent; `pytest tests -q -m "not integration"`: **2,691 passed, 23 warnings,
  134.33 s**.

## 2026-11-10 — Article bodies over plain HTTP; Crawl4AI retired

### Changed
- **News enrichment no longer starts a browser.** `ArticleBodyEnricher`
  (`src/rag/news_ingestion/article_body_enricher.py`, renamed from `crawl4ai_enricher.py`) refetches
  each short feed body over `aiohttp` with a browser User-Agent and keeps trafilatura's text, with
  `extract_html_body_text` as the second opinion; a page that yields no body above
  `news_min_body_chars`, or an unusable body, leaves the item on its feed text.
  `RSSCrawl4AINewsProvider` is now `RSSNewsProvider`.
- **The browser User-Agent is load-bearing.** CoinDesk answers `429` to aiohttp's default UA and
  `200` to the browser one; a test pins the header so the regression cannot come back.
- `news_crawl4ai_enabled` and `news_crawl_timeout` are gone from `config/config.ini`,
  `config/config.ini.example` and the loader; `RAG_NEWS_CRAWL_CONCURRENCY` became
  `RAG_NEWS_PAGE_CONCURRENCY` and the `[rag]` comments no longer mention Crawl4AI.
- `requirements.txt`: `crawl4ai~=0.9.5` removed, `playwright==1.61.0` added (Kaleido's browser pipe
  fails on this box, so the chart-export fallback is production code). `constraints.txt`: the
  `patchright` cap is removed. Uninstalled from the venv: `crawl4ai`, `patchright`,
  `playwright-stealth`, `unclecode-litellm`, `alphashape`, `rank-bm25`, `snowballstemmer`,
  `fake-useragent`; the Playwright Chromium stays for the chart fallback.
- Docs: `README.md` (feature row, architecture diagram), `AGENTS.md` (agents table, provisioner row,
  news sources), `src/AGENTS.md`, `src/rag/AGENTS.md` (class table, pipeline, enrichment chain,
  degradation), `docs/FEATURES.md`, `THIRD_PARTY_NOTICES.md` (Crawl4AI attribution and Apache entry
  replaced by ChromaDB).

### Verification
- A/B probe against the four live feeds (2026-11-10): the 12 newest article bodies all reach 400+
  characters over plain HTTP in 2.0 s at concurrency 6, while the same four pages through Crawl4AI
  cost 1.1-6.9 s each and return the same text (2,305 vs 2,093, 1,898 vs 2,054, 3,670 vs 3,670 and
  3,764 vs 3,795 characters). One `arun_many` batch of 15 URLs never returned in 11 minutes.
- End to end through the new code on the live feeds: 9/9 candidates enriched in 1.3 s, smallest body
  1,898 characters. CoinDesk page: `429` with aiohttp's default UA, `200` with the browser UA; all
  four feeds answer `200` to both.
- Footprint: 311 MB of site-packages existed only for crawl4ai (28 packages, patchright 111 MB,
  playwright 108 MB, unclecode-litellm 52 MB), the import cost 1.29 s and +87 MB RSS, and each
  enrichment batch spawned a Chromium peaking at 396 MB.
- `pytest tests -q -m "not integration"`: **2,691 passed, 23 warnings, 137.5 s** (14 crawl4ai-specific
  cases removed, 13 written for the plain-HTTP path). `ruff check src scripts start.py` clean,
  `pyright src start.py` 0 errors, `check_comment_policy`, `check_loose_helpers`,
  `check_config_contract`, `check_agents_md` and `check_dead_symbols --symbol Crawl4AIEnricher`
  (0 references) clean, `pip check` and `check_requirements.py` clean, `check_claims.py --pytest`
  consistent at 2,691 tests. No model calls, order operations or bot restart.

## 2026-11-10 — Local settings desk and in-panel documentation

### Added
- **`/admin/` is the local settings desk.** The placeholder is replaced by a three-view page:
  **Settings** lists every key present in `config/config.ini` (16 sections, 112 keys in the live
  file), with the file's comment blocks as help, loader-validated enums as selects and a
  search/filter. **API keys** lists `keys.env.example` (plus extra keys already in `keys.env`) and
  shows only `set`/`source`; it can create `keys.env` from the template. **Documentation** renders
  `docs/README.md`, `SETUP`, `CONFIGURATION`, `FEATURES`, `STATE_MACHINES`, `SETTINGS_PANEL`,
  `README.md` and `CONTRIBUTING.md`.
- **`SettingsEditor`** (`src/config/settings_editor.py`) reads/writes both files without losing
  comments, order, indentation, inline comments or line endings; it validates the candidate with
  `Config(config_path=temp)` and replaces the live file only after the loader accepts it. Every load
  returns an etag; a stale save returns `409`; keys writes never return or log a value.
- **`/api/settings/*`** (`src/dashboard/routers/settings.py`): status and docs are public, config/keys
  reads and every write require a loopback client, a local `Host` and, when present, a local
  `Origin`; `CF-Connecting-IP` and public `X-Forwarded-For` are rejected. No login, session or cookie.
- **`DocsLibrary`** (`src/dashboard/docs.py`) renders the allowlisted Markdown with `markdown-it-py`
  (raw HTML escaped), rewrites relative `.md` links into the panel, serves `docs/assets` through
  `/docs-assets`, and sends files outside the allowlist to GitHub. The page sanitizes the HTML with
  DOMPurify before inserting it.
- **`docs/README.md`** (documentation index) and **`docs/SETTINGS_PANEL.md`** (access, tabs,
  validation, restart, troubleshooting). The dashboard sidebar links to `/admin/`.

### Changed
- **`Config` accepts an optional `config_path`.** `Config(config_path=...)` reads a candidate INI
  without touching the module-level `config`; the default path is unchanged and existing callers
  keep working.
- **`docs/CONFIGURATION.md` no longer says "There is no web editor".** It describes the desk,
  validation and the restart contract. `docs/SETUP.md` now copies the tracked `config.ini` template
  in the quick start (a fresh clone previously had no config), and `README.md` links the desk and the
  documentation index.
- **`requirements.txt`** lists `markdown-it-py~=4.2.0` directly. It was already installed through
  `rich`; the settings desk now imports it.
- **AGENTS:** root §5.1/§9/§11/§12.1, `src/AGENTS.md` and `src/dashboard/AGENTS.md` record the new
  access model, the secret rule and the seven-router list.

### Verification
- `pytest tests -q`: **2,690 passed, 23 warnings, 127 s** (was 2,661 passed, 128 s; +29 tests in
  `test_settings_editor.py` and `test_dashboard_settings.py`).
- `claims consistent: 2,690 tests, ledger snapshot 2026-11-01`; the README badge/prose and
  `docs/SETUP.md` agree. The ledger snapshot was deliberately kept at 2026-11-01; this change does
  not re-publish trading results from the live data.
- `ruff check src scripts start.py`, `pyright src start.py` (0 errors), `check_dead_symbols`,
  `check_comment_policy`, `check_agents_md`, `check_config_contract`, `check_hardcoded_timeframes`,
  `check_loose_helpers`, `check_requirements`: green.
- API smoke: loopback reads `config`/`keys`, a save keeps comments and writes only the changed value,
  a bad provider returns `400` with the section/key, a stale etag returns `409`, a remote client and
  a tunnel header get `403`, docs render tables/links/assets and escape raw HTML.
- Behaviour after restart: `GET /admin/` → 200, `/api/settings/status` → 200 (`local:false` from the
  tunnel), `/api/settings/config` → 403 from the tunnel, `/docs-assets/*` → static, old
  `/api/admin/*` and `/api/monitor/github_activity` → 404.

## 2026-11-10 — Fifth pass: the admin console and the activity feed retire

### Changed
- **The GitHub Activity tab is gone (1,202).** The tab re-rendered github.com's own public event
  stream: two outbound API calls per refresh, a five-minute cache, a one-day stale fallback and a
  261-line panel module. `dashboard/github_activity.py`, the panel module, the
  `/api/monitor/github_activity` route and its heavy-path entry, the `GITHUB_USER`/`GITHUB_TOKEN`
  accessors, the `keys.env` block and every `.gh-*` rule went with it; the sidebar is back to ten
  tabs.
- **The admin console is gone (2,866).** It was the only surface that could decide an approval, the
  only writer of `config.ini` and the only reader of `ADMIN_USERNAME`/`ADMIN_PASSWORD_HASH`/
  `ADMIN_SIGNING_KEY`. Deleted: `AdminRouter` (login, sessions, config CRUD, reload, approvals,
  status, `logs/recent`, `logs/stream`, console WS), `auth.py` and its middleware,
  `WritableConfig` and its schema, the dashboard's writable-config wiring, and the four retired
  flags. `resolve_client_ip` moved to `src/dashboard/client_ip.py` because the HTTP rate limiter and
  the WS connection limiter still use it.
- **The human-approval gate retires with its only decision surface.** `PendingApprovalStore`,
  `app.py`'s wait loop, `require_approval_above_usdc`/`approval_ttl_minutes`/
  `approval_poll_seconds`, the `approval_not_granted` cycle detail and the dashboard's Approval chip
  are deleted, and `GET /api/status/state` no longer carries an `approval` block. The gate was off
  in the live config and in the template (threshold 0); keeping it without a way to approve would
  have blocked every decision above the threshold until it timed out.
- **`/admin` is now one static placeholder page** for the local settings desk (`keys.env` +
  `config.ini` from the browser) and the project documentation that are planned there: no router, no
  session, no API. `src/dashboard/AGENTS.md` records the contract for when it lands.
- **`POST /api/brain/refresh` no longer requires a session.** The anonymous 401 was the console's
  gate; the endpoint stays `no-store` and keeps the 30 writes/minute per-client limit.
- **The CSP drops `cdn.tailwindcss.com`** from `script-src` and `style-src`: no static file has ever
  loaded that host, so the entry only widened the allowlist.

### Verification
- `python scripts/check_net_diff.py --base d3ebda4 --head HEAD`: `+82 / -4150 = -4068` production
  lines across 28 files; tests `+177 / -1910 = -1733`.
- `pytest tests -q`: 2,661 passed, 128 s (was 2,820 passed, 34 deselected, 131 s); the Playwright
  admin suite was the only `integration` suite and retires with the console.
- `claims consistent: 2,661 tests`; README badge and prose, `docs/SETUP.md` and `claims.json` agree.
- `ruff`, `pyright` (0 errors), `check_dead_symbols`, `check_comment_policy`, `check_agents_md`,
  `check_config_contract`, `check_hardcoded_timeframes`, `check_loose_helpers`: green.
- Behaviour: `GET /admin/` → 200 placeholder, `/api/admin/*` → 404, `/api/monitor/github_activity`
  → 404, `POST /api/brain/refresh` → 200 without a session, capped by the write limit.

## 2026-11-09 — Fourth pass: one cache constant, one entry-signal harness

### Changed
- **`_api_cache_policies` has one no-store constant (25).** The same pair was spelled out in four branches
  of the policy table and again on the non-GET path; three of the four suffixes are matched in a single
  `endswith` tuple, the `/vectors?query=*` rule keeps its own guard.
- **The runtime tests build their entry signal once (86, tests only).** Five tests repeated the same SELL
  analysis, `TradeDecision` and executor harness, differing only in `order_id` and in whether the forward
  was delivered. `entry_signal_harness()` returns all four pieces; the duplicate-suppression assertions
  are untouched.

### Verification
- `pytest tests -m "not integration"`: 2,820 passed, 34 deselected, 131 s.
- `ruff`, `pyright`, `check_dead_symbols`, `check_comment_policy`, `check_agents_md`, `check_loose_helpers`: green.

## 2026-11-09 — Third pass: Donchian stops carrying its own copy of the rolling-range kernel

### Changed
- **Donchian channels ask `support_resistance_numba` (71).** `donchian_channels_numba` held the same 76-line
  sliding max/min loop, NaN counters included, and differed only in the midline. It now calls that kernel
  and adds the midline; output is bit-identical on six input shapes (NaN windows, `length` 1, `length`
  longer than the series). `src/indicators/AGENTS.md` says the loop has one copy.
- **Three duplicated blocks collapsed into one each (27).** `visuals.js` wired the click/keyboard lightbox
  twice, once per chart source; `vector_memory_context` formatted the exit-execution context in both of its
  metadata renderers; `components.css` styled two lightbox children (`button.active`, `.separator`) that
  `visuals.js` never creates — the `.active` rule was a copy of the `:hover` rule above it.

### Verification
- `python scripts/check_net_diff.py --base db1abfd --head HEAD`: `+150 / -1866 = -1716` production lines
  across 54 files.
- `pytest tests -m "not integration"`: 2,820 passed, 34 deselected, 135 s.
- `ruff`, `pyright`, `check_dead_symbols`, `check_comment_policy`, `check_agents_md`, `check_loose_helpers`: green.

## 2026-11-09 — The shrink continues: 216 more production lines, one dead panel and a latent test bug

A second deletion pass over the candidates the first one measured but did not take. Production is
216 lines lighter net and the suite 40; behaviour is unchanged except where a test could never have
passed.

### Removed
- **The prose synopsis (71).** `build_decision_synopsis` built an English sentence list on every
  `/api/brain/decision-summary` call, but the panel always took the structured `synopsis_data`
  branch: the string, its JS fallback and the CSS rule behind them were unreachable.
- **Three endpoints no caller asked for (43).** `/api/monitor/system_prompt`,
  `/api/brain/lifecycle` and `/api/console/recent` were registered, documented nowhere and fetched by
  nothing — the panel reads the lifecycle object from the position/status payloads and the console
  streams over the WebSocket. Their orphans went too: `AnalysisEngine.last_system_prompt`
  (write-only), the `/brain/lifecycle` cache-policy entry, and `ConsoleBuffer`/`DayPage.get_recent`.
- **Three copies of the provider chat body (82).** OpenCode, OpenRouter and LM Studio each carried
  the same ensure-call-convert-translate-error sequence; only LM Studio's role rewrite and model
  auto-select, OpenRouter's `extra_body` reasoning hop and OpenCode's nothing differed. Those are
  hooks on `BaseAIClient` now. DeepSeek keeps its own override because its decorator retries a reply
  that carried no usable JSON payload.
- **A pre-sort whose result was recomputed downstream (21).** `retrieve_context` partitioned
  candidates by body length before handing them to `add_articles_to_context`, which re-sorts by
  score and re-partitions the same way; 2,000 random index/score/body mixes produced zero different
  selections. The manager's `is_overview_stale` duplicated the cache call, and
  `register_shutdown_callback` branched on two arms that appended the same callback.

### Fixed
- **The playwright admin suite could not pass its responsive-layout tests.** Five identical `_login`
  methods existed and `TestResponsiveLayout` called `self._login` without owning one. One
  module-level helper replaces the copies; `pytest tests/test_admin_playwright.py -m integration`:
  34 passed.

### Verification
- `python scripts/check_net_diff.py --base db1abfd --head HEAD`: `+111 / -1729 = -1618` production
  lines; tests `-350`.
- `pytest tests -m "not integration"`: 2,820 passed, 34 deselected, 134 s.
- `ruff`, `pyright`, `check_dead_symbols`, `check_comment_policy`, `check_agents_md`,
  `check_config_contract`, `check_hardcoded_timeframes`, `check_loose_helpers`, `check_claims`: green.


## 2026-11-09 — The codebase shrinks: 1,712 lines that nothing ran

A deletion pass over what the audits found: the production surface lost 1,137 lines net and the
suite 310, with no behaviour change and every gate green.

### Removed
- **The dead indicator surface (691).** Twelve `@njit` kernels were reachable only through their own
  package export, an unused `TechnicalIndicators` method and one test: the whole sentiment package
  (a second Fear & Greed that the runtime never reads — Alternative.me supplies the real one),
  `price_transform_indicators.py`, and the momentum, divergence, EBSW, VHF, EOM and volume-profile
  kernels with their ten facade methods. `stdev_numba` (called internally by the volatility module)
  and `quantile` stay. `src/indicators/AGENTS.md`, `README.md`, `docs/FEATURES.md` and the landing
  page now say 65 kernels in 10 modules instead of 79 in 12.
- **The Codacy SARIF normaliser and the orphan audit script (292).** The Codacy workflow was removed
  in September; `fix_sarif.py`, three of its tests and `run_code_audit.py` (which shelled out to
  pylint, not a repository gate) survived it.
- **`NewsRepository` (92).** A pass-through whose only work was the 86400 s default that
  `RagFileHandler.load_news_articles`/`save_news_articles` already apply; `NewsManager` now calls the
  handler it holds, and no longer constructs a collaborator inside its own constructor.
- **The dead reconcile knob, the boot-time cache prune and a stale claims list (60).** `start.py`
  read a `[executor_api] reconcile_interval_seconds` key that no template or schema contains; every
  startup deleted `models--BAAI--bge-small-en-v1.5` from the user's shared Hugging Face cache.
- **Dashboard CSS no element matches (265).** The retired donation-address table, the adsense\n  placeholder, the Discord card body, the lightbox measuring overlay, the copy button, the stat\n  breakdown and the context label. A live rule that only wrapped a dead descendant\n  (`.discord-content .warning-banner`) went with it. All seven `?v=` cache strings and `index.html`\n  now agree on 7.20 — they disagreed before, so one sheet was never refetched.\n- **Duplicated plumbing in the trading package (37).** A second executor-verdict reader, a
  single-caller getter, two in-function import aliases, a Protocol with one implementation and six
  package re-exports that pulled chromadb into `import src.trading`.

### Changed
- **The four remaining state files are written through `atomic_write_json` (18).** `statistics.json`,
  `position_monitor.json`, `previous_response.json` and `last_decision.json` still shared one
  `<file>.tmp` name between concurrent writers — the race `7ed132c` fixed for `positions.json`.
- **`fmt_ta` renders through `fmt` (16).** It carried a second copy of the eight-branch precision
  ladder, so the AI prompt and the dashboard could drift on the same number.
- **The dead-letter age check reuses `LifecycleProjection.age_seconds` (7).**

### Verification
- `python scripts/check_net_diff.py --base db1abfd --head HEAD`: `+39 / -1441 = -1402` production
  lines; tests `-310`.
- `pytest tests -m "not integration"`: 2,820 passed, 34 deselected, 136 s (was 2,837 / 159 s).
- `ruff`, `pyright`, `check_dead_symbols`, `check_comment_policy`, `check_agents_md`,
  `check_config_contract`, `check_hardcoded_timeframes`, `check_loose_helpers`, `check_claims`: green.

## 2026-11-09 — The dedup reads the ledger by contract, and a phantom is not a trade

### Fixed
- **A phantom rollback no longer counts as a closed trade.** The compensating `CLOSE` at the entry
  price had no fill evidence, yet `StatisticsCalculator` paired it with its ENTRY and produced a
  0.00% trade that diluted the win rate on the next recalculate. `_record_blocked_entry_close` now
  prefixes the reasoning with `PHANTOM_CLOSE_MARKER`, and `_extract_closed_trades` skips that pair.
- **`_pending_entry_hashes()` calls `position_intents().pending_entry_hashes()` directly.** The
  previous `getattr`/`isinstance`/`except` wrapper returned an empty set on any shape change, which
  would have disabled duplicate suppression silently — the same failure class as the 09:06
  regression. AGENTS §10 requires the strategy contract to be called, not probed.

### Changed
- **Duplicate suppression and `rollback_blocked_entry` share `ENTRY_SIGNALS`** (BUY, SELL, LONG,
  SHORT) from `data_models`, instead of a literal `("BUY", "SELL")` in `app.py`. A LONG/SHORT whose
  forward failed can no longer leave a local phantom, and a repeated LONG/SHORT decision is
  suppressed like a spot one. No LONG/SHORT has ever been emitted (checked every `Bot.log`), so
  this hardens a latent path rather than fixing a live failure.
- **`.state-chip.state-neutral`** now has an explicit muted/dashed style; the flat and idle chips
  no longer fall through to the bare base rule. CSS cache version `7.19`.

## 2026-11-09 — A fresh entry no longer suppresses itself

### Fixed
- **2026-11-09 09:06 regression:** `process_analysis()` registers this cycle's ENTRY before the
  duplicate check ran, and the check read the live ledger. Every BUY/SELL therefore matched the
  intent it had just created, was logged as `duplicate_intent`, and was never forwarded. The 09:06
  SELL opened a local bookkeeping position, reached no executor, and was rolled back as a phantom
  at 13:01. The duplicate check now snapshots `pending_entry_hashes()` **before**
  `process_analysis()`, so a new entry cannot suppress itself while a genuinely older pending
  entry still does.
- **A phantom rollback resolves its ENTRY intent to `refused`** (`evidence=bot_policy`) instead of
  leaving it `pending` forever; otherwise the stale hash would keep blocking the same setup after
  the local position was dropped.
- New regression tests pin both directions: a fresh intent does not suppress itself, and an older
  pending hash still suppresses across a HOLD checkpoint.

## 2026-11-09 — Every lifecycle has a name, and the operator can see it

### Added
- **`src/trading/lifecycle.py`** is the single vocabulary for intent, reconciliation, verification,
  cycle and derived position states; the values are unchanged, so `cycle_state.json`,
  `bot_position_intents.jsonl` and `approvals.json` replay byte-for-byte. `docs/STATE_MACHINES.md`
  maps every stored and derived lifecycle, its evidence hierarchy and its user/API surface.
- **`GET /api/status/state`** returns one machine-readable snapshot (cycle, position phase,
  execution proof, approval count, executor state, brain state); `/api/brain/position` embeds the
  same object as `lifecycle`. The Overview strip and the position card render the API's labels and
  severity, including `entering`, `closing`, `divergence` and `unverified`.
- **The analysis prompt carries one execution-proof line** (`Exchange-verified`,
  `Executor-reported — NOT exchange-verified`, `unverified — do not assume exchange protection`).
- **The admin dashboard lists pending approvals** and approves/rejects them, and reports
  `restart_required` by comparing the config file mtime with the admin start.
- **The position card shows the proof's age** (`checked 12s ago`) from
  `verification.age_seconds` instead of the raw ISO timestamp; the state chips only mutate when
  their text, severity or tooltip changes, so the `aria-live` strip does not re-announce unchanged
  state.

### Changed
- **The cycle checkpoint is honest**: it stores only the last terminal state, `analyzing` is an
  in-memory marker, `skipped` explains why a decision did not run, and a duplicate entry stays
  `intent_sent` instead of being overwritten with `failed`, so duplicate suppression survives the
  next cycle. The position-intent ledger now also stores the ENTRY `decision_hash`, so a later
  HOLD cycle that overwrites the checkpoint cannot re-arm the same BUY/SELL; suppression is
  released only when that intent is confirmed or refused. An illegal intent transition is logged
  at CRITICAL and never rewrites the ledger.
- **Two dead states are gone**: `OrderLifecycle.EXECUTED` (execution proof belongs to the intent
  ledger) and `CycleState.LEARNED` (learning has its own `BrainLifecycle`).
- **Vocabulary drift is now a test failure**: every transition table, label, severity, approval
  status and cycle detail code must cover its declared states and appear in
  `docs/STATE_MACHINES.md`. The duplicate-suppression paragraph now states the real rule: the
  checkpoint holds the last finished cycle, so a later cycle that records a different state ends
  the suppression.

## 2026-11-08 — OpenCode Go joins the provider list

### Added
- **`provider = opencode` drives the OpenCode Go subscription** through its OpenAI-compatible
  endpoint (`https://opencode.ai/inference/go/openai/v1`, `OPENCODE_API_KEY`). The default model
  `deepseek-v4.1-flash` accepts charts (1M context, 384K output) and `opencode_fallback_model` adds a
  second model when the first one fails; the same client serves the pay-as-you-go Console by
  changing `opencode_base_url`, so no second provider class was written.
- **The client identifies itself the way OpenCode asks**: a `llm-trader/1.0` user agent and a
  process-stable `x-opencode-session`, both set as SDK default headers and covered by a wire test.
- **OpenCode Go cost is tracked as allowance consumption** — off-peak token rates in
  `config/model_pricing.json` and a `peak ×2` window in `src/utils/peak_rates.py`, the reverse of the
  DeepSeek convention, because the subscription is billed flat.
- **`provider = all` now walks `googleai → deepseek → opencode → local → openrouter → blockrun`**
  for text and `googleai → deepseek → opencode → openrouter → blockrun` for charts.

### Changed
- **OpenRouter's per-model fallback became the shared gateway path** (`_invoke_gateway` /
  `_request_gateway_model`), so OpenCode reuses it instead of a second copy; the response copy still
  names the provider from its metadata.

## 2026-11-03 — The cadence is the configured candle, and the traffic record survives

### Changed
- **The README no longer says "every four hours".** The cycle runs on the configured `timeframe`
  (`4h` on this deployment), so the opening paragraph now reads "on every closed candle" and the
  deployment detail stays where `crypto_pair` is named. The old phrasing had survived three releases
  because no gate covered prose.

### Added
- **`scripts/traffic_snapshot.sh`** is the POSIX twin of the PowerShell snapshot: same markdown file,
  plus `.hermes/traffic/history.jsonl` deduplicated by date, because the traffic API keeps only 14 days.
- **`check_claims.py` fails the build when the README hardcodes a four-hour cycle**, next to the
  existing stale-number checks, so the sentence cannot return unnoticed.

## 2026-11-01 — The public numbers are generated, and the README leads with the ledger

### Changed
- **The README opens with one promise, three links and a "what this is not" block.** The old opening
  described the pipeline before saying what the reader gets; the new one says the prompt, the
  retrieved memories and the verdict are public, then links the dashboard, the results page and the
  long write-up. The repository description was changed to match (135 characters, no feature list).
- **The results table was re-measured to the 2026-09-20 → 2026-10-31 window.** 14 closed trades
  (12 counted, 2 excluded for fill divergence), +0.16% (+$15.72) against -4.64% for buy-and-hold,
  $6.44 of inference, net +$9.28, 83 cycles. The previous table described the 2026-10-26 snapshot and
  had drifted two trades behind the ledger; `docs/assets/results-vs-btc.png` was regenerated from the
  same window, so the image and the table can no longer disagree.
- **The test count is one figure in every surface again: 2,786 collected, 2,752 outside the
  `integration` marker.** README, its badge, `docs/FEATURES.md`, `docs/SETUP.md`, the launch drafts
  and the four places on the site that carried 2,653 or 2,514 now agree.

### Added
- **`scripts/claims_report.py` writes `claims.json` and `.hermes/posts/_current_numbers.md`.** It
  re-collects the test suite, reads the ledger through the shared loaders, takes stars/forks/releases
  from `gh`, and carries the superseded values forward so a stale figure stays visible as a bug.
- **`scripts/check_claims.py` fails the build when a public text disagrees with `claims.json`.** It
  checks the README badge, the collected/runnable counts, the results rows, every superseded value it
  has ever seen, and — with `--site-root` — the site's `project.json` and its `TESTS_TOTAL` constants.
  `--with-posts` extends it to the private launch drafts, which is how it found four stale counts
  nobody had noticed. Wired into the `repository gates` CI step; five tests in
  `tests/domain_infra/test_claims_check.py` cover the pass, stale, badge, row and site cases.
- **`scripts/ledger_facts.py`** holds the ledger loaders that `results_report.py` and the chart
  renderer both used to carry a copy of.
- **`scripts/traffic_snapshot.ps1`** captures stars, views, clones, referrers and star timestamps into
  `.hermes/traffic/<date>.md`; the traffic API only keeps 14 days, so the launch has to be measured on
  a schedule. The launch checklist referenced this file before it existed.
- `docs/assets/dashboard-overview.png` — the X thread needs a PNG, not the webp the README uses.

### Fixed
- The launch checklist pointed at `traffic-snapshot.ps1` (hyphen, never created) and at
  `docs/RESULTS.md` (never existed); both now point at files that are in the tree.

### Verified
- `python -m pytest tests -q`: 2,786 passed, 0 skipped, 161 s. `ruff check src scripts start.py`
  clean, `check_agents_md.py`, `check_comment_policy.py`, `check_dead_symbols.py` clean.
- `python scripts/check_claims.py --with-posts --site-root D:\qrak\PythonScripts\semantic-signal-site`
  clean after the fixes; before them it named 12 stale values in the README and four documents.
- `node scripts/check-claims.mjs` (site repo) and `npm run build` clean.

## 2026-11-01 — The pair is a config value, and the documented defaults are the loader's

### Changed
- **The README no longer describes the deployment as the product.** "A single bot on BTC/USDC" read
  as "this trades BTC/USDC"; `crypto_pair` takes any CCXT market, `ConfiguredSymbolGuard` blocks a
  decision that names another pair, and this deployment happens to run BTC/USDC. The opening
  paragraph names the config key, and "every four hours" became "every cycle (4h here, `timeframe`
  is a config value)".
- **`docs/CONFIGURATION.md` documents the loader's fallbacks, not the live file.** Three rows were
  the deployment's values under a "Default" column: `crypto_pair` (loader `BTC/USDT`, the template
  ships `BTC/USDC`), `provider` (loader `googleai`, the live file runs `deepseek`) and
  `stop_loss_type` (loader `soft`, the template and the live file run `hard`). The page's own header
  says the loader wins; two of the three rows contradicted it.
- **The test count is one figure in five files.** README said 2,746 in the badge, "2,700 collected
  (2,666 run)" in the feature table and "2,712 (2,746 collected)" in the testing block, while
  `docs/FEATURES.md` and `docs/SETUP.md` carried two more variants and the landing page claimed
  2,514 twice. Everything now says 2,781 collected and 2,747 outside the `integration` marker, which
  is what `pytest --collect-only` reports on this revision.
- The landing page's closing paragraph describes "the configured market — BTC/USDC here" instead of
  asserting the pair as the project's scope.

### Added
- **`scripts/check_config_contract.py` compares the documented defaults with the loader.** It reads
  every `get_config(section, key, <literal>)` call in `src/config/loader.py`, matches the key against
  the "Core settings" table and fails when the "Default" cell disagrees; numbers compare numerically,
  and a parenthetical note after the value is allowed. Its first run named two wrong rows. The
  existing CI step and `tests/domain_infra/test_config_contract.py` cover it, so a future drift fails
  a build instead of waiting for someone to read the page.

### Fixed
- **The tracked pyright config pins `pythonPlatform: Windows`.** It broke the first CI run after it
  landed: pyright defaults the platform to the host, so on the Linux runner `ctypes.WinDLL` and
  `ctypes.get_last_error` in `start.py` do not exist and the three directives removed in `39ee468`
  turned out to be load-bearing off Windows. The bot runs on Windows and both environments now check
  the same platform; the affected block is the single-instance mutex, already guarded at its call
  site by `sys.platform == "win32"`.

### Verified
- `python scripts/check_config_contract.py` clean after the doc fixes; a probe that changed the
  `crypto_pair` fallback to `ETH/USDT` made it fail with the row named, and the file was restored.
- `pytest tests -q -m "not integration"`: 2,747 passed; `tests/domain_infra/test_public_copy.py`
  and `check_agents_md.py` clean on the edited copy.

## 2026-10-31 — Suppressions narrowed to what they actually silence

### Changed
- **`# type: ignore` dropped from 219 to 150, bare ones from 108 to 62.** Twenty-six of them
  suppressed nothing: `pyright` reports that under `reportUnnecessaryTypeIgnoreComment`, enabled once
  locally, and they are gone. The remaining 26 bare comments in `rag_engine` name their rule now
  (`reportOptionalMemberAccess`), because a bare comment also hides the next refactor's real error.
  `RagEngine._periodic_update_task` is typed `asyncio.Task[None] | None` — the one
  `reportGeneralTypeIssues` there was the attribute being inferred as `None`, not a typing wart.
- **`ModelManager` takes its six collaborators as required arguments.** `unified_parser`,
  `token_counter`, `cost_storage`, `model_pricing`, `orchestrator` and `provider_clients` were typed
  `| None` and then used unconditionally, which is what carried 42 `reportOptionalMemberAccess`
  comments. The composition root already passed every one of them; the double in
  `test_provider_orchestration` passes a `MagicMock` parser now. One suppression stays where it is
  earned: `ProviderOrchestrator.get_metadata` returns `ProviderMetadata | None`, so
  `get_metadata("local").config` in the LM Studio streaming path is a real Optional until that
  accessor gains a non-optional form.
- **The GitHub event headline is a table, not a fourteen-branch chain.** `_EVENT_HEADLINE_TEMPLATES`
  holds the ten events whose wording depends only on actor and repository; `_ref_headline`,
  `_pull_request_headline` and `_issue_headline` keep the three that read the payload. The wording is
  pinned by 29 parametrized expectations added before the rewrite, covering `PublicEvent`,
  `MemberEvent`, `DeleteEvent`, both review events, `CommitCommentEvent`, `CreateEvent` without a
  `ref_type`, a merged and an unmerged close, an unknown PR action and an unknown event type.
- **One `escapeHtml` for the dashboard.** Eight modules carried a copy; seven dropped a literal `0`
  (`if (!text) return ''`) and one did not, so the same number rendered differently per panel. The
  canonical helper lives in `modules/html_utils.js` and keeps `0` and `false`. Cache versions move
  with the files: `main.js?v=8.7` and every touched module's `?v=`.
- **Broad exception handlers: eleven silent ones, six left standing.** Removed the dead guard around
  the two `logging.getLogger(...).setLevel(...)` calls in `startup_support`; the console stream
  handler, the stale-collection drop in `vector_memory` and the Kaleido stop in the shutdown manager
  log at debug instead of passing; the event-loop pump narrows to `RuntimeError`. The six that remain
  guard teardown that must not mask the error it is unwinding from: three paths in `logger`, the
  tkinter `destroy()` in `startup_support` and in the shutdown manager, and the asyncgen shutdown in
  `crawl4ai_enricher`.
- `ConsoleRouter` receives a `logger` from `server.py`, matching every other router, so the stream
  error can be logged without a module-level logger.
- **`pyrightconfig.json` is tracked and enforces the rule.** It was gitignored, so
  `reportUnnecessaryTypeIgnoreComment` could only ever be a local experiment; the file now carries
  that setting as `error` and CI's existing `pyright` step reads it. A suppression that stops
  suppressing fails the build instead of accumulating. AGENTS §12.1 gained the row.

### Fixed
- **`escapeHtml(0)` rendered nothing in seven panels.** Any zero-valued field went out as an empty
  string; the shared helper returns `0`. A literal `null` still becomes `''`.
- **The console panel put a raw error message into `innerHTML`.** `Failed to load: ${e.message}` is
  escaped now, the one sink of the nine that had neither `DOMPurify.sanitize` nor an escaped value.

### Verified
- `pytest tests -q -m "not integration"`: 2747 passed, 0 failed, 0 skipped (baseline 2718 + 29 new
  expectations). `ruff check src scripts start.py`, `pyright src start.py` (0 errors) and all six
  `scripts/check_*.py` gates clean; `check_net_diff.py --scope code` reports −59 lines over 29 files.
- `pytest tests/test_admin_playwright.py` (the `integration` marker, Chromium): 34 passed.
- A headless browser probe loaded all eight panels through the real import graph with zero console or
  page errors, and asserted the helper's `0`, `null` and escaping cases.
- `slop-scan` on the dashboard JavaScript: 10 findings and a blended 10.46 before, 3 findings and 3.27
  after — the seven duplicate-signature findings collapsed with the copies. `ai-slop-detector`:
  902 findings and an 18.45 deficit before, 834 and 17.88 after. `aislop` moved 61 → 62 while losing
  16 diagnostics, which is why its single number is not the measurement to quote.

## 2026-10-30 — An unconfirmed-intent alert cannot outlive its intent

### Fixed
- **A stale CRITICAL was delivered under a fresh position.** The alert a background check raises
  (`_close_on_exit`, `mark_position_intent_unknown`, a refused UPDATE) was a bare string consumed
  only at the end of an analysis cycle. On 2026-10-30 01:05:19 the operator saw
  `UNCONFIRMED POSITION INTENT: ... take_profit triggered at price 79,882.79` **3 h 31 min** after
  the CLOSE it described (29.10 21:34:09), six seconds after a new LONG was confirmed — a price from
  the old position, read as "this position's target fired". No intent existed for it: the journal had
  no close row, the executor no decision, and the verdict journal had already settled that CLOSE at
  21:34:20. `UnconfirmedIntentAlert` (`data_models`) now carries `intent_key` + `created_at`;
  `take_unconfirmed_intent_alert` **drops** the warning when its intent is already `confirmed`
  (debug line), keeps it for `pending`/`unknown`/`refused` (a refused UPDATE is still news), and the
  CRITICAL logs `captured <timestamp>` so a late one is self-labelling. Every setter goes through
  `_raise_unconfirmed_intent_alert` (`executor_reconciliation`, `trading_strategy`,
  `position_management`).

### Changed
- **A CLOSE booked from the executor receipt names the venue order**: `ExitRecords.from_receipt`
  passes `exchange_order_id` into the record, which `verify` already mapped to the exit's order id, so
  the booking reads `exchange order 4206227` instead of `exchange order None`. The executor writes
  that field into the receipt and the verdict row (its half is in its own CHANGELOG). The exit journal
  stays the channel for exits the EXCHANGE detected (`source: reconcile`); a commanded CLOSE is booked
  from the receipt's fill evidence, unchanged.

### Verified
- `pytest tests -m "not integration"`: **2,718 passed, 34 deselected**; `ruff check src scripts start.py`
  clean; `pyright src start.py` 0 errors; `check_dead_symbols.py` → 0 dead, 0 test-only;
  `check_comment_policy.py` → 0; `check_agents_md.py`, `check_config_contract.py`,
  `check_hardcoded_timeframes.py`, `check_loose_helpers.py` clean.
- New tests: `TestUnconfirmedIntentAlertLifetime` (confirmed → dropped; pending, unknown, refused and a
  vanished intent → still reported) and the receipt `exchange_order_id` → booked exit.

### Why
An alert is a statement about a command in flight. Once that command is settled the statement is not
stale documentation, it is false — and it arrived attached to a different position, which is the one
way an operator warning can be worse than silence. Identity (`intent_key`) is what makes the alert
disposable; the capture time keeps the remaining ones honest.

## 2026-10-29 — A refactor deletes the code it replaces, and the gates prove it

### Added
- **`scripts/check_dead_symbols.py`**: an AST gate over every function, method and class in `src/` and `start.py`. It separates `DEAD` (no reference anywhere), `TEST-ONLY` (kept green only by its own test) and `COMPATIBILITY` (`warnings.warn(..., DeprecationWarning)` or `@deprecated`). Framework callbacks are exempt by decorator or base class; dynamic names count through `getattr`/`patch` strings; imports inside `__init__.py` are exports, not uses. `--symbol NAME` prints every call site, the deterministic replacement for an LSP `textDocument/references` query.
- **`scripts/check_net_diff.py`**: a refactor budget over `git diff --numstat`. `code` scope is `src/`, `scripts/` and `start.py`; a pull request labelled `refactor`/`cleanup` (or titled that way) must end net-negative, while a feature pull request only gets the report. `--worktree` and `--staged` measure before and after staging.
- **`scripts/check_comment_policy.py`**: no prose `#` comments in Python, no `//`/block comments in dashboard JS/CSS/HTML, and no docstring outside a class.
- **`tests/domain_infra/test_dead_symbols.py`, `test_net_diff.py`, `test_comment_policy.py`, `test_deletion_policy.py`**: 58 tests pinning the gates, their canaries, framework exemptions, dynamic references, stale allowlists, inline JS comments and the policy/CI contract.
- **`.githooks/pre-commit`** (optional): `git config core.hooksPath .githooks` runs the orphan and comment gates before a commit; CI stays the hard blocker.
- **`.github/PULL_REQUEST_TEMPLATE.md`**: a refactor must paste `Net: -N`, the dead-symbol result and the list of deleted symbols/tests.

### Changed
- **`AGENTS.md` §10/§12** now states the deletion contract: replacement surgery, no deprecation shims, a net-negative production diff, find-references before deletion, test-only symbols deleted with their tests, and a whole-feature retirement across code, tests, data accessors and claims. §12.1 names each gate; §12.9 permits the CI step that adds a named gate.
- **`CONTRIBUTING.md`, `README.md`, `docs/SETUP.md`** document the new gates, the `refactor` label and the local hook.
- **Comment/docstring policy**: only a class docstring survives. The 1,622 method, 709 function and 245 module docstrings (plus `start.py`) and 79 prose `#` comments were removed; 294 class docstrings remain (the 286 original plus the new gate classes). `__doc__`-based argparse descriptions became explicit strings, and the two FastMCP tools keep their descriptions through `@server.tool(description=...)`.
- **`.pylintrc`** no longer requires module/class/function docstrings.
- **`.github/workflows/tests.yml`** runs the dead-symbol and comment gates on every push and the deletion budget on pull requests; checkout uses full history for the merge-base.

### Removed
- `safe_array_to_scalar`, `ExecutorReconciliationMixin.position_intent_state`, `BaseNotifier.send_trading_decision` with both implementations, `LocalPositionReconciliation.is_open_confirmed`/`is_exchange_verified`, `ExecutorReconciliationMixin.unbooked_position_intents` and `TokenCounter.truncate_text`, each with its tests or doubles. The first gate run found them as two dead and six test-only definitions; `position_intent_state` and `send_trading_decision` were leftovers of earlier refactors whose call sites had already been removed (`CHANGELOG.md:690`, `docs/changelog-archive-2026.md:747`). `hash_password` and `reset_timezone_cache` stay allowlisted with a reason (documented hash generator; test seam).

### Verified
- `python scripts/check_dead_symbols.py` → `0 dead, 0 test-only, 2 allowlisted`; `python scripts/check_comment_policy.py` → 0.
- `pytest tests -q -m "not integration"`: **2,712 passed, 34 deselected, 0 failed, 0 skipped** (138 s).
- `ruff check src scripts start.py` clean; `pyright src start.py` 0 errors; `check_agents_md.py`, `check_config_contract.py`, `check_hardcoded_timeframes.py`, `check_loose_helpers.py` clean.
- `python scripts/check_net_diff.py --base origin/master --worktree --require-negative` → `+1424 / -4978 = -3554` code lines across 185 tracked files; after staging the three new gate scripts, `+2317 / -4978 = -2661` across 188 files.
- OpenAPI descriptions that lived only in FastAPI handler docstrings are gone; no runtime, wire-format, config, SQLite or Chroma change. No bot or executor process was started or restarted.

### Why
The additive instinct is cheapest when the old implementation is still green: a duplicate beside it avoids proving that nothing references the old one, and its own test keeps the ghost alive. The gates make the cheap move fail: an unreferenced definition fails the build, a test-only definition is named as such, and a refactor pull request must remove more production lines than it adds. The purge is the same rule applied to prose: a method docstring that restates its signature is not documentation, and a `//` comment is not a licence to skip a precise name.

## 2026-10-28 — The activity tab shows the release, and the feed stays inside its card

### Fixed
- **The release was invisible**: the panel read only `/users/{login}/events/public`, and GitHub files
  an event under the actor that caused it — so every `build-*` release published by
  `github-actions[bot]` and any collaborator's issue were missing. The newest build,
  `build-2026.10.28-0915`, did not appear even under its own `LLM_trader` filter. `fetch_activity`
  now requests the highlighted repository stream (`/repos/{owner}/{repo}/events`) next to the
  account stream, and `merge_events` dedupes by event id and sorts newest first: the live page went
  from 30 rows of verbs-by-qrak to 32 rows that also carry the bot release and `demonroute2`'s issue.
  A failing repository stream costs its rows only; the account stream still decides `stale`.
- **Rows escaped the card**: `.gh-list` asked for `max-height: calc(100vh - 240px)` (760 px) inside a
  600 px `.panel` that did not clip, so rows painted past the card's bottom border onto the page
  background (measured at 1600×1000: card ended at y=731, last row at y=916). The panel now clips,
  the list scrolls inside it, and the list lost its 18 px padding, which double-inset rows against
  the panel's own 24 px.
- **A narrow request pinned the cache**: the route cached whatever page the *first*
  caller asked for, so `/api/monitor/github_activity?limit=6` stored six events for 300 s and every
  panel then rendered seven rows. The cache now holds a full page (100 events, one request per
  stream) and `limit` only cuts rows when the payload is built.
- **`latest_release`**: the newest `ReleaseEvent` of the highlighted repository, with
  `/releases/latest` as the fallback when the event page predates it, rendered as a pinned header
  line — `LATEST RELEASE build-2026.10.28-0915 · Latest build — … · published 1 hour ago` — linking
  to the release page. It reports `github-actions[bot]` as the author instead of hiding the bot.

### Changed
- **`/api/monitor/github_activity`** answers with `repo` and `latest_release`; the cache key includes
  the highlighted repository, so two tabs cannot serve each other's rows.
- **Tests**: `tests/domain_infra/test_dashboard_github_activity.py` 18 → 29 tests (merge and dedupe,
  release selection across repositories, stream-failure tolerance, a narrow request not truncating
  the cached page, the new payload fields).
- **Docs**: `docs/FEATURES.md` (two streams, the release line, the clipping rule),
  `src/dashboard/AGENTS.md` (same, for the next agent); asset versions bumped to `css?v=7.17` and
  `github_activity_panel.js?v=1.1`.

### Why
The panel claims to mirror the project, but GitHub's account stream is a list of verbs *by that
account*: a release published by the Actions token and an issue opened by a collaborator are not in
it, so the tab contradicted github.com on the one thing it existed to show. The layout bug had the
same shape — the list was measured against the viewport while the card was sized independently, so
the two disagreed by 185 px and the rows won.

## 2026-10-28 — The dashboard mirrors the public GitHub event stream

### Changed
- **`src/dashboard/github_activity.py`** (new): `fetch_user_events` performs the single
  `/users/{login}/events/public` call (10 s timeout, `Accept: application/vnd.github+json`, optional
  `Bearer` token, `per_page` bounded to 100); `build_feed` renders each event as a row in GitHub's
  own wording ("qrak contributed to qrak/LLM_trader") with a detail line, a meta line and a link.
  Every event type has one headline; unknown types and malformed entries are skipped, so a new
  GitHub event cannot blank the panel. Titles are collapsed to one line and cut at 200 characters.
- **`/api/monitor/github_activity`**: `limit` (1–100, default 30), `refresh`, `highlight`. The page
  is cached 300 s per login; a failed fetch serves the last good page as `stale` for 24 h with the
  error attached, and only an empty cache answers with `error` and no rows. An exhausted
  unauthenticated quota says so instead of rendering an empty feed.
- **UI**: eleventh tab, "GitHub Activity" — `static/modules/github_activity_panel.js`,
  `css/panels.css`, `index.html` with the module and asset versions bumped: filter bar, relative
  timestamps, the highlighted repository marked, explicit empty/unavailable states. Rows are built
  with `createElement`/`textContent` and links carry `rel="noopener noreferrer"`.
- **`keys.env.example` / `Config`**: `GITHUB_USER` selects the account, `GITHUB_TOKEN` raises the
  unauthenticated 60 requests/hour to 5000; both optional, so a checkout without them keeps the tab
  behind a configuration hint.
- **Docs**: `docs/FEATURES.md` (eleven tabs, the cache and failure behaviour), `docs/SETUP.md` (the
  two variables), `src/dashboard/AGENTS.md` (module, route, cache rule).

### Why
The dashboard already shows what the bot does; the public repository is where the work is visible,
and github.com is the system of record for it. Mirroring the event stream verbatim keeps the panel
from contradicting GitHub — no second wording, no duplicated counts, no re-parsing of commit
messages the events API already returns.

### Verified
- `tests/domain_infra/test_dashboard_github_activity.py`: 18 tests covering the wording per event
  type, skipping, highlighting, truncation and the cache/stale/error paths — part of the
  2,689-collected suite that runs green.
- Live on the running bot: `GET /api/monitor/github_activity?limit=3` answered `user=qrak`,
  `count=3`, `stale=false`, `error=null` with the account's public events.

## 2026-10-28 — Dependencies state a bound, and the environment measures 1.7 GB

### Changed
- **`requirements.txt`**: every entry now states a bound instead of a bare floor. `==` marks the
  exchange, provider and chart clients whose next release changes behaviour; `~=` marks the
  libraries whose API is stable across a minor line. Bumped in the same line they were tested on:
  `ccxt 4.5.90`, `tiktoken 0.14.0`, `aiohttp 3.14.5`, `rich 14.3.4`, `pandas 2.3.3`,
  `discord.py 2.7.1`, `plotly 6.9.0`, `google-genai 2.33.0`, `blockrun-llm 1.21.3`,
  `python-dotenv 1.2.4`, `kaleido 1.5.0`, `torch 2.14.1`, `fastapi 0.162.0`, `uvicorn 0.54.0`,
  `pydantic 2.14.1`, `packaging 26.3`. Held, each with the reason on the line: `openai 2.49.0`
  (3.x client surface), `sentence-transformers 5.6.1` (embedding path), `numpy 2.2.3` +
  `numba 0.61.2` (move as a set with llvmlite), `aiodns 3.6.1` (4.x needs pycares 5),
  `aiohttp-client-cache 0.13.0` (shared backend), `trafilatura 2.1.0` (ccxt pins
  `charset_normalizer==3.4.7`, every 2.2+ release needs more).
- **`requirements-dev.txt`**: `pytest-asyncio~=1.4.1`, `pytest-timeout==2.4.0`; `pytest 9.1.1`,
  `playwright 1.61.0`, `ruff` and `pyright` keep their pins with the reason stated.
- **`requirements-mcp.txt`**: `mcp>=1.9,<2`; the 2.x line moves `mcp.server.fastmcp`.
- **`constraints.txt`**: `patchright==1.61.2` — the Chromium build Crawl4AI drives is the one
  installed, and a bump needs a new browser download.
- **Install path**: `--no-compile` in `README.md`, `docs/SETUP.md`, `CONTRIBUTING.md`, all four
  launchers and `scripts/run_all_tests.sh`. Bytecode is written on first import instead of eagerly
  for every installed module.
- **`docs/RESULTS.md` references retired** (the file is gone): `README.md` points at its own
  Results section and at `scripts/results_report.py`, `AGENTS.md` at the same, `docs/SETUP.md`
  names the script. The directory tree in `AGENTS.md` no longer lists `adr/`.
- **Measured numbers**: 2,514 → 2,689 collected tests (2,655 outside `integration`), ~126 s warm
  in `README.md`, `docs/FEATURES.md`, `docs/SETUP.md`; the disk baseline in `docs/SETUP.md` is the
  measured environment instead of 2.3 GB.
- **`.github/workflows/tests.yml`**: CPU `torch` installed from `download.pytorch.org/whl/cpu`
  before the resolver runs, so a Linux job stops pulling the ~450 MB CUDA wheel; a new
  "Repository gates" step runs `check_agents_md.py`, `check_config_contract.py`,
  `check_hardcoded_timeframes.py` and `check_loose_helpers.py` inside the existing `unit` job.
- **`scripts/check_agents_md.py`**: also scans `README.md`, `CONTRIBUTING.md` and `docs/*.md`
  (changelog archives excluded), reads markdown link targets as well as backticked paths, and
  allows the local files the docs legitimately name (`config/config.ini`, `config/peak_rates.json`).
- **`CONTRIBUTING.md`**: "Dependency Changes" states the pin policy (bounds, when a package belongs
  in `constraints.txt`, what a hold must record) and the PR checklist lists the four gates.

### Why
Half the file was eight floors (`torch>=`, `openai>=`, `pydantic>=`, …), so a fresh install resolved
a different environment than the bot ran. The CI log of 2026-10-28 proved it: the runner installed
`torch 2.14.1` and `pydantic 2.14.1` while the venv ran 2.13.0 and 2.12.5. A dependency audit of the
installed tree found no unused entry — every package is imported or used without an import
(`wsproto` by `uvicorn.Config(ws="wsproto")`, `tzdata` by `zoneinfo`, `kaleido` by the shutdown
cleanup) — but it did find 190 MB of packages no manifest names (bitsandbytes, cloudflare,
openrouter, pylint, py-spy) and 416 MB of bytecode that pip compiles eagerly for every module,
including the ones nothing imports.

### Verified
- venv rebuilt from the new files with `--no-compile` and swapped in: **2,167 MB → 1,782 MB**
  (−18%) after the full suite, 1,636 MB at creation, 224 → 215 distributions, bytecode 416 MB →
  151 MB. The manifest set is unchanged; the newer releases pull a few transitive packages the old
  environment did not have (`httpx2`, `httpcore2`, `aiohttp-retry`, `cloudpickle`, `py_ecc`,
  `bitarray`) and the packages no manifest named (bitsandbytes, cloudflare, openrouter, pylint,
  py-spy) are gone because the rebuild installs what the manifests name.
- The bot starts on the rebuilt environment: all 9 provisioning stages in 12.7 s, `bge-base`
  loaded, ChromaDB reopens the 22 stored experiences, Numba kernels compile, dashboard answers
  `200` on `127.0.0.1:8000`.
- `pytest tests -q -m "not integration"`: 2,655 passed, 34 deselected — 126 s warm versus 140 s in
  the old environment, 167 s on the cold (uncompiled) one; `tests/test_admin_playwright.py`: 34
  passed.
- Embeddings: `BAAI/bge-base-en-v1.5` loads offline and returns normalised 768-dim vectors on
  `torch 2.14.1`, `transformers 5.21.0`, `huggingface-hub 2.3.0`, `sentence-transformers 5.6.1`.
- `pip check` clean, `scripts/check_requirements.py` prints nothing, `ruff` clean,
  `pyright src start.py` 0 errors, all four repository gates exit 0.

## 2026-10-28 — CI installs the pytest that runs the suite

### Changed
- **`requirements-dev.txt`**: `pytest` pinned to `9.1.1`. The unpinned floor resolved to 9.2.0
  (uploaded 2026-10-27 21:30 UTC) on the next run, and on `ubuntu-latest` that build freezes the
  collection phase: the job spends ~6 s collecting, emits no further output and is canceled by
  GitHub — on the push and on a pull request alike, while 9.1.1 runs the same tree green on the
  same runner image with every other wheel at the version that passed the day before.

### Why
Two consecutive pushes and a diagnostic pull request were canceled inside the unit-test step with
no test output at all. A collection trace added on a throwaway branch showed ~50 collectors in
6 s and then silence; the freeze is not module-specific (ignoring the module it stopped on moved
the stop to the next one at the same elapsed time), the process did not respond to
`faulthandler` or to a watchdog thread, and the job was canceled 25–46 s later. The only
difference from the previous green run is the pytest version, so the gate stops depending on an
implicit update of the tool that runs it.

### Verified
- The 2026-10-27 green run installed `pytest-9.1.1`; the 2026-10-28 runs installed
  `pytest-9.2.0`, with identical `torch-2.14.1`, `chromadb-1.5.9`, `numba-0.61.2`,
  `pydantic-2.14.1`, `playwright-1.61.0`, `pytest-asyncio-1.4.1` and `pluggy-1.7.0`.
- Locally 9.2.0 passes (2 637 tests, 142 s) — the freeze is runner-specific, so the pin is the
  direction that keeps the gate reproducible until the upstream fix ships.

## 2026-10-28 — Every product claim matches the runtime that runs it

### Changed
- **`src/analyzer/formatters/ev_formatter.py`**: the worked example, the round-trip fee and the
  1.5× hurdle use the effective NEUTRAL cap (`RegimeRiskProfileSelector.effective_caps(MAX_POSITION_SIZE)`),
  not a hardcoded 8 %. With `max_position_size = 0.05` one prompt quoted "8 %/$1.20/$1.82" while
  Decision Rules quoted "NEUTRAL 5 %", so the hurdle sat above what a legal position can earn. The
  block also stops telling the model that P(win) "is logged and scored against the realized
  outcome" — nothing records or scores it.
- **`src/config/writable_config.py`**: the provider enum offers `deepseek`. The value the bot runs
  was missing from `options`, so the admin form rendered no selected option (the browser shows the
  first one) and the row's Save wrote `provider = local` back into `config.ini`.
- **`src/analyzer/pattern_quality_scorer.py`**: the deterministic score replaces
  `confluence_factors["pattern_quality"]` as well as `analysis["pattern_quality"]`. The factor the
  closed trade teaches the brain kept the model's self-report, so the claimed control was cosmetic.
- **`src/trading/vector_memory_rules.py`**: `corrective` rules are confirmed by a loss, like
  `anti_pattern`/`ai_mistake` (one `_warns_about_a_losing_pattern` helper drives both the counter
  and the evidence score; previously a confirming loss incremented `contradiction_count`).
  Re-storing a rule under the same id keeps its counters, `created_at` and retirement state, so
  reflection no longer resets the evidence every interval or resurrects a retired rule.
- **`src/trading/brain_context.py`**: zero retrieved experiences count as limited data, so the
  prompt stops inviting "⚠️ ANTI-PATTERN MATCH" when the store returned nothing (fresh install,
  a >56-day gap, a Chroma error or a budget drop).
- **`src/utils/prompt_budget.py`**: a block whose own end marker is absent ends at the next
  heading. Without it, the "drop semantic rules" rule silently did nothing on an install with no
  post-mortems, and a near-cap request was refused instead of degraded.
- **`src/analyzer/prompts/template_manager.py`**: the SL reference says "Historical" only when an
  average was learned; otherwise "Default … no closed-trade average yet", matching the same
  prompt's "no trade history" line.
- **`src/dashboard/routers/brain.py`** + **`src/dashboard/static/main.js`**:
  `/api/brain/status` carries the store-wide `rule_count`; the Overview KPI reads it instead of
  the length of the 20-item `/api/brain/rules` list, which disagreed with the Memory tab.
- **`src/dashboard/static/modules/websocket.js`** + **`src/dashboard/static/main.js`**: the
  WebSocket is the only writer of `#connection-status` (a successful REST poll used to overwrite
  "Disconnected"), it reconnects without a ceiling (3 s → 30 s backoff, previously it stopped after
  10 attempts), and a reconnect refreshes the countdown and the panels. `main.js?v=8.5`,
  `websocket.js?v=4.8`.
- **Docs**: `README.md` (buy & hold −1.49 %, cost/net for the 2026-10-26 measurement, provider
  chain order), `docs/FEATURES.md` (full list of injected brain blocks, rule-ranking blend,
  "three attempts (two retries)", no "expectancy from closed trades", no journal rotation),
  `docs/CONFIGURATION.md` (peak-rate example 0.5 with both windows, `data_dir` exceptions),
  `docs/RESULTS.md` (48 cycles in the window, what the mean cost covers), `AGENTS.md` (top-3
  retrieval, ranking weights, warning-rule polarity, re-store contract, new enforcement rows),
  `src/dashboard/AGENTS.md` (cache-bust rule includes `main.js?v=`, unwired admin controls
  described as API-only state).

### Why
An audit of every declaration in `docs/FEATURES.md`, `README.md`, `docs/CONFIGURATION.md`, the
UI copy and the prompts against the code, with each candidate re-verified by a failing test or an
executed path. The mismatches with a consequence ranged from the EV hurdle computed on a position
the runtime refuses (P1) and an admin form that could silently switch the provider (P1) to a
claims-validation result that never reached the learner (P1), a rule lifecycle that reset its own
evidence, and docs whose numbers no longer reproduced. Nothing was added: the fixes either made
the code do what it declared or made the declaration say what the code does.

### Verified
- `pytest tests -q -m "not integration"` → 2 636 passed, 34 deselected, 0 failed.
- `ruff check src start.py scripts` clean; `pyright src start.py` 0 errors; `check_config_contract.py`,
  `check_hardcoded_timeframes.py`, `check_loose_helpers.py`, `check_agents_md.py` clean;
  `python -c "import start"` OK.
- New pins: `test_ev_worked_example_follows_the_cap_the_runtime_enforces`,
  `test_provider_enum_offers_every_provider_the_loader_accepts`,
  `test_overwrite_llm_quality_also_corrects_the_factor_the_brain_learns`,
  `test_a_loss_confirms_a_corrective_rule_instead_of_contradicting_it`,
  `test_restoring_a_rule_keeps_the_evidence_and_retirement_it_accumulated`,
  `test_no_retrieved_experience_suppresses_the_anti_pattern_invitation`,
  `test_prompt_budget_drops_a_section_whose_end_marker_is_absent`,
  `test_brain_status_carries_the_uncapped_active_rule_count`,
  `test_overview_rule_count_is_read_from_the_status_not_the_capped_list`,
  `test_the_live_channel_owns_the_connection_badge_and_always_reconnects`,
  `test_a_learned_average_sl_is_labelled_as_history`.

## 2026-10-27 — The prompt quotes the cap and thresholds the runtime enforces

### Changed
- **`src/analyzer/prompts/template_manager.py`**: the position caps and ATR multiples in
  `build_decision_rules()` come from `RegimeRiskProfileSelector` clamped by
  `MAX_POSITION_SIZE`; an unusable cap raises instead of falling back to `0.10`. The trading-style
  block and the relevance window derive minutes through one helper and **raise** on an unknown
  frame (the private `60`/`120` fallbacks are gone). The frame is normalised on entry to
  `build_system_prompt()`/`build_response_template()`; decision thresholds come from the shared
  defaults; a dead `_thresholds` local is removed.
- **`src/trading/regime_risk_profile.py`**: `effective_caps(max_position_size)` and
  `build_profile_context(..., max_position_size=...)` render the cap `RiskManager` actually
  enforces (profile cap only tightens the config ceiling) and derive the percentages/ATR
  multiples from `_PROFILE_PARAMS` instead of three literal copies.
- **`src/trading/threshold_defaults.py`** (new): one mapping for the unlearned decision
  thresholds, shared by `BrainContextProvider` and the prompt. `learned_keys`/`trade_count`
  remain computed, so a default is never reported as learned.
- **Brain plumbing**: `TradingBrainService`/`BrainContextProvider` accept `max_position_size`;
  provisioning passes `Config.MAX_POSITION_SIZE`. A closed trade recorded without a frame omits
  `timeframe_minutes`/`timeframe_bucket` instead of storing `0`, and a rule whose
  `source_timeframe_minutes` is missing, non-numeric or non-positive is normalised to the
  service frame (value *and* bucket).
- **`src/analyzer/market_metrics_calculator.py`**: an unusable frame raises; the `1D = 24
  candles` fallback (an implicit hourly assumption) is removed.
- **`src/config/writable_config.py`**: every key is categorised `restart` — the module edits the
  file, `Config` is read once at import, and no reload signal ever existed. The docstring,
  `/api/admin/config/reload` wording, `docs/CONFIGURATION.md` and `src/dashboard/AGENTS.md` say so.
- **Gates/docs**: `AGENTS.md` §5.1/§12.1 (prompt renders effective caps and shared defaults;
  admin keys are restart-scoped), `src/analyzer/AGENTS.md`, `src/trading/AGENTS.md`.

### Why
The first two passes removed the hardcoded `4h` from the prompt, but the prompt still quoted a
second source of truth for other configured values. With `max_position_size = 0.05` it printed
"AGGRESSIVE 10%" in one sentence and "fall back to 0.05 (5%)" in the next; with a frame the
validator could not convert, `1d` was styled as "Intraday Swing" from a private 60-minute default,
and a test pinned that as expected behaviour. The admin panel advertised `hot`/`cycle` application
for edits the process never reads.

### Verified
- Reproductions before the fix: `MAX_POSITION_SIZE=0.05` → "AGGRESSIVE 10% / ..."; `1d` with a
  raising validator → `Style: Intraday Swing (1d candles)`; `13h` metrics → `1D = 24` candles.
- After: live config (`0.10`) renders the identical historic sentence and no clamp note; `0.05`
  renders `AGGRESSIVE 5% / NEUTRAL 5% / CONSERVATIVE 5% (capped by max_position_size)`; unknown
  frames raise for style and period metrics.
- `pytest tests -q` → 2 653 passed, 0 failed, 0 skipped; `ruff check src scripts start.py` clean;
  `pyright src start.py` → 0 errors, 0 warnings; `check_config_contract.py`,
  `check_hardcoded_timeframes.py`, `check_loose_helpers.py`, `check_agents_md.py`,
  `python -c "import start"` clean.

## 2026-10-27 — The prompt contract and the brain tool follow the configured frame

### Changed
- **`src/analyzer/prompts/template_manager.py`**: `build_response_template()` takes `timeframe`
  and resolves `Config.TIMEFRAME` when it is omitted. The TREND BLOCK now says "which way the
  `{timeframe}` trend points" and "whether the `{timeframe}` and daily timeframes agree"; the
  previous-analysis snapshot labels the strength with the configured frame; the JSON example uses
  `strength_primary`. `PROMPT_VERSION` is `v1.7`.
- **Trend contract**: `TrendModel.strength_primary` replaces `strength_4h`, with a before-validator
  that maps the legacy key (primary wins when both exist). `TrendValidator`, the result processor,
  `MarketConditionsExtractor` and both notifiers read primary through
  `resolve_primary_trend_strength()`; the validator's `_trend_validation` gains `timeframe`.
- **Brain learning**: production already derived decay, buckets and reflection cadence from
  `Config.TIMEFRAME_MINUTES`; the tool did not. `scripts/brain_rules.py` now imports the repo root
  (the documented `python scripts/brain_rules.py` used to fail with `No module named 'src'`),
  derives the store from `brain_store_path(Config.DATA_DIR, Config.CRYPTO_PAIR, Config.TIMEFRAME)`
  and takes `--timeframe` for an explicitly different store. `VectorMemoryService` and
  `TradingBrainService` refuse a missing/non-positive `timeframe_minutes` instead of silently using
  240; closed-trade records now carry `timeframe_minutes` + `timeframe_bucket` like UPDATE records;
  a malformed rule `source_timeframe_minutes` falls back to the service frame, not 240.
- **Residual consumer defaults**: `pattern_recency` refuses a missing frame instead of using a
  240-minute candle; `PromptBuilder` requires its `timeframe_validator` (the period-candle math no
  longer falls back to 60 minutes).
- **Gates/docs**: the config-propagation test now renders the complete system prompt and the
  response contract and asserts that only the configured frame appears; the AST gate's docstring
  names its prose blind spot; the gate-list drift and dashboard timeframe-option invariants are
  pinned; `AGENTS.md` §5.1/§12.1, the package maps and `docs/CONFIGURATION.md` record the rule.

### Why
The first timeframe pass removed seven consumer defaults and added two gates, but its system-prompt
test rendered only `build_system_prompt()`. `build_response_template()` still hardcoded
"which way the 4h trend points" and `strength_4h`, so a bot configured for `6h` was told to analyse
4h while 2 619 tests stayed green. The AST gate only matches a string constant that *is* a frame, so
the same literal inside prompt prose was invisible. In the brain path the production wiring was
correct, but the maintenance script hardcoded `brain_BTC_USDC_4h` + `240` and did not even run from
its documented command; the service constructors also carried a private 240 default.

### Verified
- Before the fix the new render tests failed: `7 failed` (response contract for `15m/1h/6h/12h/1d`,
  the complete prompt, and the legacy snapshot label). After the fix all pass.
- `pytest tests -q` → 2 642 passed, 0 failed, 0 skipped; `ruff check src scripts start.py` clean;
  `pyright src start.py` → 0 errors, 0 warnings; `check_loose_helpers.py`,
  `check_config_contract.py`, `check_hardcoded_timeframes.py` and `check_agents_md.py` clean.
- `build_response_template()` for `1h` contains no `4h`/`strength_4h`; for `4h` it names 4h and
  `strength_primary`, so the live frame's contract is unchanged in substance.
- `BrainRuleStore(tmp, 15)` passes `timeframe_minutes=15`; `VectorMemoryService(60)` derives a
  4-day half-life / 16-day window; production provisioning with `TIMEFRAME=1h` opens
  `data/trading/brain_BTC_USDC_1h` and passes 60. `python scripts/brain_rules.py --help` runs.
- No live Chroma store was written; no commit, no push.

## 2026-10-27 — The timeframe has one source, and gates prove it

### Changed
- **`src/config/loader.py`**: `Config.REQUIRED_CONFIG_KEYS` (`[general]` crypto_pair, timeframe,
  candle_limit, ai_chart_candle_limit, market_type; `[ai_providers]` provider; `[model_config]`
  max_tokens) is validated in `__init__` before anything reads it, and a failure names the file and
  every missing key. `TIMEFRAME` lost its silent `"1h"` fallback and is validated plus normalized in
  one place; the new `TIMEFRAME_MINUTES` derives the minute value once instead of the five separate
  conversions that existed in the loader, the provisioners, the analysis engine, the strategy and the
  exit monitor.
- **Timeframe consumers**: `MarketDataCollector`, `MarketMetricsCalculator`, `PromptBuilder`,
  `TemplateManager`, `ChartGenerator`, `TechnicalFormatter` and `TradingStrategy` no longer carry
  their own timeframe defaults — they take the configured value or refuse to run. `DataFetcher`
  names its macro frames (`TREND_TIMEFRAME`, `WEEK_TIMEFRAME`), the dashboard enum derives from
  `TimeframeValidator.SUPPORTED_TIMEFRAMES` (which drops the unsupported `1m`), and
  `AnalysisEngine.initialize_for_symbol` logs an explicit override warning instead of falling back
  silently when handed an invalid frame.
- **`scripts/check_config_contract.py`** (new): every `get_config` pair the loader reads must exist
  in `config.ini.example`; `config.ini` may not carry a key the loader ignores (a typo becomes a
  failure instead of a silent default); required keys must be in the template; the seeded
  provider/model must have a context window; core keys quoted in `docs/CONFIGURATION.md` must exist.
- **`scripts/check_hardcoded_timeframes.py`** (new): AST gate — a supported-frame literal outside
  `TimeframeValidator`, the `DataFetcher` macro constants and the prompt period labels fails the run.
  Docstrings are exempt.
- **`tests/domain_infra/test_config_contract.py`** and **`tests/domain_infra/test_config_propagation.py`**
  (new): pin both gates, the fail-fast startup, and a 6h/12h/1d ini reaching the system prompt,
  period candle math, exit-interval validation, the data collector and the engine.
- **`config/config.ini.example`** documents `[circuit_breaker]`, the human-approval keys under
  `[risk_management]` and `intent_journal_path` (commented: its default derives from `data_dir`).
- **`AGENTS.md`** gains §5.1 (configuration invariants) and §12.1 (enforcement map); `CONTRIBUTING.md`
  adds the loader-key rule to the PR checklist; `docs/CONFIGURATION.md` states that `timeframe` has no
  default.

### Why
The shipped `config.ini`, the template, `AGENTS.md` and `docs/CONFIGURATION.md` all described a 4h
bot, while `Config.TIMEFRAME` fell back to `1h` and seven consumers kept a private copy of a
plausible frame (`"1h"` in the collector, metrics calculator, prompt builder, chart generator and
template manager; `"4h"` in the technical formatter; `240` minutes in the strategy). A change that
takes one of those defaults instead of the configured value produces a bot that looks healthy and
trades a frame nobody chose. Benchmarks put full instruction-compliance at roughly a third of runs
even for the strongest models, so the fix is not "pick a better model": the configured value now
has one path and every divergence fails a command.

### Verified
- `python scripts/check_config_contract.py` → `config contract OK`; the gate found real drift before
  the fix (13 loader keys absent from the template, including the whole `[circuit_breaker]` section).
- `python scripts/check_hardcoded_timeframes.py` → clean after the refactor.
- Mutation proof: injecting `timeframe = "4h"` into `TemplateManager.build_system_prompt` made the
  literal gate exit 1 with `template_manager.py:324: timeframe literal '4h'` and failed
  `test_the_system_prompt_carries_the_configured_frame_and_no_other`; reverting returned both green.
- `pytest tests/ -q` → 2 619 passed, 0 failed, 0 skipped; `ruff check src start.py scripts` and
  `pyright src start.py` clean; `check_loose_helpers.py` and `check_agents_md.py` clean.

## 2026-10-27 — A broken Chrome pipe no longer costs the chart

### Changed
- **`src/analyzer/pattern_engine/chart_generator.py`**: `_image_export_with_timeout()` keeps Kaleido
  as the renderer and adds a second one. When Kaleido raises (or times out), PNG exports are
  re-rendered by `_render_with_playwright()` — the same Plotly figure, `fig.to_html` into
  Playwright's bundled Chromium, screenshot at the requested viewport and scale. The fallback logs
  a warning naming the Kaleido error, and a non-PNG format never falls back. `playwright` is an
  optional import at module level (`sync_playwright = None` when absent) because it is a dev
  dependency: without it the Kaleido error surfaces unchanged.
- **`requirements-dev.txt`** declares `playwright==1.61.0` (it was installed but undeclared; the
  admin-UI integration test and this fallback both need `python -m playwright install chromium`).

### Why
Kaleido 1.1.0 drives the **system** Chrome over a Windows debugging pipe
(`--remote-debugging-pipe --remote-debugging-io-pipes=<handles>`). On this box every launch now
exits before the DevTools handshake, so the export fails on all three attempts: at 09:03 today the
cycle ran without its chart image and the model decided from numbers alone while the bot looked
healthy. It is not new: single retry failures were logged on 22 Jul, 19 Aug, 31 Aug, 10 Sep,
17 Sep, 2 Oct, and at 05:03 today (that one recovered on retry). Chrome was the same 156 build in
both cases, and Chrome for Testing 142 and Playwright's Chromium 149 fail the same way, so the
renderer, not the browser version, is what needed a second path.

### Verified
- `ChartGenerator.create_chart_image()` on synthetic candles produces a 1920×1080 PNG in ~5 s with
  Kaleido failing first: title, min/max annotations, candles and the volume panel render, so the
  vision input is intact.
- Three new tests pin the fallback (renderer called with the export dimensions), the missing-
  Playwright path (the Kaleido error is what surfaces) and the non-PNG path (no fallback).
- `pytest tests -q` → 2 593 passed, 0 failed, 0 skipped; `ruff`, `pyright` and the helper gate clean.

## 2026-10-27 — An anti-pattern rule stops being a tautology

### Changed
- **`src/trading/brain_reflection.py`**: `trigger_loss_reflection()` groups losses on
  `direction|regime` only. The close reason and the exit profile used to be part of the key,
  which selected trades *by* the outcome the rule then described: a group of trades that left
  through a stop loss could not contain a win, so "0 wins, 0% win rate, loss rate 100%" was
  guaranteed by construction rather than observed. A group now needs at least five losses and a
  win rate below the closed-trade baseline; `rule_anti_*` additionally requires
  `TradePatternAnalyzer.is_significantly_worse()` (one-sided exact binomial, α = 0.05, baseline
  of at least ten closed trades), otherwise the rule is stored as `rule_corrective_*`. The rule
  text carries `n` and the baseline, so a thin sample is visible in the prompt.
- **`trigger_ai_mistake_reflection()`** keys on `mistake_type|confidence`, never on the
  direction or the regime, and its "win rate in comparable trades" is computed over every
  closed trade at that confidence instead of over the same losing cluster that produced the rule.
- **`src/trading/brain_patterns.py`**: `is_significantly_worse()`; `compute_group_metrics()`
  exposes `avg_loss_mfe` and `build_rule_metadata()` separates `avg_loss_mfe_pct` (losers) from
  `avg_mfe_pct` (winners), which used to read as two different numbers for the same fact; a zero
  `confluence_count` counts as unmeasured, so "weak entry confluence" only fires on real values.
- **`src/trading/vector_memory_rules.py`**: `deactivate_semantic_rule()` retires a rule with
  `active: false` plus a reason, keeping its text and counters (no read path sees it).
- **`scripts/brain_rules.py`**: `list`, `check` and `deactivate` for a brain directory. `check`
  prints the exact block a prompt would receive, which shows what a retirement changes before it
  is relied on.

### Why
The live store carried `rule_anti_pattern_long_bullish_stop_loss_sl_hard_15m_tp_hard_15m`
("2 losses, 0 wins — 0% win rate") and a matching AI-mistake rule, both built from the same
three trades, injected into every LONG-in-BULLISH decision at 65% similarity. Without the
exit-reason filter the same history is five LONG-in-BULLISH trades: 1 win / 4 losses, one of
them a +4.07% take-profit the rule would have banned. Its "weak entry confluence (avg 2 factors)"
averaged a `confluence_count` of 0 (a field not yet recorded on 20 Sep) with one of 5.

### Verified
- `pytest tests -q` → 2 544 passed, 0 failed, 0 skipped; `ruff check src scripts start.py`
  clean; `pyright src start.py` 0 errors; `python scripts/check_loose_helpers.py` reports no
  violations.
- Regression pin: `test_loss_reflection_stays_silent_on_the_2026_10_history` replays the ten
  closed trades that produced the rule and asserts the pass stores nothing.
- A dry run of all three passes on a copy of `data/trading/brain_BTC_USDC_4h` wrote no
  anti-pattern rule and re-derived the AI-mistake rule without a direction claim
  (`(4 mistake(s), 50% win rate in comparable trades)`).
- Both live rules were retired with `scripts/brain_rules.py deactivate`; `check` then printed
  an empty prompt block.

# Changelog

## 2026-11-12 — A superseded number that rounds to the current one is not stale

### Fixed
- **`check_claims.py` no longer fails when the reformatted value collides with a superseded one.** The
  ledger moved from $7.1156 to $7.1222 of inference cost, and both print as `$7.12`: the gate read its own
  current number as a stale one and blocked the publication. The token builder is now shared
  (`entry_tokens`), and `check_texts` skips a superseded token that the current ledger also produces.

### Verification
- `pytest tests/domain_infra/test_claims_check.py -q`: 6 passed, including the new
  `test_a_superseded_value_that_rounds_to_the_current_one_is_not_stale`.
- `python scripts/check_claims.py` on the real tree: consistent, exit 0, with claims at 7.1222 and the
  superseded 7.1156 still in the file.
## 2026-10-27 — A helper belongs to the class that owns it

### Changed
- **`src/utils/decorators.py`** no longer reads as classes interleaved with loose functions:
  `retry_async`/`retry_api_call` and their `_RetryContext`/`_ApiRetryContext` are grouped, and
  the four reply-shape helpers (`_first_choice`, `_finish_reason`, `_carries_no_payload`,
  `_error_label`) plus `_should_retry_api_error` are static methods of `_ApiRetryContext`.
  `_classify_retryable_error`/`_is_exchange_rate_limit_error` moved onto `_RetryContext`;
  `_add_jitter` stays module-level because both contexts use it.
- **Single-class helpers absorbed everywhere else**: `SerializableMixin` owns
  `_get_dataclass_field_meta`/`_dataclass_dict_factory`, `WritableConfig` owns
  `_validate_and_coerce`, `Logger` owns `_resolve_default_log_dir`/`_write_fallback_crash`,
  `AdminAuthMiddleware` owns `_is_lan_ip`/`_get_real_client_ip`, `DashboardServer` owns the ten
  cache/concurrency/rate-limit helpers and their constants, `PositionStatusMonitor` owns its two
  policy defaults, and `OrderLifecycle.can_transition_to()` replaces the free `can_transition()`.
- **Three new classes for families that had none**: `NewsBoilerplateCleaner` (seven publisher-chrome
  cleaners behind `strip_news_boilerplate`/`to_article_schema`), `ArticleBodyTextExtractor`
  (BeautifulSoup → tag parser → strip, behind `extract_html_body_text`) and
  `TradeHistoryReader(db_path)`, which now holds the database path instead of passing it to every
  free function; the module-level MCP tool registration stays a framework entry point.
- **`ExitRecords`** replaces sixteen free functions in `src/trading/executor_reconciliation.py`
  (`verify_exit_record` → `ExitRecords.verify`, `exit_record_from_receipt` → `ExitRecords.from_receipt`,
  `exit_event_identity` → `ExitRecords.identity`, `exit_evidence_is_learnable` → `ExitRecords.is_learnable`,
  `fill_divergence_reason` → `ExitRecords.divergence_reason`,
  `market_verification_gap_reason` → `ExitRecords.verification_gap_reason`,
  `exit_fill_is_divergent` → `ExitRecords.fill_is_divergent`,
  `fill_verification_status` → `ExitRecords.verification_status`); `close_fill_is_divergent` keeps its
  name. Call sites in `brain.py`, `trading_strategy.py` and four test modules were updated with it.
- **`scripts/check_loose_helpers.py`** is the gate that keeps it that way: it fails on a private
  helper only one class calls, on a `def` between two `class` definitions, and skips `numba`
  kernels and helpers another module imports (its `ALLOWLIST` is empty). AGENTS.md §10 gained
  *Classes before free functions* — the rule stated the intent before; now it names the exceptions
  and a machine check. `tests/domain_infra/test_loose_helpers.py` runs the same check inside the
  suite, so CI fails on the next loose helper without touching `.github/workflows/`.

### Verified
- `pytest tests -q -m "not integration" --timeout=120` → 2495 passed, 34 deselected, 0 failed;
  `ruff check src scripts start.py` clean, `pyright src start.py` 0 errors,
  `python scripts/check_loose_helpers.py` 0 violations, `python scripts/check_agents_md.py` clean.
- The count that started this: 75 module-level private helpers with in-module-only use — 22 numba
  kernels stay functions by necessity, 40 were absorbed into a class, 13 are already called by more
  than one function and stay free (`_add_jitter`, `_reference_window_seconds`, `_cached_zone`,
  `_logger_debug_enabled`, `_clip`, `_is_number`, `_positive`, `_parse_ts`, `_finite_non_negative`,
  `_normalize_title`, `_classify_st_dir`, `_normalize_exit_execution_value`, `_derive_password_hash`,
  `_verify_password` — the last two serve the module's password API, not the middleware).

## 2026-10-27 — A rejected reply says why it was rejected

### Changed
- **`src/utils/decorators.py`**: a JSON-mode reply that carries no usable payload is now
  diagnosed instead of merely rejected. `_rejection_reason()` names the shape — `reply carried
  no choices`, `empty reply`, `prose without a JSON object`, `object without a payload key`,
  `truncated at the output cap`, `reply stopped by the content filter`, `malformed JSON (...)`
  with the decoder message and position — and `retry_reason` carries it into the retry line.
  `finish_reason` (`ChoiceModel.finish_reason`, already parsed by every provider client) is what
  separates a reply cut by the output cap from one the model ended with broken JSON: the parse
  error alone cannot tell them apart.
- **A rejected reply is logged in full.** The old line kept `content[:120]`, which made a cut-off
  JSON look valid and left nothing to replay. The rejected text now goes to `Bot.log` on its own
  line, next to the accepted replies that `ModelManager` already logs in full.
- **`API returned error` is gone.** A content rejection is not an API error; the retry line now
  reads `Retrying model X in 1.00s (retry 1 of 3) — <reason>` for both classes, and every retry
  names a reason. A stale reason no longer leaks into a later attempt (`retry_reason` is reset per
  evaluation, `last_rejection_reason` keeps the one the recovery line quotes).
- **A recovered reply leaves an INFO line**: `Reply from model X accepted on attempt 2 of 4 after
  1 retry (last reason: …)`. Previously the warning had no matching "and it worked" entry.
- **`src/platforms/ai_providers/deepseek.py`**: `_log_usage` appends `finish_reason=…` to the token
  breakdown, so a reply that is short but survives the parser still shows whether it was cut.

### Verified
- Live evidence behind the change (`logs/Bot/2026_10_27/Bot.log:1411-1416`): the chart call returned
  14 433 completion tokens (11 238 of them reasoning) with a JSON that opened but never closed; the
  unchanged request succeeded 57 s later (10 109 completion, 6 595 reasoning), the cycle placed no
  order and reported `HOLD`. Four such rejections exist across the 2026-07-13 → 10-27 logs, all
  recovered on the first retry; the discarded attempt (~$0.0138 off-peak) is not part of
  `Cycle inference cost`, so the number to trust is the token breakdown, not the cycle total.
- `pytest tests -q -m "not integration" --timeout=120` → 0 failed; `ruff check src scripts start.py`
  and `pyright src start.py` clean. New tests pin every reason code, the SDK-shaped reply, the
  decoder position, the full-content log, the retry wording, the cleared stale reason and the
  recovery line (and its silence on a first-attempt success).

## 2026-10-27 — The site leaves this repository

### Changed
- **The Astro site is now its own repository**: [qrak/semantic-signal-site](https://github.com/qrak/semantic-signal-site)
  (public, history of all 38 `website/` commits preserved through `git subtree split`). The site and
  the dashboard have different release rhythms — static HTML deployed by hand to Cloudflare Pages vs a
  FastAPI process holding runtime state — and keeping them in one tree meant two build outputs in one
  CI run and a copy of the site inside this repo only for `/story`. This repository now ships the
  engine, the dashboard and the indicators; the site is documented in its own README.
- `src/dashboard/server.py`: `/story` answers **`301` to `https://qrak.org/story/`** instead of serving
  `website/dist/story/index.html` (no duplicated copy, no "run npm run build" fallback). `/landing`
  stays — that page is the dashboard's own.
- `.github/workflows/tests.yml`: the `website build` job is gone; `scripts/check_agents_md.py` no
  longer treats `website/` as a repository path; `.gitignore` lost the Astro entries.
- Repository ruleset **"master: green before merge"**: `website build` removed from the required
  status checks (leaving `python checks`, `Analyze (python)`, `Analyze (javascript-typescript)`). A
  required context whose workflow no longer exists would have kept `master` unmergeable.
- `README.md`, `docs/SETUP.md`, `AGENTS.md`, `CONTRIBUTING.md`: point at the new repository and at the
  `301` instead of naming `website/`.

### Fixed
- **Articles rewritten for tone and evidence.** `chart-vision`, `vector-memory` and
  `executor-separation` lost the comparisons with other projects, the "brain"/"conviction" framing and
  the self-assessment, and gained what the code actually enforces, the file behind each claim, and a
  closing **"What this does not prove"**. Two pages added: `/articles/measurement/` (the paper ledger,
  inference cost, benchmark and the zero-fee gap) and `/articles/offline-replay/` (55 recorded setups
  replayed through two prompt versions and two model generations — no outcomes scored, so it cannot
  rank them).
- `website/public/llms.txt`: the home-page line still described a comparison with Freqtrade /
  TradingAgents / Vibe-Trading that the site had already dropped; it now describes the pages that exist.
  `sitemap.xml` and `project.json` list the two new pages.

## 2026-10-26 — The README shows the product before explaining it, and the site stops quoting a stale suite

### Changed
- `README.md` rebuilt: a dashboard screenshot and a short GIF at the top, a nine-language switcher
  (`readme-i18n.com` renders and re-translates on every README change, so no translated files can
  rot), a **Results** section with the live paper-trading ledger and the buy-and-hold benchmark, and
  a feature table. The long-form sections were **moved, not deleted**. There is deliberately no
  "News" block (a release feed belongs in this file) and no per-project comparison table.
- New `docs/`: `FEATURES.md` (every feature with its implementing file), `SETUP.md` (requirements,
  platform scripts, secrets, website deploy), `CONFIGURATION.md` (every config key, prompt budget,
  billing windows, executor handoff), `RESULTS.md` (ledger, benchmark, caveats, reproduction).
- `scripts/results_report.py` recomputes the ledger, inference cost, cycle statistics and the BTC
  buy-and-hold benchmark from `data/trading/` plus one public Binance klines call; `docs/RESULTS.md`
  is its output, so a reader can re-run the page.
- `scripts/check_agents_md.py` fails when a repository path quoted in an `AGENTS.md` file no longer
  exists. It currently checks 83 references across the 9 files.
- Root `AGENTS.md` was verified against the code rather than trusted: the chart is **1920×1080**,
  not 4K (three claims, two files), the suite is **52** `test_*.py` files, not 48, and a
  "Documentation map" plus a drift gate were added. Everything else in the 361 lines checked out,
  including the 79 Numba kernels, the six REST routers, `ai_chart_candle_limit = 125` and the
  9 documented `BLE001` hits.
- `qrak.org` (Astro) and the legacy dashboard landing (`semanticsignal.qrak.org/landing`) now quote
  **2,514 tests measured 2026-10-26** instead of 2,419; `website/public/project.json` and `llms.txt`
  were synced with the same number.
- `.github/workflows/tests.yml` lints `start.py` too: `README.md` and `AGENTS.md` had documented
  `ruff check src scripts start.py` for months while CI only ran `ruff check src scripts`.

### Verified
- `pytest tests -q -m "not integration" --timeout=120`: **2,480 passed, 34 deselected in 143.91 s**
  (2,514 collected in total).
- `ruff check src scripts start.py`: clean. `pyright src start.py`: 0 errors.
- `python scripts/check_agents_md.py`: 83 references, all present.
- `python scripts/results_report.py`: 12 closed trades, +0.14%, BTC/USDC −1.83% over the same window.
- `npm run build` in `website/`: 11 pages plus `dist/llms-full.txt`.

### Notes
- The results section is deliberately unflattering: +0.14% against a −1.83% underlying is a two
  percentage-point difference over five weeks on 12 trades, the paper book records zero fees, and at
  a 0.15% round trip the gain would roughly disappear. `docs/RESULTS.md` says so in the same words.
- No application code, config key or test changed; the only Python changes are the two new scripts.

## 2026-10-26 — CI now enforces what the repository already claimed, and `master` is gated

### Changed
- `.github/workflows/tests.yml` runs **pyright** after ruff. Every changelog entry in this file quotes a "pyright 0 errors" result, but no workflow executed it; the gate lived only in the habit of whoever ran the command. `requirements-dev.txt` pins `pyright==1.1.414` (the current PyPI release) for the same reason `ruff` is pinned: a new release turns new rules on and would fail CI on code nobody touched. The first pin of this change was `1.1.411`, one release stale on the day it was written — the pin is a decision, not a snapshot of whatever pip happened to install.
- Action versions re-checked against `releases/latest` while adding the jobs: `actions/setup-python` was one major behind and moved `v6` → `v7` (`v7.1.0`, whose v7.0 removed the `pip-install` input this repository never used); `actions/checkout@v7` (v7.0.1), `actions/setup-node@v7` (v7.1.0), `actions/dependency-review-action@v5` (v5.0.0) and `github/codeql-action@v4` (the only major tag) were already current. There is no `dependabot.yml` in the repository, so no automation watches these — action majors have to be re-checked by hand.
- `.github/workflows/tests.yml` gains a **website** job: `npm ci && npm run build` in `website/`. Dependabot bumps `astro`, `vite`, `zod` and `undici` every few days, and nothing verified the site still built — CodeQL parses JavaScript but never runs Astro.
- `.github/workflows/tests.yml` gains `concurrency` keyed on the ref: a superseded pull request run is cancelled, a `master` run never is, so every green push still produces exactly one build (the release workflow keys off a completed run).
- `.github/workflows/dependency-review.yml` fails a pull request that introduces a high-severity dependency. Dependabot repairs what is already known to be vulnerable; this stops a new one from arriving.
- Ruleset **"master: green before merge"** (repository setting, not a file): `deletion` and `non_fast_forward` are blocked, and a pull request merge requires `python checks`, `website build`, `Analyze (python)` and `Analyze (javascript-typescript)`. Repository administrators bypass, so direct commits to `master` keep working; `CONTRIBUTING.md` documents the rule and the trap that a required context which never reports makes the branch permanently unmergeable.
- Secret scanning and push protection were already enabled on this public repository (`keys.env` is ignored and untracked); the optional `secret_scanning_non_provider_patterns` and `secret_scanning_validity_checks` return `200` from the API but stay `disabled`, so they were left alone rather than claimed.

### Verified
- Run `40390670598` on `26fccd4`: `python checks` green in 7 min (pyright inside it), `website build` green in 20 s, CodeQL `Analyze (python)` and `Analyze (javascript-typescript)` green. The same push produced its `build release` run through `workflow_run`.
- Local before pushing: `pyright --pythonpath … src start.py` → 0 errors in 13 s; `npm ci && npm run build` → 11 pages in 20 s + 10 s on a warm tree.
- Ruleset read back from the API with the four contexts and `bypass_mode: always` for the administrator role; the documentation-only commit after it was pushed to `master` under the ruleset, which is what proved the bypass works.
- Run `40394293136` on `a5d628c`: `python checks` green with `actions/setup-python@v7` (the log shows the download of `SHA 6d4fd22…`) and `pyright 1.1.414` — the "new pyright version available" line the earlier pin printed is gone, so the pin now tracks PyPI. A pull request with a planted type error (`mergeStateStatus=BLOCKED`, `Type check` red, everything else green) proved the ruleset blocks on a red check instead of only reporting it.

## 2026-10-26 — Releases stop being a second, hand-written history

### Changed
- `.github/workflows/release.yml` publishes builds. After a green `tests` run on `master` it writes an annotated `build-YYYY.MM.DD-HHMM` (UTC) tag and updates one rolling "Latest build" release whose body is generated from the commits, then deletes the older `build-*` release **entries** while keeping their tags. The page therefore shows exactly one current build, and `git checkout build-<stamp>` still works as a rollback point — the role `pre-hold-fix-2026-09-18` played by hand.
- Hand-maintained semver releases stop being a cadence; `v1.x` is milestone-only. Nothing reads a version: no `__version__`, no release artifact, no download URL, and no page on `website/` links to Releases. The four-month `v1.0` → `v1.1.5` train was prose maintained for a consumer that does not exist, and it aged within days — `v1.1.5` was already 14 commits and 8 changelog entries behind when this landed.
- `CONTRIBUTING.md` documents the model under "Branches, Versions and Releases" (single trunk, automatic releases, semver reserved for milestones) and drops the stale "create a feature branch from `main`" line: the branch is `master`.
- Deleted `develop` and `experimental`. Both were fully merged into `master` (0 commits unique to either), so the repository keeps one long-lived branch; a `nightly` branch was rejected on purpose, since it would duplicate every merge for a deploy that is `git pull`.

### Verified
- Both triggers ran end to end on the public repo. The manual dispatch published `build-2026.10.26-0228`; the green `tests` run on the same commit fired the `workflow_run` path, published `build-2026.10.26-0233`, deleted the `-0228` **release entry** and kept its tag — the Releases page then held exactly one build, and both stamps stayed checkout-able.
- No application code, test, config, dependency or runtime file changed: the set is the new workflow, `CONTRIBUTING.md` and this entry.

### Fixed
- GitHub's own generated notes list merged pull requests only: the second automated build (`build-2026.10.26-0239`, a direct push) published a "Latest build" with an empty "What's Changed" and nothing but a compare link. The body is now composed from `git log --no-merges` over the range from the previous build tag (40 commits max, `- %s (%h)`) plus the compare link, so a direct push still shows what changed and the notes are reproducible from the repository without the Release Notes API.

## 2026-10-26 — Inference cost enters the cycle journal and the EV threshold

### Changed
- Every analysis cycle now snapshots `CostStorage` before and after its LLM work and appends `inference_cost_usd`, `prompt_tokens`, `completion_tokens`, `provider` and `model` to `data/trading/metrics/cycles.jsonl`. The delta covers the chart/text analysis, the contract retry and an awaited post-mortem; a missing snapshot writes no cost fields, and a counter reset clamps to zero.
- `GET /api/performance/cycles` gains `inference_cost {analyses, total_usd, per_analysis_usd, coverage}`; existing keys are unchanged, and journal lines written before cost tracking lower `coverage` instead of failing the endpoint.
- The EV prompt now adds the rolling 20-analysis inference cost to the round-trip fee: step 5 subtracts it, `Net break-even` is `1.5 × (fee + inference)`, and the threshold sentence names both. With no measured cost the section is byte-identical to the previous fee-only version.
- `scripts/cost_report.py` prints gross P&L from `statistics.json`, recorded inference cost, net P&L, cost per analysis and cost per closed trade, with the overlapping-period caveat.
- The `cycle_metrics.py` "cost deliberately absent" TODO is closed: the journal is schemaless and now carries measured cost instead of a made-up number.

### Verified
- New tests cover positive, zero and negative inference cost plus NaN/Inf, malformed journal values, cost-aware EV output, the app's snapshot delta/rolling average, the API payload and the standalone report.
- Full suite: **2514 passed, 0 failed, 0 skipped** (`pytest tests -q` with `--basetemp` outside the repo because the default Windows temp ACL denied `tmp_path`; 155 s).
- `ruff check src scripts start.py` clean; `pyright src start.py` **0 errors**.
- Manual report on the current journal: 42 cycles, 0 analyses with cost (the old lines predate this change), gross/net `$8.54`, `cost per analysis: n/a`.
- No bot or executor process was started, restarted or signalled: the live one-cycle smoke (new `cycles.jsonl` fields and the API payload) needs the next normal restart.

## 2026-10-25 — Every timestamp names its clock; the bot core stops duck-typing itself

### Fixed
- A log line carried a bare local wall time while its day folder was named after the UTC date. Between 00:00 and 02:00 local the line landed in the previous day's folder, and on the autumn switch `02:04:15` occurred twice in the same file with nothing to tell the two hours apart. Lines now print the display zone **with its UTC offset** (`[25.10.2026 02:04:15 +02:00]`) and `logs/Bot/YYYY_MM_DD` is the same day as the lines inside it; `errors.log` and the last-resort `_write_fallback_crash` writer use the identical stamp.
- `Trading Check #N at 2026-10-25 00:00:01` printed UTC without saying so and sat next to a local-time prefix, so one line showed two clocks. It now reads `2026-10-25 00:00:01 UTC / 2026-10-25 02:00:01 CEST`.
- Time fields that lost their zone: the Discord decision footer and the console position status (`Time: …`, `Entry Time: …`), the last-analysis line fed into the prompt, the recent-history lines (`- [2026-04-14 20:00 UTC] CLOSE_LONG …`) and the chart data-range log now all carry `UTC`.
- The executor handoff sent `timestamp` as `strftime("%Y-%m-%dT%H:%M:%S.%f")`, dropping the offset; the executor only read it correctly because it guesses "naive means UTC". The payload now uses `datetime.isoformat()` (`…+00:00`), which the executor's `fromisoformat` parses unchanged — the file is self-describing for any other consumer.
- Six `datetime.fromtimestamp()` calls without a zone (format utils ×2, fear-and-greed collection ×2, chart range ×2) used the OS clock; on a host in a negative-offset zone the date shifted by a day. All are `tz=timezone.utc`, so the prompt, the chart label and the log agree with the candle timestamps.
- `BOT_INTENT_JOURNAL_PATH` existed only in test doubles: production silently fell back to `data/trading/bot_position_intents.jsonl` through a `getattr` probe. It is a real `[executor_api] intent_journal_path` key with that default, so the journal location is configured, not guessed.
- `PersistenceManager` wrote positions/statistics to a hardcoded `data/trading` while the trade DB honoured `[directories] data_dir`; a custom data directory split one deployment's state across two trees. Both follow `DATA_DIR` now.

### Changed
- `[general] timezone` (default empty = OS zone, shipped as `Europe/Warsaw`) drives every human-readable timestamp through the new `src/utils/time_utils.py`: `local_timezone()`, `local_stamp()`, `day_stamp()`, `utc_stamp()`, `utc_and_local()`. The name is validated at startup, so a typo fails on boot instead of on the first log line; `tzdata` joined `requirements.txt` because `ZoneInfo` has no IANA database on Windows without it. Trading arithmetic, TTLs and persistence remain on UTC epoch/aware-UTC datetimes — the switch changes labels only, never a candle boundary.
- `BotServices` now types `config: Config`, `market_analyzer: AnalysisEngine` and `trading_strategy: TradingStrategy` instead of `Any`. That type information is what exposed the remaining probes: pyright checks every `self.config.*` and strategy call, and the module reaches **0 errors** without a single `getattr`.
- Removed duck typing on the strategy's own contract: `take_pending_local_close_decision()`, `reconcile_local_position()`, `resolve_position_intents_after_forward()` (including the `position_intent_state()` fallback for "a strategy without the resolver"), `take_unconfirmed_intent_alert()`, `take_rejected_intent_alert()`, `position_snapshot_token()`, `protection_snapshot()` and `protection_loosening_reason()` are called directly. `inspect.isawaitable`, `callable()` probes and `isinstance(..., str)` guards around their results are gone; a double now implements the method instead of the caller guessing whether it exists. `PositionStatusMonitor` follows the same contract (`take_local_exit_request()`, `take_executor_side_exit_reason()`, the fields of the reconciliation outcome) and `sync_position_state()` returns `LocalPositionReconciliation | None` instead of `Any`.
- `executor_reconciliation` no longer lazily creates its own state through `getattr(self, "_x")`: the transition lock, the booked-exit set, the intent ledger and the two one-shot alerts are declared and initialised in `TradingStrategy.__init__`. The ledger stays lazy (it replays its journal on first use) but is typed `PositionIntentLedger | None`, and the first caller can no longer observe a half-built lock.
- Config probes replaced with direct reads in `app.py` (approval gate), `executor_reconciliation.py` (executor endpoint, exit journal), `provisioners.py` (circuit breaker) and `exchange_manager.py` (sandbox); `make_config()` carries the production defaults for all of them, so a missing key fails in the loader rather than mid-cycle. The dead `except (AttributeError, TypeError, KeyError)` around `result["analysis"]` and the duplicated exit-path default in `_executor_exit_path()` are gone as well.
- Tests that encoded the removed defences were rewritten rather than kept alive behind `getattr`: a MagicMock returned as a "reason", a strategy without the resolver, and a UTC-named log folder.
- The interactive rich console keeps the OS clock (it renders only `HH:MM:SS`); every persisted line and every notification is now unambiguous on its own.

### Verified
- 16 new tests: IANA offsets for summer/winter/half-hour/negative/UTC, the repeated autumn `02:30` hour staying distinguishable, a display-day folder that differs from the UTC date, the offset on a log line and on a crash line, naive input read as UTC, `utc_stamp(None)`, `[general] timezone` validation and the intent-journal default following `DATA_DIR`.
- Full suite: **2489 passed, 0 failed, 0 skipped** (`pytest tests -q`, 162 s); `ruff check src start.py scripts` clean; `pyright src start.py` 0 errors.
- No bot or executor process was started, restarted or signalled.

## 2026-10-25 — Macro data outlives a mislabeled Content-Type; a long Reddit reason no longer discards the assessment

### Fixed
- DefiLlama serves `/v2/chains` as valid JSON under `text/html; charset=utf-8`. `aiohttp.ClientResponse.json()` refuses that mimetype, so every macro refresh raised `Error fetching chain TVL: 200, message='Attempt to decode JSON with unexpected mimetype: text/html; charset=utf-8'`, the chain list came back empty, `get_macro_overview()` returned `None` and the cycle lost both macro and DeFi fundamentals (`Failed to fetch macro data: None`). `DefiLlamaClient._fetch_json()` now reads and decodes the body itself whatever the header says — the live endpoint yields **421 chains** and **$89.5B** TVL instead of an error pair per refresh.
- The same helper names the real failure instead of repeating the aiohttp message: `DefiLlama <url> answered HTTP <status>`, `<url> did not return JSON` with a 160-character body preview, or `<url> returned dict, expected list` when a healthy status carries the wrong JSON container; a missing snapshot logs `Macro overview unavailable: stablecoins=<n> chains=<n>` and then `Macro data unavailable, so DeFi fundamentals are skipped`, so a bare `None` no longer stands in for a diagnosis.
- A Reddit assessment whose `reason` ran past 240 characters was discarded whole (`Discarding invalid Reddit assessment: String should have at most 240 characters`) and the response contract was marked invalid with it, although `outlook` and the cited title IDs were usable. `RedditAssessmentModel` now clamps an over-long reason to the documented ceiling at a word boundary before validation — the same tolerance the evidence-ID coercion already applies — so the assessment survives and still feeds entry-time learning.

### Changed
- DefiLlama payloads are shape-checked once: `_fetch_json(url, expected_type)` owns the request, decode and container check, and `_records()` keeps the object members of an array. The five getters no longer repeat the same `isinstance` (19 checks in the module became 8), and a JSON object where an array belongs no longer surfaces as a crash inside a comprehension.
- Removed defensive branches that type checking already covers: the `except (ImportError, AttributeError, Exception)` tuple in the shutdown manager, `except AttributeError` around `Position.quote_amount` in the Discord and console notifiers, `getattr(guard, "name", …)` in the guard pipeline, `getattr(parser, "extract_json_block", None)` in the post-mortem service (the parser is now typed `UnifiedParser`), an attribute probe in the visuals router, and `getattr`/`except AttributeError` fallbacks for `FILE_MESSAGE_EXPIRY`, `MAX_POSITION_SIZE`, `DATA_DIR` and `TIMEFRAME` in the file notifier, the prompt template and the brain router. A missing config attribute now fails where it is read instead of silently falling back to `"data"`/`"unknown"`.

### Verified
- Live probe against the real endpoints: 421 chains, macro TVL `89,508,933,463.80`, no error line.
- 12 new tests: a JSON body served as `text/html`, a complete fundamentals snapshot, an HTML body, a non-200 status, the wrong JSON container, a mixed array, naming the empty source, the reason clamp (single token and word boundary), the parser contract staying valid, and the autumn DST switch (25.10.2026 Europe/Warsaw: 4 h candle bounds stay on 04:00/08:00 UTC and the label resolves to 05:00 CET).
- Full suite: **2473 passed, 0 failed, 0 skipped** (`pytest tests -q`, 158 s); `ruff check src start.py scripts` clean; `pyright src start.py` 0 errors.
- No bot or executor process was started, restarted or signalled: both keep running the previous code until the next restart.

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


## 2026-10-16 — Pattern states: durations, explicit labels, canonical ATR, prompt legend

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
- **`/story` rebuilt from `git log`.** Timeline dates are now the real ones: `chart_generator.py` 2025-12-22 (`git log --diff-filter=A`), `brain_experience` / `brain_context` / `brain_reflection` 2026-05-16, `.ai/` 2026-07-26, executor integration 2026-08-14, provider consolidation 2026-09-12. Added a status section (paper capital, testnet executor last exercised 2026-08-15, profitability not established) and a corrections section listing earlier claims that no longer matched the code.
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
- **Prompt accuracy pass** (`template_manager.py`, `ev_formatter.py`): the prompt no longer calls R/R the only hard gate, documents the executor's SL/TP normalization (SL expanded to ≥1%, clamped to ≤10%, TP clamped to ≤50%, R/R recomputed from the corrected levels), labels the historical winning-average R/R and SL figures as guidance rather than caps, and points the PRE-FLIGHT CHECKLIST at Decision Rules instead of a stale section. The ADX trend threshold is read from the brain instead of a hardcoded 25.

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
