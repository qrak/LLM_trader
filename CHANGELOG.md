# Changelog

## 2026-11-26 — The news cache stops storing every text twice

`crypto_news.json` kept `title_lower`, `body_lower`, `categories_lower`, `tags_lower` and
`detected_coins_str_lower` next to the originals, written once in `NewsManager._normalize` and
read only by `ContextBuilder._extract_article_content`, which already carried a `.lower()`
fallback. The copies were 45 % of a 376 KB file: 65, 75 and 71 archived articles showed no
divergence between the copies and `.lower()` of their originals, so the cache was paying disk for
a value it could compute.

- `_normalize` keeps only the source-name normalisation; the five copies are gone, and
  `load_cached_news` no longer backfills them on startup.
- `_extract_article_content` lowercases the originals directly, so an article written before this
  change cannot rank from a stale copy.
- No migration: articles already on disk keep their unused keys and age out inside the existing
  window, and `AGENTS.md` forbids startup rehydration loops.
- Ranking is unchanged bit for bit — the same order and every `ArticleScore` field across three
  archived windows and three query states. Cost: `0.6 ms` of `lower()` per cycle.

## 2026-11-26 — Two journals answer "why", not only "what"

The retrieval pipeline scored every article, kept five and dropped the rest, so the ranking was a
number without a reason. The 05:00 cycle printed the same news since 01:00 while newer Bitcoin
articles sat in the cache, and nothing on disk separated a conservative ranking from a broken one.

- `ArticleScoringPolicy.calculate_article_relevance` returns an `ArticleScore` — keyword, category,
  coin, importance, density and coin multipliers, recency, source weight — instead of a bare float.
- With `logger_debug` on, one `RAG ranking` line per retrieval carries the stage counts (deduped,
  pool, shortlist, rendered) and up to 25 articles: ten rendered in order, ten best-scored that did
  not fit, five that scored zero. Each entry names its `status` — `rendered`,
  `shortlisted_not_rendered`, `not_shortlisted`, `outside_pool`, `story_duplicate`, `backfill` or
  `nonpositive` — so a missing story is a stage, not a mystery.
- The line is tagged `skip_live_stream`: it lands in `Bot.log`, never in the dashboard console, and
  carries ids and titles only, no bodies. With DEBUG off the selection is unchanged and nothing is
  collected.

`metrics/cycles.jsonl` recorded the signal and the cost of 230 cycles, not the judgement behind
them: `confidence` and `confluence_factors` existed only in `trade_history.db` and only for the 23
trades that closed.

- `_record_cycle_metric` writes both for every analysed cycle, HOLD included, as the values after
  parsing and the pattern-quality overwrite — model opinions, never a calibrated `P(win)`.
- `scripts/snapshot_results.py` drops those two fields from the frozen copy, so the public ledger
  keeps the shape it already had. The live journal is untouched, and the whole journal is validated
  before the target directory exists, so a broken line stops the snapshot instead of leaving a
  directory that blocks the retry.

## 2026-11-26 — The file log survives midnight again

`DailyRotatingFileHandler` subclassed `TimedRotatingFileHandler` while switching the day directory
itself in `emit()`, so the parent's `doRollover()` fired for the same midnight and the custom
`rotator` replaced the standard rename with "create an empty file in today's directory". Measured:
`logs/Bot/2026_11_26/Bot.log` stayed at 0 bytes from 00:01:19, the last line of any `Bot.log` was
23:31:19 the day before, and the 01:00 and 05:00 cycles left no file lines at all.

- The handler now extends `logging.FileHandler`: `emit()` keeps the day-directory logic and nothing
  else rotates, so one mechanism owns the switch. `namer`, `rotator` and `_log_rotator` are gone.
- Day directories older than 30 days are pruned on the switch. `backupCount=30` never did that: the
  standard library prunes files inside one directory, not per-day directories, so `logs/Bot/` had
  kept every day since the first run.
- `tests/domain_infra/test_logger.py` covers the midnight switch, the previous day keeping its
  lines, the absence of a rotated copy, the retention sweep and the ERROR-only handler.

## 2026-11-26 — The system prompt is saved per cycle

`previous_response.json` holds the user prompt only, so an audit could not check what the model was
told in the system prompt — the EV framework, the Brain rules and `## PREVIOUS ANALYSIS CONTEXT`.
That gap produced two wrong "the model invented this number" reports in two days: the `$1.21`
round-trip fee and the `60662.90` prior resistance were both verbatim quotes from the system prompt.

- `PersistenceManager.save_system_prompt` writes `data/trading/system_prompt.json` once per cycle,
  next to `previous_response.json`, through the same `atomic_write_json`.
- `AnalysisEngine` exposes `last_generated_system_prompt` and adds it to the result as
  `generated_system_prompt`; `CryptoTradingBot._save_analysis_data` persists it.

## 2026-11-26 — Prompt version 1.9 and two number formats

- `PROMPT_VERSION` moves to `v1.9`. Two structural prompt commits (`92edb72`, `a908cd3`) shipped
  under `v1.8`, so the 17:00 and 21:00 cycles carried the same label for different prompts.
- `## Period Metrics` prints its `Δ` to two decimals (`Δ↓2.19%`, was `Δ↓2.19448997%`) and its
  per-period indicator deltas the same way (`ADX ↓ 3.75`, was `ADX ↓ 3.74657740`). Both come from
  `FormatUtils.fmt`, which keeps eight decimals below 10 to protect small prices (`0.00001234`);
  that rule is right for a price and wrong for a delta in index points.
- `Choppiness` prints two decimals in both places (`48.63` in `### Volatility` against `41.13` in
  `## Indicator Changes`, previously `48.6`). One number, one precision.
- The "uses daily candles" note names the row that actually comes from daily candles, so a token with
  less than 180 days of history reads `the 90d row uses daily candles` instead of carrying no note.

## 2026-11-25 — Clarify paper results and dashboard visibility

- The public results label gross returns and per-trade risk ratios and no longer predict a fixed result after fees. The README describes the prompt preview and separate memory view.
- Statistics warn about unmatched closes and overwritten entries without changing how trades are paired or how results are calculated.

## 2026-11-25 — One frame vocabulary in the prompt, and one table per question

The prompt carried two overlapping multi-period tables under two naming schemes, and they disagreed
about the same windows. Measured on the 2026-11-25 17:00 cycle:

| window | `## Market Data` | `## Period Metrics` |
|---|---|---|
| 24h | `-2.33%` | `Δ↓2.24921328%` |
| 3d | `-4.18%` | `Δ↓3.93586088%` |
| 7d | `-7.61%` | `Δ↓7.03782159%` |

Market Data measured from the close *before* the window, Period Metrics from the first close inside
it; the highs and lows matched, so the difference looked like a bug. `1D/2D/3D` also collided with the
daily frame's vocabulary (`### 1d OHLCV`, `Daily RSI`, `(Daily)`), and three places advertised three
different period lists: `- Analysis Includes: 4H, 1D, 7D, 30D, 365D, and WEEKLY timeframes`
(hardcoded, `4H` a third spelling, `365D` a macro line rather than a period), `periods: 4h, 12h, 24h,
3d, 7d` in analysis step 1, and the tables themselves.

1. **`## Market Data`'s Multi-Timeframe Price Summary is deleted** (`build_market_data_section` and
   `_calculate_period_candles` with it). It was a strict subset of `## Period Metrics` minus the 4h
   and 12h rows, and today's model never cited either — `12h` appears 16 times in today's log and
   every one of them is a prompt line. Three test parametrisations and one then-unused helper went
   with it.
2. **One period table, duration labels, the frame in the header.** `## Period Metrics (windows of 4h
   candles; the 180d row uses daily candles)` with rows `24h, 2d, 3d, 7d, 30d, 180d`.
   `MarketMetricsCalculator.PERIODS` maps each label to its window so the two cannot drift, the
   formatter no longer upper-cases the label, and the long row keeps real-coverage labelling (`90d`
   on a token with 90 days of history).
3. **The prompt describes the data it actually supplies.** `- Windows Supplied: …` and
   `- Raw History: 183 4h candles, 183 daily candles, 104 weekly candles` replace the hardcoded
   `Analysis Includes` line, both derived from `PERIODS` and the formatter's candle constants;
   analysis step 1 takes the same phrase, and the system prompt's "never as current 12h/24h extremes"
   now reads "24h/7d".
4. **`## Technical (4h)` owns its children.** `Price Action`, `Momentum`, `Trend`, `Volatility`,
   `Volume`, `Stats`, `Levels`, `Advanced` and `Detected Patterns` are `###`, so the frame reaches
   every number instead of being asserted once in a heading that visually ends before them.
5. **Numbers that shared a name now carry their frame and window.** `- SMAs (4h candles): …` separates
   the 4h SMA200 (71 990.23) from `SMA200: 72143.80` in `## Simple Moving Averages (Daily):`;
   `VWAP(20 bars)`, `TWAP(20 bars)`, `Rolling candle low/high (20 bars)`, `Retested S/R (120 bars)`,
   `Donchian (20 bars)` and `Close Trend (24 bars)` name the lookbacks the stop and target rules
   already depend on. The windows are class constants on `TechnicalCalculator`, so a label cannot
   outlive its length.
6. **`## Indicator Changes (4h, previous → current):` is its own section** instead of a `###` child of
   `## Period Metrics`, which made the analysis frame's cycle-over-cycle deltas read as period data.

Effect, replayed through the current code against the logged 17:00 cycle: the system prompt changes
two lines, the user prompt loses the duplicate table and gains the labels — **+51 prompt tokens**, and
every percentage now belongs to exactly one window and one frame. `3 021 passed in ~40 s`; ruff,
`pyright` (0 errors) and every `scripts/check_*.py` gate green.

## 2026-11-25 — The weekly table stops at the history it can justify

`## Raw OHLCV` gave every frame the same 183 rows, so the weekly table reached back to 2023-05-22 —
3.5 years of weekly candles for a decision whose horizon is one to three days. The three lengths were
measured on the live 2026-11-25 prompt before anything changed (`cl100k_base`):

| table | rows | window | tokens | high / low in the table |
|---|---|---|---|---|
| `### 4h OHLCV` | 183 | 2026-10-26 → 2026-11-25 (30.5 d) | 6 484 | 81 909.09 / 55 056.00 |
| `### 1d OHLCV` | 183 | 2026-05-26 → 2026-11-24 (183 d) | 6 616 | 87 500.00 / 55 056.00 |
| `### 1w OHLCV` | 183 | 2023-05-22 → 2026-11-16 (3.5 y) | 5 886 | 126 296.00 / 24 750.01 |

1. **The weekly table is capped at 104 rows (two years).** That is the shortest window that still
   holds the 2025-10-06 all-time high of 126 296.00 and the current low of 55 056.00. A one-year cap
   was tested and rejected: the highest high of the last 52 weeks is 97 965.00 (2026-01-12), so the
   model would lose the actual top. What 104 drops is 2023–2024 (24 750.01–49 001.00), which the
   prompt already states as text — `Weekly Macro Trend (200W SMA Analysis)` prints the 5.8-year
   change and the distance from the 200W SMA, `Macro Trend Analysis (365D)` the 365-day change.
2. **The analysis frame and the daily frame stay at 183.** Each window is exactly the deepest metric
   of its own scale — `30D: Range$55056.00-81909.09` and `180D: Range$55056.00-87500.00` match the
   two tables to the cent — and 183 bars also covers the longest lookback the pattern engine reports
   (the 120-bar Bollinger squeeze percentile). Cutting the daily table below 180 would leave the
   `180D` row unverifiable; doubling it to 365 would cost 6 616 more tokens to re-show a year that two
   text blocks already summarise. Extending the 4h table to 366 rows was priced at +6 476 tokens and
   not taken: it buys 61 days of intraday detail and still loses the 2026-10-02 top from the window
   by early December, because any fixed window slides.
3. **Measured effect.** The three tables go from 18 986 to 16 458 prompt tokens (−2 528, −13.3 %) and
   the block from 18 994 to 16 466, measured by replaying today's real series through
   `OhlcvTableFormatter` with the 4h and daily blocks byte-identical to the logged prompt. The
   weekly fetch stays 300 weeks (the 200W SMA and the 5.8-year text read it), the daily fetch 365 and
   the frame fetch `candle_limit`; only the rendered table is capped, so `truncate_ohlcv_section`,
   the console and the dashboard `<truncated>` rendering and every indicator are unchanged.

## 2026-11-25 — One rule for a closed trade, and a missing daily reading stays missing

An external review of the confidence table paired ledger rows by `id`: close #3 was inserted after
entry #2 but happened before it, so `HIGH 0/2` was a pairing error (chronological pairing gives
HIGH 1/2, MEDIUM 10/23 at the 2026-11-24 13:00 cutoff). Checking it surfaced the defects below;
each was reproduced on the current code first.

1. **One predicate decides whether a pair is a closed trade.** `closed_trade_is_eligible`
   (`src/trading/data_models.py`) drops a pair on an ineligible `execution_quality`, on
   `EXIT FILL DIVERGENCE` or on `PHANTOM ROLLBACK`. The statistics calculator, the prompt's
   performance summary and the notifier summary all call it. Before, the prompt ignored the phantom
   marker and the Discord/console summary filtered nothing: on the live ledger it printed
   31 closed / 48.39% / +87.24 USDC while the KPIs said 26 / 46.15% / +70.78.
2. **A compensating CLOSE is flagged by structure.** `_record_blocked_entry_close` writes
   `execution_quality={"eligible": False, "reason": "entry never executed"}` beside the marker.
3. **Three historical compensations are flagged in the ledger.** Rows #24, #26 and #68 closed a
   blocked or never-forwarded SELL at the entry price with no marker and no quality, so they counted
   as three flat trades. Each now carries `eligible: false` with `repair:
   compensation-flag-2026-11-25`; nothing is deleted, the backup is
   `data/trading/backups/compensation_flag_20261125_155504`. For #24 the executor reported the entry
   filled with failed protection and no fill evidence, so the flag says the fill is unproven, not
   that it never happened. Live statistics: **23 counted, 12W / 11L, 52.17%**, P&L unchanged at
   +70.78 USDC; Brain held no experience and `trade_post_mortem` no row for any of the three.
4. **`excluded_trades` counts the pairs the statistics dropped.** `scripts/ledger_facts.py` runs the
   calculator over the snapshot's `trade_history.db` instead of counting post-mortems that mention
   fill divergence (2 then, 8 now: five unverified fills plus the three compensations). The README
   block calls them "excluded (fill unverified or never executed)".
5. **`LongTermFormatter` omits a daily reading it does not have.** The calculator returns every
   `daily_*` key, `None` until enough history exists, and the formatter only checked the key: 10,
   20, 30 and 60 daily candles raised `TypeError` and the cycle ended before the model call; `NaN`
   printed invented labels such as `N/A (Extremely Strong Trend)`. Missing readings are now omitted,
   a real `0.0` still prints, and 85 candles render as before.
6. **A model-supplied `trend.strength` can no longer stand in for a missing ADX.**
   `TrendValidator` drops the bare `strength` alias and `resolve_primary_trend_strength` reads only
   `strength_primary`, so a `null` reading stays `null` in the log, the notifiers,
   `MarketConditions.trend_strength` and the next cycle's prompt (reproduced: `ADX 73` from the
   model's alias with Python's ADX missing).
7. **`scripts/news_feedback.py` pairs a post-mortem with the entry before it in time**, by symbol and
   `timestamp`, not by `id`; post-mortem #3 was attributed to an entry made after the close.
8. **`compute_timeframe_alignment` is a static method** — it reads only the context, and the
   offline prompt replay now calls it, so a replayed system prompt matches the live one.
9. **The README quotes the snapshot it links.** The block cited `results/2026-11-17` (20 trades,
   40.0%, +0.50%) but printed live numbers from 2026-11-22 (24 trades, 45.8%, +0.68%). The
   corrected ledger is frozen at `results/2026-11-25` and `claims.json`, the README block and the
   chart are regenerated from it: 23 counted + 8 excluded, 52.2%, +0.71% (+$70.78 gross) against
   -27.91% buy-and-hold. The site is not redeployed.
10. **The daily SMA sections of the prompt existed in the formatter and in no prompt.** `LongTermFormatter`
    read flat `sma_{period}` keys and the calculator filtered its own `sma_values`/`volume_sma_values`
    dicts out of the returned result, so `## Simple Moving Averages`, `## Volume Moving Averages` and
    `## Price Position vs SMAs` never rendered: 0 occurrences of any of the three headers in every
    `logs/Bot/*/Bot.log` ever written, while `src/analyzer/pattern_analyzer.py` was starved of the same
    dicts. The calculator now returns what it computed and the formatter reads the dicts, so the model
    gets the 20/50/100/200 daily SMAs, their volume SMAs and the distance from each. The three headers
    and the Ichimoku one name their frame as `(Daily)`, because `## Technical (4h)` prints a different
    `SMA20` and its own `Ichi` glyph under the same field names. `(Daily)` and not `(1D)`: the
    `## Period Metrics` rows call a 24-hour window of the analysis frame `1D`.
11. **The frozen snapshots keep their line endings.** `results/**` is now `-text` in `.gitattributes`,
    because `core.autocrlf=input` rewrote the captured CRLF into LF on commit and
    `scripts/snapshot_results.py --verify` then failed on a fresh clone against the manifest hashes
    (`results/2026-11-17` was in that state; `results/2026-11-12` only verified because its working
    copy had never been re-checked-out). All three snapshots now verify, including from a clean clone.

3 027 passed in 58.5 s; ruff, pyright (0 errors) and every `scripts/check_*.py` gate green.

## 2026-11-25 — The bot computes ADX, and a missing reading is no longer a number

The model was still asked to transcribe `strength_primary`/`strength_daily`, and the validator
filled a missing reading with the model's number or with 25. Reproduced on the current code:
`ADX 4h=30.83` with no daily value stored `31/73` (the model's daily) or `31/25` (the fallback),
and `0` was indistinguishable from "no reading".

1. **`TrendValidator` copies only the computed readings.** Each field is `round(ADX)` when the
   indicator stack returns a finite value inside 0–100, otherwise `null`; the model's number and
   the 25 default are gone, and one missing interval no longer touches the other.
2. **The prompt no longer asks for the transcription.** The JSON example and the TREND BLOCK drop
   `strength_*`; the result keeps the keys, so stored snapshots and the dashboard read unchanged.
3. **`timeframe_alignment` needs all four DI readings.** A missing, non-finite or out-of-range
   `+DI`/`-DI` now yields `null` and the prompt says the comparison is unavailable, instead of
   treating the gap as zero and emitting `MIXED`/`DIVERGENT`. The `1d` frame keeps its
   daily-vs-weekly fallback.
4. **`confluence_count` no longer feeds learning.** The `>50` count of opinion scores is gone from
   the experience metadata, the loss diagnostics and the `min_confluences_*` learner; the prompt
   keeps its static `2+` minimum. Factor scores stay stored for retrieval and dashboard stats.
5. **`confidence` is labelled as a model score.** Dashboard, console and Discord print
   `62/100` instead of `62%`; the value, the 60/65/75 gate and the sizing formula are unchanged.
6. **`trend_strength` no longer invents 50.** A missing ADX stores `0.0` (the field's own
   "unmeasured" value) instead of a neutral-looking 50.

2 989 passed in 43.8 s.

## 2026-11-25 — The Memory column measures conditions, not template overlap

The dashboard's Similarity column read 87–89% on every row, and the prompt printed the same
number as `[SIMILARITY 89%]`. Measured on the live store (51 documents, 23 retrievable trades,
embeddings read back from `embeddings_queue`): two random stored trades sit at **0.941 ± 0.021**
cosine, and today's market scored **0.899** against its best match — *below* the store's own
baseline. The number was a constant offset, not a match quality.

The cause is the embedded text, not the model. A trade document is ~85% shared template
(`Indicators:`, `Structure:`, `Exit Execution: SL hard/15m | TP hard/15m`, `Result: … Confidence: …`)
plus free-text reasoning, and the vocabulary is ~15 tokens, so every pair shares most of them.
Stripping the reasoning made it worse (0.955); embedding only the market context made it worse
still (0.950). Centering the vectors removed the common direction but left the same top-5 set —
the ordering was already right, only the scale was broken.

1. **`condition_match_score`** (`src/trading/vector_memory_context.py`) compares ten stored entry
   conditions against the current query document — trend, ADX bucket, RSI bucket, volatility,
   MACD, BB position, volume state, supertrend, choppiness band and ATR% within 1.5× — and returns
   the share that match, or `None` below four comparable conditions. On today's store it spans
   **11.1–70.0%** (mean 38.1) instead of 87–89%.
2. **The dashboard ranks and prints that score** (`src/dashboard/routers/brain.py`,
   `src/dashboard/static/modules/vector_panel.js`); the column is now `Match`. The cosine stays
   the candidate ranker, so the retrieved set is unchanged.
3. **The prompt prints `[CONDITION MATCH x%]`** and the thin-sample gate uses the same number;
   `get_stats_for_context` keeps only trades at or above `MIN_CONDITION_MATCH_PCT = 50`, so
   "Win Rate in similar conditions" is no longer the global win rate over 20 of 23 trades.
   When the metadata carries too few conditions the old `[SIMILARITY x%]` label is kept.

2 986 passed in 41.5 s.

## 2026-11-25 — A blocked duplicate no longer hides the executed verdict

The 08:37 TP fill on the open SHORT could not be booked for 40 minutes, and the bot kept a
phantom position while the exchange was flat. Two causes, one in each repo.

1. **The repo rename broke the executor's journal path.** The executor's `.env` still pointed
   `LLM_TRADER_DECISION_PATH` at `G:\qrak\PythonScripts\LLM_trader_private\data\trading\latest_decision.json`,
   and its verdict journal defaults to a sibling of that file. After the rename to `LLM_trader`
   the executor wrote `executor_verdicts.jsonl` into the leftover `LLM_trader_private\data\trading\`
   directory while the bot read the new one, so the close it executed at 08:37:25 never reached
   the bot. The path now points at `LLM_trader`, the executor was restarted, and the four orphaned
   verdict lines were merged into the live journal (backup in `data/trading/backups/rename_fix_*`).
2. **`_read_executor_verdict_entry` returned the newest entry for an order id, so a `blocked`
   duplicate shadowed the `executed` verdict.** The bot retries a CLOSE it has not seen confirmed;
   the executor answers `blocked: Already executed (duplicate order_id)` and journals that too.
   The last entry then said `blocked`, `ExitRecords.from_receipt` refused it, and the exit became
   permanently unbookable — the retry that was meant to recover the trade is what buried it. The
   lookup now returns the `executed` entry when one exists for that order id and only falls back
   to the newest entry otherwise. Two tests pin both branches.

The trade booked at 58,576.51 for **+4.40%**, post-mortem `plan_followed`, brain experience
`WIN +4.40%`; statistics moved to 26 trades, 46.2% win rate, Sharpe 0.29. The 09:00 cycle had
already returned CLOSE at confidence 78 and could not act on it — that decision is now moot.

2 974 passed in 47.3 s.

## 2026-11-25 — `ccxt` 4.5.94, and the seventeen upgrades that were not worth it

The dependency audit started as eighteen pins and ended as one. Every candidate was checked
against OSV and against what the bot actually does, and only `ccxt` survived.

1. **`ccxt` 4.5.90 → 4.5.94** is the one change. The bot calls only `load_markets`,
   `fetch_ohlcv`, `fetch_ticker`, `fetch_tickers`, `fetch_order_book` and `fetch_trades` —
   public market data, no orders — on `binance`, `kucoin`, `mexc` and `hyperliquid` (`gateio`
   in `config.ini:253` never loads, ccxt names it `gate`). Three fixes in 4.5.91–94 land on
   exactly those calls: `parseMarkets` skipping empty markets, Binance HTML error parsing, and
   KuCoin error mapping.
2. **numpy/numba/pandas were rejected on measurement, not taste.** OSV reports zero advisories
   for numpy 2.2.3, numba 0.61.2, pandas 2.3.3 and llvmlite 0.44.0, and the cycle log
   (`prepare 0.3s, knowledge 2.5s, context 0.1s, analysis 42.5s, settle 2.4s`) leaves 5.3 s
   that is not LLM wait — the indicators are a fraction of it. numba 0.61.2 caps numpy at
   `<2.3`, so the three move as one block across seven numba minors and nineteen importing
   files, for under a second per four-hour cycle.
3. **aiodns and aiohttp-client-cache were rejected on evidence.** The bot imports
   `aiodns.error.DNSError` only, never `DNSResolver`, so aiodns 4.x buys a pycares 5 rebuild
   and nothing else. aiohttp-client-cache 0.14/0.15 do fix a SQLite lock race and an
   event-loop-bound lock, but 61 log files from 2026-07-29 to today contain no occurrence of
   either, and the upgrade needs `autoclose=False` in `provisioners.py:189`.
4. **Python stays at 3.13.14.** 3.13 is security-only and 3.13.18 ships source only — 3.13.17
   is the last release with a Windows installer, and its one security item (pip 26.2.2 +
   certifi 2026.10.19, the eMudhra CA incident) is already covered by the venv's certifi
   2026.10.23. OSV reports no advisory for CPython 3.13.14.
5. **Known and left alone:** `chromadb` 1.5.9 carries four CVEs, all `last_affected: 1.5.9`
   with no fixed release, all network-facing against the Chroma server. The bot uses
   `chromadb.PersistentClient`, an embedded client with no server and no network exposure.

2 972 passed in 43.4 s.

## 2026-11-24 — `timeframe_alignment` stops being a guess

The model never received the daily `+DI`/`-DI`, only `Daily MACD: Bearish` and the daily ADX, so
`timeframe_alignment` was an opinion about data it could not see. The bot already computes
`daily_plus_di`/`daily_minus_di` (`technical_calculator.py:552-555`) and never used them.

1. **`classify_timeframe_alignment`** (`indicator_classifier.py`) compares the `+DI`/`-DI` dominance
   of the analysis frame with the daily one, using the same ±5 band as `classify_trend_direction`:
   same direction → `ALIGNED`, opposite → `DIVERGENT`, either side NEUTRAL → `MIXED`. No daily DI, or
   a `1d` analysis frame (daily-vs-daily proves nothing) → `None`, and the model's value stands.
2. **The value is computed before the prompt** (`AnalysisResultProcessor.compute_timeframe_alignment`,
   called from `AnalysisEngine._generate_ai_analysis`), shown in the JSON template for the model to
   copy, and overwritten after the response in `_validate_llm_claims` — the same shape as
   `pattern_quality`. The prompt's long `ALIGNED | MIXED | DIVERGENT` explanation is replaced by
   "copy the system-computed value" whenever a value exists.
3. **What the evidence actually shows.** The 4h DI gap on 24.11 was 7.79 (01:00), 0.69 (05:00), 0.67
   (09:00) and 5.13 (13:00); the model said MIXED, MIXED, MIXED, ALIGNED. Three of four match the DI
   rule, so the label was not random — it was a defensible judgement that happened to disagree with
   the bot's own data at 01:00. The fix buys determinism and one source of truth for Brain, not a
   correction of a broken model.
4. **Not changed, on purpose.** `confidence` (82/89 MEDIUM, 0 LOW, entries at 60–63 against a 60
   gate) and `confluence_count` (a >50 threshold on opinion scores; the learned `min_confluences_*`
   never fire, so it has no live effect). Both need a decision, not a patch.

2 972 passed in 44.4 s.

## 2026-11-24 — The analysis reads like a person wrote it, and Discord shows it once

The `low` narrative was three walls of numbers (1 581 chars, ~400 tokens, 15–20 figures per line) and
Discord printed the same thesis twice: the narrative message, then an embed whose description was
`analysis.reasoning`. Measured on the 13:00 cycle: 37 982 in + 4 122 out, $0.00817.

1. **`low`/`medium` narratives are written for a human reader.** The output rule and the line
   templates now ask for plain language, ≤2 sentences and ≤3 numbers per line, and the JSON-mode
   header no longer demands "quantitative data first". `analysis.reasoning` keeps the exact numbers
   and is labelled as machine memory for the next cycle. `high`/`full` stay analytical.
2. **Targets: low 300, medium 800, high 1500, full 4000** (was 400/1000/2000/8000). `full` at 8000
   tokens is ~32k chars, and `DiscordNotifier.send_message` truncates at 20 000, so a full report
   could never arrive whole. Live config moves to `medium`.
3. **One Discord card.** `_create_analysis_embed` takes the narrative as its description (≤4096
   chars, else the narrative goes as a message and the embed has no description) instead of
   repeating `analysis.reasoning`. The separate "NOT EXECUTED" banner is gone — the embed already
   carries it in the title and in its field. `extract_analysis_fields` drops the now-unused
   `reasoning` key.
4. **One verbosity path.** `build_response_template` normalizes through `_normalize_model_verbosity`
   like the other builders, so an unknown value falls back to `medium` everywhere instead of `low`
   in the template and `medium` in the system prompt. `PROMPT_VERSION` → `v1.8`.

## 2026-11-24 — The social block stops wearing the news header, and the position block stops saying "Fee" twice

Two prompt defects found while auditing the 01:00 cycle.

1. **The social-sentiment block was labelled as market context, and that disabled the news trim.**
   `wrap_untrusted` always prepended `## EXTERNAL MARKET CONTEXT (UNTRUSTED DATA)`, and it wraps
   both the social block and the news block, so the prompt carried that header twice. The label was
   wrong for social messages, but the real cost was in `PromptBudgetPolicy._remove_optional_section`:
   it bails out when its start marker matches more than once, so with social sentiment present the
   news block could never be dropped to fit the budget — the trim silently did nothing. `wrap_untrusted`
   now takes `header: str | None`, and the social block passes `None` because it already carries its
   own `## Social sentiment (untrusted message evidence)` header. The guard rule and the nonce markers
   still wrap it. Measured on a two-block prompt: before, 2 headers and `trimmed=False`; after, 1
   header, the news block trimmed, the social block kept.
2. **`- Entry Fee: Fee: $0.0000`.** `_commission_text` returns text that already starts with `Fee: `,
   and the position block prefixed it with `Entry Fee: `. The line now reads `- Entry Fee: $0.0000`.
3. **Two different prices under one label.** The prompt carried `- Current Price:` twice with different
   values: the Trading Context one is the forming candle's close (`ohlcv_array[-1, 4]`,
   `data_fetcher.py:59`) and the position block's one is the live ticker `last` (`app.py:1052`). They
   sat 0.056 % apart in the 09:00 cycle. The position block now says `- Current Price (live ticker):`
   so the model can tell which number it is reading.

2 951 passed in 65.8 s.

## 2026-11-23 — News retrieval: typed collaborators, source weights load from the shipped config, backfill respects the scorer

Refactor of the RAG news path plus two bugs found while reading it. The ranking is byte-identical
before and after: `score_probe` over a frozen `crypto_news.json` snapshot prints the same top-5 for
−5 %, 0 % and +4 % 24 h change.

1. **Collaborators are typed and required.** `NewsManager`, `IndexManager` and `ContextBuilder` took
   `article_processor=None` and then guarded every use, and `ArticleScoringPolicy` took
   `config=None` with two `if not self.config: return 1.0` escapes. Production always passes both,
   so the defaults only existed for tests. They are now required parameters with real types, and
   the four `# type: ignore` comments they forced are gone.
2. **`ContextBuilder._calculate_article_relevance` is gone.** It was a single-caller pass-through
   that also carried the `pub_time = 0.0` fallback; `keyword_search` now calls the policy directly.
3. **`calculate_coin_score` lost its regex fallback.** When `coin_patterns` was `None` it rebuilt
   the same patterns with `re.escape`. `coin` and `coin_patterns` are set together by
   `keyword_search`, so the fallback was unreachable; the policy now takes the pair or neither.
4. **`PromptBuilder` reads `self.config.RAG_NEWS_MAX_CHARS`** instead of
   `getattr(self.config, "RAG_NEWS_MAX_CHARS", DEFAULT_MAX_CHARS)` — the config type is known, so
   probing it hid a missing field instead of failing.
5. **Bug: `[rag] news_source_weights` broke with two or more pairs.** `Config._convert_value` turns
   any comma-containing value into a list, so `RAG_NEWS_SOURCE_WEIGHTS` did `str(list)` and raised
   `ValueError: could not convert string to float: "1.0'"`. `retrieve_context` catches everything
   and returns `""`, so the model silently lost every news article. `config.ini.example` ships four
   pairs, so anyone copying it hit this. The value is now parsed at load time by
   `Config._parse_source_weights`, like `news_sources`, and a malformed entry stops the bot at
   startup instead of degrading it.
6. **Bug: the coin-index backfill re-admitted rejected articles.** When fewer than `k` articles
   scored, `_expand_candidate_indices_for_symbol` appended everything `coin_index` held for the
   coin — including articles older than 24 h (the database keeps them up to ~28 h) and duplicates
   `_dedupe_by_story` had just removed. It now skips anything the scorer already ranked and
   anything older than 24 h.

`src/rag/AGENTS.md` no longer claims "no fuzzy matching" and "no source-authority weighting"; both
were false after the previous change.

2 950 passed in 55.2 s. Net diff for this change alone: +68 / −79 = −11 lines.

## 2026-11-23 — News relevance follows the market, and the cycle records what the model read

Seven changes to how the RAG layer picks news, each verified on its own:

1. **The query follows the market.** `build_context_query` returned the same
   `"<coin> price analysis market trends"` every cycle, so relevance meant overlap with five
   generic words. It now appends a state phrase from the 24 h change (`selloff liquidation crash`
   below −3 %, `decline pullback` to −1 %, `range consolidation` between, `rebound gains` to +3 %,
   `rally breakout` above), and the query is logged so cycles stay comparable.
2. **A hard 24 h window.** The recency factor was multiplied as `0.3 + 0.7 × recency`, so an
   article a day old kept 30 % of its score. It is now `recency` alone: 24 h and older scores 0,
   and a future timestamp clamps to 1.0 instead of 1.04.
3. **`cooccurrence_multiplier` is gone.** It fired only when an article contained *every* query
   word, which rewarded exactly the generic pieces the static query produced. The keyword score
   already measures term overlap.
4. **The keyword score is scaled by body length** (`min(1, sqrt(1000 / len(body)))`), so a long
   padded article no longer outranks a short specific one.
5. **Sources carry a weight** (`[rag] news_source_weights`, default
   `coindesk:1.0,cointelegraph:1.0,decrypt:0.9,cryptoslate:0.8`). cryptoslate published 23 of the
   65 stored articles, the most of the four.
6. **One story, one slot.** Articles within two hours whose titles share at least four words and
   40 % Jaccard are collapsed to the best-scored one. On the live database this drops exactly the
   two real duplicates (a repeated headline and a live-blog update) and nothing else.
7. **The cycle record carries the news.** `metrics/cycles.jsonl` gains `news: {query, articles}`
   from the prompt's own article list, and `scripts/news_feedback.py` prints, per closed trade, the
   news the model saw at entry next to the outcome.

2 947 passed in 58.5 s.

## 2026-11-23 — The news block stops being cut at 6 000 characters

`sanitize_external_text` capped every external block at a hardcoded 6 000 characters, so the news
section reached the model with 2 articles while `news_limit` asked for 5 and the prompt used
35 761 of its 100 000 tokens. The 17:00 cycle ingested 101 items from four RSS feeds, saved 65 and
showed 2. The cap is now `[rag] news_max_chars` (default 24 000, ~6 000 tokens), read by
`PromptBuilder` for the news block only; the social block keeps the 6 000-char default. One
parametrized test in `tests/domain_brain/test_prompt_contracts.py` proves the cap binds and that a
larger value keeps the whole block. 2 922 passed in 59.8 s.

## 2026-11-23 — The model stops doing ATR arithmetic; Python prints the distances

The stop rule asked the model to divide `|entry − SL|` by ATR and state the result, while
`## Levels:` carried an ATR distance only for the rolling S/R and the pivots, so a stop placed
at a retested level or a Donchian edge had to be measured by hand. `## Levels:` now prints the
1× ATR floor as two prices (LONG at or below, SHORT at or above) and gives the retested S/R and
Donchian rows the same `(delta; x.xx ATR)` form as the pivots; the SL rule tells the model to
quote a printed distance instead of computing one. `AnalysisResultProcessor` records
`sl_atr_distance` from the model's own entry/SL and the computed ATR and logs a warning below
1× ATR, so the number is Python's rather than the model's. Research behind it: context fidelity
is unsolved and larger models do not do better (FaithEval), long inputs degrade even with
perfect retrieval (Context Length Alone Hurts), and literal copying beats recomputation
(+12.2 to +24.5 pp). 2,920 passed.

## 2026-11-23 — The suite runs on half the cores, and the old entries move to the archive

`pytest tests -q` took 300 s on one core, and 190 s of that was five tests sleeping through real
retry backoff: `DeepSeekClient.retry_on_empty` re-sent a reply without a decision JSON five times
(1+2+4+8+16 s) because `asyncio.sleep` was never patched, and one StockTwits test waited the real
5 s `CHALLENGE_RETRY_SECONDS`. `tests/domain_infra/test_ai_providers.py` now zeroes
`decorators._add_jitter` for the whole module, and the StockTwits test patches its own sleep.
The suite runs under `pytest-xdist` (`-n auto` in `pytest.ini`), sized by
`pytest_xdist_auto_num_workers` in `tests/conftest.py`: half the logical cores, capped at one
worker per 1.5 GB of free RAM, because every worker imports torch and chromadb and 12 workers
exhausted the box. Measured on this machine (24 threads, ~7 GB free): 300 s serial → 112 s after
the sleep fix → 44 s with 4 workers; 8 workers ran out of memory and 6 were slower than 5.
`check_claims.py` and `claims_report.py` collect with `-n 0` so the count stays readable.
`CHANGELOG.md` keeps November (77 entries, 2,205 lines); everything up to 2026-10-31 moved
unchanged into `docs/changelog-archive-2026.md`, and a duplicated 2026-11-12 entry plus a stray
`# Changelog` heading that sat among the October entries were removed. 2,919 passed in 43.7 s.

## 2026-11-23 — A locked state file no longer loses the write

`atomic_write_json` replaced the destination in one shot, so a single `PermissionError`
(`WinError 5`) from a handle that held the file open for a moment dropped the write —
`position_monitor.json` on 2026-11-21 23:01:34 and 2026-11-23 07:02:35, each followed by a
duplicated position-status card because the loop still saw the old timestamps. The replace
now retries six times over 1.55 s on `PermissionError` only, and the target is resolved to an
absolute path so the temp file and the destination share one base. Two regression tests in
`tests/domain_infra/test_data_utils.py`; 2,919 passed.

## 2026-11-22 — One provider, no fallbacks; LM Studio and BlockRun retired; retries reach to the next candle

Retired: LM Studio (`LMStudioClient`, streaming path, `lm_studio_*`), BlockRun (`BlockRunClient`,
`blockrun-llm`, `BLOCKRUN_WALLET_KEY`, `blockrun_*`), the `all` provider chain, every fallback —
the per-provider fallback models (`openrouter_/opencode_/commandcode_fallback_model`) and the Google
paid-tier client (`GOOGLE_STUDIO_PAID_API_KEY`) — and `ProviderHealth` with `[circuit_breaker]`,
whose only job was skipping providers inside that chain. `provider = local` and `provider = all` are
now rejected by the loader; `ModelManager.send_prompt_streaming` is gone.

New slot `provider = custom` with `custom_protocol = openai | anthropic`, `custom_base_url`,
`custom_model`, `custom_max_tokens`, `custom_reasoning_effort` and the optional `CUSTOM_API_KEY`.
`OpenAICompatibleClient` replaces `OpenCodeClient` and `CommandCodeClient` (they differed only in
headers) and is the base of `OpenRouterClient` and `DeepSeekClient`; `AnthropicCompatibleClient`
posts the Messages API over `httpx`. The composition root builds only the client `provider` names,
and the orchestrator is one `get_text_response` path (449 → 165 lines).

Retries: provider errors were never retried before — `_should_retry_api_error` matched only numeric
5xx dicts, while every client returns `"connection: …"`/`"timeout: …"` strings. Those prefixes, plus
`overloaded:` (5xx, 529) and `rate_limit:` (429), are now transient. The default is 5 retries; while
offline the waits are 60 s × 4ⁿ and the fifth runs exactly to 15 minutes before the next candle
(a quarter of the frame below 1 h), after which nothing is retried, so the cycle ends and the next
analysis keeps its slot. `authentication:` is never retried. `AsyncOpenAI(max_retries=0)` removes
the SDK's two hidden retries per attempt.

Removing `blockrun-llm` also removes the 23 `PydanticDeprecatedSince20` warnings every test run
printed: `pytest tests -q -W error::DeprecationWarning` passes with no filter (2,915 passed).

## 2026-11-22 — The structural-stop exception stops at 1× ATR

The 16:00 UTC SHORT went in with the stop 0.88 ATR from entry and the invalidation 0.72 ATR away,
while the same prompt carried the `stop_too_tight` lesson against exactly that. The SL rule let a
"closer structural boundary" undercut the profile multiple with no lower bound, so the model used a
Fib pivot. The rule now ends that exception at 1× ATR — beyond the next real level and smaller size,
or HOLD — and asks the model to state the SL distance in ATR for every BUY/SELL. Prompt wording only;
no gate enforces it yet.

## 2026-11-22 — One reference price for every range position; the dashboard gets a placeholder

`RangePos` was measured from each row's own last close, so `1D`–`30D` used the last closed 4h candle
while `180D` used the last closed daily one — up to a day older. Every row now takes
`ohlcv[-1, 4]`, the last closed candle of the analysis frame, so "now" means the same thing in all of
them.

The full prompt had also started reaching the dashboard's live console and its 7-day in-memory buffer,
because the line stopped being skipped when the OHLCV section became `<truncated>`. The live stream
now carries `Full prompt content: <truncated>` through a separate `stream_message`, so the buffer
holds a placeholder instead of seven days of prompts; the terminal and `Bot.log` keep the prompt with
only the OHLCV table replaced.

## 2026-11-22 — Structure patterns carry the candle they were confirmed on

All three structure patterns reported `index = last_index`, so `is_pattern_recent` computed an age
of 0 and the 80-hour staleness target for the category never rejected anything: a double top whose
second peak sat 25 days back still reached the prompt as current, and `PatternQualityScorer` counted
it as fresh. The swing trend now reports the later of its last two pivots, and the break of structure
and the double top/bottom report `last_index - bars_ago`.

`double_top_bottom_numba` also stopped reporting a pattern the market had already invalidated: a
double top only counts while the last close is at or below its level, a double bottom only while the
close is at or above it, and when both still hold the more recent one wins instead of the top always
taking precedence. `STRUCTURE_LOOKBACK` was never read and is gone.

## 2026-11-22 — `scripts/` keeps only what something runs

Deleted the one-off harnesses and captures nothing executes: `scripts/prompt_ablation.py` (its
finding is recorded in "The prompt carries the raw candles" below), `scripts/traffic_snapshot.ps1`
and `scripts/traffic_snapshot.sh` (GitHub traffic capture, no output directory), and the gitignored
local launcher `scripts/start_script_futures.ps1` (no such branch). `scripts/pattern_edge_report.py`
stays — the candlestick entry keeps it as the evidence — and `scripts/start_script_develop.ps1`
stays for the planned `develop` branch. 30 files → 26.

## 2026-11-22 — Long-horizon range in Period Metrics; the raw tables stay out of the console

`## Period Metrics` stopped at 30 days, so nothing in the prompt said where price sat in the
half-year range. `MarketMetricsCalculator` now appends a `180D` row built from the **daily** candles
(999 4h candles cover only ~166 days, so a 4h slice could not honestly carry that label) and labels
it by its real coverage — `90D` on a token with 90 days of history. Every period row also gained
`RangePos:NN%`, the last close's position between that period's low and high. The row is computed in
Python, so it survives the prompt budget dropping the raw OHLCV tables.

The 183-row tables are now `<truncated>` in the console and on the dashboard: `truncate_ohlcv_section`
replaces the `## Raw OHLCV` block, `_ConsoleHandler` and `LogStreamHandler` honour a
`display_message` extra, and `/api/monitor/last_prompt` truncates too. The model still receives every
row and `logs/Bot/<date>/Bot.log` still keeps the full prompt.

## 2026-11-22 — Swing structure reaches the prompt; candlesticks do not

With the chart gone, "lower highs" had no way into the prompt. `structure_patterns.py` adds a
`structure` pattern category built from swing pivots (a pivot is the extreme of the ±5 bars around
it): `structure_uptrend` (higher high and higher low), `structure_downtrend` (lower high and lower
low), `structure_range` (the two disagree), `break_of_structure_bullish` / `_bearish` (the last close
beyond the most recent pivot) and `double_top` / `double_bottom` (two pivots within 0.5%). Each line
carries the levels it compared. `pattern_recency` gives the category an 80-hour staleness target, and
`PatternQualityScorer` classifies the seven types.

Candlestick patterns were implemented (TA-Lib body/shadow/doji definitions in
`candlestick_patterns.py`) and then **measured before shipping**:
`scripts/pattern_edge_report.py` replays 8,108 BTC/USDC 4h candles, compares each pattern's forward
6/12/24-bar hit rate with the unconditional base rate and applies a one-sided binomial test with a
Bonferroni correction over 13 patterns × 3 horizons (threshold p < 0.00128). **No pattern passed** —
the best was `hanging_man` at 6 bars (p = 0.0102, edge +8.7 pp). The detectors and the script stay as
the evidence; nothing in the prompt claims a candlestick edge.

## 2026-11-22 — Pattern quality counts the patterns the engine actually emits

`PatternQualityScorer` classified 27 of the types `IndicatorPatternEngine` emits as `neutral` — every
`*_divergence`, every `macd_*_crossover` and `macd_*_zero_cross`, every `stoch_*_crossover`,
`rsi_w_bottom`, `rsi_m_top`, both `ma_*_alignment` and both `*_cross_active` — while carrying 39 names
the engine never produces (`hammer`, `double_top`, `bullish_engulfing`, …). The `confirmation`
component therefore counted almost nothing but `rsi/stoch_oversold/overbought` and the two crosses, so
`pattern_quality` was blind to most of what the bot detected.

The three sets now hold exactly the emitted vocabulary: 15 bullish, 15 bearish and 9 explicitly
neutral types (`atr_spike`, the volatility trend, the three squeezes, `volume_spike`, `volume_dryup`,
`climax_volume`). A new test reads the engine source, expands the dynamic type names and fails if the
engine and the scorer ever disagree in either direction. `pattern_quality` values change as a result;
the dashboard and the prompt read the same number, so nothing else moves.

## 2026-11-22 — The prompt carries the raw candles

The numeric block describes the market but never shows it: it has a 30-day range and a 365-day
change, and no 180-day high or low at all. An 8-cycle reading test asked the model for the last
close, the 180-day high and low, the position inside that range and the 30-candle direction. With
the numeric prompt alone it scored **47%** of fields and **0/8** on the 180-day high; with the raw
tables it scored **92%** and **8/8**. The same 13-cycle, three-arm ablation that measured the chart
removal shows the tables do not move the decision beyond noise (3 flips of 13, against 3 for the
same prompt sent twice), so this is a reading fix, not a signal change.

`OhlcvTableFormatter` (`src/analyzer/formatters/ohlcv_formatter.py`) renders the last **183 closed
candles** of the analysis frame, the daily frame and the weekly frame as a CSV block
(`time,open,high,low,close,volume`, oldest first, UTC; the daily table is skipped when the frame is
daily). `PromptBuilder` appends it after the market-data section, and `PromptBudgetPolicy` drops it
first when a request would not fit. Indicators still run on the full 999/365/300 candles — only the
table is capped. Cost: ~18k prompt tokens per cycle (~$0.003 off-peak).

## 2026-11-22 — The model reads numbers, not the chart image

A 16-cycle, four-arm ablation replayed the logged cycle prompts text-only twice and with the chart.
The same prompt sent twice changed the signal in 5 of 16 cycles (mean confidence gap 3.0); adding the
chart changed it in 3 (old render) and 5 (current render), with a smaller gap (2.4 / 2.6). A separate
8-point reading test showed the model does read the image accurately — 91% of fields, last price to
the cent — so the image was redundant, not unreadable: its "P1–P4" citations restated numbers already
in the numeric block. The image is gone from every request.

Removed: `BaseAIClient.chat_completion_with_chart_analysis`, `process_chart_image`,
`_prepare_multimodal_messages`, `_attach_multimodal_content`, `_build_multimodal_content`,
`_extract_user_text_from_messages`; the same method on `DeepSeekClient`, `OpenRouterClient`,
`BlockRunClient`, `LMStudioClient`, `GoogleAIClient`; `GoogleAIClient._get_image_dimensions`,
`_estimate_image_tokens`, the `image_bytes` usage breakdown and the whole `code_execution` path
(`google_code_execution`); `ProviderMetadata.supports_chart`, `ProviderOrchestrator.supports_chart`,
`get_chart_response` and every `chart`/`chart_image` parameter; `ModelManager.supports_image_analysis`,
`send_prompt_with_chart_analysis` and the `chart` argument of `describe_provider_and_model`;
`AnalysisResultProcessor.process_analysis`'s chart branch and `fallback_system_prompt`;
`AnalysisEngine`'s `precomputed_chart`/`has_chart_analysis` and the `chart_analysis` result key;
`PromptBudgetPolicy.image_reserve_tokens` and its `chart` parameter; `PROMPT_IMAGE_RESERVE_TOKENS`;
`TemplateManager`'s `has_chart_analysis` argument, the `CHART VALIDATION` block, the `8. CHART` step
and the chart cross-check line; `scripts/chart_ablation.py`.

The chart is still rendered for the dashboard and Discord, and it no longer sits on the analysis
path: `AnalysisEngine` starts `_render_chart_for_humans()` as its own `asyncio` task before the model
call and awaits it after, so its ~12 s overlaps the inference instead of delaying it. A failed render
logs a warning and keeps the previous dashboard image. The render itself already ran off the event
loop (`asyncio.to_thread` for the figure, a dedicated thread for Playwright), so this is scheduling,
not a new thread.

Docs, dashboard copy and the public site now say the model reads text. Test count 2,934 → 2,894.

## 2026-11-22 — The position gauge keeps equal steps and merges coincident labels

With the breakeven floor the bot's stop and the entry are the same price, and the
gauge scaled the segments from the middle of a three-point list, so coincident
levels collapsed onto one another. It now places the distinct levels at equal
steps and prints levels that land on the same spot as one label
(`Bot SL = Entry`). Cache versions bumped: `main.js` 9.19, `position_panel.js`
4.19, `lifecycle_panel.js` 1.7.

## 2026-11-22 — The post-mortem sees the stop the bot moved itself

The prompt showed a planned stop of 61,400.00 next to a stop of 60,210.14 with
nothing saying who moved it, so a stopped-out trade could be read as a
discretionary close — which is exactly how the 08:35 UTC `premature_exit` verdict
was written. The post-mortem now lists every applied stop change with its own
reasoning (`Bot policy override — … breakeven floor`), read from the `UPDATE` rows
between entry and close, and states that the exit ran on the last stop above. A
trade whose stop never moved renders no block.

## 2026-11-22 — The booked close reports the trigger, not just the receipt

The 08:35 UTC SHORT closed because the bot's own stop fired, but the executor's
receipt names only `close_confirmed_by_executor`, so the post-mortem never learned
what triggered the exit and wrote `premature_exit` against a discretionary close
that never happened. The booked close now carries `— exit trigger: stop_loss at
60,233.36`; that string reaches the trade history row, the post-mortem prompt, the
exit report and the brain's `close_reason`. Separately, the reference-price check
now refreshes the in-memory excursion metrics, so `max_profit_pct` /
`max_drawdown_pct` stop being 0 on the executor-owned path — every stored WIN
carried 0.00/0.00 and Safe Drawdown read those zeros as data. It still persists
nothing between turns.

## 2026-11-22 — Separate breakeven snapshot eligibility from the live decision

In the 05:00 CET cycle the prompt showed the breakeven floor as active at
34.6% progress, but the fresh ticker after inference put it at 24.8%, below
30%. The position block now labels eligibility as a snapshot; the system
prompt distinguishes the automatic move, which rechecks both conditions on
the fresh ticker, from a model-proposed stop. The `UPDATE` rule no longer
suggests a manual breakeven/trailing move, and `HOLD(open position)` tells the
model not to propose a stop change only to trigger the automatic move. The
previous-decision snapshot now
states that it is the model's recommendation, not proof that it executed.
Execution and thresholds are unchanged.

## 2026-11-22 — Stop merging the Stochastic reading with a historical cross

The 21:00, 01:00 and 05:00 CET cycles each described a crossover from several
bars earlier as a current event; the 05:00 reply welded the 7-bars-ago cross
onto the live `%K` and called it "P2-RSI". The glossary now separates the
20/80 reading of the current `%K` from the 30/70 tag on the candle before a
cross, states that a cross outside that band still exists, and forbids merging
the historical event with today's value. The chart guidance says P2 plots RSI
only. Detection and thresholds are unchanged.

## 2026-11-22 — Report the tightening gate instead of a fake R/R

The 05:00 CET refusal reached the next prompt as `Your R/R: 2.36 | Required:
0.00` with the guard's reason labelled `Your thesis` — the entry ratio, not the
reason the UPDATE was refused. A `sl_tightening` event now prints the price at
the check, the progress toward TP, the minimum for a normal tightening and the
policy reason; missing metadata is skipped. Other guards keep their R/R block.

## 2026-11-21 — A condition bucket stops claiming a significance it does not have

The `### Market Condition Performance` block tagged a bucket from two trades with a fixed
65/35 win-rate threshold, so a three-trade sample became a verdict. On the live store the only
"weak" buckets were the long ones — `RSI 60-70 (STRONG): 0/3` and `ACCUMULATION: 1/3`, all three
trades LONG in each — so the prompt told the model that long setups fail while the pro-short
buckets carried `STRONG SIGNAL`. A tag now requires the same one-sided exact binomial test the
rule gates have used since 2026-10-27 (α = 0.05 against the closed-trade win rate); a bucket
without a significant result prints its counts and no verdict. The same change makes the
direction-bias warning fire for either side, and the prompt loses the line that granted shorts a
permission longs never had.

### Changed
- **`src/trading/brain_context.py`**: `_condition_tag()` and `_closed_win_rate()` replace the
  three copies of the 65/35 block in `_build_condition_performance_context()`; the baseline is the
  closed-trade win rate the reflection engine already uses, and a missing analyzer or baseline
  prints counts only. The `NO SHORT TRADES IN HISTORY` warning now names whichever side has no
  trades.
- **`src/analyzer/prompts/template_manager.py`**: removed `SHORT TRADES: Valid with sufficient
  confluence even in bull macro.` `MACRO CONFLICT` is the single, symmetric rule for a
  counter-trend entry.
- **Tests**: the condition-performance fixture now uses real counts and a real baseline; new pins
  cover the live 0/3 case (no tag), a bucket that beats the baseline (tag), a missing analyzer
  (counts only) and the direction-bias warning on both sides.

### Why
The 2026-10-27 change removed this class of claim from the semantic rules but left it in the
buckets, which are rebuilt from the same trade history on every cycle. A bucket tag is a claim
about the future drawn from a handful of trades, and the three trades behind `RSI 60-70 — WEAK
SIGNAL` were all longs in a market that had already turned.

### Verified
- `python -m pytest tests -q` → **2 927 passed** in 153 s.
- `ruff check src scripts start.py` clean; `pyright src start.py` 0 errors; `check_comment_policy`,
  `check_config_contract`, `check_dead_symbols`, `check_loose_helpers`, `check_hardcoded_timeframes`
  and `check_agents_md` all exit 0.
- The block rendered from a copy of `data/trading/brain_BTC_USDC_4h` prints
  `RSI 60-70 (STRONG): 0/3 wins (0% WR)` with no tag, and `scripts/brain_rules.py check` still
  prints an empty anti-pattern block.

## 2026-11-21 — Console log lines colour the header by level

The console header (`[HH:MM:SS +01:00] LEVEL file:line - `) now takes the colour of the
record's level — DEBUG blue, INFO green, WARNING orange, ERROR red, CRITICAL burgundy — so the
severity is visible before the message is read. The message keeps the default colour and the
existing highlighter, and the line stays one `Text` that wraps at the terminal width. File logs
and the dashboard live console are unchanged.

## 2026-11-21 — Clarify the timing of Stochastic crossover zones

The 17:00 CET analysis described a historical Stochastic crossover as still
oversold. The pattern now labels oversold or overbought conditions on the
candle **before** the crossover, not on the crossover candle or the latest
candle. Detection and thresholds are unchanged.

## 2026-11-21 — Align the stop guidance with the computed noise buffer

The 13:00 CET prompt quoted a fixed 0.5 ATR even though the position showed the
computed buffer. The guidance now uses the displayed distance for breakeven or
profit-side tightening and shows the breakeven floor as active at the prompt
snapshot only when both progress and distance qualify. The tightening and
execution rules are unchanged.

## 2026-11-21 — Console log lines wrap at the terminal width

After an RDP reconnect the console could be narrower than the ~48-column prefix that
rich's log table reserves for time, level and path, and the message column collapsed to
one character, printing every letter on its own line. The console line is now a single
`Text` (`[HH:MM:SS] LEVEL file:line - message`) that the terminal wraps at whatever width
it has; the level is no longer coloured. The console still follows the live terminal
width, and the file logs are unchanged.

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

## Archive

Entries up to and including 2026-10-31 were moved out to keep this file readable. Nothing was
deleted.

- [docs/changelog-archive-2026.md](docs/changelog-archive-2026.md)
- [docs/changelog-archive-2025.md](docs/changelog-archive-2025.md)
