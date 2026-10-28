# Handoff: the timeframe now has one source, and gates prove it

Date: 2026-10-27. Branch state: **uncommitted working tree**, HEAD `8664fa4`.
Read this before touching anything that reads configuration.

## TL;DR

A change hardcoded `4h` while the bot is supposed to run whatever `[general] timeframe` says. The
root cause was not one line: the loader carried a silent `1h` default, seven consumers carried
their own private copy of a plausible frame, and the test double mirrored those defaults, so the
suite was green while production and docs disagreed. All of that is removed; two mechanical gates
and two new test modules now fail when someone reintroduces a literal or a second default.

If you are an agent asked to "add a timeframe" or "read the configured timeframe": read
`Config.TIMEFRAME` / `Config.TIMEFRAME_MINUTES`, never a literal. `scripts/check_hardcoded_timeframes.py`
will fail your change otherwise.

## Symptom that started this

A model-driven edit changed analysis behaviour and introduced a hardcoded `4h`. The configuration
path was not consulted, so a bot configured for another frame would have run 4h with no error, no
warning, and a healthy-looking log. The failure mode is "plausible value instead of configured
value", not a crash.

## Root cause inventory (all of it, verified before the fix)

| Location | Problem |
|---|---|
| `src/config/loader.py` `TIMEFRAME` | fell back to `"1h"` while `config.ini`, `config.ini.example`, `AGENTS.md` §5 and `docs/CONFIGURATION.md` all said `4h` |
| `src/analyzer/market_data_collector.py` | `self.timeframe = "1h"` and `initialize(..., timeframe="1h")` |
| `src/analyzer/market_metrics_calculator.py` | `context.timeframe if context.timeframe else "1h"` |
| `src/analyzer/prompts/prompt_builder.py` | `__init__(timeframe="1h")` |
| `src/analyzer/prompts/template_manager.py` | `build_system_prompt(..., timeframe="1h")` |
| `src/analyzer/pattern_engine/chart_generator.py` | `create_chart_image(..., timeframe="1h")` |
| `src/analyzer/formatters/technical_formatter.py` | `_format_patterns_section(..., timeframe="4h")` |
| `src/trading/trading_strategy.py` | `to_minutes(config.TIMEFRAME) if config else 240` |
| `src/analyzer/analysis_engine.py` | invalid frame logged a warning and fell back to the config value |
| `tests/conftest.py` | `PRODUCTION_DEFAULTS` mirrored the loader defaults (`TIMEFRAME: "1h"`, `AI_CHART_CANDLE_LIMIT: 200`), so tests never saw the production value |
| `config/config.ini.example` | lacked 13 keys the loader reads, including the whole `[circuit_breaker]` section |

## What changed

### Single source of truth

- `Config.REQUIRED_CONFIG_KEYS` (class attribute) + `_validate_required_keys()` called from
  `__init__` before anything reads the config. Missing keys abort startup with the file path and
  every missing `[section] key`.
- `Config.TIMEFRAME` has **no default**: absent key raises `RuntimeError`; present value is
  validated and normalized through `TimeframeValidator`.
- `Config.TIMEFRAME_MINUTES` is the only place the frame becomes minutes (previously five copies:
  loader, provisioners, analysis engine, strategy, exit monitor).
- `_validate_exit_monitoring()` now uses both properties instead of re-validating.

### Consumers lost their private defaults

- `MarketDataCollector`: `timeframe` is a required positional argument of `initialize()`, normalized
  on the way in. `fetch_ohlcv()` raises when the collector was never initialized with a frame;
  `_warn_if_insufficient_history()` returns early instead of guessing.
- `MarketMetricsCalculator`: a context without a frame raises `ValueError` (silently using `1h`
  produced wrong 1D/2D/3D period metrics).
- `PromptBuilder`: `timeframe=None` resolves to `config.TIMEFRAME`; no config and no argument raises
  `ValueError("timeframe is required for PromptBuilder")`.
- `TemplateManager.build_system_prompt()`: `timeframe=None` resolves to `self.config.TIMEFRAME`.
- `ChartGenerator.create_chart_image()`: `None` raises `ValueError`.
- `TechnicalFormatter._format_patterns_section()`: `timeframe` is required (the method has no
  caller today, but the signature no longer advertises a frame).
- `TradingStrategy`: `_tf_minutes` comes from `config.TIMEFRAME`; the `else 240` fallback is gone.
- `DataFetcher`: macro frames are named constants `TREND_TIMEFRAME = "1d"`, `WEEK_TIMEFRAME = "1w"`.
- `WritableConfig`: the dashboard enum derives from `TimeframeValidator.SUPPORTED_TIMEFRAMES`
  (`1m` disappeared because the loader rejects it; `2h/6h/8h/12h` appeared because it accepts them).
- `AnalysisEngine.initialize_for_symbol()`: an explicit `timeframe` argument is validated; a frame
  different from the configured one logs `Timeframe override: configured=… effective=…`; an invalid
  frame raises instead of falling back.

### Runtime evidence

`Provisioners._display_startup_summary()` logs one line:

```
Effective configuration: pair=BTC/USDC timeframe=4h (240 min/candle) provider=deepseek chart_candles=125 candle_limit=999
```

That line is the proof the ini reached the process. Quote it in bug reports instead of assuming.

### Gates (new scripts)

- `scripts/check_config_contract.py` — `ConfigContractChecker.check() -> list[str]`:
  1. every literal `(section, key)` passed to `get_config` in `src/` exists in `config.ini.example`;
  2. every key present in `config/config.ini` is in the template (a typo such as `timefrmae`
     silently falls back to a default, so it is a violation, not a curiosity);
  3. `REQUIRED_CONFIG_KEYS` are present in the template;
  4. the seeded provider/model has an entry in `[model_context_windows]`;
  5. core keys documented in `docs/CONFIGURATION.md` exist in the template.
  `ALLOWLIST` carries accepted exceptions with a reason (sampling penalties, the commented-out
  BlockRun pair, `intent_journal_path` whose default derives from `data_dir`).
- `scripts/check_hardcoded_timeframes.py` — `HardcodedTimeframeChecker.check() -> list[str]`: AST
  scan of `src/` for supported-frame string literals outside `ALLOWLIST`
  (`timeframe_validator.py` definitions, `data_fetcher.py` macro constants, `prompt_builder.py`
  multi-timeframe period labels). Docstrings are exempt.

Both print a line per violation and exit 1; both are safe to run from the repo root without the
application import graph.

### Tests

- `tests/domain_infra/test_config_contract.py` (12) — gates are clean on the repo; each violation
  kind is proved on a `tmp_path` tree; missing `timeframe` aborts startup by name; the shipped
  `config/config.ini` is internally consistent (supported frame, chart limit ≤ candle limit,
  exit interval ≤ frame, context window for the seeded model).
- `tests/domain_infra/test_config_propagation.py` (13) — a real `Config` built from a temp ini using
  `6h`/`12h`/`1d`; the system prompt carries the configured frame and **no** other frame token
  (`\b4h\b` and `\b1h\b` asserted absent); period candle math (`24h` → 4 candles at `6h`); exit
  interval longer than the frame rejected at load; collector requires the frame; metrics refuse an
  empty one; the engine uses the configured frame, logs an override, and rejects `13h`.
- Updated: `tests/conftest.py` (`TIMEFRAME_MINUTES` derived via `TimeframeValidator`; docstring
  explains why the double stays at `1h`), `tests/domain_infra/test_config_and_pricing.py`
  (`REQUIRED_GENERAL` snippet prepended to the partial inis), `tests/domain_infra/test_dashboard_admin.py`
  (`1m` replaced by a supported frame), `tests/domain_rag_indicators/test_rag_market.py` (three
  collectors that bypass `initialize()` now set `timeframe` explicitly).

### Docs and policy

- `AGENTS.md`: **§5.1 Configuration invariants** (single source, no consumer defaults, new-key
  checklist, startup evidence) and **§12.1 Enforcement map** (rule → the command that disproves it).
  Both new scripts are in the §12 "Fast gate first" list.
- `CONTRIBUTING.md`: PR checklist line for loader keys.
- `docs/CONFIGURATION.md`: `timeframe` is documented as **required, no default**.
- `config/config.ini.example`: gained `[circuit_breaker]`, the human-approval keys
  (`require_approval_above_usdc`, `approval_ttl_minutes`, `approval_poll_seconds`) and a commented
  `intent_journal_path`.
- `CHANGELOG.md`: entry at the top with Changed/Why/Verified.

## Verification (what was actually run)

```
python -m py_compile <touched files>            → OK
python scripts/check_config_contract.py         → config contract OK
python scripts/check_hardcoded_timeframes.py    → no hardcoded timeframes outside the allowlist
python scripts/check_loose_helpers.py           → none
python scripts/check_agents_md.py               → 88 paths in 9 files, all exist
python -c "import start"                        → OK
ruff check src start.py scripts                 → All checks passed
pyright src start.py                            → 0 errors, 0 warnings
pytest tests/ -q                                → 2619 passed, 0 failed, 0 skipped
```

Manual proofs:

1. An ini without `[general] timeframe` → `RuntimeError: Missing required configuration keys in
   …\config.ini: [general] timeframe`.
2. Mutation: setting `timeframe = "4h"` inside `TemplateManager.build_system_prompt` made the
   literal gate exit 1 (`template_manager.py:324: timeframe literal '4h'`) and failed
   `test_the_system_prompt_carries_the_configured_frame_and_no_other`; reverting returned both green.

## Deliberate exclusions (do not "fix" these casually)

- **Not committed, not pushed.** AGENTS §12.8 requires explicit consent. The working tree is the
  change; four new files are untracked by design:
  `scripts/check_config_contract.py`, `scripts/check_hardcoded_timeframes.py`,
  `tests/domain_infra/test_config_contract.py`, `tests/domain_infra/test_config_propagation.py`.
  Do not delete them as stray files.
- **CI untouched** (AGENTS §12.9). Adding the two scripts to `.github/workflows/tests.yml` is a
  one-line job step; it needs the owner's approval.
- **`tests/conftest.py` keeps `TIMEFRAME = "1h"`.** Aligning it with the production `4h` reddened
  four tests that encode sub-4h semantics (SL tightening bands 20% vs 15%, exit fallbacks). The
  production frame is covered by `test_config_propagation.py` instead of a mirrored default. The
  docstring says so.
- **`Config.TIMEFRAME` has no default on purpose.** If a test builds `Config` from a partial ini, add
  the required `[general]` keys (`REQUIRED_GENERAL` in `test_config_and_pricing.py` is the pattern)
  instead of reintroducing a fallback.

## Checklist for the next change that touches configuration

1. Read through `Config` only. No literal frame, no local default, no `or "4h"`.
2. Adding a key: `config/config.ini.example` + `Config.REQUIRED_CONFIG_KEYS` (if the runtime cannot
   start without it) + `docs/CONFIGURATION.md` + the contract test, in the same change.
3. Renaming/removing a key: same three artefacts plus the contract gate run.
4. Run the fast gate from AGENTS §12.3; the two new scripts are part of it.
5. If a gate fires on something legitimate, add an `ALLOWLIST` entry **with a reason string** —
   never silence the script, never widen the rule.
6. A new "the agent forgot X" bug: add a row to the §12.1 enforcement map and the gate that would
   have caught it, in the same change. That is how the guarantee compounds.
7. Quote the `Effective configuration:` startup line as evidence instead of assuming the ini was read.

## Known rough edges

- `MarketDataCollector` tests that bypass `initialize()` must set `collector.timeframe` themselves;
  `fetch_ohlcv()` now raises (and the surrounding `except` turns that into `False`) when it is unset.
  That is intended: it fails loudly instead of fetching a frame nobody chose.
- `PromptBuilder` already required every formatter (`overview_formatter`, `long_term_formatter`,
  `technical_formatter`, `market_formatter`); a new test must pass all of them, not just the two
  that are directly validated.
- `prompt_builder.py` still contains `"4h"`/`"12h"`/`"24h"` keys — they are multi-timeframe summary
  periods, not the analysis frame, which is why the file is allowlisted. Do not "fix" them by
  deleting; the summary needs those buckets.
- `WritableConfig` no longer offers `1m` in the dashboard. A config that somehow contains `1m` will
  fail at load (`Unsupported timeframe`), which is the correct outcome: the loader always rejected it.

## Follow-up 2026-10-27 — the prompt contract and the brain tool

The first pass missed two consumers of the old frame:

- `TemplateManager.build_response_template()` still said "which way the 4h trend points" and emitted
  `strength_4h`; the tree was green because `test_the_system_prompt_carries_the_configured_frame_and_no_other`
  rendered only `build_system_prompt()`. The AST gate also missed it: it matches a string constant
  that *is* a frame, not a frame embedded in prose. `tests/domain_infra/test_config_propagation.py`
  now renders the complete prompt and the response contract (`-k response_contract`) and asserts that
  only `Config.TIMEFRAME` appears. Treat a green AST gate as "no assigned literal", not "no prompt
  prose"; the render test is the gate for text.
- `scripts/brain_rules.py` hardcoded `brain_BTC_USDC_4h` and `240`, and did not run from its own
  documented command (`No module named 'src'`). It now bootstraps the repo root, resolves
  `brain_store_path(Config.DATA_DIR, Config.CRYPTO_PAIR, Config.TIMEFRAME)`, accepts `--timeframe`
  for an explicitly different store, and passes minutes into `VectorMemoryService`.
- `VectorMemoryService` / `TradingBrainService` no longer default `timeframe_minutes` to 240; they
  refuse a missing/non-positive frame. Closed-trade records now carry the frame and bucket, and a
  malformed rule `source_timeframe_minutes` falls back to the service frame.
- `pattern_recency` and `PromptBuilder` lost their private 240/60-minute fallbacks for the same
  reason. `PROMPT_VERSION` is v1.7; `strength_4h` is accepted only as a legacy input alias for
  `strength_primary`.

Verification for this follow-up: `pytest tests -q` → 2 642 passed; the new render test is red on the
pre-fix tree and green here; ruff, pyright, both config gates and the helper/AGENTS gates are clean.

## Second audit 2026-10-27 — other configured values the prompt quoted from the wrong place

The owner asked what else the same defect class had missed. The answer was four more paths plus
two truthfulness gaps; all are fixed in the same working tree.

| Location | Why it was a defect | Proof before the fix |
|---|---|---|
| `template_manager.build_decision_rules()` | caps `AGGRESSIVE 10% / NEUTRAL 8% / CONSERVATIVE 5%` and `1.5x/2x/2.5x` were literals, while `RiskManager` enforces `min(profile_cap, MAX_POSITION_SIZE)` | `max_position_size=0.05` → the prompt said 10% and "fall back to 0.05 (5%)" in one paragraph |
| `regime_risk_profile.build_profile_context()` | the `ACTIVE RISK PROFILE` block repeated the same percentages/multiples as separate literals, with no access to `MAX_POSITION_SIZE` | same config → "max position 10%" |
| `template_manager._build_timeframe_context()` / relevance window | `timeframe_minutes = 60` and `window_minutes = 120` were silent fallbacks when the frame could not be converted | `1d` + raising validator → `Style: Intraday Swing (1d candles)`; a test pinned it |
| `market_metrics_calculator.update_period_metrics()` | on a bad frame it fell back to `1D=24, 2D=48, …`, i.e. an implicit hourly frame, and reported wrong metrics | `13h` → metrics computed from 24-candle "days" |
| `brain_experience.record_closed_trade()` | wrote `timeframe_minutes: 0` when the frame was omitted; `_timeframe_bucket(0)` called it `scalping` | a record without a frame carried a numeric frame of 0 |
| `vector_memory_rules` | a rule with a non-numeric `source_timeframe_minutes` kept the garbage value while deriving the bucket from the service frame | metadata said `"garbage"` |
| `WritableConfig` + dashboard docs | categories `hot`/`cycle` and a "hot-reload through asyncio.Event" claim; no runtime reload exists (`Config` is read once, at import) | editing `timeframe` in the panel changed the file, not the running bot |

What changed:

- `RegimeRiskProfileSelector.effective_caps()` / `build_profile_context(..., max_position_size=)`
  are the only place a cap is turned into prose; `TradingBrainService` → `BrainContextProvider`
  carry `Config.MAX_POSITION_SIZE` from provisioning.
- `TemplateManager._timeframe_minutes()` and `_configured_position_cap()` raise instead of
  substituting `60`/`120`/`0.10`; the frame is normalised on entry.
- `src/trading/threshold_defaults.py` holds the ~15 unlearned decision defaults once
  (`apply_defaults()`); the prompt and the provider render the same mapping, `learned_keys` stays
  derived from the raw learned dict.
- Brain records omit frame keys when unknown; rule metadata is normalised to the service frame.
- Period metrics raise on an unusable frame.
- Every `WritableConfig` key is `restart`; the docstrings, `/api/admin/config/reload` response,
  `docs/CONFIGURATION.md` and `src/dashboard/AGENTS.md` state that the file changes now and the
  process applies it on the next start (`Shift+R`). A real cycle-level reload is deliberately not
  attempted: `Config` is a singleton and frame-bound services are built at startup, so a partial
  reload would recreate two frames at once — the exact bug this work removes. That needs its own
  design (re-provision frame-dependent services).

Verification: the full suite is **2 653 passed**; the live config (`0.10`) renders byte-identical
cap text (no clamp note), while `0.05` renders the clamped caps and profile percentages; unknown
frames raise in both prompt and metrics paths; `ruff check src scripts start.py` clean; `pyright`
0 errors; all four scripts and `python -c "import start"` clean.

Known edges left alone on purpose: the `240` literals in `brain.py`/`stop_loss_tightening_policy.py`
are category boundaries (`<240` intraday vs swing), not fallback frames; `landing.html`'s demo log
line is marketing copy; the AST timeframe gate still cannot see a frame inside prose, so the
render tests remain the prose gate and a new config-derived sentence needs its own render
assertion.

