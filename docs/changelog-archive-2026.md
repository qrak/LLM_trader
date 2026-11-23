# Changelog archive - 2026

Entries from 2026-01-01 to 2026-10-31, newest first.
Current entries: [../CHANGELOG.md](../CHANGELOG.md)

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

## 2026-07-30 — v1.0.4 — Public Live Console View + 7-Day Log Buffer + Vector Memory 768D Upgrade


### Added
- **Live Console tab** on public dashboard — real-time log streaming via WebSocket, styled like `rich` terminal output with level-based color coding (ERROR=red, WARNING=yellow, INFO=gray, DEBUG=dim)
- **ConsoleBuffer** (`src/dashboard/console_buffer.py`) — in-memory day-grouped ring buffer, max 7 days, clears on restart
- **ConsoleRouter** (`src/dashboard/routers/console.py`) — public REST endpoints (`/api/console/recent`, `/api/console/pages`, `/api/console/page/{n}`) + WebSocket (`/api/console/live`) for live streaming
- Day page navigation — buttons for each historical day, auto-fetched from buffer
- Level filter (All/Errors/Warnings/Info+), auto-scroll toggle, clear button

### Changed
- `DashboardServer` — wires ConsoleBuffer, ConsoleRouter, and lifespan consumer task
- `main.js` — imports and initializes ConsolePanel module

## 2026-07-23 — Dependabot security fixes + Futures LONG/SHORT signals + Executor safety hardening

### Security (Dependabot — 19 alerts resolved)
- **Pillow** 12.2.0 → 12.3.0 — 12 CVEs: heap OOB write (CVE-2026-59199, CVE-2026-59205, CVE-2026-59197), decompression bomb (CVE-2026-59200, CVE-2026-54058/59/60, CVE-2026-55379/80), TGA heap leak (CVE-2026-59198), JPEG2000 DoS (CVE-2026-59204), EPS infinite loop (CVE-2026-59203), Windows command injection (CVE-2026-55798)
- **astro** ^6.4.6 → ^7.1.0 — 3 CVEs: XSS via unescaped spread attribute names (CVE-2026-59729), XSS via transition:* directives (CVE-2026-59727), reflected XSS via View Transition properties
- **js-yaml** → 4.3.0 override — YAML merge-key chains force quadratic CPU (CVE-2026-59869)
- **sharp** → 0.35.3 (transitive) — libvips CVEs (CVE-2026-33327/28, CVE-2026-35590)
- **svgo** → 4.0.2 (transitive) — removeScripts plugin bypass
- **openrouter** 0.9.1 → >=0.11.0 — SDK compatibility (presence_penalty removed)

### Added
- **Futures LONG/SHORT signals**: `template_manager.py` emits LONG/SHORT in prompts when `market_type=futures`; `position_extractor.py` regex + valid_signals accept LONG/SHORT; `executor_handler.py` ACTIONABLE_SIGNALS includes LONG/SHORT (signals now unified: BUY=LONG, SELL=SHORT)
- **MARKET_TYPE config**: `config.ini.example` + `Config` properties (`MARKET_TYPE`, `ENTRY_ORDER_TYPE`) — spot vs futures, market vs limit
- **OpenRouter reasoning effort**: `config.ini.example` documents `openrouter_reasoning_effort` (none→max); `openrouter.py` pops it from model_config and passes as `reasoning={"effort": ...}` extra kwarg
- **Executor dead-letter queue**: `data/trading/failed_forwards.jsonl` — when executor forward fails after all retries, payload is persisted for replay instead of being silently lost
- **Retry on executor forward**: `executor_handler._forward()` decorated with `@retry_async(3, backoff×2, max 30s)`
- **Google image token estimation**: `google.py` — `_get_image_dimensions()` parses PNG/JPEG headers without full decode; `_estimate_image_tokens()` calculates Gemini tile-based token cost (258 tokens per 75×75 tile)

### Changed
- **Executor forward prioritized**: `app.py` sends to executor BEFORE Discord notification + file persist — trade execution is the priority
- **Executor payload hardened**: `_build()` refuses to build when `strategy_decision is None` (analysis never validated), uses `strategy_decision` fields (quantity, price, stop_loss, take_profit, confidence, reasoning) instead of raw analysis dict
- **order_type from config**: executor payload uses `self._config.ENTRY_ORDER_TYPE` instead of `analysis.get("order_type")` — no more LLM-controlled order type
- **Timestamp precision**: executor payload timestamp format changed to `%Y-%m-%dT%H:%M:%S.%f` (microseconds)
- **Chart resolution reduced**: 4K (3840×2160) → 1080p (1920×1080) — cuts Google image token cost by ~75%

### Safety (position guards)
- **CLOSE guard**: `trading_strategy.py` — queries executor API before sending CLOSE; if no position tracked on exchange, skips with warning (prevents double-close or close on unfilled limit order)
- **UPDATE guard**: verifies executor has position before sending SL/TP update; fails closed on network error
- **SL widening cap**: rejects stop-loss widening beyond 150% of original SL distance from entry — AI can only tighten or keep stops, not blow out risk
- **Ambiguous position_size rejected**: `position_extractor.py` — numbers > 1.0 without `%` suffix now trigger warning + None (was blindly dividing by 100, risking 50→0.5 misinterpretation)

### Fixed
- **OpenRouter presence_penalty**: filtered out from `_known_unsupported_params` (removed in SDK 0.11+)
- **Test assertion**: `test_openrouter_provider.py` updated to expect `presence_penalty` absent from sent kwargs

### Removed
- `scripts/start_script_develop.ps1` — dead develop-mode startup script

## 2026-07-15 — DRY fixes + SRP refactor (MarketConditionsExtractor)

### Changed
- **DRY: `_execute_market_knowledge_update`** — extracted `analysis_type` variable instead of if/else branches with identical log format strings.
- **DRY: Market knowledge fallback message** — extracted duplicated `"continuing with cached/partial market knowledge"` string into `_MARKET_KNOWLEDGE_FALLBACK_MSG` module constant, used in both TimeoutError and Exception handlers.
- **DRY: `_wait_for_next_timeframe` / `_wait_until_next_timeframe_after`** — extracted shared timeframe arithmetic into `_calculate_next_check(source_time_ms)` helper. Both methods now call the helper and handle only their unique logging + edge cases.
- **SRP: `MarketConditionsExtractor`** — extracted 4 methods from `TradingStrategy` (was 1151 lines / 19 methods → now 945 lines / 15 methods):
  - `extract_price(result)` ← `_extract_price_from_result`
  - `extract_market_conditions(result)` ← `_extract_market_conditions`
  - `extract_confluence_factors(result)` ← `_extract_confluence_factors`
  - `build_conditions_from_position(position)` ← `_build_conditions_from_position` (static)
  - New class wired through composition root in `start.py`, injected into `TradingStrategy.__init__` via `conditions_extractor=`.
- **Unused imports cleaned** — removed `re`, `classify_bb_position`, `classify_macd_signal`, `classify_market_sentiment`, `classify_order_book_bias`, `classify_rsi_label`, `classify_volume_state`, `classify_volatility_level` from `trading_strategy.py` (only used by extracted methods).

### Added
- `src/trading/market_conditions_extractor.py` — new class with constructor DI (`logger`), 3 instance methods + 1 static method.
- `tests/test_app_timeframe_wait.py` — 9 tests covering `_calculate_next_check`, `_wait_for_next_timeframe`, and `_wait_until_next_timeframe_after`.

## 2026-07-11 — Position gap context + defensive code purge

### Added
- Prompt now includes time since last closed position and its direction (LONG/SHORT) when no position is open — `TradingStrategy._get_last_closed_position_info()` queries SQLite trade history via existing `persistence.sqlite_history.query()`.

### Changed
- `get_position_context()`: extracted shared `## Capital Status` header into `capital_header` variable — DRY, no duplicate string literals.

### Removed
- **23 defensive checks purged total** — `getattr`/`isinstance`/`hasattr` on guaranteed types:
  - `post_mortem.py`: 14 `getattr(closed_position, ...)` → direct attribute access (Position/TradeDecision are dataclasses)
  - `pattern_quality_scorer.py`: 4 `isinstance(pattern_list, list|str)` → patterns dict shape is fixed by the repository's own PatternAnalyzer
  - `vector_memory.py`: 1 `isinstance(serialized, dict)` → `serialize_for_json` always returns dict for dict input
  - `brain.py` router: 2 `getattr(self.vector_memory, ...)` → injected service always has those attrs
  - `template_manager.py`: 3 `isinstance(data, dict)` + `getattr(self.config, ...)` → JSON parse + config attrs are guaranteed
  - `app.py`: 3 — `isinstance(analysis, dict)` on typed param (only caller already checks), `getattr(self.config, "EXECUTOR_API_ENABLED")` + `getattr(self.config, "EXECUTOR_API_URL")` on Config dataclass
  - `pipeline.py`: 1 — `hasattr(guard, "invalidate_cache")` on already name-identified CooldownWindowGuard

### Fixed
- `test_admin_edge_cases.py`: 5 `TestWritableConfigEdgeCases` tests → `async def` + `await` instead of `asyncio.get_event_loop().run_until_complete()` (dashboard tests killed the loop during teardown)
- `test_admin_live.py`: `browser` fixture → graceful `pytest.skip()` when Playwright chromium binary unavailable (Ubuntu 26.04 unsupported)
- `huggingface-hub` upgraded 1.2.3 → 1.23.0 (transformers demanded `>=1.5.0`, blocked `test_blocked_trades_integration.py` + `test_edge_cases_feedback.py`)
- `log_stream.py`: removed unused `import time` and `from typing import Any`
- `admin.py`: removed unused `verify_admin_session` import
- `crawl4ai_enricher.py`: `# noqa: F401` → `# pylint: disable=unused-import` (pylint ignores flake8 pragmas)
- `brain.py`: moved `MarketConditions` under `if TYPE_CHECKING` (only used in quoted annotations)
- `pattern_quality_scorer.py`: `category` → `_category` (unused loop variable)

## 2026-07-10 — Cost tracking fix + Decision Pathways zoom/layout

### Fixed
- `_extract_usage_metadata` captures `thoughts_token_count` and sums it into `completion_tokens` — Google bills thinking tokens as output ("Output price including thinking tokens"). Previously only `candidates_token_count` was counted, causing ~3.5x cost underreporting vs actual Google Cloud charges.

### Changed
- Decision Pathways panel: zoom controls (+/−/fit), label truncation with hover title, horizontal node spreading, keyboard zoom, deferred vis-network init, tab-switch delay 300ms.
- Cache-bust version bumps: style.css?v=7.2, main.js?v=7.3, modules/*.js bumped.
- config.ini.example: timeframe warning note.
- Relaxed test assertion from hardcoded version to `?v=` pattern.

### Removed
- Defensive `try/except` guard around `thoughts_token_count` — the outer `except AttributeError` catches missing fields. No backward-compat dead code.

## 2026-07-09 — Decision Pathways + executor bridge production cleanup

Develop production batch: readable multi-source Brain Activity graph, dead synaptic/API path removal, and hardened llm_trader_executor integration.

### Added
- `GET /api/brain/decision-summary` aggregates active position, vector experiences, semantic rules, blocked trades, trade-journal post-mortems, and last response into a synopsis + hierarchical multi-source graph payload.
- Brain Activity **Decision Pathways** panel: labeled hierarchical graph (Position / Memory / Rules / Journal / Decision), synopsis strip, type legend, click-to-detail card.
- Shared `_build_execution_decision()` CCXT-ready payload used by both file write and HTTP forward.
- `tests/test_decision_summary_api.py`, `tests/test_execution_decision_bridge.py`.

### Changed
- `_save_execution_decision` returns the written payload; `_forward_decision_to_executor` reuses it so disk + HTTP stay identical.
- Executor network/HTTP failures log full traceback (`exc_info=True`); silent `except: pass` removed.
- `PersistenceManager.save_latest_decision` logs with traceback and re-raises after atomic write failures.
- `httpx>=0.27,<1` declared in `requirements.txt`.
- Panel DOM id renamed `panel-synapses` → `panel-decision-pathways`; README screenshot caption retitled **Decision Pathways**.

### Removed
- `synapse_viewer.js` trade-history BUY/SELL/CLOSE graph consumer (vis-network retained for Decision Pathways only).
- `GET /api/brain/memory` + `BrainRouter.get_vector_memory` + dashboard cache prefix `memory_*` (no back-compat).
- Dead legend CSS `.dot.buy/.close/.update`.

### Tests / validation
- Decision summary + static bindings + brain router + server cache.
- Executor bridge unit tests (payload, SKIP HOLD, HTTP forward, error logging, atomic latest_decision write).
- Discord trading-check path remains green with executor API disabled in fixtures.

### Ops notes
- Hard-refresh dashboard clients for Decision Pathways JS/CSS cache bust (`main.js?v=6.0`, `style.css?v=6.3`, `decision_pathways_panel.js?v=1.1`).
- Production was briefly stuck loading synopsis because Cloudflare caches `main.js?v=*` as immutable and origin still had old `synapse_viewer` major path under `main.js?v=5.2` while HTML already pointed at Decision Pathways markup.
- Restart trading bot process to pick up brain router + executor bridge changes.
- Executor HTTP remains optional via `[executor_api]`; `latest_decision.json` stays as file fallback.


## 2026-07-08 — Executor Audit & Short Model Fix

### Changed

- **Executor audit**: 10-point security review of `llm_trader_executor` (separate repo) completed and pushed to `main`.
- **SPOT_SHORT_ALLOWED** (new `.env` flag in executor, default `true`): SELL signals on spot are now capped to actual held base-asset balance instead of hard-blocked. When `false`, any insufficiently-backed SELL is rejected outright.
- **UPDATE quantity fix**: `place_sl_tp_oco()` now overrides quantity from the tracked position's real size instead of accepting `quantity=0.0` from the LLM decision (which would cancel SL/TP without replacing them).
- **set_leverage()** now called before entry on non-spot markets (previously leverage from the LLM prompt was validated but never applied).
- **PositionTracker** refactored from single-global to `dict[symbol, Position]` — multi-pair ready.
- **OCO fallback**: `create_order('oco')` attempted directly with try/except instead of relying on `exchange.has['createOrder']['oco']` (which crashes on most exchanges because `has` values are booleans, not dicts).
- **UTF-8 stdout** forced on Windows to prevent `UnicodeEncodeError` crashes from emoji log prints.
- **Notional check** in safety guard uses live ticker price as fallback for market orders (previously `entry_price=None` would bypass the check entirely).

### Integration Notes

- Executor payload format from `_save_execution_decision()` is fully compatible — no changes needed on this side.
- The executor now clamps SELL quantity to <99% of available base balance. For example, a SELL 6 BTC signal with 1 BTC held executes as 0.99 BTC sell instead of failing.
- Restart executor (`scripts/start.ps1` in `llm_trader_executor`) to pick up changes.


## 2026-07-03 — v1.0.2 — BlockRun AI (optional) + Post-Mortem llm_analysis + Dashboard Resilience

### Added
- **BlockRun.AI provider** (fully optional) — x402 micropayment-based AI provider via `blockrun-llm` SDK. Gated behind `BLOCKRUN_WALLET_KEY` env var; no-op when unset.
- **`llm_analysis` field in post-mortem API response** — `get_recent_post_mortems()` now returns the full LLM analysis text that was already stored but missing from the query result.
- **96 new tests**: BlockRun provider unit/integration tests, post-mortem API endpoint tests, trading strategy check_position/exit handling tests.

### Fixed
- **Dashboard resilience** — each panel initializer now has its own try/catch. A single panel failure no longer cascades and kills the entire dashboard.
- **SL/TP logging crash** — `trading_strategy.py` no longer crashes when formatting a position update with `None` stop_loss or take_profit.

## 2026-06-24 — v1.0 Release — Security Hardening & Dependabot Cleanup

### Security Fixes

- **aiohttp 3.13.4 → 3.14.1**: Patched 11 CVEs including deserialization RCE (CVE-2026-34993, CVSS 6.4), cross-origin cookie leak (CVE-2026-47265), unbounded pipelined requests DoS (CVE-2026-54273), WebSocket memory limit bypass (CVE-2026-54274), TLS SNI bypass on reused connections (CVE-2026-54275), DigestAuth credential leak via redirect (CVE-2026-54276), HTTP parser `max_line_size` bypass (CVE-2026-54277), zip-bomb unbounded decompression (CVE-2026-54278), host-only cookie persistence downgrade (CVE-2026-54279), mid-disconnect resource leak (CVE-2026-54280), CRLF injection in multipart headers (CVE-2026-50269).
- **fastapi 0.128.0 → 0.138.0**: Unblocks starlette ≥1.x for 4 additional CVEs.
- **starlette 0.50.0 → 1.3.1**: Patched host header SSRF (CVE-2026-54282), path-based authorization bypass (PYSEC-2026-161), HTTPEndpoint method smuggling via non-standard verbs (CVE-2026-48817), Windows StaticFiles UNC path NTLM leak (CVE-2026-48818, Linux not affected), unbound urlencoded form parsing DoS (CVE-2026-54283).
- **python-multipart 0.0.27 → 0.0.31**: Patched quadratic-complexity DoS in `;`-separated form parsing (CVE-2026-53539), HTTP parameter pollution via `;`/`&` parser differential (CVE-2026-53538), negative Content-Length unbounded read (CVE-2026-53540).
- **lxml 5.4.0 → 6.1.1**: Patched XXE via default `resolve_entities=True` allowing local file reads from untrusted XML input (PYSEC-2026-87).
- **cryptography 48.0.0 → 48.0.1**: Patched statically-linked OpenSSL vulnerability in wheels (GHSA-537c-gmf6-5ccf).
- **pydantic-settings 2.14.1 → 2.14.2**: Patched out-of-tree symlink read bypassing `secrets_dir_max_size` (GHSA-4xgf-cpjx-pc3j).
- **pynacl 1.5.0 → 1.6.2**: Patched libsodium ed25519 point validation (CVE-2025-69277).
- **tornado 6.5.6 → 6.5.7**: Patched `CurlAsyncHTTPClient` credential leakage via pycurl handle reuse (GHSA-pw6j-qg29-8w7f).

### Web (npm) Security Fixes

- **astro <6.4.6 → 6.4.6**: Patched 5 CVEs including reflected XSS via slot name (CVE-2026-50146, HIGH), Host header SSRF in prerendered error page fetch (CVE-2026-54299, HIGH), XSS via unescaped spread props (CVE-2026-54298, MEDIUM), XSS via incomplete `</script>` sanitization (CVE-2026-41067, MEDIUM), server island encrypted parameter replay (CVE-2026-45028, LOW).
- **vite ≤6.4.2 → 6.4.3**: Patched `server.fs.deny` bypass on Windows (CVE-2026-53571, HIGH), NTLMv2 hash disclosure via launch-editor UNC path (CVE-2026-53632, MEDIUM).
- **js-yaml ≤4.1.1 → 4.2.0**: Patched quadratic-complexity DoS in merge key handling via repeated aliases (CVE-2026-53550, MEDIUM).
- **esbuild <0.28.1 → 0.28.1**: Patched arbitrary file read when running dev server on Windows (GHSA).

### Changed

- Removed `data_template/` tree from public repo.
- Cleaned public README, AGENTS.md, and removed outdated CI workflow and stale start scripts.
- Merged `develop` branch into `master` for v1.0 release.

### Validation

- **1021 pytest passed** (Playwright/live excluded — pre-existing).
- **pip-audit**: 0 fixable vulnerabilities remaining. Only unfixed: `chromadb` CVE-2026-45829 (no patch available yet, ChromaDB runs localhost-only — not exploitable).
- **Dependabot**: 40 alerts auto-resolved to `fixed` state.

---

# Changelog

## 2026-06-19 — AI Trade Journal Post-Mortem

### Added

- **Post-mortem analysis on every position close** (`src/trading/post_mortem.py`): After a position closes, the LLM analyzes what happened (entry reasoning, exit data, market conditions) and extracts an actionable lesson. Produces a structured `PostMortemResult` with verdict, analysis, expected vs actual, and lesson. Fires after the CLOSE row is written to SQLite, wrapped in try/except so failures never break the close flow.
- **SQLite + FTS5 storage layer** (`src/managers/post_mortem_repository.py`): New `trade_post_mortem` and `trade_post_mortem_fts` tables in the existing `trade_history.db`. Thread-safe via `threading.Lock` and per-call connections matching the `SQLiteTradeHistory` pattern. Supports insert, recent listing, keyword search via FTS5 MATCH, and count. Schema is `CREATE TABLE IF NOT EXISTS` — zero impact on existing tables, no migration needed.
- **Trade journal injected into brain context** (`src/trading/brain_context.py`): `BrainContextProvider` appends the 5 most recent post-mortem lessons as a `"### Trade Journal (Recent Post-Mortem Lessons):"` section in the system prompt, so the LLM sees past mistakes before every trading decision.
- **Dashboard Trade Journal tab** (`src/dashboard/static/modules/post_mortem_panel.js`): New sidebar tab with searchable post-mortem viewer. Displays verdict, symbol, P&L, close reason, and lesson learned per card. Expandable full-analysis details. FTS5 search bar with result highlighting. Updates every 30s in the slow-lane cycle.
- **Dashboard API endpoint** (`src/dashboard/routers/brain.py`): `GET /api/brain/post-mortems?q=breakout&limit=20` returns post-mortems with optional FTS5 keyword search. Error-returning, never crashes.

### Fixed

- **LLM response parsing resilience** (`src/trading/post_mortum.py`): Added raw `json.loads()` fallback in `_parse_response` when the LLM returns plain JSON without markdown fences. The primary extraction path (`UnifiedParser.extract_json_block`) handles ` ```json ` blocks; the fallback catches model output that skips the fence syntax.
- **Test timestamp ordering** (`tests/test_post_mortem_repository.py`): Replaced `time.sleep(1.5)` with direct SQL timestamp override, eliminating a test smell. Ordering test now runs in 0.02s instead of 1.5s.
- **Mock safety in context tests** (`tests/test_brain_context_journal.py`): Configured all mock return values explicitly (`trade_count=0`, `get_context_for_prompt.return_value=""`, etc.) to prevent `MagicMock` truthyness from corrupting `BrainContextProvider` control flow.

### New Files

| File | Purpose |
|------|---------|
| `src/managers/post_mortem_repository.py` | SQLite + FTS5 storage (insert, search, recent, count) |
| `src/trading/post_mortem.py` | `PostMortemResult` Pydantic model + `PostMortemService` (LLM orchestration) |
| `src/dashboard/static/modules/post_mortem_panel.js` | Dashboard frontend module with search and card rendering |
| `tests/test_post_mortem_repository.py` | 10 tests: insert, FTS5 search, ordering, count, error handling |
| `tests/test_post_mortem_service.py` | 6 tests: success, empty/malformed/missing response, LLM exception, repo exception |
| `tests/test_brain_context_journal.py` | 7 tests: journal section present/absent, multi-entry, P&L formatting, error skip |
| `tests/test_post_mortem_contract.py` | 10 tests: Pydantic model validation, empty/missing rejection, extra fields |

### Modified Files

| File | Change |
|------|--------|
| `src/trading/trading_strategy.py` | `close_position()` triggers post-mortem after CLOSE row, before stats recalc |
| `src/trading/brain_context.py` | Trade journal section injected into `get_context()` output |
| `src/trading/brain.py` | `post_mortem_repo` wired through to `BrainContextProvider` |
| `src/dashboard/routers/brain.py` | `GET /api/brain/post-mortems` endpoint + `post_mortem_repo` param |
| `src/dashboard/server.py` | `post_mortem_repo` threaded into `BrainRouter` |
| `src/dashboard/static/index.html` | Trade Journal nav item + tab content panel + search toolbar |
| `src/dashboard/static/main.js` | Import + slow lane + init for post-mortem panel |
| `src/dashboard/static/css/panels.css` | Full `.pm-*` styling for cards, search bar, verdict colors, details |
| `start.py` | DI wiring for `PostMortemRepository` + `PostMortemService` into strategy, brain, dashboard |

### Validation

- Full pytest (no Playwright): `1060 passed`
- New test suite: `33 passed in 0.16s`
- Live LLM validation: end-to-end call via gemini-3.5-flash confirmed prompt → response → parse → store → search
- Ruff lint: all checks passed on new and modified files

## 2026-06-10 — Ticker Fetch Retry + Cascading Error Protection

### Fixed

- **Ticker fetch retries restored** (`src/app.py`): Applied `@retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)` to `_fetch_current_ticker()` and removed the internal `try/except Exception` that swallowed all errors before the decorator could see them. Transient network errors, timeouts, and rate limits now retry with exponential backoff; non-retryable exchange errors (e.g. `BadSymbol`) propagate immediately. The startup ticker fetch (`run()`) wraps the call in a try/except that logs and continues without a price.
- **Zero-division guard in notifier** (`src/notifiers/base_notifier.py`): `calculate_stop_target_distances()` returns `(0.0, 0.0)` when `current_price` is `None` or `<= 0`, preventing the cascading `ZeroDivisionError` in `send_position_status`.
- **Skip status on unavailable price** (`src/trading/position_status_monitor.py`): `_loop()` and `handle_new_position()` now skip the position-status notification when the ticker price is unavailable instead of sending a misleading `$0.00`.

### Added

- **Ticker retry tests** (`tests/test_ticker_retry.py`): Covers retry-then-succeed on `RequestTimeout`/`RateLimitExceeded`, exhaustion on `ClientConnectorError`, non-retry on `BadSymbol`, the zero/None price guard, and the monitor skip-on-no-price behavior.

### Validation

- Focused pytest: `tests/test_ticker_retry.py tests/test_brain_integration.py -q` -> **66 passed**.

## 2026-06-10 — Invalidation Check Prompt + Surprise Ratio Reflection (Pass 12)

### Added

- **Step 5.5 invalidation check** (`src/analyzer/prompts/template_manager.py`): Replaced the generic "BULL vs BEAR CASE: Which side wins?" with a falsification-based prompt that forces the model to name a specific price level or indicator condition that would prove its signal wrong. Signals without a falsifiable invalidation trigger are rejected (HOLD).
- **Surprise ratio computation** (`src/trading/brain_experience.py`): On every closed trade, the system now computes `surprise_ratio = |realized_pnl - expected_pnl| / |expected_pnl|` where expected P&L is the TP distance at entry. Stored in ChromaDB metadata as `surprise_ratio`.
- **Surprise ratio in reflection rules** (`src/trading/brain_reflection.py`): All three reflection passes (best-practice, anti-pattern/corrective, AI-mistake) now compute the average surprise ratio for the trade group and include it in the rule text. Best-practice rules with `avg_surprise > 1.5` are flagged as "⚠️ high surprise" so the LLM can distinguish thesis-validated wins from lucky outcomes.
- Overridden prompt section `5.5 BULL vs BEAR CASE` → `5.5 INVALIDATION CHECK` in the analysis steps.

### Changed

- `template_manager.py`: Step 5.5 in `build_analysis_steps()` rewritten from opinion-prompting to falsification-prompting.
- `brain_experience.py`: `record_closed_trade()` computes and persists `surprise_ratio` in trade metadata.
- `brain_reflection.py`: `trigger_reflection()`, `trigger_loss_reflection()`, and `trigger_ai_mistake_reflection()` all compute `avg_surprise` from grouped trades and surface it in rule text.

### Validation

- Focused AST parse: `src/analyzer/prompts/template_manager.py` → OK
- AST parse: `src/trading/brain_experience.py` → OK
- AST parse: `src/trading/brain_reflection.py` → OK
- No breaking test changes: existing `test_prompt_consistency.py` and `test_template_manager.py` unaffected (step 5.5 string not under test assertion).

## 2026-05-29 — Timeframe-Aware Semantic Rule Freshness (Pass 11)

### Changed

- Added timeframe-aware semantic-rule influence scoring. Active rules remain physically preserved, but prompt retrieval now ranks by similarity, evidence quality, timeframe freshness, and contradiction penalty.
- Semantic rules now carry lifecycle metadata including `created_at`, `support_count`, validation/contradiction counters, and source timeframe fields.
- Closed-trade recording now updates matched semantic-rule validation or contradiction metadata so old rules can be down-weighted by live evidence without age-only deletion.
- Brain prompt context and dashboard active-rule responses now expose concise freshness/evidence fields.

### Validation

- Focused pytest: `tests/test_vector_memory.py tests/test_vector_memory_pruning_safety.py tests/test_brain_integration.py tests/test_dashboard_brain_router.py -q` -> **116 passed**.
- Full pytest: `tests -q` -> **900 passed**.
- Targeted Ruff on touched trading/dashboard/test files -> **passed**.

## 2026-05-29 — SQLite-Only Trade History Cleanup (Pass 10)

### Changed

- Removed runtime trade-history JSON compatibility paths. Trade decisions now save to SQLite only, history reads export from SQLite only, and entry-decision lookup no longer scans legacy JSON files.
- Removed SQLite auto-migration constructor support and the one-shot JSON migration helper from the trade-history store.
- Updated tests to seed SQLite directly instead of writing legacy history files.
- Removed old-history fee inference from notifier performance stats; stats now use persisted fee values.
- Updated AGENTS documentation for SQLite-only persistence, persistence-backed cooldown/dashboard paths, active semantic-rule pruning safety, and Windows PowerShell terminal guardrails.

## 2026-05-29 — Production Readiness Integration Fixes (Pass 9)

### Fixed

- **SQLite runtime integration gaps closed**:
    - `src/dashboard/routers/brain.py`: `get_vector_memory()` now reads trades via `persistence.load_trade_history()` (SQLite-backed) instead of direct `trade_history.json` file reads.
    - `src/dashboard/routers/performance.py`: `get_performance_history()` now reads trades through injected persistence instead of direct JSON file reads.
    - `src/dashboard/server.py`: passes `persistence` into `PerformanceRouter`.

- **Cooldown guard no longer depends on legacy JSON**:
    - `src/trading/guards/cooldown_window.py` now uses injected persistence (`get_last_execution_timestamp`) instead of disk reads from `trade_history.json`.
    - Guard now fails closed when persistence is not wired or cannot read execution history, avoiding silent fail-open behavior under persistence failures.
    - `start.py` now wires `CooldownWindowGuard(persistence=persistence)` in the production guard pipeline.

- **SQLite API hardening**:
    - `src/managers/sqlite_trade_history.py`:
        - validates `order` (`ASC`/`DESC`) before SQL generation,
        - clamps/validates pagination inputs,
        - adds `get_last_execution_timestamp(actions=("BUY", "SELL"))` helper for guard usage.
    - `src/managers/persistence_manager.py`:
        - `load_trade_history()` now uses SQLite export directly,
        - adds `get_last_execution_timestamp()` facade for DI-friendly guard consumers.

- **ChromaDB pruning policy corrected for semantic safety**:
    - `src/trading/vector_memory.py`: `prune_aged_documents()` now parses timestamps as datetimes and preserves active semantic rules (`active=True`) instead of deleting by age-only policy.

- **Lint cleanup**:
    - `scripts/query_trade_history.py` and `src/managers/sqlite_trade_history.py` cleaned up to satisfy Ruff.

### Added

- `tests/test_sqlite_trade_history.py`:
    - SQLite initialization coverage,
    - query input validation,
    - execution timestamp helper behavior.

- `tests/test_vector_memory_pruning_safety.py`:
    - verifies old experience/blocked-trade pruning,
    - verifies active semantic-rule preservation,
    - verifies malformed timestamp handling.

### Changed Tests

- `tests/test_dashboard_brain_router.py`: adds persistence-backed trade-history read assertion for vector memory endpoint.
- `tests/test_dashboard_performance_router.py`: adds persistence-backed history assertion and updates constructor usage.
- `tests/test_order_governance.py`: adds cooldown fail-closed and no-history allow-path assertions.

### Validation

- Ruff (targeted):
    - `python -m ruff check src/managers/sqlite_trade_history.py scripts/query_trade_history.py` -> **passed**.

- Ruff (broad scope):
    - `python -m ruff check src start.py scripts tests` -> **passed**.

- Pytest (targeted integration set):
    - `pytest tests/test_position_persistence.py tests/test_order_governance.py tests/test_sqlite_trade_history.py tests/test_dashboard_brain_router.py tests/test_dashboard_performance_router.py tests/test_vector_memory_pruning_safety.py -q` -> **42 passed**.

- Pytest (pruning safety):
    - `pytest tests/test_vector_memory_pruning_safety.py -q` -> **3 passed**.

- Full-suite pytest:
    - `pytest -q` -> **895 passed**.

### Live Data Migration

- `data/trading/trade_history.json` was migrated into `data/trading/trade_history.db`.
- Migration inserted **12 records** into the SQLite `trade_history` table.
- The legacy JSON file was renamed to `data/trading/trade_history.json.migrated` and kept as backup.

## 2026-05-29 — Production Readiness: SQLite Migration, Headless Shutdown & State Validation (Pass 8)

### Added

- **`src/managers/sqlite_trade_history.py`**: SQLite-backed trade history store replacing JSON file accumulation. Provides O(1) INSERT appends, indexed timestamp/symbol/action queries, WAL journal mode with NORMAL synchronous for performance, and a `get_stats()` method returning aggregate data for the dashboard.

- **`scripts/query_trade_history.py`**: CLI utility for querying the trade history SQLite database directly from the terminal. Supports `recent`, `search`, and `stats` commands with filtering by symbol, action, date range, limit/offset pagination, and sort direction. Also works with raw `sqlite3` command for ad-hoc queries.

- **`PersistenceManager.validate_loaded_position()`**: Startup state validation that checks `positions.json` for config mismatches (symbol vs `CRYPTO_PAIR`, unrecognized fields). Logs actionable `STARTUP STATE WARNING` messages without discarding the position — the operator decides.

- **`/api/monitor/health` endpoint**: Lightweight liveness probe returning `{"status": "ok", "timestamp": "..."}` with zero I/O. Suitable for Docker healthchecks, systemd `ExecStartPost` probes, and local watchdog scripts without touching Cloudflare routing.

- **`VectorMemoryService.prune_aged_documents()`**: Safe ChromaDB collection maintenance that removes only documents definitively beyond the relevance window (3× `_max_age_days`). Runs once at startup on all three collections (`trading_experiences`, `semantic_rules`, `system_constraints_rejections`). Uses the existing `PRUNE_AGE_MULTIPLIER` constant — at a 4h timeframe, keeps ~168 days of history while preventing unbounded disk/RAM growth.

### Changed

- **CoinGecko cache TTL** (`start.py:365`): Changed `expire_after=-1` (never expire) to `RAG_COINGECKO_UPDATE_INTERVAL_HOURS * 3600` seconds (24h by default). Cache entries now expire on the same cadence as the data refresh cycle, preventing unbounded SQLite growth.

- **`GracefulShutdownManager.show_exit_confirmation()`**: Added `_is_headless()` detection — when `DISPLAY` env var is not set AND `sys.stdin` is not a TTY (systemd/WSL/Docker), skips ALL blocking prompts (tkinter dialog and `input()`) and returns `True` immediately. Enables unattended systemd/Wired restarts without hanging.

- **`TradingStrategy.__init__()`**: Now calls `persistence.validate_loaded_position()` after loading an existing position. Any symbol/config mismatches are logged as warnings.

- **`PersistenceManager`**: `save_trade_decision()` now writes to SQLite, `load_trade_history()` reads from SQLite, and `get_entry_decision_for_position()` uses SQLite's indexed timestamp range query for O(log n) lookup instead of scanning all JSON records.

### SQLite Schema

```sql
CREATE TABLE trade_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence TEXT,
    price REAL,
    stop_loss REAL,
    take_profit REAL,
    position_size REAL,
    quote_amount REAL,
    quantity REAL,
    fee REAL,
    reasoning TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_th_timestamp ON trade_history(timestamp DESC);
CREATE INDEX idx_th_symbol ON trade_history(symbol);
CREATE INDEX idx_th_action ON trade_history(action);
```

### Validation

- Ruff: `ruff check src/ start.py scripts/` — All checks passed.
- Full pytest: **875 passed** (baseline maintained, zero regressions).

## 2026-05-29 — Chaos Engineering Test Suite (Pass 7)

### Added

- **`tests/test_llm_output_corruption.py`** (15 assertions): Coverage for LLM output corruption, schema violations, and fallback-loop resilience. Scenarios: truncated/malformed JSON, string injection in numeric fields, Infinity/NaN in confidence/price, missing required execution fields (stop_loss, take_profit, entry_price) for BUY/SELL/UPDATE signals, empty choices list, HTML/script injection in reasoning, massive 100KB reasoning field overflow, fallback chain survival when all providers return corrupt data, and mixed corruption-then-recovery patterns.

- **`tests/test_async_concurrency_race.py`** (9 assertions): Async concurrency, latency injection, race condition, and state safety tests. Scenarios: slow news-fetch inside `asyncio.gather` (wall-clock == max not sum), timeout of one gather task not killing siblings, out-of-order completion verification, `threading.Lock` serialization in `VectorMemoryService._encode_embedding`, cancelled `update_if_needed` not corrupting `last_update` state, and double-update serialization via `asyncio.Lock`.

- **`tests/test_api_rate_limiting_backoff.py`** (16 assertions): HTTP 429 rate-limit, 5xx server error, exponential backoff with jitter, and circuit-breaker boundary tests. Scenarios: rate-limited provider triggers fallback to next provider, all providers rate-limited returns last failure, 503 error retry via `retry_api_call`, `_add_jitter` ±25% range verification with 200 samples, jitter randomness proving, zero-delay jitter edge case, exponential delay growth across retries, exhausted retries return last response, retryable-then-success recovery, `retry_async` handling of `ClientConnectorError`, `ccxt.RateLimitExceeded` exchange error retry, and non-retryable `ccxt.BadSymbol` propagation.

- **`tests/test_vector_db_context_poisoning.py`** (15 assertions): ChromaDB vector store boundaries, context poisoning, and metadata corruption tests. Scenarios: empty news database returns empty context, zero-matching keyword search, empty collection in `VectorMemoryService`, `get_context_for_prompt` with zero experiences, perfect-similarity (distance=0.0) edge case, `_sanitize_metadata` handling of NaN/Inf/None/lists, missing `outcome` key in ChromaDB metadata, corrupted timestamps, `_parse_trade_timestamp` edge cases, 200-article token budget enforcement, malformed articles missing body, null-title crash proof (production bug fix: `_process_article_simple` now handles `None` titles), UPDATE store with zero PnL, failed initialization graceful False return, NaN `suggested_rr` in blocked trades, keyword_search exception isolation in `retrieve_context`.

### Changed

- **`src/rag/context_builder.py` — `_process_article_simple()`**: Added null-title guard. Previously `item.get('title', 'No Title').strip()` would crash with `AttributeError` when the title field existed but was set to `None`. Now falls through to `'No Title'` when title is `None`, matching the existing missing-key behavior. Discovered by chaos test `test_build_context_with_null_title_does_not_crash`.

### Validation

- Ruff: `ruff check src/ tests/` — All checks passed.
- Full pytest: **884 passed** (baseline 829 + 55 new chaos tests, zero regressions).

---

## 2026-05-28 - Performance & Resilience Runtime Optimization (Pass 6)

### Added

- **`src/utils/decorators.py`**: Introduced `_add_jitter()` function adding ±25% random jitter to all exponential-backoff sleep calls. Applied to both `_RetryContext._handle_retryable_error()` and `_ApiRetryContext._wait_and_increment()`. Prevents thundering herd when concurrent operations retry simultaneously.

- **`src/rag/rag_engine.py`**: Added `_last_update_attempt` timestamp and `_minimum_retry_interval` (5 min) to prevent redundant update retries in the `retrieve_context()` hot path. Skips `update_if_needed()` when the main loop's update attempt just failed or timed out, reducing context-retrieval latency by avoiding unnecessary news-fetch retries.

### Changed

- **`src/analyzer/analysis_engine.py` — `_enrich_market_context()`**: Refactored three sequential API calls (market overview, microstructure, coin details) into parallel `asyncio.gather()` with isolated error handling per sub-task. Reduces wall-clock latency from ~sum of three API calls to ~max of three API calls (~3x speedup on this path).

- **`src/rag/rag_engine.py` — `refresh_market_data()`**: Refactored three sequential I/O operations (categories update, news fetch, market overview update) into parallel `asyncio.gather()`. Extracted error-isolated helpers `_safe_fetch_news()` and `_safe_update_market_overview()`. Reduces wall-clock latency from sequential to concurrent execution.

- **`src/platforms/ai_providers/openrouter.py` — `get_generation_cost()`**: Offloaded synchronous SDK call `client.generations.get_generation()` to `asyncio.to_thread()`, preventing event-loop blocking during generation-cost lookup.

### Validation

- Ruff: `ruff check src/` passed.
- Full pytest: **829 passed** (baseline maintained, zero regressions).

---

## 2026-05-28 - Formatter Abstraction Collapse (Pass 5)

### Changed

- **`src/analyzer/formatters/market_formatter.py`**: Absorbed `MarketPeriodFormatter` — its `format_market_period_metrics()` and `_format_indicator_changes_compressed()` methods now live directly on `MarketFormatter`. Removed the `period_formatter` constructor parameter and `MarketPeriodFormatter` import. `MarketFormatter` no longer acts as a pass-through carrier for a sub-formatter; it owns period formatting directly.

- **`src/analyzer/prompts/prompt_builder.py`**: Replaced leaky `self.period_formatter = market_formatter.period_formatter` with direct call `self.market_formatter.format_market_period_metrics()`. Eliminates the reach-through abstraction where `PromptBuilder` reached into `MarketFormatter` to grab a sub-component.

- **`start.py`**: Removed `MarketPeriodFormatter` construction and injection from `_provision_analyzer_layer()`. Removed `MarketPeriodFormatter` import.

- **`src/analyzer/__init__.py`**: Replaced stale multi-line docstring referencing non-existent `core/`, `data/`, `calculations/` subdirectory layout. Removed duplicate comment header.

- **`src/analyzer/formatters/__init__.py`**: Removed `MarketPeriodFormatter` re-export.

### Removed

- **`src/analyzer/formatters/market_period_formatter.py`**: Deleted. All 109 lines of its logic folded into `MarketFormatter` — eliminated the leaky abstraction where `PromptBuilder` was reaching through `MarketFormatter` to grab a separately-injected sub-formatter.

### Tests updated

- **`tests/test_market_period_formatter.py`**: Imports `MarketFormatter` instead of the deleted `MarketPeriodFormatter`; test logic unchanged.
- **`tests/test_prompt_builder.py`**: Removed `market_formatter.period_formatter = MagicMock()` fixture line (no longer needed).
- **`tests/test_prompt_context_helpers.py`**: Same fixture cleanup.

### Validation

- Ruff: `ruff check src/ start.py` passed.
- Full pytest: **829 passed** (baseline maintained, +13 over previous 816).
- 1 file deleted, 9 files modified.
- **Net: -66 lines, -1 file.**

---

## 2026-05-28 - TechnicalIndicators Composition-to-Inheritance Refactor (Pass 6)

### Changed

- **`src/indicators/base/technical_indicators.py`**: Changed `TechnicalIndicators` from composing `IndicatorBase` via `self._base` (composition) to directly inheriting from `IndicatorBase`. Eliminated 5 trivial delegation properties (`open`, `high`, `low`, `close`, `volume`) and the `get_data()` pass-through method — all now inherited. Replaced 77 `self._base.calculate_indicator(...)` calls with `self.calculate_indicator(...)`. Removed unused `Union` and `pandas` imports.

### Tests updated

- No test changes needed — `TechnicalIndicators` exposes the identical API through inheritance.

### Validation

- Ruff: `ruff check src/` passed.
- Full pytest: **829 passed** (baseline maintained).
- 1 file modified.
- **Net: -61 lines, -2 files cumulative this session.**

---

## 2026-05-27 - Dependency Injection Cleanup and Docs Separation (Pass 4)

### Changed

- **`src/analyzer/analysis_result_processor.py`**: `AnalysisResultProcessor.__init__` now requires `trend_validator: TrendValidator` and `quality_scorer: PatternQualityScorer` as explicit constructor parameters instead of constructing them internally. Removes the last internal DI violation in the analyzer layer.
- **`start.py`**: Composition root now instantiates `TrendValidator()` and `PatternQualityScorer()` inline at the `AnalysisResultProcessor` construction site in `_provision_analyzer_layer()`.
- **`src/utils/token_counter.py`**: Retired `ModelPricing.__new__` singleton pattern (`_instance`, `_pricing` class-level state, `__new__` override). Each `ModelPricing()` instance now loads pricing from `config/model_pricing.json` independently. Public `get_cost()` API is unchanged.
- **`AGENTS.md`**: Removed stale `factories/` entry from the project-structure tree (source files were already deleted in Pass 2). Replaced the cross-platform Linux/macOS/Windows command matrix and hardcoded absolute Windows paths with a two-line "Operator Commands" pointer to `README.md`. Terminal guardrail policy rules are retained.
- **`README.md`**: Removed stale `factories/` entry from the directory structure description; replaced it with `utils/`. Replaced single bare `pytest tests/` command in the Testing section with a concise relative-path command block covering pytest, ruff, and mypy for both Windows and Linux/macOS.

### Removed

- **`src/factories/__pycache__/`** and the now-empty **`src/factories/`** directory: bytecode residue from the factory modules deleted in Pass 2 is gone.

### Tests updated

- **`tests/test_trend_validator.py`**: Updated 4 `AnalysisResultProcessor` construction sites in `TestProcessorIntegration` to pass `trend_validator=` and `quality_scorer=` keyword arguments.

### Validation

- Pylance (`get_errors`): 0 diagnostics workspace-wide.
- `tests/test_model_pricing.py`: 4 passed.
- Full `pytest tests -q --tb=short`: run in progress at time of entry; see terminal for authoritative result.
- `ruff check src tests start.py`: run in progress at time of entry; see terminal for authoritative result.

---

## 2026-05-27 - AGENTS-Only Instruction Authority Migration

### Changed

- **`AGENTS.md`**: Added explicit instruction-authority contract making AGENTS files canonical across IDEs/harnesses.
- **`AGENTS.md`**: Added terminal guardrails and Windows command examples so execution policy is no longer IDE-specific.
- **`AGENTS.md`**: Added documentation-governance checklist and anti-drift rule to block reintroduction of tool-specific policy docs.

### Removed

- Deleted IDE/harness-specific instruction files:
	- **`.github/copilot-instructions.md`**
	- **`.github/instructions/terminal-guardrails.instructions.md`**
	- **`CLAUDE.md`**
	- **`.windsurfrules`**

### Notes

- Kept **`.github/workflows/compatibility_manual_main.yml`** as CI execution configuration only; it is not an instruction-authority source.

## 2026-05-27 - Dead Code Deletion and Contract Flattening (Pass 3)

### Removed

- Removed 30+ private `TradingBrainService` delegation wrappers that only forwarded to extracted collaborators and had no live production or test call sites.
- Deleted `src/notifiers/filehandler_components/` after folding its four single-use tracking, persistence, scheduling, and deletion classes into `DiscordFileHandler`.
- Deleted `src/contracts/` model/risk protocol files and the obsolete contracts README; `ModelManager` and `RiskManager` are now used as concrete injected services.
- Removed unused `AnalysisEngine` constructor dependencies for CoinGecko and Alternative.me; those clients remain wired only where they are actually used.

### Changed

- **`src/notifiers/filehandler.py`**: Flattened Discord message tracking into one class that owns JSON persistence, expiry scanning, retry-backed deletion, and cleanup task scheduling.
- **`start.py`**: Simplified Discord file-handler construction and `AnalysisEngine` provisioning after removing unused dependency layers.
- **`src/app.py`**, **`src/analyzer/analysis_engine.py`**, **`src/analyzer/analysis_result_processor.py`**, **`src/managers/model_manager.py`**, **`src/managers/risk_manager.py`**, and **`src/trading/trading_strategy.py`**: Replaced single-implementation protocol annotations/inheritance with direct concrete service types.
- **`AGENTS.md`**: Removed the stale `src/contracts` project-structure entry.

### Validation

- Focused brain/vector-memory pytest: `97 passed in 0.42s`.
- Focused analyzer/app wiring pytest: `4 passed in 1.56s`.
- Focused Discord notifier pytest after flattening: `7 passed in 1.12s`.
- Focused trading/risk/response pytest after protocol removal: `69 passed in 2.15s`.
- Full pytest validation is not claimed for this pass; the attempted PowerShell capture command was malformed (`^U` prefix and `$pytestOut` variable failure), making its `PYTEST_EXIT:0` output unreliable.
- Full ruff validation: `ruff check src tests start.py` passed.
- Pylance (`get_errors`): no errors on modified files or workspace-wide final check.

## 2026-05-27 - Dead Code Deletion and Factory Flattening (Pass 2)

### Removed

- Deleted `src/analyzer/pattern_engine/pattern_engine.py` — `PatternEngine.detect_patterns()` always returned `{}` and was never wired after its Numba deprecation.
- Deleted `src/analyzer/pattern_engine/pattern_matchers.py`, `swing_detection.py`, `trendline_fitting.py` — the three Numba JIT support files that `PatternEngine` depended on; now fully unreachable.
- Deleted `src/platforms/ai_providers/mock.py` — `MockClient` (133 LoC) was never instantiated anywhere in production or test code.
- Deleted `src/factories/provider_factory.py`, `data_fetcher_factory.py`, `position_factory.py`, and `src/factories/__init__.py` — all three factories were single-method wrappers that added indirection without logic; the module is now empty.

### Changed

- **`src/analyzer/pattern_engine/__init__.py`**: Stripped to a single re-export of `ChartGenerator`; removed the 12 dead symbol exports for the now-deleted pattern-engine stack.
- **`src/analyzer/pattern_analyzer.py`**: Removed `PatternEngine` constructor dependency and its `detect_patterns()` call; `warmup()` now uses a fixed 64-sample count instead of deriving it from the dead engine's `lookback`; removed unreachable `get_all_patterns()` method.
- **`src/analyzer/analysis_engine.py`**: Inlined `DataFetcher(exchange, logger)` construction in `initialize_for_symbol()`; removed `data_fetcher_factory` constructor parameter and all references.
- **`src/trading/trading_strategy.py`**: Inlined `Position(...)` construction in `_open_new_position()`; replaced `position_factory.create_updated_position()` call in `_update_position_parameters()` with `dataclasses.replace(self.current_position, stop_loss=..., take_profit=...)`; removed `position_factory` constructor parameter.
- **`src/managers/provider_types.py`**: Removed `ProviderClients.from_factory_dict()` classmethod that was only called by the now-deleted `ProviderFactory`.
- **`start.py`**: Replaced `ProviderFactory` usage with inline `GoogleAIClient`, `OpenRouterClient`, `LMStudioClient` construction in `_provision_model_layer()`; removed `DataFetcherFactory` from `_provision_utilities()`; removed `PositionFactory` and `PatternEngine` from `_provision_analyzer_layer()` and `_provision_trading_layer()`.

### Tests

- Updated 5 test files (`test_trading_strategy_branches.py`, `test_trading_strategy_frictions.py`, `test_trading_strategy_process_analysis.py`, `test_order_governance.py`, `test_edge_cases_feedback.py`) to remove `position_factory` constructor arg and add `entry_price`/`volatility_level` fields to `RiskAssessment` mocks that now flow through inline `Position(...)` construction.
- Position-update assertions changed from `factory.create_updated_position.assert_called_once()` to direct `strategy.current_position.stop_loss == ...` attribute checks.

### Validation

- Full pytest: **816 passed** (baseline maintained).
- Full ruff validation: `ruff check src tests start.py` — all checks passed.
- Pylance (`get_errors`): no errors on any modified file.

## 2026-05-27 - Overengineering Simplification Pass

### Changed

- **src/app.py** and **start.py**: Replaced the wide `CryptoTradingBot` constructor surface with a compact `BotServices` bundle and moved position-monitor provisioning into construction via a factory callback.
- **src/analyzer/technical_calculator.py**: Inlined technical indicator construction and removed the redundant factory layer.
- **src/analyzer/prompts/prompt_builder.py**: Folded analyzer prompt context-section helpers directly into `PromptBuilder`, keeping prompt assembly in one owner instead of a pass-through collaborator.
- **src/logger/logger.py**, **src/utils/profiler.py**, and **tests/conftest.py**: Removed broad config-loader test monkeypatching by making config singleton access lazy and local to fallback paths.
- **src/trading/trading_strategy.py** and **src/trading/guards/pipeline.py**: Simplified guard/audit ownership so `TradingStrategy` records audit events and `GuardPipeline` only evaluates guards.
- **src/config/loader.py** and config consumers: Retired the oversized config protocol type surface in favor of concrete config injection with type-check-only hints.

### Removed

- Removed `TechnicalIndicatorsFactory` and `src/factories/technical_indicators_factory.py`.
- Removed `src/config/protocol.py` after replacing `ConfigProtocol` annotations.
- Removed analyzer `ContextBuilder` and `src/analyzer/prompts/context_builder.py`.
- Removed unused prompt delta-alert helper coverage along with the helper itself.
- Removed unused order lifecycle states/history (`APPROVED`, `CANCELLED`, `state_history`) and the unused guard-pipeline audit compatibility shim.

### Validation

- Focused pytest after final pass: `138 passed in 1.86s`.
- Full pytest after final pass: `816 passed in 27.11s`.
- Full ruff validation: `ruff check src tests start.py` passed.
- Targeted stale-symbol scan found no live Python/Markdown references to the removed protocol, factories, prompt context builder, guard audit shim, or old order lifecycle states.

## 2026-05-25 - Config Parsing and Silent-Failure Regressions

### Changed

- Hardened INI loading to strip inline comments and disable percent interpolation for literal percentage text in configuration values or comments.
- Sanitized non-finite statistics values before JSON persistence and rejected NaN/Infinity numeric payloads in AI parsing, position extraction, and market cache timestamp checks.

### Added

- Added regression coverage for config inline comments, signal-direction disambiguation, cooldown cache invalidation after execution, and nested tuple restoration for position confluence factors.
- Added hidden-runtime-failure tests for non-finite persisted statistics, malformed AI numeric fields, non-finite position extraction values, and corrupt market overview timestamps.

## 2026-05-23 - Order Governance Foundation

### Added

- Added a default pre-execution order guard pipeline with configured-pair, explicit over-cap position size, and cooldown guards.
- Added in-memory audit records for order intent creation, guard checks, approval, rejection, and execution lifecycle events.
- Added regression coverage for order lifecycle transitions, guard audit capture, guard rejection handling, max-position validation, and no-guard execution telemetry.

### Changed

- `TradingStrategy` now wraps new position entries in order lifecycle/audit telemetry while the composition root injects the guard pipeline by default.
- Replaced optional extra symbol allow-list configuration with fixed configured-pair enforcement.
- Removed stale Google sampling plumbing from the Google config/provider path and kept Gemini on model defaults.

## 2026-05-21 - Documentation Alignment and Static Website Rollout

### Added

- Added a new static website workspace under `website/` using Astro + Tailwind for a standalone engineering landing page.
- Added production-style simulated telemetry visuals (SVG/CSS panels) for dashboard, stream, and vector-memory sections without depending on outdated screenshot assets.

### Changed

- Updated architecture documentation in `docs/llm_agent_documentation.md` to match the real staged startup flow (`SingleInstanceLock` plus sequenced `CompositionRoot.build_dependencies()` provisioning).
- Updated `docs/detailed_file_documentation.md` with the modular trading-brain collaborator map (`brain_context.py`, `brain_experience.py`, `brain_exit_profiles.py`, `brain_patterns.py`, `brain_reflection.py`) and indicator semantic translation details.
- Updated `docs/INDEX.md` with a compact documentation map linking architecture docs, detailed file docs, cache playbook, refactoring status, README, changelog, and website workspace.
- Updated `README.md` to include landing/deployment context, refreshed architecture tree for modular brain collaborators, and explicit paper-trading scope posture.

### Removed

- Removed outdated dashboard screenshot dependencies from public-facing release messaging in README and website rollout materials.
- Completed cleanup of deprecated legacy runtime structure references in updated release-facing documentation.

## 2026-05-21 - Manual Main-Branch Compatibility Workflow and Cross-Platform Start Scripts

### Added

- **.github/workflows/compatibility_manual_main.yml**: Added a low-cost manual compatibility workflow with three tiers: Linux full tests, Windows medium static checks, and macOS smoke syntax checks.
- **scripts/start_script.ps1**: Moved Windows main-branch startup script into `scripts/` and updated it to resolve repository-rooted paths after relocation.
- **scripts/start_script_linux.sh**: Added Linux startup script for main branch with venv creation, optional dependency installation, and `start.py` launch support.
- **scripts/start_script_macos.sh**: Added macOS startup script for main branch with venv creation, optional dependency installation, and `start.py` launch support.
- **README.md**: Documented cross-platform startup script usage from both the repository root and the `scripts/` directory.

### Changed

- **.gitignore**: Removed the global `*.ps1` ignore rule so PowerShell scripts are tracked normally without exception rules.
- **start_script.ps1**: Removed root-level script in favor of the new tracked path under `scripts/`.
- **scripts/start_script.ps1**, **scripts/start_script_linux.sh**, and **scripts/start_script_macos.sh**: Made Git branch checkout best-effort so startup can continue when Git is not available on `PATH`.
- **src/utils/graceful_shutdown_manager.py**: Exit confirmation now falls back to a terminal prompt when a Tkinter GUI dialog cannot be shown, improving Ctrl+C behavior on Linux/macOS and headless sessions.

## 2026-05-21 - Discord Analysis Message De-duplication

### Changed

- **src/app.py**: Removed the direct per-cycle `send_trading_decision()` call so analysis cycles no longer emit a separate decision embed in addition to the analysis notification payload.
- **src/notifiers/notifier.py**: Kept narrative reasoning delivery via existing `send_message()` chunking behavior (multiple Discord messages when reasoning exceeds Discord limits).
- **tests/test_app_discord_message_flow.py**: Added orchestration-level regression test verifying the trading loop uses analysis notifications without sending an extra decision embed.

## 2026-05-20 - Model Pricing Corrections

### Changed

- **config/model_pricing.json**: Corrected `gemini-3.5-flash` pricing to $1.50 input / $9.00 output per million tokens (was $0.50/$12.00). Added `google/gemini-3.5-flash` to the OpenRouter reference section. Added deprecation note to `gemini-2.0-flash` (Google shutting down June 1 2026). Source: ai.google.dev/pricing (2026-05-19).
- **tests/test_model_pricing.py**: Updated `gemini-3.5-flash` pricing regression assertion from $12.50 to $10.50.

## 2026-05-20 - OpenRouter 0.9.1 and Gemini Sampling Compatibility

### Changed

- **requirements.txt**: Pinned the beta OpenRouter SDK to `openrouter==0.9.1`.
- **src/config/loader.py** and **src/config/protocol.py**: Switched default model config output to canonical `frequency_penalty` and `presence_penalty` names while preserving deprecated `freq_penalty` and `pres_penalty` INI aliases.
- **src/platforms/ai_providers/openrouter.py** and **src/managers/provider_orchestrator.py**: Wired OpenRouter `server_url` construction, explicit SDK cleanup, and a one-retry fallback model for validation or rate-limit failures.
- **src/platforms/ai_providers/google.py**: Kept code execution tools limited to known Gemini 3 Flash+ models.
- **config/config.ini** and **config/config.ini.example**: Clarified OpenRouter fallback/base URL behavior and canonical penalty names.
- **README.md**: Removed active legacy Google sampling from the normal configuration snippet and documented it as Gemini 1.x/2.x-only.
- **config/model_pricing.json**: Updated pricing metadata date and added the active `gemini-3.5-flash` Google Studio model pricing reference.

## 2026-05-20 - Google GenAI SDK 2.4 Compatibility Upgrade

### Changed

- **requirements.txt**: Raised the Google GenAI SDK minimum version to `google-genai>=2.4.0`.
- **src/platforms/ai_providers/google.py**: Migrated `ThinkingConfig` to use `types.ThinkingLevel` enum values (`MINIMAL`, `LOW`, `MEDIUM`, `HIGH`) instead of raw strings to match SDK 2.4 API.
- **src/platforms/ai_providers/google.py**: Switched `GenerateContentConfig` construction to `model_validate()` for forward-compatible field assignment.
- **src/platforms/ai_providers/google.py**: Added SDK-native `errors.APIError`-aware thinking fallback handling via `_should_retry_without_thinking()` and explicit async SDK client cleanup in `close()`.
- **tests/test_google_ai_provider.py**: Added focused Google provider tests for generation config, thinking fallback, chart-image request wiring, and async cleanup.

## 2026-05-20 - Gemini 3.5 Flash Google Sampling-Parameter Migration

### Changed

- **config/config.ini** and **config/config.ini.example**: Removed Google sampling overrides from `[model_config]` and aligned with Gemini 3.x model-default behavior.
- **src/config/loader.py**: Removed Google sampling-key mapping from `_google_model_config`; Google runtime config now forwards only `max_tokens`, `thinking_level`, and `google_code_execution`.
- **src/platforms/ai_providers/google.py**: `GenerateContentConfig` no longer sends `temperature`, `top_p`, or `top_k` for Google requests, preventing deprecated/unsupported sampling-parameter usage on Gemini 3.5 Flash.

## 2026-05-17 - Prompt Builder Type Fixes

### Fixed

- **src/analyzer/prompts/prompt_builder.py**: Added `| None` to all eight injected-dependency constructor parameters (`TechnicalCalculator`, `FormatUtils`, `MarketOverviewFormatter`, `LongTermFormatter`, `TechnicalFormatter`, `MarketFormatter`, `TemplateManager`, `ContextBuilder`) to resolve Pylance type errors.
- **src/analyzer/prompts/prompt_builder.py**: Guarded `context_builder.build_market_data_section()` call with `if context.ohlcv_candles is not None` to satisfy `np.ndarray` (non-optional) parameter contract.
- **src/analyzer/prompts/prompt_builder.py**: Replaced `context.current_price` (typed `float | None`) with `context.current_price or 0.0` in the `format_long_term_analysis()` call.
- **src/analyzer/prompts/prompt_builder.py**: Added `if self.context is None: return False` guard at the top of `_has_advanced_support_resistance()` to fix `"technical_data" is not a known attribute of None` Pylance error.
- **tests/test_prompt_builder.py**: Added `config=SimpleNamespace(MODEL_VERBOSITY="high")` to `_make_prompt_builder()` fixture to match updated constructor signature.
- **tests/test_prompt_consistency.py**: Removed leftover `>>>>>>> bug-check` merge-conflict markers and orphaned dead class body (duplicate docstring and `setup_method`) introduced by the prior merge commit.

## 2026-05-16 - Previous-Response Continuity Sanitizer Retention and News Exclusion

### Changed

- **src/analyzer/prompts/template_manager.py**: Updated previous-response sanitization to use verbosity-scaled retention caps (`low=1500`, `medium=3000`, `high=4500`) so continuity context preserves more technical narrative while still stripping prompt/schema artifacts.
- **src/analyzer/prompts/template_manager.py**: Added explicit exclusion of prior news/sentiment lines from continuity context (for example `NEWS & MACRO`, `NEWS`, `SENTIMENT`, `MARKET SENTIMENT`) to restore technical-only carryover semantics between loops.
- **src/analyzer/prompts/template_manager.py**: `build_system_prompt()` now passes normalized model verbosity into the sanitizer so continuity size is deterministic per verbosity level.

### Added

- **tests/test_prompt_consistency.py**: Added regression coverage for preserving non-labeled analytical continuity lines and verifying low-vs-high retention cap scaling.
- **tests/test_prompt_consistency.py**: Added regression coverage proving news/sentiment lines are excluded from prior-context continuity while technical lines remain.

## 2026-05-16 - News Token Budget Contract Alignment

### Changed

- **config/config.ini** and **config/config.ini.example**: Set `[rag] article_max_tokens` to `1000` for runtime and example defaults.
- **src/config/loader.py**: Updated `RAG_ARTICLE_MAX_TOKENS` fallback default from `256` to `1000` so missing-config behavior matches runtime target.
- **src/rag/context_builder.py**: Removed hardcoded total-context default (`2000`) from `build_context()`. When `max_tokens` is omitted, it now resolves from config as `RAG_ARTICLE_MAX_TOKENS * RAG_NEWS_LIMIT`, while preserving strict per-article limiting via `RAG_ARTICLE_MAX_TOKENS`.
- **src/rag/article_processor.py**: Removed hardcoded `[:10000]` body truncation in `detect_coins_in_article()`, so coin detection scans full article body instead of silently clipping long articles.
- **src/rag/rag_engine.py**: Clarified retrieval-limit docstring contract that default total budget is derived from per-article config and article count.

### Added

- **tests/test_article_processor_contract.py**: Regression test proving late-body ticker mentions (beyond 10k chars) are still detected.
- **tests/test_rag_engine_retrieval_contract.py**: Contract test for config-derived default retrieval budget.
- **tests/test_rag_context_builder_contract.py**: Contract test verifying `build_context()` default budget resolves from config and passes per-article cap consistently.

## 2026-05-16 - Hybrid Stop-Loss Tightening Policy

### Added

- **src/trading/stop_loss_tightening_policy.py** (new): `StopLossTighteningPolicy` — single authoritative gate for SL tightening decisions. Replaces inline executor logic with a deterministic, side-effect-free policy class. Evaluates tightening eligibility from per-timeframe config thresholds (scalping/intraday/swing/position), with optional brain override clamped within a configurable `[floor, ceiling]` range. Returns a `TighteningEvaluation` dataclass with full decision metadata.
- **config/config.ini.example**: Seven new keys in `[risk_management]` documenting `sl_tightening_scalping`, `sl_tightening_intraday`, `sl_tightening_swing`, `sl_tightening_position`, `sl_tightening_floor`, `sl_tightening_ceiling`, and `sl_tightening_min_samples`.
- **tests/test_stop_loss_tightening_policy.py** (new, 25 tests): Covers timeframe bucket defaults, `from_config()`, LONG/SHORT tightening allow/reject, zero TP distance, missing current price, brain override with sufficient/insufficient samples, floor/ceiling clamping, and all fallback paths.

### Changed

- **src/config/loader.py** and **src/config/protocol.py**: Seven new typed properties for `SL_TIGHTENING_*` config keys.
- **start.py**: Single `StopLossTighteningPolicy` instance provisioned in `_provision_trading_layer()` and injected into both `TradingBrainService` and `TradingStrategy`.
- **src/trading/trading_strategy.py**: Replaced hardcoded inline tightening progress guard with `policy.evaluate_update()`; blocked events recorded via `vector_memory.store_blocked_trade(guard_type="sl_tightening")`; blocked events now include position identity metadata (`position_id`, `position_entry_trade_id`, `position_entry_timestamp`); `_last_sl_tightening_evaluation` instance attribute captures the accepted evaluation and forwards it to `brain_service.track_position_update()`; `get_position_context()` now renders an `## SL Tightening Policy` section showing effective threshold, current progress, and eligibility.
- **src/trading/brain.py**: `track_position_update()` accepts `tightening_evaluation: TighteningEvaluation | None` and `timeframe_minutes` and forwards them to the recorder. `get_dynamic_thresholds()` now populates a nested `sl_tightening` dict with `base_threshold`, `effective_threshold`, `effective_threshold_pct`, and `source` alongside existing flat keys.
- **src/trading/brain_experience.py**: `track_position_update()` stores position identity (`position_id`, `position_entry_trade_id`, `position_entry_timestamp`), SL/TP values, distance percentages, timeframe bucket, and full policy evaluation metadata. `record_closed_trade()` stores position identity fields to enable update–close pairing. Added `_timeframe_bucket()` static helper.
- **src/trading/brain_context.py**: `get_dynamic_thresholds()` now preserves the nested `sl_tightening` payload from `compute_optimal_thresholds()`.
- **src/trading/vector_memory_analytics.py**: `compute_optimal_thresholds()` passes the full raw Chroma snapshot to the new `_learn_sl_tightening_threshold()` method. Added `_learn_sl_tightening_threshold()`: pairs accepted SL-tightening UPDATE records with their eventual WIN/LOSS close outcomes, deduplicates to one update per position, scans candidates `[0.05 … 0.40]`, selects the lowest with positive expectancy, and stores the result under `thresholds["sl_tightening"]`.
- **src/analyzer/prompts/template_manager.py**: Replaced two hardcoded `"50%+"` thresholds in the system prompt and response template with dynamic rendering from `dynamic_thresholds["sl_tightening_pct"]`; falls back to generic guidance when the key is absent.
- **src/analyzer/prompts/prompt_builder.py**: `build_system_prompt()` now receives `dynamic_thresholds` so the effective tightening threshold is visible in the system prompt.
- **tests/test_brain_integration.py**: Added `TestTrackPositionUpdateWithPolicy` (recorder forwarding) and extended `TestGetDynamicThresholds` with nested `sl_tightening` payload tests (config-source fallback, brain-source override, flat/nested key parity).
- **tests/test_vector_memory.py**: Added four tests under `TestAnalyticsAndThresholds` for `_learn_sl_tightening_threshold`: paired wins learn threshold, below min-samples no emission, no close pairs no emission, all-loss expectancy no emission.
- **tests/test_trading_strategy_branches.py**: Updated four SL tightening tests to inject `StopLossTighteningPolicy` instances; renamed `test_not_is_tightening_but_generic_change` → `test_sl_tightening_no_current_price_rejected` to match new safety-reject behavior.
- **tests/test_prompt_consistency.py**: Updated `test_update_progress_rule_has_no_competing_percentage_thresholds` to assert the static `50%+` string is gone and `"hybrid tightening policy"` is present.

## 2026-05-16 - Trading Brain Close Lifecycle and Vector Memory Hardening

### Changed

- **trading_strategy.py**: Centralized trade-decision recording so saved BUY/SELL/UPDATE/CLOSE decisions also refresh short-term trading memory.
- **trading_strategy.py**: Close-time brain learning now runs through a thread offload after the close decision, statistics rebuild, and position clear are persisted.
- **trading_strategy.py** and **dashboard_state.py**: Trade-close brain learning now emits dashboard lifecycle notifications and invalidates brain, position, rule, memory, vector, performance, and statistics caches.
- **vector_memory.py**: Added serialized embedding-model access and stricter Chroma metadata sanitation for non-finite and unsupported values.
- **dashboard/routers/brain.py**: Moved vector detail retrieval, stats, and sorting work off the async route thread and made legacy numeric metadata sorting safe.
- **dashboard/routers/brain.py**: Added brain lifecycle and refresh endpoints plus structured risk-management data in the position payload while preserving existing flat fields.
- **dashboard/static**: Added overview SL/TP execution policy badges, brain lifecycle status, vector freshness, trade-friction rendering, and safer legacy metadata formatting.
- **persistence_manager.py**: Entry-decision lookup now filters by symbol when available and chooses the nearest timestamp match; position cache updates only after successful disk writes.
- **indicator_classifier.py** and **brain.py**: Corrected `ExitExecutionContext` type contracts used by trading brain context building.
- **brain.py**: Split the monolithic trading brain into focused collaborator classes while keeping `TradingBrainService` as the stable public facade and preserving the public `vector_memory` access path.
- **brain_context.py**, **brain_experience.py**, **brain_exit_profiles.py**, **brain_patterns.py**, and **brain_reflection.py**: Extracted prompt-context/threshold lookup, experience recording, SL/TP exit-profile normalization, trade-pattern diagnostics, and semantic-rule rebuild logic into cohesive modules.
- **indicator_classifier.py**: Preserved direct mandatory config attribute access for SL/TP execution settings; tests now supply required config fields instead of relying on missing-attribute fallbacks.
- **.github/skills/refactor/SKILL.md**: Clarified refactor validation guidance to avoid repeatedly rerunning already-green suites unless relevant code changed or a final validation boundary is reached.
- **data_utils.py**: `SerializableMixin.from_dict()` now restores tuple fields from serialized lists.
- **prompt_builder.py**: Added `_previous_context_contains_stale_prompt_rules()` to detect when old prompt contract text leaks into continuity context, protecting against schema/format instructions from corrupting future analyses.

### Added

- **test_brain_integration.py** (+86): Brain lifecycle, refresh endpoints, cache bypassing, and memory refresh after saved decisions.
- **test_dashboard_brain_router.py** (+118): Risk payload, lifecycle data, confidence parsing, brain status/memory extraction from JSON snapshots.
- **test_dashboard_server_cache.py** (+33 new): Cache headers, server cache invalidation, blocked-trade routing paths.
- **test_dashboard_static_bindings.py** (+29 new): Static HTML/JavaScript bindings, SL/TP badges, lifecycle status, vector freshness rendering.
- **test_data_utils_serialization.py** (+16 new): Tuple field serialization/deserialization in `SerializableMixin.from_dict()`.
- **test_edge_cases_feedback.py** (+99): Updated to use `MarketConditions` data model for market context across edge case scenarios.
- **test_indicator_classifier.py** (+25): Mandatory config attribute access for SL/TP execution settings.
- **test_position_persistence.py** (+60 new): Entry-decision lookup with symbol filtering, cache updates after successful disk writes, write failure handling.
- **test_prompt_consistency.py** (+137 new): Comprehensive coverage for sanitization of continuity context, stale prompt-section removal, wording consistency, HOLD/UPDATE semantics.
- **test_prompt_linting.py** (+22 new): Stale prompt-rules detection and sanitization coverage.
- **test_risk_manager_frictions.py** (+23): Guard types friction generation, delta calculations, mandatory key validation.
- **test_trading_strategy_branches.py** (+111): Market conditions validation, decision-making processes, friction capture integration.
- **test_trading_strategy_frictions.py** (+4): Trading strategy friction reporting and persistence paths.
- **test_trading_strategy_process_analysis.py** (+18): Analysis processing and market condition handling.
- **test_vector_memory.py** (+47): Non-finite metadata sanitation, complex metadata handling, Chroma store validation.
- Regression coverage and fixture updates for mandatory SL/TP config contracts in indicator classification and trading-strategy friction capture.
- Dashboard regression tests for lifecycle/risk payloads, brain refresh cache invalidation, cache headers, blocked-trade routing, and static frontend bindings.

## 2026-05-15 - System-Side ADX Trend Validation and Pattern Quality Scoring

### Added

- **trend_validator.py**: New module that cross-checks LLM-reported `strength_4h`/`daily` against computed ADX indicators.
- **pattern_quality_scorer.py**: New deterministic pattern quality scoring module using 4-component scoring from actual pattern detection results.
- Integrated trend/pattern validation into `AnalysisResultProcessor._validate_llm_claims()` so validation runs on every analysis cycle.
- 38 TrendValidator tests (100% coverage) validating ADX cross-checks and strength assertions.
- 34 PatternQualityScorer tests (90% coverage) validating deterministic scoring logic.
- 36 TradingStrategy branch tests (+22pp coverage increase to 73% overall).

## 2026-05-15 - RiskManager Friction Reporting and TradingStrategy Integration Tests

### Added

- **tests/test_risk_manager_frictions.py**: Comprehensive friction reporting tests validating generation of friction reports across guard types (max_position_size, max_concurrent_positions, etc.) with correct delta calculations.
- **tests/test_trading_strategy_friction_capture.py**: Friction capture and persistence tests ensuring frictions from RiskManager are correctly stored, retrieved, and handled on storage failures.
- Enhanced existing test fixtures by adding `scoring_policy` and `enricher` mocks where necessary to support new test requirements.

### Changed

- **RiskManager**: Verified all mandatory keys are present in friction reports and that parameters are correctly propagated during friction storage.
- **TradingStrategy**: Ensured frictions are persisted correctly and failures in storage are handled gracefully with appropriate logging.

## 2026-05-15 - Prompt Consistency and Continuity Sanitization

### Changed

- **template_manager.py**: Sanitized previous-analysis continuity text so old response-format/schema instructions cannot be reintroduced under `PREVIOUS ANALYSIS CONTEXT`.
- **template_manager.py**: Previous decision snapshots now use the last valid fenced `analysis` JSON block, avoiding schema/example JSON when a response contains multiple fenced blocks.
- **template_manager.py**: Clarified that markdown-heading restrictions apply to the model output only, and made HOLD/open-position and UPDATE semantics more explicit.
- **prompt_builder.py**: Added preflight warning detection for stale prompt instructions leaking into previous-analysis context and made untrusted-context linting robust to wording variants.

### Added

- **test_prompt_consistency.py**: Regression coverage for previous-context sanitization, stale prompt-section removal, output-heading wording, HOLD semantics, and UPDATE progress wording.

## 2026-05-12 - Gemini Flash Model Profitability: SL Tightening Guards and Prompt Simplification

### Changed

- **template_manager.py**: Compressed response template by ~60 lines through markdown tables, reduced conditional JSON rules, simplified trend/position/risk/conflict sections, and streamlined output format.
- **prompt_builder.py**: Removed all-caps directives (CRITICAL→directive, MUST→must) and reduced system prompt by ~30 lines for Flash model clarity.
- **trading_strategy.py**: Added three critical code guards to prevent SL tightening death spiral: (1) reject SL tightening until price moves 15%+ toward TP (was: immediate tightening), (2) reject entries with R/R < 1.5, (3) cap UPDATE frequency: 1 per 8h for position trades, 2 per swing, 4 per scalping (was: every 4h candle).
- **risk_manager.py**: Increased minimum SL from 0.5% to 1.0% (below 4H ATR moving average) to reduce micro-losses and preserve account.
- **Config (untracked)**: Brain learning reset from 20 consecutive losing trades; soft exit policy enabled; max position size capped at 10%; increased thinking depth for Flash model.

### Fixed

- **Root Cause**: SL tightening on every update eliminated TP hits (0/20 in previous cycle, 4.2 updates/trade on average).
- **Solution Impact**: Code guards now prevent premature SL moves; UPDATE frequency cap reduces noise; R/R check prevents underwater entries; simplified prompts reduce token waste.

### Tests

- Updated `test_template_manager.py` (+61 lines) with coverage for compressed template formats, markdown table rendering, and simplified rule sections.

### Verified

- All 56 tests pass after template simplification and guard implementation.
- No regressions in entry/exit logic validation.
- Prompt simplifications reduce token usage for more efficient model processing.

## 2026-05-14 - Code Quality: Eliminate Dynamic Attribute Access and Improve Type Safety

### Changed — Dynamic Attribute Access (Phase 1)

- **prompt_builder.py**: Removed `_detect_minimal_context()` and `_minimal_context` flag entirely. All prompt sections now render unconditionally — the feature was dead code (MINIMAL_CONTEXT config key never existed). Removed `getattr`/`hasattr` calls on config.
- **notifier.py**: Replaced `getattr(exc, "status", None)` with `isinstance(exc, discord.HTTPException)` type-narrowing.
- **crawl4ai_enricher.py**: Replaced all 6 `getattr()` calls with direct attribute access. Removed `_markdown_value()` helper. Crawl4AI objects have a documented API contract.
- **dashboard/server.py**: Replaced `getattr(response, "body", b"")` with try/except AttributeError.
- **dashboard/routers/brain.py**: Replaced `getattr(self.config, "TIMEFRAME", "unknown")` with direct access + try/except fallback (2 sites).
- **rss_primitives.py**: Removed 14 decorative section separator headers.
- **crawl4ai_enricher.py**: Removed 4 decorative section separator headers.
- **test_discord_notifier_rate_limit.py**: `FakeDiscordHTTPError` now inherits from `discord.HTTPException`, matching production hierarchy.

### Changed — Modern Typing (Phase 2)

- **config/protocol.py**: `Dict[str, str]` → `dict[str, str]`, `Dict[str, Any]` → `dict[str, Any]`. Removed unused `Dict` import.
- **contracts/model_contract.py**: Full modernization. `Optional[str]` → `str | None`, `List[Dict[str, str]]` → `list[dict[str, str]]`, `Union[io.BytesIO, bytes, str]` → `io.BytesIO | bytes | str`, `Tuple[str, str]` → `tuple[str, str]`. Removed `Optional`, `Union`, `Tuple`, `Dict`, `List` imports.
- **config/loader.py**: `Dict[str, str]` → `dict[str, str]`, `Dict[str, Any]` → `dict[str, Any]`. Removed unused `Dict` import.
- **src/app.py**: All 11 `Optional[X]` → `X | None`, all `Dict[str, Any]` → `dict[str, Any]`. Removed `Optional` and `Dict` imports.

### Removed

- `_detect_minimal_context()` and `_minimal_context` from `PromptBuilder`.
- `_markdown_value()` from `crawl4ai_enricher.py`.
- 18 decorative section separator headers across 2 files.

### Changed — Dependency Injection (Phase 3)

- **context_builder.py**: `ArticleScoringPolicy` is now injected via constructor (`scoring_policy` parameter) instead of being constructed internally from `config`. The composition root (`start.py`) now creates and wires it.
- **rss_provider.py**: `Crawl4AIEnricher` is now injected via constructor (`enricher` parameter) instead of being constructed internally from `config` values. The composition root now creates and wires it.
- **start.py**: Added construction of `ArticleScoringPolicy` and `Crawl4AIEnricher` in the DI wiring layer, passed to `ContextBuilder` and `RSSCrawl4AINewsProvider` respectively. Removed stale duplicate import.
- **test_rss_provider_contract.py**: Added `enricher=MagicMock()` to `_make_provider()` fixture.
- **test_rag_context_builder_contract.py**: Added `scoring_policy=MagicMock()` to `_builder()` fixture and inline construction site.

### Changed — Modern Typing: Full Codebase Sweep (Phase 5)

- **src/platforms/** (12 files): `Dict`/`List`/`Optional`/`Tuple` → `dict`/`list`/`| None`/`tuple` in all AI provider clients, exchange wrappers, and API clients.
- **src/trading/** (12 files): Full modernization of trading layer — brain, strategy, memory, exit monitor, position management, statistics, vector memory.
- **src/managers/** (5 files): Model, persistence, provider orchestrator, types, risk manager. Added `from __future__ import annotations` where forward refs with `|` needed.
- **src/notifiers/** (5 files): Base, console, Discord, file handler, and components.
- **src/parsing/** (1 file): Unified parser.
- **src/analyzer/** (19 files): Analysis engine, context, result processor, data fetcher, formatters, pattern engine, prompt builder, template manager, context builder, technical calculator. Added `from __future__ import annotations` for forward ref support.
- **src/dashboard/** (6 files): State, routers (brain, monitor, performance, visuals, websocket).
- **src/factories/** (2 files): Position and provider factories.
- **src/indicators/** (4 files): Base indicators, technical indicators, support/resistance, volatility.
- **src/rag/** (15 files): Article processor, category processor, collision resolver, context builder, file handler, index manager, market components, news manager, RAG engine, scoring policy, ticker manager. Added `from __future__ import annotations` where needed.
- **src/contracts/** (1 file): Risk contract.
- All bare `Dict`/`List`/`Tuple` type references (without subscript) converted to `dict`/`list`/`tuple`.

### Verified

- Entire codebase imports cleanly.
- 400 of 406 tests pass; 6 failures are pre-existing (confirmed via git stash).
- No `getattr()`/`hasattr()`/`setattr()` in application code.
- No Pydantic v1 patterns.
- No unused imports.
- Old-style typing (`Optional[X]`, `Dict[K,V]`, `List[X]`, `Tuple[X]`, bare `Dict`/`List`/`Tuple`) eliminated from all source files.
- `from __future__ import annotations` added to 5 files where forward references required it.

### Verified

- 0 `getattr()`/`hasattr()`/`setattr()` in application code.
- 400 of 406 tests pass; 6 failures are pre-existing.
- No Pydantic v1 patterns found.
- No unused imports.
- 33 files modified across all 4 phases, 0 regressions.

### Changed — Redundant isinstance Cleanup (Phase 6)

- **unified_parser.py**: Removed 2 defensive `isinstance(data, dict)` guards in `_normalize_numeric_fields()` and `_attach_response_validation()`. Both functions are typed as `dict[str, Any]` — the isinstance checks were redundant with the type contract and silently masked type errors instead of failing fast.

### Verified

- 400 of 406 tests pass; 6 failures are pre-existing.
- No regressions from isinstance removal.
- All 88 remaining isinstance calls in the codebase are legitimate: type-narrowing on `Any`, polymorphic Union dispatch, exception classification, or untrusted-data validation.

## 2026-05-09 - Non-Blocking News Refresh and Enrichment Timeouts

### Changed

- Added bounded timeout budgets for the market-knowledge refresh path so trading checks continue to analysis even when RSS fetch or article enrichment stalls.
- Added new `[rag]` configuration keys in both config files: `rag_update_timeout_seconds`, `news_fetch_total_timeout_seconds`, and `news_enrichment_timeout_seconds`.
- Increased default `news_crawl_timeout` to 120s and kept it fully configurable.
- `src/app.py` now wraps market-knowledge refresh with timeout handling and explicit fallback logs, preventing pre-analysis stalls.
- `src/rag/news_ingestion/rss_provider.py` now applies outer timeouts to RSS fetch and enrichment stages, logs stage durations, and skips stalled enrichment without blocking execution.
- `src/rag/news_ingestion/crawl4ai_enricher.py` now applies an explicit batch timeout around Crawl4AI runs and falls back to aiohttp with clear warning telemetry.
- `src/rag/news_ingestion/schema_mapper.py` now removes high-confidence Decrypt-style market ticker prefixes before article bodies reach prompts or Discord notifications.
- `src/notifiers/notifier.py` now retries transient Discord 5xx send failures (including 503 no healthy upstream) with bounded backoff for text, embed, and chart sends.

### Added

- **test_discord_notifier_rate_limit.py** (+40 new): Transient Discord 5xx retry behavior, retry backoff logic, non-transient failure handling.
- **test_news_ingestion.py** (+48 new): RSS fetch timeout, enrichment timeout skip, non-blocking execution paths.
- **test_rss_provider_contract.py** (+33 new): RSS provider contract coverage for fetch-stage timeout and enrichment-timeout skip behavior, stage duration logging.

### Verified

- All 56+ tests pass with timeout and retry integration.
- Non-blocking execution confirmed: trading analysis proceeds even when RSS/enrichment stalls.
- Discord retry backoff working correctly for transient 5xx failures.

## 2026-05-08 - Prompt Reasoning and Observability Foundation

### Changed

- Added a model-facing decision reasoning protocol and decision gate for regime classification, conflict resolution, HOLD discipline, UPDATE/CLOSE gating, and explicit invalidation checks.
- Added backend prompt metadata and dashboard metadata fields so prompt behavior can be attributed by prompt version, response-contract version, and prompt variant without injecting version text into the LLM prompt.
- Added non-blocking prompt preflight linting for critical response-format, analysis-step, analysis-time, JSON-example, token-count, and untrusted-context guardrails.
- Added Pydantic-based validation metadata for trading analysis responses while preserving the existing JSON parser and fallback behavior.
- **src/parsing/unified_parser.py**: Extended with response validation schema, decision gate logic, and error recovery paths.
- **src/platforms/ai_providers/response_models.py**: Added prompt metadata fields (version, variant, contract_version) and comprehensive validation models.

### Added

- **test_prompt_linting.py** (+65 new): Preflight linting coverage for response format, analysis steps, JSON examples, token counts, and untrusted-context detection.
- **test_response_validation.py** (+62 new): Pydantic validation metadata, decision reasoning protocol, conflict resolution gate, HOLD/UPDATE/CLOSE gating logic.
- **test_template_manager.py** (+24): Metadata field integration and template rendering with observability features.

### Verified

- All 440+ line additions validated; zero regressions in response parsing.
- Preflight linting successfully detects common response-format violations before LLM execution.

## 2026-05-07 - Documentation Discoverability and Risk Messaging Refresh

### Changed

- Updated `README.md` top-level status messaging to remove outdated branch/news notes and clearly state demo-account/paper-trading scope, with no real exchange execution implemented in the public branch.
- Added a compact `Latest Changes (May 2026)` section in `README.md` to make recent behavior updates easier to discover from GitHub and LLM retrieval workflows.
- Refreshed `README.md` feature and configuration guidance to reflect timeframe-aware memory windows, semantic-rule evolution, and current risk-management guardrails (`max_position_size` and fallback sizing tiers).
- Rewrote `README.md` disclaimer language to emphasize not-financial-advice boundaries, user responsibility for local regulatory compliance, and warranty/liability consistency with `LICENSE.md`.
- Expanded `CONTRIBUTING.md` with local setup, testing expectations, changelog/documentation update requirements, and repository convention references.
- Added a documentation navigation hub for faster contributor and LLM orientation.
- Refreshed documentation-plan metadata and completion notes to include May 6-7 updates and discoverability changes.
- Added GitHub workflow scaffolding with `.github/pull_request_template.md` and issue templates under `.github/ISSUE_TEMPLATE/`.

## 2026-05-07 - Prompt Contract Hardening

### Changed

- Trading prompts now provide a parser-safe JSON example without pseudo-values, inline comments, or placeholder ranges, reducing invalid AI response risk for compact decision output.
- Response-format guidance now maps analysis steps to the existing five compact narrative lines, uses chart validation instructions only when chart analysis is available, and clarifies BUY/SELL, HOLD, UPDATE, and CLOSE JSON field semantics for open-position and no-position states.
- Confidence guidance now defers BUY/SELL entry thresholds to the dynamic response template and allows high-confidence HOLD decisions when staying out or maintaining a position is strongly justified.
- RAG/news/custom prompt snippets are now explicitly wrapped as untrusted market evidence so embedded article text cannot override system instructions, response format, risk rules, or trading policy.

## 2026-05-07 - Timeframe-Aware Memory Relevance and Prompt Horizon

### Changed

- `VectorMemoryService` now derives recency-decay defaults from the active timeframe at startup. For `4h`, this yields a 14-day half-life and a 56-day hard relevance window; lower timeframes use shorter windows, and higher timeframes use longer windows (capped).
- `TradingBrainService` now derives its closed-trade reflection scan cadence from the active timeframe: lower/noisier timeframes wait for more closed trades, `4h` keeps the existing 5-trade baseline, and daily/weekly timeframes scan sooner while preserving semantic-rule sample gates.
- `retrieve_similar_experiences` now over-fetches vector candidates before ranking, applies a hard age cutoff for prompt relevance, and then hybrid-ranks fresh candidates by similarity and recency. Older entries remain stored but are no longer used as fallback filler when fresh candidates are insufficient.
- Vector prompt context headers now include the active relevance window (`active window: last N days`) so the model can reason with explicit memory freshness constraints.
- System prompts now include a new `Trading Style & Horizon` section that adapts guidance to the active timeframe (scalping, intraday swing, swing, or position context), including expected hold horizon, noise tolerance, and news relevance window.

## 2026-05-06 - Internal Contract Cleanup

### Changed

- Removed redundant `hasattr()`, `getattr()`, and `isinstance()` checks from typed production paths, including config access, dashboard brain status/memory handling, prompt snapshot formatting, RAG retrieval, exit monitoring, retry logging, and AI provider response conversion.
- Tightened internal contracts so services now use direct attribute and method access for known collaborators such as config objects, strategy exit checks, logger-bearing instances, semantic-rule memory, and canonical article/analysis payloads.
- Config loader now normalizes admin IDs, dashboard CORS origins, and RAG news sources at load time so downstream services receive concrete lists instead of re-checking input shapes.
- Removed verified unused internal code, including stale trading statistics dataclasses, token request tracking state, obsolete dashboard state broadcast helpers, an obsolete CCXT multi-price fallback, a dead CoinGecko coin-data fetcher, unused provider metadata, and unused helper methods/constants.
- Fixed dashboard countdown datetime arithmetic and cache eviction typing surfaced during cleanup validation.

## 2026-05-06 - Latest Change Audit Fixes

### Fixed

- Semantic-rule refresh now checks all stored semantic rules for legacy unknown exit profiles and retires both stale loss-rule prefixes when a refreshed loss pattern changes between anti-pattern and corrective classification.
- Dashboard brain helpers now ignore malformed non-object JSON snapshots instead of relying on broad exception handling when building market context, brain status, memory, or current-position fallback price data.

## 2026-05-06 - Semantic Rule Exit Profile Repair

### Fixed

- `src/trading/brain.py` now fills missing semantic-rule SL/TP execution metadata from the configured exit profile before grouping trades, storing new rules, or rendering prompt context. This prevents legacy closed trades from producing `SL unknown/unknown | TP unknown/unknown` when the bot has configured hard/soft exit settings.
- `src/trading/vector_memory_rules.py` now supports targeted semantic-rule deactivation so refreshed resolved-profile rules can retire only their matching stale unknown-profile predecessors.
- `start.py` now injects config-derived exit execution defaults into the trading brain and triggers a one-shot refresh when active semantic rules still contain unknown exit profiles.
- `src/dashboard/routers/brain.py` now exposes dominant SL/TP intervals and rewrites legacy unknown exit-profile text for the Semantic Rules dashboard response.

## 2026-05-06 - Position Size Guardrails

### Changed

- LLM `position_size` parsing now treats explicit percent strings correctly, including values below 1% such as `0.5%`.
- RiskManager position-size fallbacks for LOW/MEDIUM/HIGH confidence are now configurable under `[risk_management]` and are used only when the AI omits or returns an invalid `position_size`.
- Brain and prompt defaults now keep `max_position_size` as a hard cap without nudging normal entries up to the cap as a minimum.

## 2026-05-06 - Trading Performance Summary P&L Percent Fix

### Fixed

- Discord and console performance summaries now calculate `Total P&L (%)` from realized quote P&L divided by configured demo quote capital, matching persisted trading statistics. Previously the notifier summed per-trade percentages, so variable position sizes could show a positive total percent while quote P&L was negative.
- Trading memory context now uses the same capital-based total P&L percentage when configured, while preserving average per-trade percentage separately.

## 2026-05-02 - Previous Analysis Context: Structured Decision Snapshot

### Changed

- `src/analyzer/prompts/template_manager.py` — `## PREVIOUS ANALYSIS CONTEXT` system-prompt section now includes a compact structured snapshot extracted from the previous AI response JSON block. The snapshot surfaces prior signal, confidence, entry/SL/TP/R/R/position size, trend direction and strength, confluence factor scores, and up to two key support/resistance levels alongside the existing narrative reasoning. Falls back to narrative-only when the JSON block is absent or malformed.
- `src/analyzer/prompts/template_manager.py` — A JSON-only previous response (no narrative text) now correctly produces a `## PREVIOUS ANALYSIS CONTEXT` section; previously it was silently skipped.
- `src/analyzer/prompts/template_manager.py` — `"reasoning"` field instruction in the response template now asks for 3-4 sentences covering decision thesis, market regime, key invalidation trigger, and next watch condition, improving data quality for vector-memory similarity and subsequent analysis cycles.
- `src/trading/trading_strategy.py` now uses the shared indicator classifiers when converting analysis results into entry market conditions, so real `technical_data` keys such as `atr_percent`, `macd_line`/`macd_signal`, `obv_slope`, and Bollinger Band levels are preserved for trading-brain similarity storage.
- `src/trading/data_models.py`, `src/factories/position_factory.py`, and `src/managers/persistence_manager.py` now persist `order_book_bias_at_entry`, allowing stop-loss/take-profit exits to reconstruct the same order-book context used by vector similarity queries.

## 2026-05-02 - Test Stability Fixes: Template Config Fallback and Short Article Extraction

### Fixed

- `src/analyzer/prompts/template_manager.py` now handles lightweight config objects safely when building the response template. If `MAX_POSITION_SIZE` is missing or invalid, it falls back to `0.10` instead of raising `AttributeError`.
- `src/rag/news_ingestion/rss_primitives.py` now accepts short but valid content from `article`/article-content/main selectors, avoiding fallback parser output that could include unrelated sidebar text.

## 2026-05-02 - Position Sizing: 10% Hard Cap and Corrected Prompt Formula

### Fixed

- **Prompt formula corrected**: The AI was instructed to use `position_size = confidence / 100`, turning a 50% confidence into a 50% capital allocation. Formula is now `(confidence / 100) × max_position_size` (default 0.10), so confidence 75 → 0.075 (7.5% of capital).
- **Hard cap added in `RiskManager`**: `calculate_entry_parameters` now clamps any AI-provided `position_size` to `config.MAX_POSITION_SIZE`. A warning is logged when clamping occurs.
- **`min_pos_size` floor corrected**: The brain-learned floor for position sizing now defaults to `0.02` (was `0.10`) and is capped at `max_position_size` to prevent floor > cap.
- **New config property `MAX_POSITION_SIZE`**: Added to `src/config/loader.py`, reading `[risk_management] max_position_size` from `config.ini` with a default of `0.10`.
- **`config.ini` / `config.ini.example`**: Added `max_position_size = 0.10` under `[risk_management]`.

## 2026-05-02 - Dashboard Confidence Parsing Regression Fix

### Fixed

- Restored dashboard Confidence KPI updates in `src/dashboard/routers/brain.py` by safely parsing `analysis.confidence` from JSON blocks in `response.text_analysis` even when `technical_data` is present.
- This fixes the regression introduced after the Apr 30 confidence-path fix, where confidence fell back to `--` when LLM output used JSON-only confidence (for example `"confidence": 75`) instead of a `Confidence: 75%` text line.
- Added/updated coverage in `tests/test_dashboard_brain_router.py` to ensure confidence is extracted from both regex fallback text and JSON analysis payloads when indicator fields are present.

## 2026-05-02 - Hard Exit Monitor Duplicate Position Save Fix

### Fixed

- Updated `src/trading/exit_monitor.py` so when both hard stop-loss and hard take-profit checks are due in the same tick, the monitor performs a single combined strategy check instead of two separate checks.
- This prevents duplicate `Saved position` writes/logs for unchanged open positions while preserving the same SL-first/TP-second evaluation semantics and timestamp persistence.

## 2026-05-01 - Security: Dependency Upgrades and requirements.txt Cleanup

### Security

- Upgraded `aiohttp` from `3.13.3` to `3.13.4` (CVE-2026-34513–34520, CVE-2026-22815 — 10 CVEs fixed).
- Upgraded `cryptography` to `≥46.0.7` (CVE-2026-26007, CVE-2026-34073, CVE-2026-39892).
- Upgraded `lxml` to `≥6.1.0` (CVE-2026-41066).
- Upgraded `orjson` to `≥3.11.6` (CVE-2025-67221).
- Upgraded `protobuf` to `≥6.33.5` (CVE-2026-0994).
- Upgraded `pyasn1` to `≥0.6.3` (CVE-2026-23490, CVE-2026-30922).
- Upgraded `pygments` to `≥2.20.0` (CVE-2026-4539).
- Upgraded `pytest` to `≥9.0.3` (CVE-2025-71176) — dev dependency.
- Upgraded `python-dotenv` to `≥1.2.2` (CVE-2026-28684).
- Upgraded `requests` to `≥2.33.0` (CVE-2026-25645).
- Upgraded `transformers` to `≥4.53.0` (CVE-2025-3777, CVE-2025-3933, CVE-2025-5197, CVE-2025-6051, CVE-2025-6638, CVE-2025-6921).
- Remaining unfixed: `pip==26.0.1` (CVE-2026-3219) — no patched version published by PyPA yet.

### Changed

- `start.py` now reads optional `HF_TOKEN` from `keys.env` and exports it to process environment (`HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`) before `SentenceTransformer` initialization. This removes repeated unauthenticated Hugging Face Hub warnings when a token is configured and enables higher rate limits.
- `keys.env.example` now documents optional `HF_TOKEN` setup.

## 2026-05-01 - Outcome-Aware Semantic Rules and AI Mistake Learning

### Changed

- `src/trading/brain.py` — `_trigger_reflection()` now evaluates all trades matching a pattern (wins and losses) via a shared `_compute_group_metrics()` helper. The win gate was lowered from 10 to 5 and rule IDs are now deterministic (`rule_best_{pattern_key}`) to prevent duplicate rules across repeated reflections. Stored `best_practice` metadata now includes `wins`, `losses`, `win_rate`, `loss_rate`, `avg_pnl_pct`, `profit_factor`, `expectancy_pct`, `avg_mae_pct`, and `avg_mfe_pct`. Rules are only stored when win rate is ≥ 60%.
- `src/trading/brain.py` — `_trigger_loss_reflection()` fully overhauled: loss gate lowered from 5 to 3, occurrence gate from 3 to 2. Uses `_compute_group_metrics`, `_derive_failure_reason`, and `_derive_recommended_adjustment`. Stores `rule_type` of either `anti_pattern` (loss rate ≥ 60%) or `corrective` (mixed). Deterministic IDs prevent duplicate rules.
- `src/trading/brain.py` — Added `_trigger_ai_mistake_reflection()` so repeated HIGH-confidence failures, sideways/choppy failures, and reasoning/premise mismatches produce `ai_mistake` semantic rules. These rules store `mistake_type`, `entry_confidence`, `failed_assumption`, `failure_reason`, `recommended_adjustment`, and win/loss/P&L metrics.
- `src/trading/brain.py` — Reflection keys now include hard/soft SL/TP execution profile. Stop-like close reasons such as `hard_stop`, `hard_stop_loss`, `soft_stop_loss`, and `stop_loss_hit` are normalized to the stop-loss diagnostic bucket while preserving the exact dominant exit profile in metadata.
- `src/trading/brain.py` — `get_context()` now renders rule type tags (`[⚠️ AVOID]`, `[⚡ IMPROVE]`, `[🧠 AI MISTAKE]`) and a recommended adjustment line for each learned rule. Apply Insights prompt now explicitly instructs the LLM to balance wins/losses, compare current reasoning to failed AI assumptions, and account for hard/soft SL/TP execution settings.
- `src/trading/vector_memory_rules.py` — `get_anti_patterns_for_prompt()` now fetches `corrective` and `ai_mistake` rules in addition to `anti_pattern` rules and appends `→ Why:` / `→ Fix:` lines when diagnostic metadata is present.
- `src/dashboard/routers/brain.py` — `get_active_rules` endpoint now maps all new metadata fields: `rule_type`, `loss_rate`, `wins`, `losses`, `avg_pnl_pct`, `profit_factor`, `expectancy_pct`, `failure_reason`, `recommended_adjustment`, `mistake_type`, `entry_confidence`, `failed_assumption`, `dominant_close_reason`, `dominant_exit_profile`, `dominant_stop_loss_type`, and `dominant_take_profit_type`.
- `src/dashboard/routers/brain.py` and `src/utils/indicator_classifier.py` — Dashboard brain status and exit-execution context building now tolerate lightweight config objects by falling back to `unknown`/timeframe defaults when optional SL/TP execution attributes are absent.
- `src/dashboard/static/modules/vector_panel.js` — `renderSemanticRules()` updated: empty-state text no longer references "wins only"; rule cards now show a color-coded type badge (Best Practice / Avoid / Corrective / AI Mistake), wins/losses split, avg P&L, profit factor, exit profile, mistake type, failure reason, and an optional recommended adjustment line.
- `src/trading/brain.py`, `src/trading/vector_memory.py`, and `src/trading/vector_memory_context.py` — Brain reflection now runs every 5 closed trades instead of 10, and recency weighting for similar experiences now uses a 30-day half-life instead of 90 days so fresh market behavior affects prompts and semantic-rule synthesis faster.

### Added

- `src/trading/brain.py` — New private helpers compute combined win/loss statistics, normalize stop-like close reasons, preserve hard/soft exit profiles, detect sideways AI overconfidence, derive failed AI assumptions from stored reasoning, and map those diagnoses to concrete LLM-actionable instructions. Shared metadata and loss-diagnostic helpers keep reflection rules DRY across best-practice, loss, and AI-mistake paths.

## 2026-05-01 - VS Code Python Terminal Startup Workaround

### Changed

- Updated workspace settings in `.vscode/settings.json` to disable `python.terminal.activateEnvironment` and `python.terminal.activateEnvInCurrentTerminal`, and set `python-envs.terminal.autoActivationType` to `command`. This avoids Python Envs shell-startup profile injection failures (`EPERM` during `mkdir`) observed on some Windows user-profile paths with non-ASCII characters, while keeping interpreter selection pinned to `./.venv/Scripts/python.exe`.

## 2026-04-30 - RSS News Pipeline, Exit Monitoring, Security Fixes, and Public Cleanup

Primary commits: `926056e`, `349d49a`, `d6a72cc`, `551f67a`, `1e244da`, `be5e190`

### Added

- Replaced the old legacy-provider-driven news flow with a full RSS ingestion stack under `src/rag/news_ingestion/`, including `rss_provider.py`, `rss_primitives.py`, `schema_mapper.py`, and optional Crawl4AI enrichment via `crawl4ai_enricher.py`.
- Added `src/rag/scoring_policy.py`, `src/rag/news_repository.py`, and `src/rag/local_taxonomy.py` to support article scoring, storage, and local category handling without legacy provider category dependencies.
- Started tracking `data/categories.json` in Git as the local category/taxonomy snapshot used by the new RSS-driven news flow.
- Added `src/platforms/ccxt_market_api.py` to formalize exchange market access outside the removed legacy market path.
- Added `src/trading/exit_monitor.py` and `src/trading/position_status_monitor.py` so stop-loss and take-profit execution can be checked on bot-side intervals instead of waiting only for candle-close logic.
- Added two debugging scripts, `scripts/fetch_free_news_preview.py` and `scripts/compare_news_body_quality.py`, to inspect raw RSS ingestion and enrichment quality.
- Added broad contract coverage for the new pipeline, including tests for news ingestion, RSS provider behavior, RAG retrieval/scoring contracts, exit monitoring, notifier rate limits, and CCXT migration.

### Changed

- Reworked the configuration contract for news ingestion and exit handling. `config/config.ini.example` gained RSS source whitelisting, per-source feed URLs, page-enrichment controls, density/co-occurrence scoring knobs, lower-timeframe guidance, and the new `[risk_management]` hard-exit interval settings. Existing local `config/config.ini` files need the same keys added manually because `config/config.ini` is intentionally ignored.
- Removed the obsolete legacy-news API key entry from `keys.env.example` to match the news migration away from the retired provider path.
- Narrowed the `data/` ignore rule in `.gitignore` so the repo now keeps `data/categories.json` under version control while still ignoring the rest of the local runtime data directory.
- Updated the bot runtime in `src/app.py`, `start.py`, `src/config/loader.py`, and `src/config/protocol.py` to wire the new RSS/news services, low-timeframe scheduling, exit-monitor settings, and richer config contracts.
- Extended `src/dashboard/routers/brain.py` so dashboard state is derived from parsed analysis JSON and richer live-trading context instead of fragile regex-only extraction. The Apr 30 diff also shows position SL/TP distance fields being carried through the API more explicitly.
- Reworked `src/rag/news_manager.py` and `src/rag/context_builder.py` so RAG assembly now pulls from the RSS repository, local taxonomy, and scoring policy instead of the previous legacy-provider-oriented path.
- Enhanced notifier output in `src/notifiers/base_notifier.py`, `src/notifiers/console_notifier.py`, and `src/notifiers/notifier.py` so trade alerts can include chart images and last-trade context.
- Broadened `src/utils/indicator_classifier.py` and `src/utils/timeframe_validator.py` to support exit-execution context, lower timeframes, and more explicit runtime scheduling rules.

### Performance

- Landed a major O(N) optimization wave across indicator code. The release touched momentum, stochastic, moving-average crossover, support/resistance, and classifier-related paths, while the release notes and file diffs align on sliding-window rewrites for RSI, MACD, CCI, VHF, MAD, MFI, Z-score, Donchian, Ichimoku, Fibonacci Bollinger Bands, Vortex, Choppiness, and rolling volume filters.
- Tightened market data and persistence code around `src/analyzer/analysis_engine.py`, `src/analyzer/data_fetcher.py`, `src/analyzer/market_data_collector.py`, and `src/managers/persistence_manager.py` so the new ingestion and exit-monitor flows fit into the existing analysis loop.

### Security and Fixes

- Fixed a critical path traversal issue in `src/rag/file_handler.py` and shipped a cluster of front-end hardening changes across dashboard modules for DOM XSS, mutation-XSS, clickjacking, and template interpolation bugs.
- Fixed the dashboard confidence KPI path by moving the brain router toward parsed JSON extraction rather than depending on a `Confidence: XX%` regex shape that no longer matched the prompt contract.
- Fixed position sizing for values greater than 1% and hardened vector-threshold learning and reflection-key parsing in trading logic.

### Removed

- Deleted legacy provider categories/data/market clients plus `src/rag/category_fetcher.py` and `src/rag/news_category_analyzer.py` in the main release.
- Deleted the remaining legacy provider news client and package stub in follow-up commit `1e244da`, completing the migration away from retired-provider-backed news.
- Deleted the public `data_template/` tree in `be5e190` after it became clear that the directory no longer matched the private mainline.

### Notes

- The Apr 30 release was published in two steps: the main feature sync landed first, then same-day cleanup commits removed merge-conflict markers from `README.md`, source files, and `config/config.ini.example`, and finished deleting leftover legacy-provider artifacts and `data_template/` artifacts. This section describes the final cleaned-up public state.

## 2026-04-14 - Dependency Maintenance

Primary commit: `554a507` (Dependabot merge of `8319f44`)

- Bumped Pillow from `12.1.1` to `12.2.0` on the public branch. This was later carried forward into the Apr 30 release notes as the image-processing/security maintenance update.

## 2026-03-28 - Release Sync: Caching, Vector Memory, Dashboard and Trading Improvements

Primary commit: `80389c1`

### Added

- Added `tests/test_dashboard_server_cache.py` and expanded dashboard brain router coverage to validate new cache and state behavior.

### Changed

- Implemented endpoint-aware dashboard caching with ETag/header strategy in `src/dashboard/server.py`.
- Updated dashboard client/runtime flow in `src/dashboard/static/main.js` and `src/dashboard/routers/brain.py` to align vector queries and state handling with richer trading context.
- Added fallback recomputation for SL/TP distance percentages in dashboard position output when persisted values are zero or missing.

## 2026-03-27 - Release Sync: Dashboard, Indicators, and Vector-Memory Overhaul

Primary commit: `3493598`

### Added

- Split dashboard styling into modular CSS files under `src/dashboard/static/css/`.
- Added `src/trading/vector_memory_analytics.py`, `src/trading/vector_memory_context.py`, and `src/trading/vector_memory_rules.py`.
- Added `src/utils/indicator_classifier.py` and new focused tests for context builder, dashboard routes, template manager, indicator classifier, and vector memory.

### Changed

- Reworked dashboard layout and panel modules in `index.html` and `main.js`.
- Updated analyzer, formatter, and indicator modules across momentum, sentiment, statistical, trend, volatility, support/resistance, and volume packages.
- Refined RAG and market-data processing paths and aligned dashboard/trading context behavior with vector-memory logic.

### Removed

- Removed the old monolithic `src/dashboard/static/style.css`.
- Removed many internal diagnostic scripts and broad internal verification tests from the public release snapshot.

## 2026-03-04 - News Architecture Simplification, Public Packaging Pass, and README/License Cleanup

Primary commits: `f03304b`, `8621585`, `5ba8169`

### Changed

- Simplified the legacy news-provider stack by collapsing the earlier multi-file `news_components` structure into a single provider news client paired with a leaner `src/rag/news_manager.py`.
- Expanded dashboard behavior again in `src/dashboard/routers/brain.py`, `monitor.py`, and `performance.py`, while also improving logging and monitor-route behavior.
- Updated technical-calculation and risk-manager behavior alongside the dashboard and news refactor.
- Reworked public-facing documentation and licensing so the Mar 4 release shipped with a cleaned-up README structure and MIT-oriented release files.

### Added

- Added `tests/benchmark_monitor_news.py` during the refactor phase to benchmark news monitoring before the public release tree was trimmed back.

### Removed

- Removed the older provider news API module and legacy `news_components/` modules after the refactor consolidated news fetching into a single client.
- Removed additional dashboard, RAG, start-up, and test code in the public release commit to produce a smaller public release snapshot.
- Removed the remaining Polyform references from public documentation in the follow-up README cleanup commit.

## 2026-03-02 - Major Update: Performance, Security, and Quality Enhancements

Primary commit: `2f1a6ad`

### Changed

- Improved dashboard server binding/runtime handling and startup orchestration in `start.py`.
- Updated market/platform integrations around Alternative.me, CoinGecko, and legacy-provider API handling.
- Extended configuration behavior and associated test coverage (including dashboard config/cors paths).

## 2026-02-24 - Data-Model and Provider Infrastructure Refactor

Primary commit: `d89a1f6`

### Changed

- Renamed `src/trading/dataclasses.py` to `src/trading/data_models.py` and expanded trade/position snapshots for richer brain and vector-memory context.
- Strengthened AI provider base clients and retry/error-classification behavior across provider adapters.
- Added additional optimized statistical-indicator work (including kurtosis/skewness paths) with related regression coverage.

## 2026-02-22 - Major Update: Performance, Security, and Quality Enhancements

Primary commit: `576ddf8`

### Changed

- Improved dashboard routing/state behavior and refined server/static module implementations.
- Continued O(N)-oriented indicator and analyzer optimization pass.
- Expanded validation and security-related test coverage, including websocket origin and dashboard route checks.

## 2026-02-21 - Comprehensive System Overhaul, UI Enhancements, and Performance Optimizations

Primary commits: `a6e6af4`, `9195a34`

### Added

- Added `.pylintrc`, broad regression/integration test expansion, and refreshed dashboard image assets.
- Added a `data_template/` snapshot for the public release packaging at that time.

### Changed

- Overhauled dashboard UI/UX into a richer multi-panel surface with improved accessibility and interaction behavior.
- Reworked analyzer/trading/runtime layers for non-blocking I/O paths, stronger context handling, and improved orchestration.
- Applied multiple indicator/runtime correctness and performance improvements across trend, statistical, and momentum paths.

### Security and Fixes

- Hardened dashboard security paths (including XSS/DoS-related fixes and websocket safeguards).
- Fixed a duplicate live dashboard link in README (`9195a34`).

### Removed

- Removed older agent/workflow artifacts such as `AGENTS.md`, `CONTRIBUTING.md`, and obsolete Jules prompt files from the release snapshot.

## 2026-01-25

Primary commits: `a60020d`, `04ff96d`, `a2ac568`

- Moved runtime wiring toward a composition-root architecture in `start.py` and tightened dependency boundaries across analyzer, RAG, and trading services.
- Added `PositionFactory` and `DataFetcherFactory`, improved graceful shutdown and position lifecycle handling, and cleaned imports/readability across touched modules.
- Improved technical-analysis formatting, dashboard accessibility (aria labels and focus indicators), `UnifiedParser` behavior, and LM Studio model caching.

## 2026-01-24

Primary commit: `59b6b71`

- Added dashboard full-stack caching (HTTP cache headers plus RAM state cache) and refactored the brain router for high-traffic endpoint efficiency.
- Added cache verification tooling for headers and dashboard state TTL behavior.

## 2026-01-23

Primary commits: `bab3f48`, `df589da`, `f18e98d`, `967b71a`, `5ce1f67`, `81a9c01`, `77f935a`, `5276279`, `ae2f983`

- Delivered a broad analyzer/prompt/chart quality pass: pattern deduplication, markdown/prompt cleanup, chart visibility/tooltips, and indicator-tracking improvements.
- Added performance-focused refactors in market metrics and statistical indicators (including linear regression and O(N) optimizations).

## 2026-01-21

Primary commit: `ecd3eab`

- Introduced the first usable dashboard layer with brain/performance monitoring views, dashboard state, and core UI modules.

## 2026-01-20

Primary commit: `f877508`

- Added an initial integrated web dashboard build (UI, styling, and backend server wiring).

## 2026-01-19

Primary commits: `6256e60` plus merged PR batch (`500e7e3`, `151b710`, `7a0547e`, `f63d742`, `9264bb9`, `08f9fa6`, `de1f3a0`)

- Added comprehensive trade-lifecycle regression coverage and merged a broad optimization/fix batch across indicators, async testing, and vector-memory integration paths.

## 2026-01-18

Primary commits: `593c61b`, `c8ecc6f`, `65d0bda`, `1cb788d` plus merged PR batch (`80671e9`, `61a05d4`, `cdb72b0`)

- Added DefiLlama fundamentals integration and expanded market-overview/RAG market-data building.
- Merged security/accessibility/performance PRs and removed outdated AI agent guideline docs.

## 2026-01-17

Primary commits: `fe454f3`, `b6374e8`, `ce36abb`, `f9797fe` plus merges (`b1c1920`, `6c22ccc`)

- Expanded dashboard capabilities (statistics/news/positions panels), added dashboard security hardening and Cloudflare Tunnel support, and refreshed dependencies.

## 2026-01-16

Primary commits: `2858c28`, `036954a`, `81f5c64`, `47c9dec` plus merged PRs (`71d83a6`, `fb26593`, `c64edfe`)

- Unified AI provider responses with Pydantic models and reduced redundant indicator calculations.
- Expanded AI chart generation and dashboard visual-analysis support while merging security/performance/accessibility PRs.

## 2026-01-15

Primary commits: `d12bdfe`, `9684675`, `c31d755`, `4c034bc` (plus related updates)

- Improved LONG-bias mitigation logic and upgraded security dependencies (aiohttp) with safer dashboard error handling.
- Enforced UTC consistency in temporal/trading operations and tightened provider compatibility.

## 2026-01-14

Primary commits: `f0bf950`, `1d73168`, `66450ce`, `4532bdf`, `09f868a` (plus refactor/cleanup updates)

- Added provider orchestration, stronger typed analysis context, expanded provider test coverage, and major dashboard feature growth (including new panels and synapse-style views).
- Continued large refactor wave across retrieval, parsing, and prompt/brain paths.

## 2026-01-13

Primary commits: `614f685`, `cf168d1`, `8e6d1f8`, `e24a04c`, `bdaba75`, `a67b9fc`, `17a9f93`, `a3b551e`, `ff97d8e`, `1e578d1`, `c4dafeb`, `9a26fcd`

- Major refactor/bugfix sweep across brain context, provider SDK integration, and performance tuning.

## 2026-01-12

Primary commit: `8feb100`

- Applied a focused bugfix/refactor pass across in-flight analyzer and trading logic.

## 2026-01-11

Primary commit: `1d0141a`

- Added non-breaking dynamic threshold learning enhancements in trading brain/vector memory using optional metadata and sample-size safeguards.

## 2026-01-10

Primary commits: `a061786`, `a30c942`, `b7b6730`

- Delivered bugfix/refactor updates and richer brain context extraction (weekend awareness, sentiment/order-book pressure, and serialization cleanup).

## 2026-01-09

Primary commit: `0271f7f`

- Applied dashboard bugfixes after early rollout.

## 2026-01-07

Primary commit: `f2062e7`

- Added core position-management strategy logic and dashboard routes for brain and position panels.

## 2026-01-06

Primary commits: `450bce6`, `3871980`, `3e9d0a8`, `7f26be1`, `a5c9955`, `e94ce36`

- Added vector-memory service foundations and integrated analysis-engine/config-loader improvements.
- Expanded real-time dashboard capabilities with websocket-driven monitoring and chart management endpoints.

## 2026-01-05

Primary commits: `6e8b0d0`, `6a69c17`, `029db43`, `538ffd7`, `85c884d` (plus related updates)

- Added single-instance protection and shutdown confirmation flow.
- Improved prompt-building with position context, strengthened statistics update ordering, and refined trading insights handling.

## 2026-01-04

Primary commits: large foundation batch including `dcb6244`, `4844ff9`, `5bb354a`, `301527b` and related trading/notifier refactors

- Added foundational trading modules for memory, persistence, statistics, strategy execution, and brain context.
- Added vector-memory integration and application composition-root wiring in `app.py`/`start.py`.
- Refactored notifier/formatting and logger internals while cleaning circular-import pressure.

## 2026-01-01

Primary commits: `3f08d9f`, `cff20ce`

- Enhanced trading behavior with dynamic parameters and improved data handling.
- Updated ignore rules for research-paper artifacts.
