# Changelog

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
