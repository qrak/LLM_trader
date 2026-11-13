# Configuration

Runtime configuration lives in `config/config.ini` (gitignored); the tracked template is
`config/config.ini.example`. The loader in `src/config/loader.py` is the single source of truth
for key names and validation — if this page and the loader disagree, the loader wins.

Edits take effect on the next start of the process (the dashboard's `Shift+R` in-place restart).
`Config` is read once at import and services bind their frame/provider/dependencies during
provisioning, so every key is `restart`-scoped. The local settings desk at
`http://localhost:8000/admin/` edits these keys through a form, uses the comments in this file as
field help, and validates the candidate with this loader before an atomic write; it is
loopback-only and never promises hot reload. Hand-editing the file remains supported. See
[SETTINGS_PANEL.md](SETTINGS_PANEL.md).

## Core settings

| Setting | Default | Description |
|---------|---------|-------------|
| `crypto_pair` | `BTC/USDT` (the template ships `BTC/USDC`) | Any CCXT market the exchange lists; `ConfiguredSymbolGuard` rejects a decision that names another pair. The landing page's status pill is the only copy of this value outside the config, so change both together |
| `timeframe` | required | Analysis candle timeframe. No default: startup aborts naming the key when it is missing. 4h is the frame the bot was designed and tested for; smaller frames degrade signal quality. The prompt's trend labels and the brain's decay/freshness/reflection windows all derive from this value |
| `model_verbosity` | full | New-analysis detail: low (~400), medium (~1000), high (~2000), full (~8000 narrative output tokens) |
| `provider` | `googleai` | AI provider (`local`, `googleai`, `openrouter`, `deepseek`, `opencode`, `all`; BlockRun is a fallback-chain member only). The live deployment runs `deepseek` |
| `demo_quote_capital` | 10000 | Simulated capital (USDC) |
| `max_position_size` | 0.10 | Hard ceiling for a position as a fraction of capital; a regime profile may only tighten it, and the prompt renders the resulting effective caps |
| `min_rr_entry` | 0.0 | Baseline R/R floor for the entry gate (0 = no fixed floor; the brain can raise it) |
| `stop_loss_type` | `soft` | `hard` (interval polling) or `soft` (candle close). The template and the live deployment run `hard` |
| `stop_loss_check_interval` | 15m | Hard-exit polling interval; must not exceed the timeframe |
| `reference_exit_check_seconds` | 10 | How often the SL/TP is checked against the REAL market while a position is open and the executor is live (`0` = off). This is the stop that runs; the exchange leg is only a catastrophe backstop |

## Verbosity and prompt budget

`[model_config] model_verbosity` controls the **new answer**, not how much history is carried
forward. `full` requests a complete Markdown research report: metrics, timeframe comparison,
indicator categories and interpretations, key levels, news impact, chart evidence, educational
context, scenarios, risks and synthesis. Other modes retain their compact 3/5/13-line formats.
The token figures are narrative writing targets, excluding decision JSON and hidden reasoning —
not hard API limits. `max_tokens` / provider-specific output caps still apply; no post-generation
truncation or extra formatting call is used. Previous reasoning has **no separate length cap**;
the structured decision thesis is preferred when available, otherwise the sanitized narrative is
kept without length-based truncation.

`[prompt_budget] request_max_tokens` (default **100000 estimated input tokens**) protects the
whole request. Before each provider/model call, including fallbacks, the guard also reserves
output capacity, an image allowance and a safety margin within the configured model context
window. Under pressure it omits lower-priority sections rather than cutting arbitrary tokens; if
protected content still cannot fit, no request is sent. Prior Python R/R feedback remains
protected. Response-schema/R/R checks validate decisions, not prompt length.

Local token budgets use `tiktoken`'s `cl100k_base` as an **estimate**, not a Gemini/DeepSeek-native
counter. Upgrading tiktoken does not supply those tokenizers or pricing. Recorded request usage
comes from the provider API (including Google thinking tokens); OpenRouter cost comes from its
generation endpoint when available. Google/DeepSeek dollar figures are rate-table estimates,
not invoice reconciliation; the current calculation does not apply cache-hit discounts.

## Billing windows (`config/peak_rates.json`)

Some providers charge different rates at different hours — DeepSeek is half price off-peak
(Mon–Fri 01:00–04:00 and 06:00–10:00 UTC are peak; nights and the whole weekend are off-peak).
OpenCode Go bills the same windows in reverse: `config/model_pricing.json` carries its off-peak
rates and the peak window multiplies them by 2, because the subscription is consumed in monthly
dollar allowances rather than per-request invoices.
The bot applies that window when reporting cost, so the log shows the price actually charged and
the UTC clock of the request: `Request cost: $0.007000 (off-peak x0.5 @ 08:16 UTC)`. The clock is
UTC because the windows are defined in UTC: a cycle that starts at the same local time every day
lands on the other side of a boundary when DST changes, and a `:04` cycle misses a window that
ends at `:00` by four minutes. On OpenCode Go the dollar figure is not an estimate — the response
carries `usage.opencode.allowance`, the amount the request drew from the monthly allowance, and
the bot reports that number with `, from provider` after the window; the window label then
describes the request, not the arithmetic.

Per-token rates live in `config/model_pricing.json` and are treated as the **base rates** the
multiplier scales: the DeepSeek API is stored at its peak price (off-peak halves it), OpenCode Go
stores its two DeepSeek rows at the off-peak price (peak doubles them), and every other model has
no window at all. The table carries every model Go offers, the models at the top of OpenRouter's
public token ranking, and the current Gemini family, each row naming its source and date. A model
that is not listed reports no cost instead of guessing. `peak_rates.json` only declares *when* a
window applies and how it scales those rates. The file
is optional — copy `config/peak_rates.example.json` to `config/peak_rates.json`; without it the
defaults from `src/utils/peak_rates.py` apply and anything unlisted is billed flat. It is read at
startup, and each provider entry is merged over its built-in default, so list only what you
change:

```json
{
  "deepseek": {
    "models": ["deepseek-flash", "deepseek-v4-flash-vision-exp", "deepseek-v4-pro"],
    "peak_multiplier": 1.0,
    "off_peak_multiplier": 0.5,
    "peak_windows_utc": [
      {"days": ["mon-fri"], "start_utc": "01:00", "end_utc": "04:00"},
      {"days": ["mon-fri"], "start_utc": "06:00", "end_utc": "10:00"}
    ]
  }
}
```

| Field | Meaning |
|-------|---------|
| `models` | Models the entry covers (omitted = every model of that provider) |
| `peak_multiplier` / `off_peak_multiplier` | Scale applied to the base rates inside / outside the peak windows |
| `peak_windows_utc` | Peak windows. `days` takes single names or inclusive ranges (`["mon-fri"]`, `["sat","sun"]`); clock values are UTC, start-inclusive and end-exclusive |
| `_default` | Entry used for providers and models nobody listed (multiplier `1.0` = flat rates) |

## Decisions gate

`[decisions_gate]` asks a second model — OpenAI GPT-6 Luna Decisions through OpenRouter's alpha
`/api/alpha/decisions` path — to score the decision the bot just produced. It runs after the Python
validators and before the response is rendered, and it may only shrink the position size.

| Key | Meaning |
|---|---|
| `mode` | `off` makes no call at all; `shadow` calls the gate and writes the journal while the decision stays byte-identical; `size` additionally multiplies `position_size` |
| `url` / `model` | The endpoint and the decisions model; the alpha path is the only Decisions shape the configured `OPENROUTER_API_KEY` reaches, so a move to the GA `/v1/decisions` changes these two values and the client |
| `min_confidence_pct` | Below this confidence in its own answer the gate leaves the size alone |
| `timeout_seconds` | Hard limit on one gate call; a timeout fails open |

The state is compact on purpose (~1k tokens against the 19.5k-token analysis prompt): the decision
without its `confidence`, the entry snapshot's indicators, the last 20 candles and the
`direction / regime / ADX bucket` triple the brain's rules declare. Every call appends one line to
`data/trading/decisions_gate.jsonl` — mode, answers, multiplier, latency, tokens and the API's own
`usage.cost`; nothing in that file reaches the public snapshot. A refusal, timeout or malformed body
leaves the decision and the size untouched and logs one warning.

## Executor handoff

`[executor_api]` points at the separate executor service. `enabled = false` keeps the bot a pure
paper/simulator: decisions are booked locally and nothing is sent over HTTP. With
`enabled = true`, a decision is POSTed to `executor_api_url`; HTTP 200 means **queued**, not
executed, and the verdict journal (`data/trading/executor_verdicts.jsonl`) is the evidence of
what actually happened.

## Directories

`[directories] data_dir` moves the runtime tree: trade database, statistics, brain ChromaDB,
metrics journals and the intent journal all derive their path from it, so a custom data directory
does not split that state across two trees. Three files stay outside it by design:
`[executor_api] verdict_path` and `exit_path` are a cross-service contract with the executor
(both sides must name the same file, so `config.ini.example` pins them), and the forwarding
dead-letter queue is `data/trading/failed_forwards.jsonl` relative to the process working
directory. Leave the two executor paths alone unless the executor's `LLM_TRADER_VERDICT_PATH` /
`LLM_TRADER_EXIT_PATH` move too.
