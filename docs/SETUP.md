# Setup

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.13 | 3.13+ |
| **RAM** | 4 GB | 8+ GB |
| **Disk** | 3 GB | 8+ GB (a full environment measures ≈2.3 GB, plus browser binaries for news enrichment, logs and trade data) |
| **CPU** | 2 cores | 4+ cores |
| **GPU** | Not required | Not required — embeddings run on CPU |
| **OS** | Windows 10+, Linux, macOS | Linux (WSL2) |
| **Internet** | Required (API calls) | Required |

Python 3.12 or older will fail at import: the code uses 3.13 syntax and `asyncio` APIs.

## Quick start

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt -c constraints.txt
cp keys.env.example keys.env  # optional; alternatively set your selected provider key in the OS environment
python start.py               # dashboard at http://localhost:8000
```

`constraints.txt` carries the version floors and compatibility caps for packages that only
arrive as dependencies; always install with `-c constraints.txt` (see `CONTRIBUTING.md`).

The first run provisions ChromaDB, the CoinGecko cache and the news database, and downloads the
embedding model — expect a few minutes before the first analysis appears.

## Platform scripts

| Script | Purpose |
|--------|---------|
| `scripts/start_script_main.ps1` | Start the bot (Windows) |
| `scripts/start_script_main_linux.sh` | Start the bot (Linux) |
| `scripts/start_script_main_macos.sh` | Start the bot (macOS) |
| `scripts/run_all_tests.sh` | Run the full test suite in `.venv` |
| `scripts/query_trade_history.py` | CLI utility to inspect the SQLite trade history |
| `scripts/cost_report.py` | Gross vs net P&L after recorded LLM inference cost |
| `scripts/results_report.py` | Full ledger + inference cost + buy-and-hold benchmark (source of `docs/RESULTS.md`) |

Each launcher checks `requirements.txt` and `constraints.txt` against the installed environment
and only runs pip when something is actually unmet.

## Runtime controls

| Key | Action |
|-----|--------|
| `a` | Force analysis — run an immediate market check |
| `d` | Toggle the dashboard on/off |
| `h` | Help — show available commands |
| `q` | Quit — graceful shutdown with state preservation |
| `Shift+R` | In-place reload (relaunches `start.py` through the launcher) |

`Shift+R` only works when the process was started by one of the platform launchers: they export
`LLM_TRADER_RELOAD_SUPPORTED=1` and restart the bot when it exits with code `42`.

## Secrets

Secrets come from **OS / service environment variables (recommended)** or an optional `keys.env`
in the repository root; process variables take precedence, including explicitly empty values.
Both sources are read at startup, so restart after changes (on Windows, reopen the launcher
terminal after editing persistent variables). Environment variables are not encrypted — keep them
out of logs and shell history. `keys.env.example` lists every variable.

Only the selected provider needs its key. OpenRouter-hosted Google or DeepSeek models use **only
`OPENROUTER_API_KEY`**, not the direct-vendor keys. With `provider = all`, unavailable clients
are skipped; `provider = local` needs no cloud key.

| Variable | Required | For |
|----------|----------|-----|
| `GOOGLE_STUDIO_API_KEY` | Only for `googleai` | Google AI Studio provider |
| `GOOGLE_STUDIO_PAID_API_KEY` | If used | Paid-tier Google AI |
| `OPENROUTER_API_KEY` | Only for `openrouter` | Every model served through OpenRouter |
| `DEEPSEEK_API_KEY` | Only for `deepseek` | DeepSeek official API |
| `BOT_TOKEN_DISCORD` | If used | Discord notifications |
| `MAIN_CHANNEL_ID` | If used | Discord notification channel |
| `COINGECKO_API_KEY` | No | Market metrics (rate-limit boost) |
| `HF_TOKEN` | No | HuggingFace model download |

## Running the tests

```bash
python -m pytest tests -q                 # 2,514 tests, ~145 s
python -m pytest tests/domain_trading -q  # one area
ruff check src scripts start.py
pyright src start.py
```

`tests/test_admin_playwright.py` needs Playwright plus Chromium, is marked `integration` (excluded
from CI), and skips itself when Playwright is not installed. On Windows the default temp directory can deny `tmp_path`; pass
`--basetemp=C:\Users\Public\_pytest` if collection fails on permissions.

## Deploying the website

`website/` is an Astro site deployed to Cloudflare Pages (`qrak.org`). The project is a
direct-upload project: a push does **not** deploy it.

```bash
cd website
npm run build
npx wrangler pages deploy dist --project-name semantic-signal-landing --branch main
```
