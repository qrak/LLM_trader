# Security policy

## Supported version

`master` only. There are no maintained release branches: the rolling `build-*` tags are rollback
anchors for `master`, not supported versions.

## What is a vulnerability here

- **The settings desk stops being loopback-only.** `http://localhost:8000/admin/` edits
  `config.ini` and `keys.env`; any route, tunnel, LAN path or missing `Origin`/`Host` check that lets
  a remote client read or write those files is a vulnerability.
- **A path to the exchange that skips the guards.** The model's answer reaches an order only through
  the guard pipeline, the risk pipeline and the separate executor. Anything that lets a prompt, a web
  page, an MCP client or a dependency place or modify an order on its own is a vulnerability.
- **A secret in the open.** API keys, cookies or account identifiers in the repository, in a log, in
  `claims.json`, in the published dashboard payloads, or in a snapshot under `results/`.
- **The read-only MCP server writing.** `src/mcp_tools/history_server.py` opens the ledger with
  `mode=ro` and `PRAGMA query_only`; a query or parameter that writes is a vulnerability.
- **Remote code execution** through config values, prompts, RSS articles or chart rendering.

## What is not

- **Losing a paper trade, or the strategy never becoming profitable.** The book is simulated and the
  README says the result is not evidence of an edge.
- **A market-data or inference provider being down.** The bot degrades to the next provider and then
  to `HOLD`, by design.
- **Reading published data.** The ledger, the prompts and the results page are intentionally public;
  they need no authentication.
- **The absence of fees, slippage or a track record.** Documented, not a vulnerability.

## How to report

Use GitHub's private reporting: **Security → Advisories → New draft security advisory** on
`qrak/LLM_trader`. Please do not open a public issue, and do not include an actual API key, cookie or
`keys.env` content in the report — the real secrets live in an OS-level store outside this
repository, so a report never needs one.

This is a one-maintainer project: expect a first answer within a few days, not within hours.

## What happens next

The fix ships on `master` with a test and a `CHANGELOG.md` entry, and the advisory names the commit.
The repository runs CodeQL and a dependency review on every push and pull request, so a report that
points at one of those findings is welcome as an ordinary issue instead.
