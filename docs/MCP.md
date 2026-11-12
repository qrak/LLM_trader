# Read-only MCP server over the trade ledger

`src/mcp_tools/history_server.py` exposes the bot's own SQLite ledger to an MCP client (Claude
Desktop, Cursor, VS Code Copilot, any SDK host). It has two tools and no way to write:

| tool | arguments | returns |
|---|---|---|
| `recent` | `symbol`, `action`, `limit` (default 20, capped at 200) | the newest `trade_history` rows: timestamp, symbol, action, confidence, price, stop loss, take profit, position size, quote amount, quantity, fee |
| `stats` | `symbol`, `days` (default 30) | coverage of the window (rows, first and last timestamp) plus decision counts per action |

`stats` counts decisions, **not** outcomes: it is the shape of what the bot did, not a win rate.
The win rate lives in [`results/`](../results) and `claims.json`, where it is computed from closed
trades only.

## Install and run

The server is optional: the bot does not import it and does not need it at runtime.

```bash
pip install -r requirements-mcp.txt          # mcp>=1.9,<2 — 2.x moves FastMCP out of mcp.server
python -m src.mcp_tools.history_server --db data/trading/trade_history.db
```

`--db` defaults to `data/trading/trade_history.db`; nothing else is configurable, and there is no
network listener — the transport is stdio, started by the client.

## Client configuration

```json
{
  "mcpServers": {
    "llm-trader-history": {
      "command": "python",
      "args": ["-m", "src.mcp_tools.history_server", "--db", "data/trading/trade_history.db"],
      "cwd": "/absolute/path/to/LLM_trader"
    }
  }
}
```

## Why it cannot damage the book

- Every query opens the database with `mode=ro` and sets `PRAGMA query_only = ON`, so a write fails
  inside SQLite instead of relying on the caller to behave.
- `limit` is clamped to 200 rows, and the two tools have no parameters that reach a file path.
- There is no write tool, no order tool and no config tool. Adding one would put an LLM between the
  ledger and the exchange, which is exactly what the risk pipeline and the separate executor exist
  to prevent.

## Tests

```bash
python -m pytest tests/domain_infra/test_mcp_history.py -q
```

The reader is covered against a temporary database. The `build_server` test is skipped while the
`mcp` package is absent (`pytest.importorskip`), so the suite stays green on a bare install.
