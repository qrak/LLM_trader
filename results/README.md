# Frozen ledger snapshots

Each directory here is one immutable copy of the bot's runtime ledger, taken with
`python scripts/snapshot_results.py`. `manifest.json` records the capture time, the source directory
and the `sha256` of every file, so a copy can be checked instead of trusted.

```
python scripts/snapshot_results.py                          # writes results/<UTC date>/
python scripts/snapshot_results.py --verify results/<date>   # re-hashes it, exit 1 on any difference
python scripts/results_report.py --data-dir results/<date>   # reprints the numbers the README quotes
```

The snapshot is what the public numbers are generated from: `claims.json` is regenerated with
`--data-dir results/<date>`, and the site's `src/data/ledger.json` comes from the same directory. That
is why the README table, the results page and `claims.json` cannot disagree — see
[docs/PUBLIC-NUMBERS.md](../docs/PUBLIC-NUMBERS.md).

A snapshot is never edited and never deleted. A new one means a new directory and a new
`claims.json`; `data/trading/` itself stays out of git.

## What is inside

| file | contents |
|---|---|
| `statistics.json` | the ledger totals the bot last wrote |
| `api_costs.json` | per-provider inference cost and token counts |
| `trade_history.db` | every recorded decision, plus the post-mortems |
| `metrics/cycles.jsonl` | one line per analysis cycle: timestamp, signal, latency |
| `manifest.json` | capture time, source path, `sha256` and size of each file |

The ChromaDB brain (`brain_<pair>_<timeframe>/`), the logs and the backups stay in `data/trading/`:
they are the bot's working state, not the ledger the numbers come from.
