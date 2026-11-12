# Public numbers: one chain, no retyping

Every public text in this project quotes numbers that come from one place. This page is the contract;
if a command below changes, this page changes in the same commit.

```
data/trading/  (live runtime ledger, never in git)
     │  python scripts/snapshot_results.py
     ▼
results/<date>/ (frozen copy + manifest.json with sha256)
     │  python scripts/claims_report.py --data-dir results/<date> --benchmark
     ↓                                    ↓
claims.json                            python scripts/results_report.py --json <site>/src/data/ledger.json
(README, docs, gate)                       --data-dir results/<date>
                                        │
                                        ▼
                                       semantic-signal-site: src/pages/results.astro
                                       renders the ledger block and the trade table from it
```

The rule: **no number is typed twice.** A number enters through `claims.json` or through
`results_report.py --json`, and a page renders it. The site's own [check-claims.mjs](https://github.com/qrak/semantic-signal-site/blob/master/scripts/check-claims.mjs) runs on
every `npm run build` and fails when a page disagrees with the `public/claims.json` it carries.

## The automated path

```bash
python scripts/publish_public_numbers.py --dry-run   # print the plan, write nothing
python scripts/publish_public_numbers.py             # publish today's numbers
python scripts/publish_public_numbers.py --force     # redo today's publication
```

One run: runs the full suite (a red suite publishes nothing), freezes the ledger into
`results/<date>/`, re-renders the chart, regenerates `claims.json`, rewrites the block between
`<!-- numbers:start -->` and `<!-- numbers:end -->` in `README.md` (plus the badge and the test
count), writes the site's ledger JSON, its claims copy and its project summary, and
only after `check_claims --pytest`, `check_claims --site-root`, `check_agents_md` and ruff pass does it
commit, push, build and deploy. A dirty worktree — somebody else mid-change — aborts before anything is
written, and a second run in the same day is a no-op.

It runs from Task Scheduler as `LLM_trader publish numbers` (daily 04:30, hidden, `IgnoreNew`), through
`scripts/publish_public_numbers.ps1`, which loads the Cloudflare credentials from the DPAPI store and
logs to `logs/publish_public_numbers.log`.

The task runs non-elevated while both repositories are owned by `BUILTIN\\Administratorzy`, so git
refuses to touch them until both paths are listed in the user's `safe.directory` (git says exactly
that in its error). Without it the run aborts at the first `git status`.

## The manual path

The same steps by hand, when the automation is unavailable:

```bash
# 1. freeze the ledger (once per publication, not per cycle)
python scripts/snapshot_results.py

# 2. regenerate the numbers every public text quotes
python scripts/claims_report.py --data-dir results/<date> --benchmark

# 3. hand the same ledger to the site
python scripts/results_report.py --json <site>/src/data/ledger.json --data-dir results/<date>
cp claims.json <site>/public/claims.json

# 4. gate both repositories before deploying
python scripts/check_claims.py --pytest
python scripts/check_claims.py --site-root <site>
python scripts/snapshot_results.py --verify results/<date>
```

`--site-root` fails when the site still carries the previous `claims.json`, when
its ledger JSON disagrees with `claims.json`, or when a page quotes a stale test count or
commit count. It is a local gate: the site is a separate repository, so it cannot run in this
repository's CI.

## What is gated where

| gate | covers | runs in |
|---|---|---|
| `scripts/check_claims.py --pytest` | README, docs and CONTRIBUTING against `claims.json`; test count against a real collection | CI, every push |
| `scripts/check_claims.py --site-root <site>` | the site's copy of `claims.json`, its ledger JSON, test and commit counts in pages | local, before `wrangler pages deploy` |
| site check-claims.mjs | every site page against the `public/claims.json` it ships | site build |
| `scripts/snapshot_results.py --verify results/<date>` | the frozen ledger itself against its manifest | when a snapshot is quoted |

## What this does not cover

The buy-and-hold figure needs one public Binance klines request. When the network is unavailable the
snapshot still reproduces the ledger, and `benchmark_error` says why the benchmark column is missing.
