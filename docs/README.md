# Documentation

The [README](../README.md) is the landing page; these are the long-form pages. The settings desk at
`http://localhost:8000/admin/` (Documentation tab) renders the allowlisted pages — this index, Setup,
Configuration, Features, State machines, Settings panel, the project README and Contributing — while
the bot runs. Public numbers, the MCP guide and the changelog archives are outside that allowlist;
their links open on GitHub.

## Start here

| Page | What it covers |
|------|----------------|
| [Setup](SETUP.md) | Requirements, quick start, platform launchers, secrets, runtime controls |
| [Configuration](CONFIGURATION.md) | Core settings, prompt budget, billing windows, decision gate and executor handoff |
| [Settings panel](SETTINGS_PANEL.md) | The localhost settings desk: access, tabs, validation, restart, troubleshooting |

## Reference

| Page | What it covers |
|------|----------------|
| [Features](FEATURES.md) | Every feature in detail, with the implementing file |
| [State machines](STATE_MACHINES.md) | Lifecycle states, their evidence and their user/API surface |
| [Public numbers](PUBLIC-NUMBERS.md) | The chain from the runtime ledger to `claims.json`, the site and the gate that keeps them equal |
| [MCP server](MCP.md) | The read-only MCP server over the trade ledger: tools, install, client config, safety |

## Project

| Page | What it covers |
|------|----------------|
| [Project overview](../README.md) | What the bot is, how it performs, quick start |
| [Contributing](../CONTRIBUTING.md) | Branches, releases, test gates, changelog rules |
| [Changelog](../CHANGELOG.md) | Every change, dated, with the run that verified it |
| [Agent rules](../AGENTS.md) | Policy for coding agents working in this repository |

## Where a fact lives

A fact belongs in exactly one page and is linked from the others. The public numbers live in
[`claims.json`](../claims.json) and are regenerated from a frozen ledger snapshot in
[`results/`](../results) by `python scripts/claims_report.py --data-dir results/<date>`; the results
page on the site renders the same snapshot. Do not copy them into new prose without that script. The
configuration page is checked against the loader and the template by
`python scripts/check_config_contract.py`.
