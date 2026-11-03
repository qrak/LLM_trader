<#
.SYNOPSIS
  Snapshot GitHub traffic and star counts into .hermes/traffic/<date>.md

.DESCRIPTION
  The traffic API keeps only 14 days, so the numbers have to be captured on a schedule.
  Run this before a launch and at D+3, D+7, D+14 and D+30.
  POSIX twin for WSL/Linux: scripts/traffic_snapshot.sh (same markdown, plus history.jsonl).

.EXAMPLE
  pwsh -NoProfile -File scripts/traffic_snapshot.ps1
  pwsh -NoProfile -File scripts/traffic_snapshot.ps1 -Slug qrak/LLM_trader -OutDir .hermes/traffic
#>
param(
    [string]$Slug = 'qrak/LLM_trader',
    [string]$OutDir = '.hermes/traffic'
)

$ErrorActionPreference = 'Stop'
$stamp = Get-Date -Format 'yyyy-MM-dd'
$target = Join-Path $OutDir "$stamp.md"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$totals = gh api "repos/$Slug" --jq '"| item | value |\n|---|---|\n| stars | \(.stargazers_count) |\n| forks | \(.forks_count) |\n| open issues | \(.open_issues_count) |\n| watchers | \(.subscribers_count) |\n| last push | \(.pushed_at) |"'
$views = gh api "repos/$Slug/traffic/views" --jq '"- \(.count) views, \(.uniques) unique visitors (14 days)"'
$clones = gh api "repos/$Slug/traffic/clones" --jq '"- \(.count) clones, \(.uniques) unique cloners (14 days)"'
$referrerRows = gh api "repos/$Slug/traffic/popular/referrers" --jq '.[] | "| \(.referrer) | \(.count) | \(.uniques) |"'
$pathRows = gh api "repos/$Slug/traffic/popular/paths" --jq '.[] | "| \(.title) | \(.count) |"' | Select-Object -First 10
$starRows = @(gh api "repos/$Slug/stargazers?per_page=100" -H 'Accept: application/vnd.github.v3.star+json' --jq '.[] | .starred_at')

$text = @(
    "# GitHub traffic and stars - $stamp",
    "",
    "Repository: https://github.com/$Slug",
    "",
    "## Totals",
    "",
    ($totals -join "`n"),
    "",
    $views,
    $clones,
    "- $($starRows.Count) starred_at timestamps captured below",
    "",
    "## Referrers (14 days)",
    "",
    "| referrer | views | uniques |",
    "|---|---|---|",
    ($referrerRows -join "`n"),
    "",
    "## Popular paths (14 days)",
    "",
    "| path | views |",
    "|---|---|",
    ($pathRows -join "`n"),
    "",
    "## Star timestamps (page 1, oldest first)",
    "",
    '```',
    ($starRows -join "`n"),
    '```'
) -join "`n"

Set-Content -Path $target -Value $text -Encoding UTF8
Write-Host "wrote $target"
Get-Content $target -TotalCount 20
