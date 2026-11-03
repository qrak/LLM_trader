#!/usr/bin/env bash
# POSIX twin of scripts/traffic_snapshot.ps1: same markdown file, plus a deduplicated
# history.jsonl so the 14-day GitHub traffic window stops erasing the record.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG="qrak/LLM_trader"
OUT_DIR="$ROOT/.hermes/traffic"

while [ $# -gt 0 ]; do
    case "$1" in
        --slug) SLUG="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -n "${GH:-}" ]; then
    GH_BIN="$GH"
elif command -v gh >/dev/null 2>&1; then
    GH_BIN="$(command -v gh)"
elif [ -x "/mnt/c/Program Files/GitHub CLI/gh.exe" ]; then
    GH_BIN="/mnt/c/Program Files/GitHub CLI/gh.exe"
else
    echo "gh not found; set GH=/path/to/gh" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

"$GH_BIN" api "repos/$SLUG" > "$TMP/repo.json"
"$GH_BIN" api "repos/$SLUG/traffic/views" > "$TMP/views.json"
"$GH_BIN" api "repos/$SLUG/traffic/clones" > "$TMP/clones.json"
"$GH_BIN" api "repos/$SLUG/traffic/popular/referrers" > "$TMP/refs.json"
"$GH_BIN" api "repos/$SLUG/traffic/popular/paths" > "$TMP/paths.json"
"$GH_BIN" api -H "Accept: application/vnd.github.star+json" \
    "repos/$SLUG/stargazers?per_page=100&page=1" > "$TMP/stars1.json"
"$GH_BIN" api -H "Accept: application/vnd.github.star+json" \
    "repos/$SLUG/stargazers?per_page=100&page=2" > "$TMP/stars2.json"

python3 - "$TMP" "$OUT_DIR" "$SLUG" <<'PY'
import collections
import datetime
import json
import pathlib
import sys

src, out, slug = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
stamp = datetime.date.today().isoformat()


def load(name):
    with open(src / name, encoding="utf-8") as handle:
        return json.load(handle)


repo = load("repo.json")
views = load("views.json")
clones = load("clones.json")
refs = load("refs.json")
paths = load("paths.json")
stars = [item["starred_at"] for item in load("stars1.json") + load("stars2.json")]

by_day = collections.Counter(moment[:10] for moment in stars)
window = [(datetime.date.today() - datetime.timedelta(days=offset)).isoformat() for offset in range(7)]
last_week = sum(by_day[day] for day in window)

lines = [
    f"# GitHub traffic and stars - {stamp}", "",
    f"Repository: https://github.com/{slug}", "",
    "## Totals", "",
    "| item | value |", "|---|---|",
    f"| stars | {repo['stargazers_count']} |",
    f"| forks | {repo['forks_count']} |",
    f"| open issues | {repo['open_issues_count']} |",
    f"| watchers | {repo['subscribers_count']} |",
    f"| last push | {repo['pushed_at']} |", "",
    f"- {views['count']} views, {views['uniques']} unique visitors (14 days)",
    f"- {clones['count']} clones, {clones['uniques']} unique cloners (14 days)",
    f"- {last_week} new stars in the last 7 days", "",
    "## Referrers (14 days)", "", "| referrer | views | uniques |", "|---|---|---|",
]
lines += [f"| {row['referrer']} | {row['count']} | {row['uniques']} |" for row in refs]
lines += ["", "## Popular paths (14 days)", "", "| path | views |", "|---|---|"]
lines += [f"| {row['title']} | {row['count']} |" for row in paths[:10]]
lines += ["", "## Star timestamps (last 100, oldest first)", "", "```"]
lines += stars[-100:] + ["```", ""]
(out / f"{stamp}.md").write_text("\n".join(lines), encoding="utf-8")

record = {
    "date": stamp,
    "stars": repo["stargazers_count"],
    "forks": repo["forks_count"],
    "views_14d": views["count"],
    "view_uniques_14d": views["uniques"],
    "clones_14d": clones["count"],
    "clone_uniques_14d": clones["uniques"],
    "stars_7d": last_week,
    "referrers": {row["referrer"]: row["count"] for row in refs},
}
history = out / "history.jsonl"
rows = []
if history.exists():
    for line in history.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("date") != stamp:
            rows.append(row)
rows.append(record)
history.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
print(f"wrote {out / f'{stamp}.md'} and {history}")
PY
