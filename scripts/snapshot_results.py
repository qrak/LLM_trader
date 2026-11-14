from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

LEDGER_FILES = ("statistics.json", "api_costs.json", "metrics/cycles.jsonl")
DATABASE = "trade_history.db"


class LedgerSnapshot:
    def __init__(self, source: Path, target: Path) -> None:
        self.source = source
        self.target = target

    def write(self) -> dict:
        self.target.mkdir(parents=True, exist_ok=True)
        (self.target / "metrics").mkdir(exist_ok=True)
        names = []
        for name in LEDGER_FILES:
            origin = self.source / name
            if not origin.is_file():
                raise FileNotFoundError(f"ledger file missing: {origin}")
            shutil.copy2(origin, self.target / name)
            names.append(name)
        self._copy_database()
        names.append(DATABASE)
        manifest = {
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(self.source),
            "files": {name: self._entry(name) for name in names},
        }
        (self.target / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    def verify(self) -> list[str]:
        manifest_path = self.target / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = [
            f"{name}: sha256 differs from the manifest"
            for name, entry in manifest["files"].items()
            if self._entry(name)["sha256"] != entry["sha256"]
        ]
        expected = set(manifest["files"])
        for path in self.target.rglob("*"):
            relative = path.relative_to(self.target).as_posix()
            if path.is_file() and relative != "manifest.json" and relative not in expected:
                problems.append(f"{relative}: not in the manifest")
        return problems

    def _copy_database(self) -> None:
        origin = self.source / DATABASE
        if not origin.is_file():
            raise FileNotFoundError(f"ledger database missing: {origin}")
        destination = self.target / DATABASE
        with (
            sqlite3.connect(f"file:{origin.resolve()}?mode=ro", uri=True) as source,
            sqlite3.connect(destination) as target,
        ):
            source.backup(target)
        with sqlite3.connect(destination) as target:
            target.execute("PRAGMA journal_mode = DELETE")
        for sidecar in ("-wal", "-shm"):
            leftover = destination.with_name(DATABASE + sidecar)
            if leftover.exists():
                leftover.unlink()

    def _entry(self, name: str) -> dict:
        path = self.target / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"sha256": digest, "bytes": path.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the runtime ledger into results/<date>/ so a clone can reproduce the numbers."
    )
    parser.add_argument("--source", default=Path("data/trading"), type=Path)
    parser.add_argument("--target", default=None, type=Path, help="defaults to results/<UTC date>")
    parser.add_argument("--verify", default=None, type=Path, help="re-hash an existing snapshot")
    args = parser.parse_args(argv)

    if args.verify is not None:
        problems = LedgerSnapshot(args.source, args.verify.resolve()).verify()
        for problem in problems:
            print(problem)
        print(f"{args.verify.resolve()}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        return 0 if not problems else 1

    target = args.target or Path("results") / datetime.now(timezone.utc).date().isoformat()
    manifest = LedgerSnapshot(args.source, target.resolve()).write()
    for name, entry in manifest["files"].items():
        print(f"{name}: {entry['bytes']} bytes, sha256 {entry['sha256'][:16]}")
    print(f"snapshot written to {target.resolve()} (captured {manifest['captured_at']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
