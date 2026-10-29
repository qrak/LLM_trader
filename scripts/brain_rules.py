

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chromadb

from src.trading.vector_memory import (
    VectorMemoryService,
    brain_store_path,
)
from src.utils.timeframe_validator import TimeframeValidator

RULES_COLLECTION = "semantic_rules"


def _fingerprint(path: Path) -> str:

    database = path / "chroma.sqlite3"
    if not database.is_file():
        return "missing"
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    return f"{digest[:16]} ({database.stat().st_size} B)"


class BrainRuleStore:
    """Read and retire the semantic rules stored in one brain directory."""

    def __init__(
        self,
        brain_dir: Path,
        timeframe_minutes: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.brain_dir = brain_dir
        self.logger = logger or logging.getLogger("brain_rules")
        self.client = chromadb.PersistentClient(path=str(brain_dir))
        self.collection = self.client.get_collection(RULES_COLLECTION)
        self.service = VectorMemoryService(
            self.logger,
            self.client,
            embedding_model=None,
            timeframe_minutes=timeframe_minutes,
        )

    def rules(self) -> list[dict]:

        stored = self.collection.get(include=["documents", "metadatas"])
        entries: list[dict] = []
        for index, rule_id in enumerate(stored["ids"]):
            entries.append({
                "rule_id": rule_id,
                "text": stored["documents"][index] if stored["documents"] else "",
                "metadata": stored["metadatas"][index] if stored["metadatas"] else {},
            })
        return entries

    def prompt_block(self) -> str:

        return self.service.get_anti_patterns_for_prompt(k=2)

    def deactivate(self, rule_ids: list[str], reason: str, dry_run: bool = False) -> bool:

        if dry_run:
            for rule_id in rule_ids:
                print(f"would deactivate {rule_id} ({reason or 'no reason given'})")
            return True
        results = [self.service.deactivate_semantic_rule(rule_id, reason) for rule_id in rule_ids]
        return all(results)


def _print_rules(store: BrainRuleStore) -> None:
    entries = store.rules()
    active = [entry for entry in entries if entry["metadata"].get("active", True)]
    print(f"\n=== {store.brain_dir} ===")
    print(f"rules: {len(entries)} total, {len(active)} active, fingerprint {_fingerprint(store.brain_dir)}\n")
    if not entries:
        print("(no rules stored)")
        return
    for entry in entries:
        metadata = entry["metadata"]
        state = "ACTIVE" if metadata.get("active", True) else "inactive"
        print(f"{entry['rule_id']}")
        print(
            f"  {state} · {metadata.get('rule_type', '?')} · "
            f"source_pattern={metadata.get('source_pattern', '?')} · "
            f"support_count={metadata.get('support_count', '?')}"
        )
        print(f"  {entry['text']}")
        if metadata.get("deactivation_reason"):
            print(f"  retired because: {metadata['deactivation_reason']}")
        print()


def _print_prompt_block(store: BrainRuleStore) -> None:
    block = store.prompt_block()
    print(f"\n=== prompt block for {store.brain_dir} ===\n")
    if block:
        print(block)
    else:
        print("(empty — no AVOID / IMPROVE / AI-MISTAKE section will be injected)")
    print(f"\nfingerprint {_fingerprint(store.brain_dir)}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="List, inspect and retire the semantic rules of a trading brain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A rule is never deleted: deactivate sets active=false and every read path filters on active.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("list", "show every stored rule and whether it is active"),
        ("check", "print the anti-pattern block this service renders"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--brain-dir", type=Path, default=None)
        command.add_argument("--timeframe", default=None)

    retire = sub.add_parser("deactivate", help="retire one or more rules by id")
    retire.add_argument("--id", dest="ids", action="append", required=True)
    retire.add_argument("--reason", default="")
    retire.add_argument("--dry-run", action="store_true")
    retire.add_argument("--brain-dir", type=Path, default=None)
    retire.add_argument("--timeframe", default=None)

    args = parser.parse_args()

    if args.brain_dir is not None and args.timeframe is not None:
        timeframe = TimeframeValidator.validate_and_normalize(args.timeframe)
        brain_dir = args.brain_dir
    else:
        from src.config.loader import config as live_config

        timeframe = TimeframeValidator.validate_and_normalize(args.timeframe or live_config.TIMEFRAME)
        brain_dir = args.brain_dir or Path(
            brain_store_path(live_config.DATA_DIR, live_config.CRYPTO_PAIR, timeframe)
        )
    timeframe_minutes = TimeframeValidator.to_minutes(timeframe)

    if not brain_dir.is_dir():
        print(f"Error: no brain directory at {brain_dir} (timeframe={timeframe})")
        return 1

    store = BrainRuleStore(brain_dir, timeframe_minutes)

    if args.command == "list":
        _print_rules(store)
        return 0

    if args.command == "check":
        _print_prompt_block(store)
        return 0

    ok = store.deactivate(args.ids, args.reason, dry_run=args.dry_run)
    if not args.dry_run:
        _print_rules(store)
        _print_prompt_block(store)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
