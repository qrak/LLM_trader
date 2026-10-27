"""List, inspect and retire the semantic rules of a trading brain's Chroma store.

Usage:
  python scripts/brain_rules.py list      [--brain-dir PATH]
  python scripts/brain_rules.py check     [--brain-dir PATH]
  python scripts/brain_rules.py deactivate --id ID [--id ID ...] [--reason TEXT]
                                           [--dry-run] [--brain-dir PATH]

A rule is never deleted: `deactivate` sets `active: false`, so the text and the
evidence stay in the store while every read path — which filters on `active` — stops
seeing it. Nothing in `src/` expires rules by age: an aged rule merely loses retrieval
score and is labelled `legacy`, so retiring a rule needs this command.

`check` prints exactly the block a prompt receives, which is the honest verification:
when it returns an empty string, no AVOID / IMPROVE / AI-MISTAKE section can appear.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import chromadb

from src.trading.vector_memory import VectorMemoryService

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRAIN_DIR = REPO_ROOT / "data" / "trading" / "brain_BTC_USDC_4h"
RULES_COLLECTION = "semantic_rules"


def _fingerprint(path: Path) -> str:
    """Short digest of the Chroma database file, to prove what changed in a repair."""
    database = path / "chroma.sqlite3"
    if not database.is_file():
        return "missing"
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    return f"{digest[:16]} ({database.stat().st_size} B)"


class BrainRuleStore:
    """Read and retire the semantic rules stored in one brain directory."""

    def __init__(self, brain_dir: Path, logger: logging.Logger | None = None) -> None:
        self.brain_dir = brain_dir
        self.logger = logger or logging.getLogger("brain_rules")
        self.client = chromadb.PersistentClient(path=str(brain_dir))
        self.collection = self.client.get_collection(RULES_COLLECTION)
        self.service = VectorMemoryService(
            self.logger,
            self.client,
            embedding_model=None,
            timeframe_minutes=240,
        )

    def rules(self) -> list[dict]:
        """Every stored rule, active or not, as Chroma returns it."""
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
        """The exact rule block injected into the next prompt."""
        return self.service.get_anti_patterns_for_prompt(k=2)

    def deactivate(self, rule_ids: list[str], reason: str, dry_run: bool = False) -> bool:
        """Retire each rule; returns False when any id was unknown or the write failed."""
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
    # Rule text carries emoji; a cp1250 console would otherwise raise on print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="List, inspect and retire the semantic rules of a trading brain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("list", "show every stored rule and whether it is active"),
        ("check", "print the exact rule block the next prompt would receive"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--brain-dir", type=Path, default=DEFAULT_BRAIN_DIR)

    retire = sub.add_parser("deactivate", help="retire one or more rules by id")
    retire.add_argument("--id", dest="ids", action="append", required=True)
    retire.add_argument("--reason", default="")
    retire.add_argument("--dry-run", action="store_true")
    retire.add_argument("--brain-dir", type=Path, default=DEFAULT_BRAIN_DIR)

    args = parser.parse_args()

    if not args.brain_dir.is_dir():
        print(f"Error: no brain directory at {args.brain_dir}")
        return 1

    store = BrainRuleStore(args.brain_dir)

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
