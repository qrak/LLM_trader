"""Journal Rotation & Archiving Script.

Rotates .ai/*-journal.md files when they exceed a line threshold (default: 300 lines).
Keeps recent entries in the active journal file and archives older entries to .ai/archive/.
All archived entries remain indexed in ChromaDB vector memory.
"""

import os
import re
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AI_DIR = _PROJECT_ROOT / ".ai"
_ARCHIVE_DIR = _AI_DIR / "archive"
_MAX_ACTIVE_LINES = 300
_KEEP_RECENT_ENTRIES = 15


def rotate_journal(journal_path: Path, max_lines: int = _MAX_ACTIVE_LINES) -> bool:
    """Rotate a single journal file if it exceeds max_lines.

    Returns:
        True if rotated, False otherwise.
    """
    if not journal_path.is_file():
        return False

    text = journal_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if len(lines) <= max_lines:
        return False

    # Extract header (all lines before the first '## ')
    header_lines: list[str] = []
    entries: list[tuple[str, list[str]]] = []  # list of (header_title, entry_lines)
    current_title = ""
    current_entry: list[str] = []

    in_header = True
    for line in lines:
        if line.startswith("## "):
            in_header = False
            if current_entry:
                entries.append((current_title, current_entry))
            current_title = line
            current_entry = [line]
        elif in_header:
            header_lines.append(line)
        else:
            current_entry.append(line)

    if current_entry:
        entries.append((current_title, current_entry))

    if len(entries) <= _KEEP_RECENT_ENTRIES:
        return False

    # Split into active (recent) vs archived (older)
    archived_entries = entries[: -_KEEP_RECENT_ENTRIES]
    active_entries = entries[-_KEEP_RECENT_ENTRIES:]

    # Prepare archive file
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    year = datetime.now().year
    archive_name = f"{journal_path.stem}-{year}.md"
    archive_path = _ARCHIVE_DIR / archive_name

    # Write archived entries
    archive_content = []
    if archive_path.exists():
        archive_content.append(archive_path.read_text(encoding="utf-8"))
    else:
        archive_content.append(f"# Archived Journal Entries — {journal_path.name}\n\n")

    for title, entry_body in archived_entries:
        archive_content.append("\n".join(entry_body) + "\n\n")

    archive_path.write_text("".join(archive_content), encoding="utf-8")

    # Rewrite active journal file
    active_content = []
    active_content.extend(header_lines)
    if active_content and not active_content[-1].strip() == "":
        active_content.append("")

    for title, entry_body in active_entries:
        active_content.append("\n".join(entry_body))
        active_content.append("")

    journal_path.write_text("\n".join(active_content), encoding="utf-8")
    print(f"[ROTATE] {journal_path.name}: {len(archived_entries)} entries archived to .ai/archive/{archive_name}")
    return True


def rotate_all_journals() -> int:
    """Scan and rotate all .ai/*-journal.md files."""
    if not _AI_DIR.exists():
        return 0

    rotated_count = 0
    for journal_path in _AI_DIR.glob("*-journal.md"):
        if rotate_journal(journal_path):
            rotated_count += 1

    return rotated_count


if __name__ == "__main__":
    count = rotate_all_journals()
    if count == 0:
        print("[ROTATE] All journal files are under threshold (no rotation needed).")
