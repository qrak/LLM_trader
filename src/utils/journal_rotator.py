"""Automatic journal rotation service for .ai/ agent journal files."""

import shutil
from datetime import datetime, timezone
from pathlib import Path


class JournalRotator:
    """Service handling automatic agent journal rotation when size limits are exceeded."""

    def __init__(
        self,
        ai_dir: Path | str | None = None,
        max_lines: int = 100,
        max_bytes: int = 50 * 1024,
    ):
        """Initialize journal rotator settings.

        Args:
            ai_dir: Root directory for agent prompt/journal files (.ai/).
            max_lines: Maximum allowed line count before rotation.
            max_bytes: Maximum allowed byte size before rotation (default 50 KB).
        """
        if ai_dir is None:
            self.ai_dir = Path(__file__).resolve().parent.parent.parent / ".ai"
        else:
            self.ai_dir = Path(ai_dir)
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def rotate_journal_file(self, journal_path: Path) -> bool:
        """Rotate a single journal file if it exceeds size limits.

        Moves old content to .ai/archive/YYYY-MM/<name>-journal-<timestamp>.md
        and keeps a fresh active journal with header and current summary.

        Args:
            journal_path: Path to target journal markdown file.

        Returns:
            True if rotated, False otherwise.
        """
        if not journal_path.exists():
            return False

        try:
            content = journal_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            size_bytes = journal_path.stat().st_size

            if len(lines) < self.max_lines and size_bytes < self.max_bytes:
                return False

            # Archive path: .ai/archive/YYYY-MM/
            now = datetime.now(timezone.utc)
            month_str = now.strftime("%Y-%m")
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")

            ai_dir = journal_path.parent
            archive_dir = ai_dir / "archive" / month_str
            archive_dir.mkdir(parents=True, exist_ok=True)

            stem = journal_path.stem
            archived_filename = f"{stem}_{timestamp_str}.md"
            archived_path = archive_dir / archived_filename

            # Move original file to archive
            shutil.copy2(journal_path, archived_path)

            # Create clean active journal with top header lines
            header_lines = []
            for line in lines[:10]:
                if line.startswith(("# ", "## ", ">")):
                    header_lines.append(line)
                else:
                    break

            new_content = (
                "\n".join(header_lines)
                + "\n\n*Previous journal archived to `.ai/archive/"
                + month_str
                + "/"
                + archived_filename
                + "`*\n"
            )
            journal_path.write_text(new_content, encoding="utf-8")
            return True

        except Exception:  # noqa: BLE001
            return False

    def rotate_all_journals(self) -> int:
        """Scan and rotate all .ai/*-journal.md files if they exceed thresholds."""
        if not self.ai_dir.exists():
            return 0

        rotated_count = 0
        for journal_file in self.ai_dir.glob("*-journal.md"):
            if self.rotate_journal_file(journal_file):
                rotated_count += 1

        main_journal = self.ai_dir / "journal.md"
        if main_journal.exists() and self.rotate_journal_file(main_journal):
            rotated_count += 1

        return rotated_count

