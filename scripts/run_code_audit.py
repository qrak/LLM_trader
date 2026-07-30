"""Run full static analysis (Ruff, Pyright, Pylint) and save findings to data/static_analysis_report.txt."""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_FILE = DATA_DIR / "static_analysis_report.txt"

TARGET_PATHS = ["src", "start.py"]


class CodeAuditor:
    """Orchestrates static analysis tools and writes consolidated audit reports."""

    def __init__(self, output_path: Path = DEFAULT_OUTPUT_FILE):
        self.output_path = output_path
        self.venv_python = self._find_venv_python()

    def _find_venv_python(self) -> str:
        """Locate the virtual environment Python interpreter."""
        if sys.platform == "win32":
            candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        else:
            candidate = PROJECT_ROOT / ".venv" / "bin" / "python"

        if candidate.exists():
            return str(candidate)
        return sys.executable

    def _run_tool_command(self, cmd: list[str]) -> tuple[int, str]:
        """Execute a tool command and return (exit_code, combined_output)."""
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            combined = (stdout + "\n" + stderr).strip()
            return result.returncode, combined
        except Exception as err:
            return 1, f"Failed to execute command {' '.join(cmd)}: {err}"

    def run_ruff(self) -> dict[str, str | int]:
        """Run Ruff check on target paths."""
        cmd = [self.venv_python, "-m", "ruff", "check"] + TARGET_PATHS
        code, output = self._run_tool_command(cmd)
        return {
            "name": "Ruff",
            "exit_code": code,
            "output": output or "No issues found.",
        }

    def run_pyright(self) -> dict[str, str | int]:
        """Run Pyright type checking on target paths."""
        pyright_bin = Path(self.venv_python).parent / ("pyright.exe" if sys.platform == "win32" else "pyright")
        if pyright_bin.exists():
            cmd = [str(pyright_bin)] + TARGET_PATHS
        else:
            cmd = [self.venv_python, "-m", "pyright"] + TARGET_PATHS

        code, output = self._run_tool_command(cmd)
        return {
            "name": "Pyright",
            "exit_code": code,
            "output": output or "No issues found.",
        }

    def run_pylint(self) -> dict[str, str | int]:
        """Run Pylint on target paths (focusing on Errors and Warnings)."""
        cmd = [
            self.venv_python,
            "-m",
            "pylint",
            "--disable=C,R",  # Disable Convention and Refactor messages to focus on E and W
            "--score=n",
        ] + TARGET_PATHS

        code, output = self._run_tool_command(cmd)
        return {
            "name": "Pylint",
            "exit_code": code,
            "output": output or "No issues found.",
        }

    def generate_report(self, tools: list[str] | None = None) -> Path:
        """Run static analysis tools and write the consolidated report."""
        if tools is None:
            tools = ["ruff", "pyright", "pylint"]

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        report_sections: list[str] = [
            "=" * 80,
            f"  LLM_trader Code Audit & Static Analysis Report",
            f"  Generated: {timestamp}",
            f"  Target Paths: {', '.join(TARGET_PATHS)}",
            "=" * 80,
            "",
        ]

        results: list[dict[str, str | int]] = []

        if "ruff" in tools:
            print("[AUDIT] Running Ruff...")
            results.append(self.run_ruff())

        if "pyright" in tools:
            print("[AUDIT] Running Pyright...")
            results.append(self.run_pyright())

        if "pylint" in tools:
            print("[AUDIT] Running Pylint...")
            results.append(self.run_pylint())

        summary_lines: list[str] = ["SUMMARY OF AUDIT RESULTS:", "-" * 40]
        for res in results:
            status = "PASSED (0 issues)" if res["exit_code"] == 0 else f"ISSUES FOUND (exit code {res['exit_code']})"
            summary_lines.append(f"  • {res['name']}: {status}")

        report_sections.extend(summary_lines)
        report_sections.append("")

        for res in results:
            report_sections.extend([
                "=" * 80,
                f"SECTION: {res['name'].upper()} OUTPUT (Exit Code: {res['exit_code']})",
                "=" * 80,
                str(res["output"]),
                "",
            ])

        full_report = "\n".join(report_sections)

        self.output_path.write_text(full_report, encoding="utf-8")
        print(f"[AUDIT] Audit complete! Report saved to: {self.output_path}")

        return self.output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ruff, Pyright, and Pylint audit and write report to data/static_analysis_report.txt"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to save the output report file",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="ruff,pyright,pylint",
        help="Comma-separated list of tools to run (default: ruff,pyright,pylint)",
    )

    args = parser.parse_args()
    selected_tools = [t.strip().lower() for t in args.tools.split(",") if t.strip()]

    auditor = CodeAuditor(output_path=args.output)
    auditor.generate_report(tools=selected_tools)


if __name__ == "__main__":
    main()
