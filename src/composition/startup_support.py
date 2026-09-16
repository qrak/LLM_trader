"""
Startup helpers for the bot entry point.

Environment/GPU probing, the optional Tk error dialog, the console banner and the startup
summary table. Kept out of start.py so the composition root stays readable.

"""

"""
Crypto Trading Bot - Entry Point
Automated trading with AI-powered decisions.
"""

import logging
import os
import warnings
from pathlib import Path

import torch
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config.loader import config
from src.logger.logger import Logger

# pylint: disable=wrong-import-position

warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="discord")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.genai")


def configure_hf_hub_auth() -> None:
    """Expose optional Hugging Face token and suppress raw tqdm progress bar leaks."""
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    except Exception:  # noqa: S110, BLE001
        pass

    hf_token = config.get_env("HF_TOKEN")
    if not hf_token:
        return

    token = str(hf_token).strip()
    if not token:
        return

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
def cleanup_legacy_embedding_cache(logger: Logger | None = None) -> None:
    """Remove legacy bge-small embedding models from HuggingFace cache to free disk space."""
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        old_model_dir = cache_dir / "models--BAAI--bge-small-en-v1.5"
        new_model_dir = cache_dir / "models--BAAI--bge-base-en-v1.5"

        if new_model_dir.exists() and old_model_dir.exists():
            import shutil

            shutil.rmtree(old_model_dir, ignore_errors=True)
            if logger:
                logger.info(
                    "  -> Pruned legacy BAAI/bge-small-en-v1.5 embedding model from cache"
                )
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.debug("Failed to prune legacy embedding cache: %s", e)
def get_best_device() -> str:
    """Auto-detect best available hardware accelerator for embeddings.

    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
try:
    import tkinter as tk
    from tkinter import messagebox

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
def show_error_dialog(title: str, message: str) -> bool:
    if not TKINTER_AVAILABLE:
        return False

    root = None
    try:
        root = tk.Tk()  # type: ignore[reportPossiblyUnboundVariable]
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(title, message, parent=root)  # type: ignore[reportPossiblyUnboundVariable]
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:  # noqa: S110, BLE001
                pass
def build_startup_banner(project_root: Path) -> Panel:
    """Styled 'LLM TRADER v1.1' banner using rich panel + unicode block drawing."""
    logo_llm = [
        "██╗     ██╗     ███╗   ███╗",
        "██║     ██║     ████╗ ████║",
        "██║     ██║     ██╔████╔██║",
        "██║     ██║     ██║╚██╔╝██║",
        "███████╗███████╗██║ ╚═╝ ██║",
        "╚══════╝╚══════╝╚═╝     ╚═╝",
    ]
    logo_trader = [
        "████████╗██████╗  █████╗ ██████╗ ███████╗██████╗ ",
        "╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗",
        "   ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝",
        "   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗",
        "   ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║",
        "   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝",
    ]
    logo_version = [
        "██╗   ██╗ ██╗   ██╗",
        "██║   ██║███║  ███║",
        "██║   ██║╚██║  ╚██║",
        "╚██╗ ██╔╝ ██║   ██║",
        " ╚████╔╝  ██║██╗██║",
        "  ╚═══╝   ╚═╝╚═╝╚═╝",
    ]
    canvas_width = max(len(line) for line in [*logo_llm, *logo_trader, *logo_version])

    text = Text()
    for line in logo_llm:
        text.append(line.center(canvas_width) + "\n", style="bold bright_cyan")
    text.append("\n")
    for line in logo_trader:
        text.append(line.center(canvas_width) + "\n", style="bold bright_cyan")
    text.append("\n")
    for line in logo_version:
        text.append(line.center(canvas_width) + "\n", style="bold bright_yellow")
    text.append("\n")
    tagline = "AI-Powered Crypto Trading Bot"
    text.append(tagline.center(canvas_width) + "\n", style="bold yellow")
    text.append(("─" * len(tagline)).center(canvas_width) + "\n", style="dim white")
    text.append(str(project_root.resolve()).center(canvas_width) + "\n", style="dim cyan")

    return Panel(
        Align.center(text),
        title="[bold blue]🚀  LLM TRADER  🚀[/]",
        border_style="bright_blue",
        padding=(1, 2),
    )
def print_summary_table(
    console: Console, elapsed: float, stats: dict[str, str]
) -> None:
    """Show a compact summary table after all provisioning stages complete."""
    table = Table(
        title=f"[bold green]✅  Initialization Complete  ({elapsed:.1f}s)[/]",
        title_justify="center",
        border_style="green",
        box=None,
        padding=(0, 2),
        show_header=False,
    )
    table.add_column("Metric", style="dim cyan", no_wrap=True)
    table.add_column("Value", style="bold white", justify="right")

    for metric, value in stats.items():
        table.add_row(metric, value)

    table.add_row("", "", style="dim")
    table.add_row("Init time", f"{elapsed:.2f}s", style="green")

    console.print()
    console.print(Panel(table, border_style="green"))
