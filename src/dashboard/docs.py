from pathlib import Path
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ASSETS_DIR = REPO_ROOT / "docs" / "assets"
REPO_URL = "https://github.com/qrak/LLM_trader"


class DocsLibrary:
    DOCS = {  # noqa: RUF012
        "docs": ("docs/README.md", "Documentation home"),
        "setup": ("docs/SETUP.md", "Setup"),
        "configuration": ("docs/CONFIGURATION.md", "Configuration"),
        "features": ("docs/FEATURES.md", "Features"),
        "state-machines": ("docs/STATE_MACHINES.md", "State machines"),
        "settings-panel": ("docs/SETTINGS_PANEL.md", "Settings panel"),
        "overview": ("README.md", "Project overview"),
        "contributing": ("CONTRIBUTING.md", "Contributing"),
    }

    def __init__(self, root: str | Path = REPO_ROOT):
        self.root = Path(root)
        self.assets_dir = self.root / "docs" / "assets"
        self._md = MarkdownIt(
            "commonmark",
            {"html": False, "linkify": False, "typographer": False},
        )
        self._md.enable(["table", "strikethrough"])

    def nav(self) -> list[dict[str, str]]:
        return [
            {"slug": slug, "title": title} for slug, (_path, title) in self.DOCS.items()
        ]

    def render(self, slug: str) -> dict[str, str] | None:
        entry = self.DOCS.get(slug)
        if entry is None:
            return None
        path = self.root / entry[0]
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        tokens = self._md.parse(text)
        self._rewrite(tokens, path)
        html = self._md.renderer.render(tokens, self._md.options, {})
        return {
            "slug": slug,
            "title": entry[1],
            "html": html,
            "source_url": f"{REPO_URL}/blob/master/{entry[0]}",
        }

    def _rewrite(self, tokens: list[Token], doc_path: Path) -> None:
        for token in tokens:
            if not token.children:
                continue
            for child in token.children:
                if child.type == "link_open":
                    href = child.attrGet("href")
                    target, external = self._href(
                        href if isinstance(href, str) else "", doc_path
                    )
                    child.attrSet("href", target)
                    if external:
                        child.attrSet("target", "_blank")
                        child.attrSet("rel", "noopener noreferrer")
                elif child.type == "image":
                    source = child.attrGet("src")
                    child.attrSet(
                        "src",
                        self._image_src(
                            source if isinstance(source, str) else "",
                            doc_path,
                        ),
                    )

    def _href(self, href: str, doc_path: Path) -> tuple[str, bool]:
        if not href:
            return "#", False
        if href.startswith("#"):
            return href, False
        if href.startswith("mailto:"):
            return href, False
        parsed = urlsplit(href)
        if parsed.scheme in ("http", "https"):
            return href, True
        if parsed.scheme or href.startswith("//"):
            return "#", False
        base = href.split("#", 1)[0]
        target = (doc_path.parent / base).resolve()
        if target.is_relative_to(self.assets_dir) and target.is_file():
            relative = target.relative_to(self.assets_dir).as_posix()
            return f"/docs-assets/{relative}", False
        if target.is_dir() and target.is_relative_to(self.root):
            relative = target.relative_to(self.root).as_posix()
            return f"{REPO_URL}/tree/master/{relative}", True
        if target.suffix.lower() == ".md":
            slug = self._slug_for(target)
            if slug is not None:
                return f"#docs/{slug}", False
            if target.is_relative_to(self.root):
                relative = target.relative_to(self.root).as_posix()
                return f"{REPO_URL}/blob/master/{relative}", True
            return "#", False
        if target.is_file() and target.is_relative_to(self.root):
            relative = target.relative_to(self.root).as_posix()
            return f"{REPO_URL}/blob/master/{relative}", True
        return "#", False

    def _image_src(self, src: str, doc_path: Path) -> str:
        if not src:
            return "#"
        if src.startswith("data:image/"):
            return src
        parsed = urlsplit(src)
        if parsed.scheme in ("http", "https"):
            return src
        if parsed.scheme:
            return "#"
        target = (doc_path.parent / src).resolve()
        if target.is_relative_to(self.assets_dir) and target.is_file():
            relative = target.relative_to(self.assets_dir).as_posix()
            return f"/docs-assets/{relative}"
        return "#"

    def _slug_for(self, target: Path) -> str | None:
        for slug, (path, _title) in self.DOCS.items():
            if (self.root / path).resolve() == target:
                return slug
        return None
