"""Codebase vector semantic search engine using ChromaDB.

Indexes Python modules (AST-level) and Markdown documentation into a
local ChromaDB collection for instant semantic search across the codebase.
"""

import ast
import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.logger.logger import Logger


@dataclass(slots=True)
class CodeChunk:
    """A semantic chunk extracted from a source file."""

    file_path: str
    symbol_name: str
    symbol_type: str  # class, function, method, docstring, module_doc, markdown_section
    start_line: int
    end_line: int
    content: str
    docstring: str = ""


@dataclass(slots=True)
class SearchResult:
    """A single search result from the codebase vector index."""

    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    score: float
    snippet: str


COLLECTION_NAME = "codebase_semantic_index"

# Directories and patterns to skip during indexing
_SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "data",
    "logs", ".ai", ".agents",
}

_PYTHON_GLOBS = ["src/**/*.py", "start.py", "scripts/*.py"]
_MARKDOWN_GLOBS = [
    "AGENTS.md", "README.md", "CHANGELOG.md",
    "src/*/AGENTS.md",
    ".ai/*.md",
    ".ai/archive/*.md",
    "docs/*.md",
]


class CodebaseVectorIndexer:
    """AST-aware codebase indexer with ChromaDB vector storage.

    Parses Python source files into semantic chunks (classes, functions,
    methods, docstrings) and Markdown documentation into section chunks.
    Stores embeddings in a dedicated ChromaDB collection for instant
    semantic search.

    Uses SHA-256 delta hashing so re-indexing skips unchanged files.
    """

    def __init__(
        self,
        logger: Logger,
        chroma_client: Any,
        embedding_model: Any,
        project_root: Path,
    ):
        """Initialize the codebase vector indexer.

        Args:
            logger: Logger instance (DI).
            chroma_client: Injected ChromaDB PersistentClient.
            embedding_model: Injected SentenceTransformer model.
            project_root: Root directory of the project to index.
        """
        self.logger = logger
        self._client = chroma_client
        self._embedding_model = embedding_model
        self._embedding_lock = threading.Lock()
        self._project_root = project_root.resolve()
        self._collection: Any | None = None

    def _ensure_collection(self) -> Any:
        """Get or create the codebase semantic index collection with auto dimension recovery."""
        if self._collection is None:
            collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            try:
                count = collection.count()
            except Exception:
                count = 0
            if isinstance(count, int) and count > 0 and self._embedding_model is not None:
                try:
                    test_vector = self._encode("dimension_check")
                    collection.query(query_embeddings=[test_vector], n_results=1)
                except Exception as err:
                    err_msg = str(err).lower()
                    if any(kw in err_msg for kw in ("dimension", "expect", "incompatible", "invalid")):
                        self.logger.warning(
                            "Codebase vector index dimension mismatch (%s). "
                            "Re-creating collection '%s'...",
                            err, COLLECTION_NAME
                        )
                        try:
                            self._client.delete_collection(name=COLLECTION_NAME)
                        except Exception:
                            pass
                        collection = self._client.create_collection(
                            name=COLLECTION_NAME,
                            metadata={"hnsw:space": "cosine"},
                        )
                    else:
                        raise
            self._collection = collection
        return self._collection

    def _encode(self, text: str) -> list[float]:
        """Encode text to embedding vector with thread-safe model access."""
        return self._encode_batch([text])[0]

    def _encode_single(self, text: str) -> list[float]:
        """Encode a single text string with thread-safe model access."""
        with self._embedding_lock:
            encoded = self._embedding_model.encode(text)
        try:
            return encoded.tolist()
        except AttributeError:
            return list(encoded)

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts to embedding vectors with thread-safe model access."""
        if not texts:
            return []
        try:
            with self._embedding_lock:
                encoded = self._embedding_model.encode(texts, batch_size=32)
            try:
                return encoded.tolist()
            except AttributeError:
                return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in encoded]
        except (TypeError, AttributeError):
            return [self._encode_single(t) for t in texts]

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    # ------------------------------------------------------------------
    # AST Python Parsing
    # ------------------------------------------------------------------

    def _parse_python_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a Python file into semantic code chunks using AST.

        Extracts classes (with member overview), functions/methods
        (with decorators and docstrings), and module-level docstrings.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            self.logger.debug("Skipping %s: %s", file_path, exc)
            return []

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            self.logger.debug("Skipping %s (syntax error): %s", file_path, exc)
            return []

        rel_path = str(file_path.relative_to(self._project_root)).replace("\\", "/")
        lines = source.splitlines()
        chunks: list[CodeChunk] = []

        # Module docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            chunks.append(CodeChunk(
                file_path=rel_path,
                symbol_name="(module)",
                symbol_type="module_doc",
                start_line=1,
                end_line=min(len(lines), 10),
                content=module_doc,
                docstring=module_doc,
            ))

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._extract_class(node, rel_path, lines))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_function(node, rel_path, lines)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _extract_class(
        self, node: ast.ClassDef, rel_path: str, lines: list[str]
    ) -> list[CodeChunk]:
        """Extract a class definition and its methods as separate chunks."""
        chunks: list[CodeChunk] = []
        class_doc = ast.get_docstring(node) or ""

        # Build a member overview (method signatures)
        members = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async " if isinstance(child, ast.AsyncFunctionDef) else ""
                args = self._format_args(child.args)
                members.append(f"  {prefix}def {child.name}({args})")

        member_text = "\n".join(members[:20])  # Cap at 20 members for embedding size
        class_content = f"class {node.name}:\n{class_doc}\n\nMembers:\n{member_text}"

        chunks.append(CodeChunk(
            file_path=rel_path,
            symbol_name=node.name,
            symbol_type="class",
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            content=class_content,
            docstring=class_doc,
        ))

        # Extract each method as its own chunk
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_function(
                    child, rel_path, lines, class_name=node.name
                )
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        rel_path: str,
        lines: list[str],
        class_name: str | None = None,
    ) -> CodeChunk | None:
        """Extract a function or method definition as a code chunk."""
        end_line = node.end_lineno or node.lineno
        # Extract the actual source lines (capped at 50 lines for embedding)
        func_lines = lines[node.lineno - 1 : min(end_line, node.lineno + 49)]
        func_source = "\n".join(func_lines)

        doc = ast.get_docstring(node) or ""
        symbol_name = f"{class_name}.{node.name}" if class_name else node.name
        symbol_type = "method" if class_name else "function"
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        args = self._format_args(node.args)

        # Build a rich content string for embedding
        content = f"{prefix}def {symbol_name}({args})\n{doc}\n{func_source}"

        return CodeChunk(
            file_path=rel_path,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            start_line=node.lineno,
            end_line=end_line,
            content=content,
            docstring=doc,
        )

    @staticmethod
    def _format_args(args: ast.arguments) -> str:
        """Format function arguments to a concise signature string."""
        parts = []
        for arg in args.args:
            annotation = ""
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                annotation = f": {arg.annotation.id}"
            parts.append(f"{arg.arg}{annotation}")
        if len(parts) > 6:
            parts = parts[:6] + ["..."]
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Markdown Parsing
    # ------------------------------------------------------------------

    def _parse_markdown_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a Markdown file into section chunks delimited by headers."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            self.logger.debug("Skipping %s: %s", file_path, exc)
            return []

        rel_path = str(file_path.relative_to(self._project_root)).replace("\\", "/")
        lines = content.splitlines()
        chunks: list[CodeChunk] = []

        current_header = "(top)"
        current_start = 1
        current_lines: list[str] = []

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#") and " " in stripped:
                # Flush previous section
                if current_lines:
                    section_text = "\n".join(current_lines)
                    if len(section_text.strip()) > 20:  # Skip trivially empty sections
                        chunks.append(CodeChunk(
                            file_path=rel_path,
                            symbol_name=current_header,
                            symbol_type="markdown_section",
                            start_line=current_start,
                            end_line=i - 1,
                            content=section_text,
                        ))
                current_header = stripped.lstrip("#").strip()
                current_start = i
                current_lines = [line]
            else:
                current_lines.append(line)

        # Flush last section
        if current_lines:
            section_text = "\n".join(current_lines)
            if len(section_text.strip()) > 20:
                chunks.append(CodeChunk(
                    file_path=rel_path,
                    symbol_name=current_header,
                    symbol_type="markdown_section",
                    start_line=current_start,
                    end_line=len(lines),
                    content=section_text,
                ))

        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _collect_files(self) -> tuple[list[Path], list[Path]]:
        """Collect Python and Markdown files to index."""
        py_files: list[Path] = []
        md_files: list[Path] = []

        for pattern in _PYTHON_GLOBS:
            for path in self._project_root.glob(pattern):
                if any(skip in path.parts for skip in _SKIP_DIRS):
                    continue
                if path.is_file():
                    py_files.append(path)

        for pattern in _MARKDOWN_GLOBS:
            for path in self._project_root.glob(pattern):
                if any(skip in path.parts for skip in _SKIP_DIRS):
                    continue
                if path.is_file():
                    md_files.append(path)

        return py_files, md_files

    def _get_indexed_hashes(self) -> dict[str, str]:
        """Get the file_hash for all currently indexed files from ChromaDB metadata."""
        collection = self._ensure_collection()
        try:
            count = collection.count()
        except Exception:
            return {}
        if count == 0:
            return {}

        try:
            results = collection.get(include=["metadatas"])
        except Exception:
            return {}

        hashes: dict[str, str] = {}
        for meta in results.get("metadatas", []):
            if meta:
                fp = meta.get("file_path", "")
                fh = meta.get("file_hash", "")
                if fp and fh:
                    hashes[fp] = fh
        return hashes

    def _delete_file_chunks(self, rel_path: str) -> None:
        """Delete all chunks for a given file from the collection."""
        collection = self._ensure_collection()
        try:
            results = collection.get(
                where={"file_path": rel_path},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
        except Exception as exc:
            self.logger.debug("Failed to delete chunks for %s: %s", rel_path, exc)

    def index_codebase(self, force: bool = False) -> dict[str, int]:
        """Index the codebase into the ChromaDB vector collection.

        Uses SHA-256 delta hashing: unchanged files are skipped.

        Args:
            force: If True, re-index all files regardless of hash.

        Returns:
            Dict with counts: {"indexed": N, "skipped": N, "total_chunks": N}
        """
        collection = self._ensure_collection()
        py_files, md_files = self._collect_files()
        all_files = [(f, "python") for f in py_files] + [(f, "markdown") for f in md_files]

        if not force:
            indexed_hashes = self._get_indexed_hashes()
        else:
            indexed_hashes = {}

        indexed_count = 0
        skipped_count = 0
        total_chunks = 0

        for file_path, file_type in all_files:
            rel_path = str(file_path.relative_to(self._project_root)).replace("\\", "/")
            current_hash = self._compute_file_hash(file_path)

            # Delta check: skip if hash matches
            if not force and indexed_hashes.get(rel_path) == current_hash:
                skipped_count += 1
                continue

            # Delete old chunks for this file before re-indexing
            self._delete_file_chunks(rel_path)

            # Parse into chunks
            if file_type == "python":
                chunks = self._parse_python_file(file_path)
            else:
                chunks = self._parse_markdown_file(file_path)

            if not chunks:
                skipped_count += 1
                continue

            # Batch insert
            ids = []
            documents = []
            metadatas = []
            embed_texts = []

            for i, chunk in enumerate(chunks):
                doc_id = f"{rel_path}::{chunk.symbol_name}::{i}"
                embed_texts.append(chunk.content[:500])
                ids.append(doc_id)
                documents.append(chunk.content[:2000])  # Store up to 2000 chars
                metadatas.append({
                    "file_path": chunk.file_path,
                    "symbol_name": chunk.symbol_name,
                    "symbol_type": chunk.symbol_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "file_hash": current_hash,
                    "docstring": chunk.docstring[:300] if chunk.docstring else "",
                })

            embeddings = self._encode_batch(embed_texts)

            try:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                indexed_count += 1
                total_chunks += len(chunks)
            except Exception as exc:
                self.logger.warning("Failed to index %s: %s", rel_path, exc)

        self.logger.info(
            "Codebase index: %d files indexed, %d skipped, %d total chunks",
            indexed_count, skipped_count, total_chunks,
        )
        return {
            "indexed": indexed_count,
            "skipped": skipped_count,
            "total_chunks": total_chunks,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_codebase(
        self,
        query: str,
        top_k: int = 5,
        symbol_type: str | None = None,
    ) -> list[SearchResult]:
        """Search the codebase vector index semantically.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            symbol_type: Optional filter by symbol type
                         (class, function, method, markdown_section).

        Returns:
            List of SearchResult ordered by relevance (highest score first).
        """
        collection = self._ensure_collection()
        try:
            count = collection.count()
        except Exception:
            count = 0

        if count == 0:
            self.logger.warning("Codebase index is empty. Run index_codebase() first.")
            return []

        query_embedding = self._encode(query)
        where_filter = {"symbol_type": symbol_type} if symbol_type else None

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            self.logger.warning("Codebase search failed: %s", exc)
            return []

        search_results: list[SearchResult] = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids_list):
            meta = metas_list[i] if i < len(metas_list) else {}
            doc = docs_list[i] if i < len(docs_list) else ""
            dist = dists_list[i] if i < len(dists_list) else 1.0

            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1.0 = perfect match, 0.0 = no match
            score = max(0.0, 1.0 - dist)

            search_results.append(SearchResult(
                file_path=meta.get("file_path", ""),
                symbol_name=meta.get("symbol_name", ""),
                symbol_type=meta.get("symbol_type", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                score=round(score, 4),
                snippet=doc[:300] if doc else "",
            ))

        return search_results

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the codebase index."""
        collection = self._ensure_collection()
        try:
            count = collection.count()
        except Exception:
            count = 0

        return {
            "collection_name": COLLECTION_NAME,
            "total_chunks": count,
            "project_root": str(self._project_root),
        }
