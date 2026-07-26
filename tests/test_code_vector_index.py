"""Tests for the codebase vector semantic search engine."""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.rag.code_vector_index import (
    COLLECTION_NAME,
    CodeChunk,
    CodebaseVectorIndexer,
    SearchResult,
)


class _FakeCollection:
    """In-memory fake ChromaDB collection for testing."""

    def __init__(self):
        self._data: dict[str, dict] = {}

    def count(self) -> int:
        return len(self._data)

    def get(self, include=None, where=None):
        ids = []
        metadatas = []
        documents = []
        for doc_id, entry in self._data.items():
            if where:
                match = all(
                    entry["metadata"].get(k) == v for k, v in where.items()
                )
                if not match:
                    continue
            ids.append(doc_id)
            if include and "metadatas" in include:
                metadatas.append(entry["metadata"])
            if include and "documents" in include:
                documents.append(entry["document"])

        result = {"ids": ids}
        if metadatas:
            result["metadatas"] = metadatas
        if documents:
            result["documents"] = documents
        return result

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._data[doc_id] = {
                "embedding": embeddings[i],
                "document": documents[i],
                "metadata": metadatas[i],
            }

    def delete(self, ids):
        for doc_id in ids:
            self._data.pop(doc_id, None)

    def query(self, query_embeddings, n_results, where=None, include=None):
        # Return all items sorted by insertion order (not real similarity)
        items = list(self._data.values())
        if where:
            items = [
                item for item in items
                if all(item["metadata"].get(k) == v for k, v in where.items())
            ]
        items = items[:n_results]
        return {
            "ids": [[f"id_{i}" for i in range(len(items))]],
            "documents": [[item["document"] for item in items]],
            "metadatas": [[item["metadata"] for item in items]],
            "distances": [[0.2 * i for i in range(len(items))]],
        }


class _FakeChromaClient:
    """Fake ChromaDB client that returns in-memory collections."""

    def __init__(self):
        self._collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = _FakeCollection()
        return self._collections[name]


class _FakeEmbeddingModel:
    """Fake SentenceTransformer that returns deterministic embeddings."""

    def encode(self, text: str) -> list[float]:
        # Simple hash-based deterministic embedding (384-dim like bge-small)
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(c, 16) / 15.0 for c in h] + [0.0] * (384 - 32)


@pytest.fixture
def indexer(tmp_path: Path) -> CodebaseVectorIndexer:
    """Create a CodebaseVectorIndexer with fakes."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    client = _FakeChromaClient()
    model = _FakeEmbeddingModel()
    return CodebaseVectorIndexer(
        logger=logger,
        chroma_client=client,
        embedding_model=model,
        project_root=tmp_path,
    )


class TestPythonParsing:
    """Tests for AST-based Python file parsing."""

    def test_parse_module_docstring(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "example.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text('"""Module docstring."""\n\nx = 42\n', encoding="utf-8")

        chunks = indexer._parse_python_file(py_file)
        doc_chunks = [c for c in chunks if c.symbol_type == "module_doc"]
        assert len(doc_chunks) == 1
        assert "Module docstring" in doc_chunks[0].content

    def test_parse_class_and_methods(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "myclass.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text(textwrap.dedent('''\
            class MyService:
                """Service for doing things."""

                def process(self, data: str) -> bool:
                    """Process input data."""
                    return True

                async def fetch(self):
                    """Fetch data async."""
                    pass
        '''), encoding="utf-8")

        chunks = indexer._parse_python_file(py_file)

        class_chunks = [c for c in chunks if c.symbol_type == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].symbol_name == "MyService"
        assert "Service for doing things" in class_chunks[0].docstring

        method_chunks = [c for c in chunks if c.symbol_type == "method"]
        assert len(method_chunks) == 2
        names = {c.symbol_name for c in method_chunks}
        assert "MyService.process" in names
        assert "MyService.fetch" in names

    def test_parse_top_level_function(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "utils.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text(textwrap.dedent('''\
            def calculate_rsi(prices: list, period: int = 14) -> float:
                """Calculate RSI indicator."""
                return 50.0
        '''), encoding="utf-8")

        chunks = indexer._parse_python_file(py_file)
        func_chunks = [c for c in chunks if c.symbol_type == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].symbol_name == "calculate_rsi"
        assert "Calculate RSI" in func_chunks[0].docstring

    def test_parse_syntax_error_file_returns_empty(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "broken.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text("def broken(\n", encoding="utf-8")

        chunks = indexer._parse_python_file(py_file)
        assert chunks == []


class TestMarkdownParsing:
    """Tests for Markdown section parsing."""

    def test_parse_sections(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        md_file = tmp_path / "README.md"
        md_file.write_text(textwrap.dedent('''\
            # Project Title

            This is the project overview with enough content to pass the threshold.

            ## Installation

            Run pip install to set up the project and its dependencies correctly.

            ## Usage

            Start the bot with python start.py and configure via config.ini file.
        '''), encoding="utf-8")

        chunks = indexer._parse_markdown_file(md_file)
        assert len(chunks) >= 3
        section_names = {c.symbol_name for c in chunks}
        assert "Project Title" in section_names
        assert "Installation" in section_names
        assert "Usage" in section_names

    def test_empty_sections_skipped(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        md_file = tmp_path / "sparse.md"
        md_file.write_text("# Title\n\n## Empty\n\n## Also Empty\n\n", encoding="utf-8")

        chunks = indexer._parse_markdown_file(md_file)
        # All sections have <= 20 chars of content, so all should be skipped
        assert len(chunks) == 0


class TestDeltaIndexing:
    """Tests for SHA-256 delta indexing behavior."""

    def test_unchanged_files_are_skipped(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        # Create a file and index it
        py_file = tmp_path / "src" / "service.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text(textwrap.dedent('''\
            def my_function():
                """Do something useful."""
                return 42
        '''), encoding="utf-8")

        # First index
        result1 = indexer.index_codebase(force=True)
        assert result1["indexed"] >= 1

        # Second index (same content) — should skip
        result2 = indexer.index_codebase(force=False)
        assert result2["skipped"] >= 1
        assert result2["indexed"] == 0

    def test_force_reindexes_all(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "service.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text("def f(): pass\n", encoding="utf-8")

        indexer.index_codebase(force=True)
        result = indexer.index_codebase(force=True)
        # With force=True, should re-index even unchanged files
        assert result["indexed"] >= 1


class TestSemanticSearch:
    """Tests for the search_codebase API."""

    def test_search_returns_results(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "calculator.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text(textwrap.dedent('''\
            def calculate_rsi(prices: list) -> float:
                """Calculate Relative Strength Index."""
                return 50.0

            def calculate_macd(prices: list) -> tuple:
                """Calculate MACD indicator line and signal."""
                return (0.0, 0.0)
        '''), encoding="utf-8")

        indexer.index_codebase(force=True)
        results = indexer.search_codebase("RSI indicator calculation", top_k=5)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.file_path for r in results)
        assert all(r.score >= 0.0 for r in results)

    def test_search_empty_index_returns_empty(self, indexer: CodebaseVectorIndexer):
        results = indexer.search_codebase("anything")
        assert results == []

    def test_search_with_type_filter(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "service.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text(textwrap.dedent('''\
            class MyService:
                """A service class."""
                def run(self):
                    pass
        '''), encoding="utf-8")

        indexer.index_codebase(force=True)

        # Search only for classes
        class_results = indexer.search_codebase("service", symbol_type="class")
        assert all(r.symbol_type == "class" for r in class_results)

    def test_search_result_has_line_numbers(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "lines.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text(textwrap.dedent('''\
            def first():
                pass

            def second():
                pass
        '''), encoding="utf-8")

        indexer.index_codebase(force=True)
        results = indexer.search_codebase("first or second function")

        assert len(results) > 0
        for r in results:
            assert r.start_line > 0
            assert r.end_line >= r.start_line


class TestGetStats:
    """Tests for index statistics."""

    def test_stats_empty_index(self, indexer: CodebaseVectorIndexer):
        stats = indexer.get_stats()
        assert stats["collection_name"] == COLLECTION_NAME
        assert stats["total_chunks"] == 0

    def test_stats_after_indexing(self, indexer: CodebaseVectorIndexer, tmp_path: Path):
        py_file = tmp_path / "src" / "module.py"
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text("def func(): pass\n", encoding="utf-8")

        indexer.index_codebase(force=True)
        stats = indexer.get_stats()
        assert stats["total_chunks"] > 0


class TestFileHashComputation:
    """Tests for SHA-256 file hash computation."""

    def test_same_content_same_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        content = "def foo(): pass\n"
        f1.write_text(content, encoding="utf-8")
        f2.write_text(content, encoding="utf-8")

        h1 = CodebaseVectorIndexer._compute_file_hash(f1)
        h2 = CodebaseVectorIndexer._compute_file_hash(f2)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("def foo(): pass\n", encoding="utf-8")
        f2.write_text("def bar(): pass\n", encoding="utf-8")

        h1 = CodebaseVectorIndexer._compute_file_hash(f1)
        h2 = CodebaseVectorIndexer._compute_file_hash(f2)
        assert h1 != h2
