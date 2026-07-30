"""CLI tool for semantic codebase search via the vector index.

Usage:
    python scripts/query_codebase.py "your natural language query"
    python scripts/query_codebase.py --reindex "your query"
    python scripts/query_codebase.py --reindex --force
    python scripts/query_codebase.py --stats
    python scripts/query_codebase.py --type function "RSI calculation"

Self-contained: initialises its own ChromaDB client and SentenceTransformer,
queries the persisted codebase_semantic_index collection, prints results, exits.
No bot startup required.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _get_best_device() -> str:
    """Auto-detect best available hardware accelerator."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search across the LLM_trader codebase via ChromaDB vector index.",
    )
    parser.add_argument("query", nargs="?", default=None, help="Natural language search query.")
    parser.add_argument("--reindex", action="store_true", help="Re-index changed files before searching.")
    parser.add_argument("--force", action="store_true", help="Force full re-index (ignore file hashes).")
    parser.add_argument("--stats", action="store_true", help="Show index statistics and exit.")
    parser.add_argument("--top", type=int, default=5, help="Number of results to return (default: 5).")
    parser.add_argument("--type", type=str, default=None,
                        choices=["class", "function", "method", "markdown_section", "module_doc"],
                        help="Filter results by symbol type.")
    args = parser.parse_args()

    # --- Lazy imports (heavy) ---
    import chromadb
    from sentence_transformers import SentenceTransformer
    from src.logger.logger import Logger
    from src.rag.code_vector_index import CodebaseVectorIndexer

    # Resolve paths
    project_root = _PROJECT_ROOT

    # Read config for index dir (fallback to data/codebase_index)
    try:
        from src.config.loader import config as app_config
        index_dir = app_config.CODEBASE_INDEX_DIR
    except Exception:
        index_dir = str(project_root / "data" / "codebase_index")

    os.makedirs(index_dir, exist_ok=True)

    # Initialise services
    logger = Logger()
    chroma_client = chromadb.PersistentClient(path=index_dir)

    device = _get_best_device()
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    indexer = CodebaseVectorIndexer(
        logger=logger,
        chroma_client=chroma_client,
        embedding_model=embedding_model,
        project_root=project_root,
    )

    # --- Stats ---
    if args.stats:
        stats = indexer.get_stats()
        print("\n[STATS] Codebase Vector Index Stats")
        print(f"   Collection:   {stats['collection_name']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Project root: {stats['project_root']}")
        return

    # --- Reindex ---
    if args.reindex or args.force:
        print("[INDEX] Indexing codebase...")
        result = indexer.index_codebase(force=args.force)
        print(f"   [OK] Indexed: {result['indexed']} files, "
              f"Skipped: {result['skipped']}, "
              f"Total chunks: {result['total_chunks']}")
        if not args.query:
            return

    # --- Search ---
    if not args.query:
        parser.print_help()
        return

    results = indexer.search_codebase(
        query=args.query,
        top_k=args.top,
        symbol_type=args.type,
    )

    if not results:
        print(f"\n[!] No results found for: \"{args.query}\"")
        print("   Tip: Run with --reindex to build/update the index first.")
        return

    print(f"\n[SEARCH] Results for: \"{args.query}\"\n")
    for i, r in enumerate(results, start=1):
        score_pct = int(r.score * 100)
        print(f"  {i}. [{r.score:.2f}] ({score_pct}%)  {r.file_path}:L{r.start_line}-L{r.end_line}")
        print(f"     {r.symbol_type}: {r.symbol_name}")
        # Show first 2 lines of snippet
        snippet_lines = r.snippet.strip().splitlines()[:2]
        for line in snippet_lines:
            print(f"     | {line[:120]}")
        print()


if __name__ == "__main__":
    main()
