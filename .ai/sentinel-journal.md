# Sentinel 🛡️ Security Audit Journal

## 2026-07-26 - Codebase Vector Search & Startup Maintenance Security Audit
**Audit Scope:** Codebase Vector Indexer (`src/rag/code_vector_index.py`), CLI Query Tool (`scripts/query_codebase.py`), Journal Rotation (`scripts/rotate_journals.py`), Startup Wiring (`start.py`), Config Loader (`src/config/loader.py`), and `.gitignore` updates.

**Security Findings & Safeguards Verified:**
1. **Zero Secret Leakage in ChromaDB**:
   - Audited ChromaDB collection `codebase_semantic_index` (160 files, 2,306 chunks). Verified 0 sensitive files (`keys.env`, `.env`, `config.ini`, API keys, or certificates) were indexed.
   - Indexing globs are strictly restricted to source files (`src/**/*.py`, `start.py`, `scripts/*.py`) and curated Markdown documentation (`*.md`).
2. **Git & Storage Boundaries**:
   - Verified `data/codebase_index/` is fully covered by `/data/*` in `.gitignore`. Vector database files cannot be committed or pushed to GitHub.
   - Verified `.ai/plans/` remains ignored in `.gitignore`.
3. **Safe AST Parsing**:
   - Code chunking uses `ast.parse()` static analysis only. Zero dynamic code evaluation (`eval`, `exec`, or module imports).
4. **Path Traversal Protection**:
   - File path resolution is anchored to `_PROJECT_ROOT` with relative sanitization and directory skipping (`_SKIP_DIRS`).

**Verdict:** 🛡️ **APPROVED** — Zero security violations or vulnerability regressions detected.

## 2026-07-27 - Admin Auth Rate Limiting, RAG SSRF Guards & Config Input Sanitization
**Audit Scope:** Admin authentication (`src/dashboard/auth.py`), Admin control console router (`src/dashboard/routers/admin.py`), and RAG Crawl4AI enricher (`src/rag/news_ingestion/crawl4ai_enricher.py`).

**Security Safeguards Added:**
1. **IP Sliding-Window Rate Limiting (`auth.py` / `admin.py`)**:
   - Added `check_login_rate_limit()` and `record_login_attempt()` sliding-window attempt tracker (max 5 attempts per 60 seconds per IP).
   - Prevents PBKDF2 (100k iterations) CPU exhaustion and brute-force dictionary attacks against `/api/admin/login`.
2. **RAG News Crawler SSRF Protection (`crawl4ai_enricher.py`)**:
   - Added `_is_safe_external_url()` static validator.
   - Enforces scheme checks (`http`/`https`) and strictly blocks attempts to crawl internal loopback/LAN IP ranges (`127.0.0.1`, `::1`, `10.0.0.0/8`, `192.168.0.0/16`, `169.254.169.254`).
3. **Config Update Length Bounds (`admin.py`)**:
   - Added `@field_validator` to `ConfigUpdateRequest` capping string configuration inputs at 4,000 characters.

**Verdict:** 🛡️ **APPROVED** — Security posture hardened with rate limiting, SSRF checks, and input bounds.

