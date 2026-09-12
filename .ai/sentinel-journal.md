# Sentinel 🛡️ Security Audit Journal

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

## 2026-07-27 - Rate Limiting In-Memory Key Eviction Hardening
**Audit Scope:** Admin authentication rate limiting (`src/dashboard/auth.py`).

**Security Safeguard Hardened:**
1. **Sliding-Window In-Memory Key Eviction (`check_login_rate_limit`)**:
   - Added automated key eviction for IP entries with no active window attempts.
   - Prevents memory growth and dictionary bloat from stale client IP tracking entries over long-running daemon lifecycles.

**Verdict:** 🛡️ **APPROVED** — Rate limiter memory safety verified.

