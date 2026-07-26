# Bugfixer 🐛 Regression & Bug Journal

## 2026-07-26 - Merge Conflict Resolution in Vector Memory Rules
**Learning:** Merging `origin/develop` (commit `fd0265a`) into local `develop` (commit `9295459`) created a conflict in `src/trading/vector_memory_rules.py`. Removing module-level `_LOG2` and `_LOG1P_FULL_SAMPLE` pre-computed constants caused `NameError` runtime exceptions in `_rule_freshness_score` and `_rule_evidence_score`.
**Action:** Resolved the merge conflict by retaining `_LOG2: float = math.log(2)` and `_LOG1P_FULL_SAMPLE: float = math.log1p(20)`. Verified that all 48 vector memory unit tests and 1,220 total workspace unit tests pass with zero failures.

## 2026-07-26 - Post-Refactoring Regression Suite Verification
**Learning:** Verified full workspace test suite (1,220 tests passed cleanly) following multi-domain refactoring, speedup, and accessibility updates.
**Action:** Confirmed zero breaking changes across brain router, vector memory rules, and static dashboard assets.

## 2026-07-26 - Secondary Refactoring & UX Pass Verification
**Learning:** Verified full workspace test suite and linter after secondary config `getattr` cleanup, primitive set lookup optimization, and admin login accessibility updates.
**Action:** All 1,220 unit tests pass cleanly with zero regressions.

## 2026-07-26 - Codebase Vector Search Engine & Security Audit
**Learning:** Verified Codebase Vector Semantic Search Engine (`src/rag/code_vector_index.py`, `scripts/query_codebase.py`, `.agents/skills/codebase-vector-search/SKILL.md`) and audited ChromaDB vector collection `codebase_semantic_index` for security boundaries.
**Action:** Audited all 160 indexed files and 2,304 chunks in ChromaDB. Verified **0 sensitive files** (`keys.env`, `.env`, `config.ini`, API keys, or credentials) were included — globs are strictly constrained to Python source files (`src/**/*.py`, `start.py`, `scripts/*.py`) and curated documentation (`*.md`). Verified full workspace test suite (1,236 tests passed cleanly, 0 failures) and linter (0 errors).

## 2026-07-26 - Test Mock Config Fix for Clean Direct Property Access
**Learning:** Defensive `getattr` calls in production `.py` files were previously masking incomplete test mock definitions. Updating test helper functions (`_make_config`, `_make_manager`, `_make_mgr_with_verbosity`) across `test_template_manager.py`, `test_trading_strategy_branches.py`, `test_trading_strategy_process_analysis.py`, and `test_rss_provider_contract.py` to supply complete config attributes allows production code to strictly enforce clean, direct property access (`self.config.FIELD`) without defensive `getattr` fallbacks.
**Action:** Updated test suite mock helpers with standard config attributes (`MARKET_TYPE`, `ENTRY_ORDER_TYPE`, `EXECUTOR_API_ENABLED`, `EXECUTOR_API_URL`, `RAG_NEWS_LIMIT`) and converted all production source files to direct property access. Verified 269 component unit tests pass 100% cleanly (0 failures).


