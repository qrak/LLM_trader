# Refactor ✨ Clean Code Journal

## 2026-07-26 - Class-Scope Mixin Type Annotations & Dict Lookup Dispatch
**Learning:** `VectorMemoryRulesMixin` referenced class attributes (`logger`, `_semantic_rules_collection`, `_decay_half_life_days`, etc.) in mixin methods without explicit class-level type annotations, raising IDE unresolved attribute warnings. In addition, `SerializableMixin._convert_value()` in `data_utils.py` executed redundant `if-elif` checks for primitive default values.
**Action:** Added explicit class-level attribute type annotations to `VectorMemoryRulesMixin` in `vector_memory_rules.py`, and refactored `SerializableMixin._convert_value()` to use a constant `_PRIMITIVE_DEFAULTS` dictionary lookup.

## 2026-07-26 - Helper Extraction for Position State in Dashboard Brain Router
**Smell:** Repeated `position if isinstance(position, dict) else {"has_position": False}` pattern was evaluated three separate times in `get_decision_summary()`.
**Cleanup:** Extracted `pos_dict = position if isinstance(position, dict) else {"has_position": False}` once right before synopsis, decision graph, and result dict construction.
**Lesson:** Normalizing structural dictionary input once at the top of a response builder prevents noise and redundant runtime type checks down the call stack.

## 2026-07-26 - Config Property Direct Access Cleanups
**Smell:** Defensive `getattr(self.config, "EXECUTOR_API_ENABLED", False)` in `trading_strategy.py` and `getattr(self.config, "MARKET_TYPE", "spot")` in `template_manager.py` bypassed known `Config` properties.
**Cleanup:** Replaced defensive `getattr()` calls with direct property access `self.config.EXECUTOR_API_ENABLED`, `self.config.EXECUTOR_API_URL`, `self.config.MARKET_TYPE`, and `self.config.ENTRY_ORDER_TYPE`.
**Lesson:** Known configuration properties defined on injected `Config` instances should be accessed directly according to project DI conventions.

## 2026-07-27 - Purge In-Function Lazy Imports in Dashboard Admin Router
**Smell:** `login()` in `src/dashboard/routers/admin.py` executed in-function lazy imports (`from ..auth import _sign_token`, `import time as _time`), violating Section 3.1 of master `AGENTS.md`.
**Cleanup:** Promoted `_sign_token` to top-level module import in `admin.py` and used top-level `time.time()`.
**Lesson:** All module dependencies must be imported at top-level scope or injected via constructor parameters at the CompositionRoot (`start.py`).

## 2026-07-27 - Direct Dict Defaulting & Defensive Check Reduction in Vector Memory Analytics
**Smell:** Imperative `if key not in groups: groups[key] = []` and multi-branch `if/elif` threshold checks in `_append_trade_to_group()`, `_factor_bucket_for_score()`, and `compute_adx_performance()`.
**Cleanup:** Replaced manual dictionary initialization with `groups.setdefault(key, [])` and simplified score/ADX bucketing with short-circuiting ternary expressions.
**Lesson:** Standard dictionary primitives like `dict.setdefault()` eliminate repetitive condition branches and decrease cyclomatic complexity without changing execution semantics.

## 2026-07-27 - Parameterized Loop Extraction in Indicator Pattern Engine
**Smell:** Duplicate code blocks for RSI and MACD bullish/bearish divergence checks in `_detect_divergence_patterns()` and multi-branch timestamp formatting logic in `_format_pattern_time()`.
**Cleanup:** Replaced 45 lines of copy-pasted divergence check blocks with a 12-line parameterized loop over `[("rsi", "rsi"), ("macd_line", "macd")]` and simplified timestamp formatting.
**Lesson:** Iterating over tuple definitions for repeated indicator evaluation patterns removes structural duplication and eliminates risk of missing edge-case updates in individual branches.






