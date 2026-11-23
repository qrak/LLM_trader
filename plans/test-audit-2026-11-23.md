# Test-suite audit — 2026-11-23

Read-only review of `tests/` (2,919 collected tests, 69 files) plus the runner configuration.
Nothing in this report has been deleted; every candidate below needs an explicit decision.

## What the time actually went to

Measured on this machine (Ryzen 9 7900X, 24 threads, ~7 GB free RAM), `.venv`, Python 3.13.14:

| Run | Wall time | Result |
| --- | --- | --- |
| `pytest tests -q` before any change | 300 s | 2,919 passed |
| after patching the five sleeping tests | 112 s | 2,919 passed |
| after `-n 4` (xdist) | 44 s | 2,919 passed |
| `-n 5` | 53 s | 2,919 passed |
| `-n 6` | 71 s | 2,919 passed |
| `-n 8` | — | workers killed, 1.7 GB free left |
| `-n 12` | — | out of memory |

Five tests were 190 s of the original 300 s, all of them sleeping through real retry backoff:

| Test | Time | Cause |
| --- | --- | --- |
| `test_ai_providers.py::TestMalformedPayloadDegradation::test_malformed_sdk_payloads_have_pinned_degradation` | 90 s | `retry_on_empty` re-sent a non-JSON reply 5× (1+2+4+8+16 s) |
| `test_ai_providers.py::TestDeepSeekRequestWiring::test_chat_completion_forwards_reasoning_effort_and_logs_cache_counters` | 35 s | same |
| `test_ai_providers.py::TestDeepSeekRequestWiring::test_json_output_mode_travels_with_the_request` | 33 s | same |
| `test_ai_providers.py::TestDeepSeekRequestWiring::test_chat_completion_drops_rejected_parameter_and_retries` | 33 s | same |
| `test_social_context.py::test_the_reported_error_is_the_last_status` | 5 s | real `CHALLENGE_RETRY_SECONDS` after a 403 |

The other 2,914 tests took ~105 s together, about 0.04 s each. **Deleting tests was never going to
fix the runtime**; the count is not the problem.

## Candidates for deletion or trimming

Ranked by how much they actually buy. None of them is a correctness risk to remove, but none of
them saves meaningful time either — after the fixes above the whole suite is 44 s.

### 1. Tests that re-run a CI gate on the whole repository (5 tests, ~8.5 s of work)

These call the same checker that `scripts/` runs in CI and in the pre-commit hook, so a green CI
already proves them:

- `tests/domain_infra/test_dead_symbols.py:26` `test_the_repository_has_no_dead_symbols` (2.6 s)
- `tests/domain_infra/test_dead_symbols.py:30` `test_known_live_symbols_are_not_reported` (2.1 s)
- `tests/domain_infra/test_comment_policy.py:19` `test_the_repository_has_no_comments_or_docstrings` (1.7 s)
- `tests/domain_infra/test_loose_helpers.py:73` `test_the_repository_has_no_loose_helpers` (1.3 s)
- `tests/domain_infra/test_config_contract.py:40` and `:44` (0.9 s)

The *unit* tests in the same files (tmp_path fixtures that prove the checker catches a violation)
must stay — they are the only thing testing the checker itself. Only the "scan the real repo"
assertions are duplicated by CI. Keeping them is defensible (they fail locally before a push);
deleting them removes ~8.5 s of work, which is ~2 s of wall time on 4 workers.

### 2. `tests/domain_infra/test_public_copy.py` — 46 tests for 3 assertions

`test_markdown_has_no_personal_identifiers` is parametrized over 12 phrases and re-reads every
markdown file for each one; `test_shipped_dashboard_copy_has_no_personal_identifiers` adds 24 more
(2 files × 12 phrases) and `test_shipped_dashboard_copy_is_not_first_person` 10 more. One test per
assertion that collects all offenders into a list would cover the same ground in 3 tests. Cheap
either way (<1 s), so this is a readability change, not a speed one.

### 3. `tests/domain_brain/test_vector_memory.py` — duplicated classifier tables

- `:262` `test_experience_document_labels_rsi_with_classifier` (11 cases) asserts
  `classify_rsi_label(rsi) == label` — the same function and the same thresholds that
  `test_indicators_and_patterns.py:224` already covers with 15 cases including NaN and inf. Its own
  added value is the second assertion, `f"RSI={rsi:.1f} ({label})" in build_document(rsi=rsi)`.
  Trimming to 2–3 cases keeps the document coverage and drops 8 duplicated ones.
- `:320` `test_adx_label_tiers` (9 cases) tests `VectorMemoryService._adx_label`
  (`src/trading/vector_memory_context.py:195`), a private re-implementation of
  `classify_adx_label` that `test_indicators_and_patterns.py:196` already covers with 10 cases.
  The test is not redundant — the *production* duplication is the defect. Fix the source, then
  delete this test.

### 4. Parametrized tables worth a second look

Largest collected groups (collected ids per test function):

| Cases | Test |
| --- | --- |
| 32 | `test_indicators_and_patterns.py::test_dict_classifiers_map_raw_indicators_to_labels` |
| 24 | `test_public_copy.py::test_shipped_dashboard_copy_has_no_personal_identifiers` |
| 16 | `test_indicators_and_patterns.py::test_indicator_alignment_table` |
| 16 | `test_indicators_and_patterns.py::test_pattern_direction_classification_table` |
| 15 | `test_indicators_and_patterns.py::test_rsi_label_follows_its_threshold_table` |
| 13 | `test_strategy_flow.py::TestUpdatePositionParameters::test_each_proposal_is_gated_and_persisted` |
| 13 | `test_rag_news.py::test_enricher_ssrf_guard_rejects_internal_and_non_http_targets` |

These are threshold tables where every case is a distinct boundary, so they are cheap and worth
keeping. No case here is in the same equivalence class as another.

## What must not be deleted

- The five tests that were slow. They were slow because of a missing patch, not because they were
  redundant; they cover the retry contract and the provider wiring.
- The unit tests of the `scripts/` checkers (tmp_path based). They are the only coverage of the
  gates themselves.
- Anything in `domain_trading` that asserts on money, fees, stops or booking. Those are the paths
  where a silent regression costs real money.

## Recommendation

1. Keep the suite as it is. The runtime problem is solved (300 s → 44 s) and the remaining
   candidates are worth ~2 s of wall time and a few dozen lines.
2. If the count itself is the annoyance, do items 2 and 3 only: they are the two places where the
   same assertion is genuinely written twice.
3. Fix `VectorMemoryService._adx_label` to call `classify_adx_label` and delete
   `test_adx_label_tiers` in the same change — that is a production defect, not a test problem.
