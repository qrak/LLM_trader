## 2025-02-27 - Numba Sliding Window Optimization
**Learning:** Numba handles loops efficiently, but algorithmic complexity still matters. Replacing O(N*K) re-summing with O(N) sliding window updates inside a Numba function yielded a ~4.5x speedup.
**Action:** Look for repetitive window calculations inside @njit functions and replace them with incremental updates.
