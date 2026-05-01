## 2024-03-02 - [Replace np.isnan with math.isnan for scalars]
**Learning:** `np.isnan()` has a large dispatch overhead when called on scalar values (like `arr[-1]` or single floats). Using the built-in `math.isnan()` is ~13x faster for these scalar operations.
**Action:** When checking if a single value is NaN (especially in tight loops or heavy indicator calculation methods), prefer `math.isnan()` over `np.isnan()`. Reserve `np.isnan()` for vectorized operations on full numpy arrays.

## 2024-05-15 - [Refactor np.isnan to math.isnan everywhere for scalars]
**Learning:** Extending the previous learning, replacing `np.isnan` with `math.isnan` consistently across the entire codebase for scalar values yields micro-performance gains, even inside numba compiled loops, although the python gains are more significant (~13x). Be careful to only replace `np.isnan(scalar)` and not vectorized operations like `~np.isnan(arr)`.
**Action:** Always default to `math.isnan` for any scalar NaN checks.

## 2024-05-16 - [Offload file reading from async logic to thread pools]
**Learning:** Calling `json.load()` and `open()` synchronously on large files like `previous_response.json` from within a heavily async method (such as `_build_analysis_context` inside the main trading loop or dashboard APIs) blocks the main asyncio event loop, causing severe latency and degraded bot performance.
**Action:** Always offload synchronous file reads and parsing into a thread pool worker using `asyncio.to_thread` when inside an async environment.

## 2024-05-18 - [Offload CostStorage initialization to a thread pool]
**Learning:** Instantiating classes that perform synchronous file I/O operations (like `open()` and `json.load()`) in their `__init__` method (e.g., `CostStorage()`) within an async route blocks the main `asyncio` event loop. This creates severe performance bottlenecks and potential timeouts in FastAPI endpoints.
**Action:** When inside an async environment, instantiating classes with synchronous `__init__` methods involving file I/O must be offloaded to a thread pool using `asyncio.to_thread(ClassName, args...)`.
## 2024-05-17 - [Offload DefiLlama file I/O from async logic to thread pools]
**Learning:** Calling synchronous file I/O operations (`open`, `json.load`, `json.dump`, `os.path.exists`, `os.remove`, `os.replace`) inside the `get_defi_fundamentals` async function directly blocks the main asyncio event loop. Even atomic file operations wait on disk, degrading concurrency.
**Action:** Extract all file read/write operations (e.g. cache reads and writes) into synchronous helper methods and run them using `await asyncio.to_thread()` when called from within an async context.

## 2024-05-18 - [Add caching to PersistenceManager.get_last_analysis_time]
**Learning:** Calling `get_last_analysis_time()` rapidly in tight loops blocks on disk I/O each time, creating unnecessary overhead. A simple in-memory cache with validity flags can reduce read time by roughly 300x.
**Action:** Identify frequently read data that only needs an update when a write occurs, and cache it in memory via class variables, updating the cache synchronously during write operations.

## 2024-05-19 - [Replace O(N*K) sliding window loops in MFI and CMF with O(N)]
**Learning:** Functions like `mfi_numba` and `chaikin_money_flow_numba` in `src/indicators/volume/volume_indicators.py` used an inner loop over the lookback window `length` inside the main loop over the data array `n`. This O(N*K) complexity scales poorly for large arrays and large lookback windows.
**Action:** Replace nested loops in sliding window indicators with single-pass O(N) algorithms by maintaining running sums, adding the newest element and subtracting the oldest element that drops out of the window.

## 2025-03-12 - [Refactor Fibonacci Bollinger Bands to O(N)]
**Learning:** `np.sum`, `np.mean`, and list slicing in sliding window calculations via Numba inside tight loops result in O(N * window_size) performance and negate significant parts of Numba's speedups.
**Action:** Use running sums (e.g. `sum_pv += new_val; sum_pv -= old_val`) to drop the complexity to O(N). When keeping a running sum of squares for variance, it's vital to periodically recenter the sums (e.g., every 1000 items) to prevent floating-point precision loss due to drifting time series.

## 2025-03-12 - [Refactor cci_numba to avoid np.roll]
**Learning:** `np.roll()` inside a Numba loop is incredibly slow because it allocates and returns a new array on every iteration rather than modifying the original array in place. If its output is ignored, the underlying array (like `tp_window`) never changes, completely breaking correctness (meaning values drift to nonsense).
**Action:** Replace `np.roll` with single-pass O(N) sliding sum variables or slice expressions. Never assume `np.roll()` operates in place inside a Numba `@njit` loop.

## 2025-03-12 - [Refactor loops to avoid np.mean on array slices]
**Learning:** Using `np.mean(arr[i:j])` inside Numba compiled loops for calculating moving averages (such as in `eom`, `stochastic` and `rmi`) creates massive performance overhead due to implicit memory allocations in each loop iteration, resulting in O(N*L) time complexity where L is the lookback period.
**Action:** Refactor instances of `np.mean` on slices to single-pass O(N) sliding sum windows using a running total that adds the incoming value and subtracts the outgoing one, taking care of explicitly checking and propagating NaN values when necessary.

## 2025-03-12 - [Refactor np.max and np.min on slices to O(N) sliding window]
**Learning:** Using `np.max(high[i - length:i])` and `np.min(low[i - length:i])` inside a Numba loop creates severe O(N*K) performance bottlenecks. This is because Numba implicitely allocates a slice slice and iterates it for max/min *every single loop iteration*.
**Action:** Replace `np.max` and `np.min` on moving windows inside Numba loops with an amortized O(N) sliding window that maintains the max/min and only scans the subset of elements when the max/min value drops out of the window.
## 2025-03-16 - [Refactor np.min and np.max on array slices to explicit loops]
**Learning:** Using `np.min` and `np.max` on array slices (e.g., `np.max(high[i - length + 1:i + 1])`) inside Numba compiled loops implicitly allocates new arrays on every iteration, destroying performance. Refactoring this to use explicit single-pass element-wise loops completely avoids these allocations and yields up to ~3x performance boosts in sliding window calculations.
**Action:** When searching for min or max elements within a sliding window inside Numba `@njit` loops, avoid creating array slices. Instead, initialize the min/max variable to the first element in the window and manually iterate over the rest to update it.

## 2025-04-18 - [Replace np.sum on slices with O(N) sliding window sum]
**Learning:** Using `np.sum(array[i - length:i])` inside a Numba `@njit` loop results in implicit slice allocations and iterating over the window on every single step, creating severe O(N*L) performance bottlenecks (where L is the lookback length).
**Action:** Replace `np.sum` on moving windows inside Numba loops with a running total. Initialize the sum for the first window, then in the main loop, add the newest incoming element to the sum and subtract the oldest element falling out of the window to achieve an amortized O(N) time complexity.

## 2025-05-18 - [Replace np.sum on slices with O(N) sliding window sum for zscore]
**Learning:** Using `np.sum` inside a Numba loop with array slices (e.g., `np.sum(window)`) iterates over the window on every single step. For rolling statistical measures like Z-Score, this creates severe O(N*L) bottlenecks.
**Action:** Replace `np.sum` inside Numba loops with a running total (e.g., updating sums of elements and sums of squares elements) to drop complexity from O(N*L) to amortized O(N), occasionally recalculating values exactly (e.g., every 1000 items) to prevent float precision drift.

## 2025-06-12 - [Refactor np.sum(diff) and array slicing in Numba loops to explicit loops]
**Learning:** Using operations like `np.max`, `np.min`, and `np.abs(sliced_close[1:] - sliced_close[:-1])` on array slices inside a Numba loop implicitly allocates new arrays on every iteration. This causes significant performance bottlenecks, creating O(N*K) scaling.
**Action:** Avoid creating slices in Numba loops for simple window calculations. Manually calculate differences (`abs(val - prev_val)`) and maintain maximum and minimum states with an explicit inner loop over the window indices.
