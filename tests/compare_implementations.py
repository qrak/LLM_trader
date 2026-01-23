import numpy as np
import time
from tests.legacy_indicators import entropy_numba_legacy, linreg_numba_legacy, skew_numba_legacy
from src.indicators.statistical.statistical_indicators import entropy_numba, linreg_numba, skew_numba


def warmup(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception:
        pass

def compare_implementations():
    print("Generating random market data (N=20,000)...")
    np.random.seed(42)
    N = 20_000
    close = np.random.uniform(100, 200, N)
    high = close + 5
    low = close - 5
    length = 14
    
    print("Warming up JIT compilation...")
    # Warmup calls
    warmup(entropy_numba, close, length)
    warmup(entropy_numba_legacy, close, length)
    warmup(linreg_numba, close, length, True)
    warmup(linreg_numba_legacy, close, length, True)
    
    # Imports for warmup
    from src.indicators.volume.volume_indicators import twap_numba
    from tests.legacy_indicators import twap_numba_legacy
    warmup(twap_numba, high, low, close, length)
    warmup(twap_numba_legacy, high, low, close, length)
    
    from src.indicators.volatility.volatility_indicators import atr_numba
    from tests.legacy_indicators import atr_wma_numba_legacy
    warmup(atr_numba, high, low, close, length, 'wma')
    warmup(atr_wma_numba_legacy, high, low, close, length)

    from src.indicators.volatility.volatility_indicators import choppiness_index_numba
    from tests.legacy_indicators import choppiness_index_numba_legacy
    warmup(choppiness_index_numba, high, low, close, length)
    warmup(choppiness_index_numba_legacy, high, low, close, length)

    from src.indicators.volatility.volatility_indicators import bollinger_bands_numba
    from tests.legacy_indicators import bollinger_bands_numba_legacy
    warmup(bollinger_bands_numba, close, 20, 2.0)
    warmup(bollinger_bands_numba_legacy, close, 20, 2.0)

    from src.indicators.statistical.statistical_indicators import variance_numba
    from tests.legacy_indicators import variance_numba_legacy
    warmup(variance_numba, close, 30)
    warmup(variance_numba, close, 30)
    warmup(variance_numba_legacy, close, 30)

    warmup(skew_numba, close, 30)
    warmup(skew_numba_legacy, close, 30)

    from src.indicators.momentum import uo_numba
    from tests.legacy_indicators import uo_numba_legacy
    config = {
        'fast': 7, 'medium': 14, 'slow': 28,
        'fast_w': 4.0, 'medium_w': 2.0, 'slow_w': 1.0,
        'drift': 1
    }
    warmup(uo_numba, high, low, close, config)
    warmup(uo_numba_legacy, high, low, close)

    print("Warmup complete.\n")

    print("\n")
    
    # Legacy (Old)
    t0 = time.time()
    res_old_ent = entropy_numba_legacy(close, length)
    t_old_ent = time.time() - t0
    print(f"Old Entropy Time: {t_old_ent:.4f}s")
    
    # New (Optimized)
    t0 = time.time()
    res_new_ent = entropy_numba(close, length)
    t_new_ent = time.time() - t0
    print(f"New Entropy Time: {t_new_ent:.4f}s ({t_old_ent/t_new_ent:.1f}x faster)")
    
    # Correctness Check
    # We expect a mismatch due to the fix of the 1-bar lag.
    # New[i] should roughly equal Old[i+1] (because Old used i-L:i, New uses i-L+1:i+1)
    mask = ~np.isnan(res_old_ent) & ~np.isnan(res_new_ent)
    
    # Direct comparison (should fail)
    is_match = np.allclose(res_old_ent[mask], res_new_ent[mask], atol=1e-7)
    print(f"Direct Match: {is_match}")
    
    # Shifted comparison (New[i] vs Old[i+1])
    # Check if Old[i] == New[i-1] ? 
    # Old at i uses data up to i-1. New at i-1 uses data up to i-1.
    # So Old[i] should equal New[i-1].
    valid_indices = np.where(mask)[0]
    valid_indices = valid_indices[valid_indices > 0] # Avoid index 0
    
    # Compare Old[i] with New[i-1]
    diffs = np.abs(res_old_ent[valid_indices] - res_new_ent[valid_indices - 1])
    is_lag_match = np.all(diffs < 1e-7)
    print(f"Lagged Match (Old[i] == New[i-1]): {is_lag_match}")
    if is_lag_match:
        print(">> CONFIRMED: Old implementation had a 1-bar lag. New one is real-time.")

    print("\n--- LINREG (r=True, N=100k, L=14) ---")
    
    # Legacy (Old)
    t0 = time.time()
    res_old_lr = linreg_numba_legacy(close, length, r=True)
    t_old_lr = time.time() - t0
    print(f"Old LinReg Time: {t_old_lr:.4f}s")
    
    # New (Optimized)
    t0 = time.time()
    res_new_lr = linreg_numba(close, length, r=True)
    t_new_lr = time.time() - t0
    print(f"New LinReg Time: {t_new_lr:.4f}s ({t_old_lr/t_new_lr:.1f}x faster)")
    
    # Correctness
    print(f"Debug: res_old_lr type={type(res_old_lr)}")
    if hasattr(res_old_lr, 'shape'):
        print(f"Debug: res_old_lr shape={res_old_lr.shape}, dtype={res_old_lr.dtype}")
    if hasattr(res_old_lr, 'flags'):
        print(f"Debug: res_old_lr flags={res_old_lr.flags}")
    
    try:
        mask_lr = ~np.isnan(res_old_lr)
        is_lr_match = np.allclose(res_old_lr[mask_lr], res_new_lr[mask_lr], atol=1e-7)
        print(f"Direct Match: {is_lr_match}")
        if is_lr_match:
            print(">> CONFIRMED: Results are identical.")
    except Exception as e:
        print(f"CRASH in LinReg verification: {e}")


    from src.indicators.volume.volume_indicators import twap_numba
    from tests.legacy_indicators import twap_numba_legacy
    
    print("\n--- TWAP (N=100k, L=14) ---")
    high = close + 5
    low = close - 5
    
    t0 = time.time()
    res_old_twap = twap_numba_legacy(high, low, close, length)
    t_old_twap = time.time() - t0
    print(f"Old TWAP Time: {t_old_twap:.4f}s")
    
    t0 = time.time()
    res_new_twap = twap_numba(high, low, close, length)
    t_new_twap = time.time() - t0
    print(f"New TWAP Time: {t_new_twap:.4f}s ({t_old_twap/t_new_twap:.1f}x faster)")
    
    mask_twap = ~np.isnan(res_old_twap)
    is_twap_match = np.allclose(res_old_twap[mask_twap], res_new_twap[mask_twap], atol=1e-7)
    print(f"Direct Match: {is_twap_match}")
    if is_twap_match:
        print(">> CONFIRMED: Results are identical.")

    from src.indicators.volatility.volatility_indicators import atr_numba
    from tests.legacy_indicators import atr_wma_numba_legacy

    print("\n--- ATR WMA (N=100k, L=14) ---")
    
    t0 = time.time()
    res_old_atr = atr_wma_numba_legacy(high, low, close, length)
    t_old_atr = time.time() - t0
    print(f"Old ATR(WMA) Time: {t_old_atr:.4f}s")
    
    t0 = time.time()
    res_new_atr = atr_numba(high, low, close, length, mamode='wma')
    t_new_atr = time.time() - t0
    print(f"New ATR(WMA) Time: {t_new_atr:.4f}s ({t_old_atr/t_new_atr:.1f}x faster)")
    
    mask_atr = ~np.isnan(res_old_atr)
    # ATR logic might have small floating point diffs due to sliding window sums vs fresh sums
    is_atr_match = np.allclose(res_old_atr[mask_atr], res_new_atr[mask_atr], atol=1e-7)
    print(f"Direct Match: {is_atr_match}")
    if not is_atr_match:
        diff = np.max(np.abs(res_old_atr[mask_atr] - res_new_atr[mask_atr]))
        print(f"Max Diff: {diff:.2e}")
        if diff < 1e-5:
             print(">> CONFIRMED: Results match within epsilon (cleaner precision in new version).")
    else:
        print(">> CONFIRMED: Results are identical.")

    from src.indicators.volatility.volatility_indicators import choppiness_index_numba
    from tests.legacy_indicators import choppiness_index_numba_legacy

    print("\n--- CHOPPINESS INDEX (N=100k, L=14) ---")
    t0 = time.time()
    res_old_ci = choppiness_index_numba_legacy(high, low, close, length)
    t_old_ci = time.time() - t0
    print(f"Old Choppiness Time: {t_old_ci:.4f}s")
    
    t0 = time.time()
    res_new_ci = choppiness_index_numba(high, low, close, length)
    t_new_ci = time.time() - t0
    print(f"New Choppiness Time: {t_new_ci:.4f}s ({t_old_ci/t_new_ci:.1f}x faster)")
    
    mask_ci = ~np.isnan(res_old_ci)
    is_ci_match = np.allclose(res_old_ci[mask_ci], res_new_ci[mask_ci], atol=1e-7)
    print(f"Direct Match: {is_ci_match}")
    if is_ci_match:
         print(">> CONFIRMED: Results are identical.")

    from src.indicators.volatility.volatility_indicators import bollinger_bands_numba
    from tests.legacy_indicators import bollinger_bands_numba_legacy

    print("\n--- BOLLINGER BANDS (N=100k, L=20) ---")
    length_bb = 20
    num_std = 2.0
    t0 = time.time()
    res_old_bb = bollinger_bands_numba_legacy(close, length_bb, num_std)
    t_old_bb = time.time() - t0
    print(f"Old BB Time: {t_old_bb:.4f}s")
    
    t0 = time.time()
    res_new_bb = bollinger_bands_numba(close, length_bb, num_std)
    t_new_bb = time.time() - t0
    print(f"New BB Time: {t_new_bb:.4f}s ({t_old_bb/t_new_bb:.1f}x faster)")
    
    # BB returns tuple (upper, mid, lower)
    # Check upper band for simplicity
    mask_bb = ~np.isnan(res_old_bb[0])
    is_bb_match = np.allclose(res_old_bb[0][mask_bb], res_new_bb[0][mask_bb], atol=1e-7)
    print(f"Direct Match (Upper Band): {is_bb_match}")
    if is_bb_match:
        print(">> CONFIRMED: Results are identical.")

    from src.indicators.statistical.statistical_indicators import variance_numba
    from tests.legacy_indicators import variance_numba_legacy

    print("\n--- VARIANCE (N=100k, L=30) ---")
    length_var = 30
    t0 = time.time()
    res_old_var = variance_numba_legacy(close, length_var)
    t_old_var = time.time() - t0
    print(f"Old Variance Time: {t_old_var:.4f}s")
    
    t0 = time.time()
    res_new_var = variance_numba(close, length_var)
    t_new_var = time.time() - t0
    print(f"New Variance Time: {t_new_var:.4f}s ({t_old_var/t_new_var:.1f}x faster)")
    
    mask_var = ~np.isnan(res_old_var)
    is_var_match = np.allclose(res_old_var[mask_var], res_new_var[mask_var], atol=1e-7)
    print(f"Direct Match: {is_var_match}")
    if is_var_match:
        print(">> CONFIRMED: Results are identical.")
    else:
        diff = np.max(np.abs(res_old_var[mask_var] - res_new_var[mask_var]))
        print(f"Max Diff: {diff:.2e}")
        if diff < 1e-5:
            print(">> CONFIRMED: Results match within epsilon.")

    from src.indicators.momentum import uo_numba
    from tests.legacy_indicators import uo_numba_legacy

    print("\n--- ULTIMATE OSCILLATOR (N=100k) ---")
    # Need to config wrapper for new UO or call _uo_numba directly if exposed?
    # New UO wrapper expects config object or dict.
    # We will pass a dict.
    config = {
        'fast': 7, 'medium': 14, 'slow': 28,
        'fast_w': 4.0, 'medium_w': 2.0, 'slow_w': 1.0,
        'drift': 1
    }
    
    t0 = time.time()
    res_old_uo = uo_numba_legacy(high, low, close)
    t_old_uo = time.time() - t0
    print(f"Old UO Time: {t_old_uo:.4f}s")
    
    t0 = time.time()
    res_new_uo = uo_numba(high, low, close, config)
    t_new_uo = time.time() - t0
    print(f"New UO Time: {t_new_uo:.4f}s ({t_old_uo/t_new_uo:.1f}x faster)")
    
    mask_uo = ~np.isnan(res_old_uo)
    is_uo_match = np.allclose(res_old_uo[mask_uo], res_new_uo[mask_uo], atol=1e-7)
    print(f"Direct Match: {is_uo_match}")
    if is_uo_match:
        print(">> CONFIRMED: Results are identical.")

    print("\n--- SKEWNESS (N=100k, L=30) ---")
    length_skew = 30
    t0 = time.time()
    res_old_skew = skew_numba_legacy(close, length_skew)
    t_old_skew = time.time() - t0
    print(f"Old Skewness Time: {t_old_skew:.4f}s")
    
    t0 = time.time()
    res_new_skew = skew_numba(close, length_skew)
    t_new_skew = time.time() - t0
    print(f"New Skewness Time: {t_new_skew:.4f}s ({t_old_skew/t_new_skew:.1f}x faster)")
    
    mask_skew = ~np.isnan(res_old_skew)
    # Skewness involves cubic powers, so precision diffs are expected.
    # Also we updated the bias factor in the new implementation to be more standard,
    # whereas the legacy one had a suspicious factor.
    # Let's check if they match closely first.
    is_skew_match = np.allclose(res_old_skew[mask_skew], res_new_skew[mask_skew], atol=1e-5)
    print(f"Direct Match: {is_skew_match}")
    
    if not is_skew_match:
        diff = np.max(np.abs(res_old_skew[mask_skew] - res_new_skew[mask_skew]))
        print(f"Max Diff: {diff:.2e}")
        # Check if the difference is due to the bias factor change
        # Legacy factor: len*(len+1) / ((len-1)*(len-2)*(len-3))
        # New factor: len / ((len-1)*(len-2))
        legacy_factor = (length_skew * (length_skew + 1)) / ((length_skew - 1) * (length_skew - 2) * (length_skew - 3))
        new_factor = length_skew / ((length_skew - 1) * (length_skew - 2))
        ratio = legacy_factor / new_factor
        print(f"Expected Ratio mismatch due to factor correction: {ratio:.4f}")
        
        # Try to correct for the factor to see if the core logic is correct
        res_new_skew_adjusted = res_new_skew * (legacy_factor / new_factor)
        is_adjusted_match = np.allclose(res_old_skew[mask_skew], res_new_skew_adjusted[mask_skew], atol=1e-5)
        print(f"Match after factor adjustment: {is_adjusted_match}")
        
        if is_adjusted_match:
            print(">> CONFIRMED: Core logic runs correctly in O(N). Mismatch is intentional (fixed bias factor).")
        else:
             print(">> WARNING: Mismatch persists even after factor adjustment. Check implementation details.")
    else:
        print(">> CONFIRMED: Results are identical.")

if __name__ == "__main__":
    compare_implementations()
