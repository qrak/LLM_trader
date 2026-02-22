import numpy as np
import time
from src.indicators.statistical.statistical_indicators import skew_numba, entropy_numba, linreg_numba

def title(t):
    print(f"\n{'='*50}\n{t}\n{'='*50}")

def ref_skewness_population(data):
    """Reference implementation of Population Skewness (bias=True in scipy)."""
    n = len(data)
    if n < 3: return np.nan  # noqa: E701
    mean = np.mean(data)
    diffs = data - mean
    var_pop = np.mean(diffs**2)
    std_pop = np.sqrt(var_pop)
    moment3 = np.mean(diffs**3)
    if std_pop == 0: return 0.0  # noqa: E701
    return moment3 / (std_pop**3)

def ref_skewness_sample(data):
    """Reference implementation of Sample Skewness (bias=False in scipy/pandas).
    Fisher-Pearson coefficient of skewness.
    """
    n = len(data)
    if n < 3: return np.nan  # noqa: E701
    # Use population mean for moments calculation, but adjust factor
    mean = np.mean(data)
    diffs = data - mean
    
    # Method 1 using moments:
    # m3 = sum(diffs^3) / n
    # m2 = sum(diffs^2) / n
    # g1 = m3 / m2^(1.5)
    # G1 = \frac{\sqrt{n(n-1)}}{n-2} g1
    
    m2 = np.mean(diffs**2)
    m3 = np.mean(diffs**3)
    g1 = m3 / (m2**1.5)
    
    factor = np.sqrt(n * (n - 1)) / (n - 2)
    return factor * g1

def ref_entropy(data, base=2.0):
    """Reference Price Entropy."""
    s = np.sum(data)
    if s <= 0: return 0.0  # noqa: E701
    probs = data / s
    # Avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs)) / np.log(base)

def ref_linreg_r(data):
    """Reference Linear Regression R^2."""
    n = len(data)
    y = data
    x = np.arange(1, n + 1)
    
    # Pearson Correlation Coefficient Squared
    # r = cov(x,y) / (std_x * std_y)
    numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    denom = np.sqrt(np.sum((x - np.mean(x))**2)) * np.sqrt(np.sum((y - np.mean(y))**2))
    
    if denom == 0: return 0.0  # noqa: E701
    r = numerator / denom
    return r

def verify_math():
    title("MATHEMATICAL CORRECTNESS VERIFICATION")
    
    np.random.seed(123)
    N_TOTAL = 1000
    L = 30
    
    # Generate data (Positive for entropy)
    data = np.random.uniform(10, 100, N_TOTAL)
    
    # 1. VERIFY SKEWNESS
    print("\n")
    
    # Run user implementation
    start_t = time.time()
    res_opt = skew_numba(data, L)
    print(f"Computed 1000 windows in {(time.time()-start_t)*1000:.4f}ms")
    
    # Verify last window
    last_window = data[-L:]
    
    target_pop = ref_skewness_population(last_window)
    target_sample = ref_skewness_sample(last_window)
    
    actual = res_opt[-1]
    
    print(f"Window Data Mean: {np.mean(last_window):.4f}")
    print(f"Target Population Skew: {target_pop:.8f}")
    print(f"Target Sample Skew:     {target_sample:.8f}")
    print(f"Actual Optimized Skew:  {actual:.8f}")
    
    # Determine which one it matches
    err_pop = abs(actual - target_pop)
    err_sample = abs(actual - target_sample)
    
    if err_sample < 1e-6:
        print(">> MATCH: Implementation matches SAMPLE Skewness (Unbiased, Pandas default) ✅")
    elif err_pop < 1e-6:
        print(">> MATCH: Implementation matches POPULATION Skewness (Biased, Scipy default) ✅")
    else:
        print(f">> FAIL: Does not match either standard definition.\n   ErrSample={err_sample:.8f}\n   ErrPop={err_pop:.8f} ❌")
        
        # Debugging factor
        # What factor is needed?
        print("Debug analysis:")
        # My code uses factor: L / ((L-1)(L-2)) * (sum_z3 / std_pop^3)?? No.
        # My code: skew_val = bias_factor * (sum_cubed_diff / std_dev**3)
        # where sum_cubed_diff = sum((x-mean)^3) = n * m3
        # std_dev = sqrt(m2) (population)
        # So Actual = bias_factor * (n * m3) / (m2^1.5)
        # Actual = bias_factor * n * g1_pop
        # My bias_factor = n / ((n-1)(n-2))
        # Thus Actual = n^2 / ((n-1)(n-2)) * g1_pop
        
        # Formula for Sample Skewness: G1 = sqrt(n*(n-1))/(n-2) * g1_pop
        
        expected_sample_from_pop = (np.sqrt(L*(L-1))/(L-2)) * target_pop
        print(f"Theoretical Sample Skew from Pop: {expected_sample_from_pop:.8f}")
        
        # Check if my code actually implemented: n^2 / ((n-1)(n-2))
        code_impl_factor = (L*L) / ((L-1)*(L-2))
        current_g1_pop_approx = actual / code_impl_factor
        print(f"Implied Pop Skew: {current_g1_pop_approx:.8f}")

    # 2. VERIFY ENTROPY
    print("\n")
    res_ent = entropy_numba(data, L)
    actual_ent = res_ent[-1]
    target_ent = ref_entropy(last_window) 
    
    print(f"Target Entropy: {target_ent:.8f}")
    print(f"Actual Entropy: {actual_ent:.8f}")
    if abs(actual_ent - target_ent) < 1e-6:
        print(">> MATCH: Entropy formula is mathematically correct ✅")
    else:
        print(f">> FAIL: Entropy mismatch. Diff={abs(actual_ent-target_ent):.8e} ❌")

    # 3. VERIFY LINREG
    print("\n")
    # r=True
    res_linreg = linreg_numba(data, L, r=True)
    actual_r = res_linreg[-1]
    target_r = ref_linreg_r(last_window)
    
    print(f"Target Pearson r: {target_r:.8f}")
    print(f"Actual LinReg r:  {actual_r:.8f}")
    if abs(actual_r - target_r) < 1e-6:
        print(">> MATCH: Pearson Correlation is correct ✅")
    else:
        print(f">> FAIL: LinReg mismatch. Diff={abs(actual_r-target_r):.8e} ❌")

if __name__ == "__main__":
    verify_math()
