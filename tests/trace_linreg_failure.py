import numpy as np
from src.indicators.statistical.statistical_indicators import linreg_numba

def ref_pearson(data):
    n = len(data)
    x = np.arange(1, n + 1)
    y = data
    numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    denom = np.sqrt(np.sum((x - np.mean(x))**2)) * np.sqrt(np.sum((y - np.mean(y))**2))
    return numerator / denom

def trace_check():
    np.random.seed(123)
    N_TOTAL = 1000
    L = 30
    data = np.random.uniform(10, 100, N_TOTAL)
    
    # Last window
    last_window = data[-L:]
    
    # 1. Pearson
    pearson = ref_pearson(last_window)
    print(f"Target Pearson R: {pearson}")
    
    # 2. Formula with Reference Sums
    offset = data[0]
    y = last_window - offset
    x = np.arange(1, L + 1)
    
    y_sum = np.sum(y)
    xy_sum = np.sum(x * y)
    y2_sum = np.sum(y**2)
    x_sum = np.sum(x)
    x2_sum = np.sum(x**2)
    
    length = L
    divisor = length * x2_sum - x_sum * x_sum
    rn = length * xy_sum - x_sum * y_sum
    term1 = length * y2_sum - y_sum * y_sum
    rd = np.sqrt(divisor * max(0.0, term1))
    
    calc_r = rn / rd
    print(f"Formula R (Ref Sums): {calc_r}")
    
    # 3. Numba Result
    res = linreg_numba(data, L, r=True)
    actual_r = res[-1]
    print(f"Numba Result: {actual_r}")
    
    if abs(pearson - actual_r) > 1e-6:
        print(">> MISMATCH! Pearson != Numba")
        if abs(pearson - calc_r) < 1e-6:
             print(">> Pearson == Formula. So Numba is wrong.")
        else:
             print(">> Pearson != Formula. Mathematical impossibility?")

if __name__ == "__main__":
    trace_check()
