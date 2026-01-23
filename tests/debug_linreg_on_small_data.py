import numpy as np
from src.indicators.statistical.statistical_indicators import linreg_numba

def ref_linreg_r(data):
    n = len(data)
    x = np.arange(1, n + 1)
    y = data
    
    numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    denom = np.sqrt(np.sum((x - np.mean(x))**2)) * np.sqrt(np.sum((y - np.mean(y))**2))
    
    if denom == 0: return 0.0
    return numerator / denom

def debug():
    print("DEBUGGING LINREG O(N)")
    # Simple linear data + noise
    data = np.array([10.0, 11.0, 12.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0], dtype=np.float64)
    length = 5
    
    # Run O(N)
    res = linreg_numba(data, length, r=True)
    
    print("\nResults:")
    for i in range(length-1, len(data)):
        window = data[i-length+1 : i+1]
        target = ref_linreg_r(window)
        actual = res[i]
        print(f"Index {i}: Target={target:.6f}, Actual={actual:.6f}, Diff={actual-target:.6f}")
        
    # Check if actual is completely wrong (e.g. inverted sign or huge)
    
if __name__ == "__main__":
    debug()
