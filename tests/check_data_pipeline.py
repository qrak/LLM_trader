
import numpy as np
import time
from numba import njit
import sys

@njit
def simple_sma(close, length):
    n = len(close)
    out = np.empty(n)
    for i in range(length, n):
        out[i] = np.sum(close[i-length:i]) / length
    return out

def check_pipeline():
    print("--- PIPELINE VERIFICATION ---")
    
    print("\n1. Dtype Performance Check")
    N = 100_000
    timestamps = np.arange(1000, 1000 + N, dtype=np.int64)
    closes = np.random.uniform(100, 200, N)
    
    raw_data = []
    print("Generating raw list data...")
    for i in range(1000): # Small batch for testing construction
         raw_data.append([int(timestamps[i]), 100.0, 105.0, 95.0, closes[i], 1000.0])
    
    # Check implicit array creation
    arr_implicit = np.array(raw_data)
    print(f"Implicit np.array(raw_data) dtype: {arr_implicit.dtype}")
    
    # Check explicit float64
    arr_explicit = np.array(raw_data, dtype=np.float64)
    print(f"Explicit np.array(..., dtype=np.float64) dtype: {arr_explicit.dtype}")
    
    # If implicit is object, Numba fails/slows.
    # If implicit is float64 (because ints promoted), it's fine.

    print("\n1b. None Handling Check")
    try:
        raw_none = [[160000000000, 100, 110, 90, 105, None]] # Vol is None
        arr_none_safe = np.array(raw_none, dtype=np.float64)
    except Exception as e:
        print(f"Error casting None to float64: {e}")
        print(">> WARNING: DataFetcher needs to handle None -> NaN conversion explicitly.")
    
    print("\n2. Column Index Check")
    # CCXT Standard: 0=TS, 1=Open, 2=High, 3=Low, 4=Close, 5=Vol
    mock_ohlcv = np.zeros((1, 6))
    mock_ohlcv[0, 3] = 999.0 # LOW
    mock_ohlcv[0, 4] = 888.0 # CLOSE
    
    # Emulate TechnicalCalculator logic
    extracted_close_at_3 = mock_ohlcv[:, 3]
    extracted_close_at_4 = mock_ohlcv[:, 4]
    
    print(f"Data at Index 3 (Low): {extracted_close_at_3[0]}")
    print(f"Data at Index 4 (Close): {extracted_close_at_4[0]}")
    
    if extracted_close_at_3[0] == 999.0:
        print(">> VERIFIED: Index 3 corresponds to LOW. Usage for 'close_prices' is Wrong.")
    
    raw_with_none = [[1, 2.0], [2, None]]
    arr_none = np.array(raw_with_none)
    print(f"Array with None dtype: {arr_none.dtype}") # Should be object
    
if __name__ == "__main__":
    check_pipeline()
