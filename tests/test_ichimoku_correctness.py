
import numpy as np
from src.indicators.trend.trend_calculation_utils import calculate_ichimoku_lines, calculate_ichimoku_spans

def test_ichimoku_correctness():
    # Create sample data
    n = 200
    high = np.random.rand(n) + 10
    low = high - 1

    conversion_length = 9
    base_length = 26
    lagging_span2_length = 52
    displacement = 26

    conversion_line, base_line = calculate_ichimoku_lines(high, low, conversion_length, base_length)

    # Check conversion line
    first_valid_conversion_idx = -1
    for i in range(n):
        if not np.isnan(conversion_line[i]):
            first_valid_conversion_idx = i
            break

    print(f"Conversion Length: {conversion_length}")
    print(f"Base Length: {base_length}")
    print(f"First valid Conversion Line index: {first_valid_conversion_idx}")

    expected_conv_idx = conversion_length - 1
    assert first_valid_conversion_idx == expected_conv_idx, \
        f"Expected valid conversion line data from index {expected_conv_idx}, but got {first_valid_conversion_idx}"
    print("✅ PASS: Conversion line starts at correct index")

    # Check base line
    first_valid_base_idx = -1
    for i in range(n):
        if not np.isnan(base_line[i]):
            first_valid_base_idx = i
            break

    print(f"First valid Base Line index: {first_valid_base_idx}")
    expected_base_idx = base_length - 1
    assert first_valid_base_idx == expected_base_idx, \
        f"Expected valid base line data from index {expected_base_idx}, but got {first_valid_base_idx}"
    print("✅ PASS: Base line starts at correct index")

    # Check Spans
    span_a, span_b = calculate_ichimoku_spans(high, low, conversion_line, base_line, lagging_span2_length, displacement)

    first_valid_span_a = -1
    for i in range(n):
        if not np.isnan(span_a[i]):
            first_valid_span_a = i
            break

    # Span A depends on Conversion (9) and Base (26). So valid from index 25.
    # Displaced by 26. So valid from 25 + 26 = 51.
    expected_span_a_idx = (base_length - 1) + displacement

    print(f"First valid Span A index: {first_valid_span_a}")

    assert first_valid_span_a == expected_span_a_idx, \
        f"Expected valid Span A data from index {expected_span_a_idx}, but got {first_valid_span_a}"
    print("✅ PASS: Span A starts at correct index")

    # Span B depends on Lagging Span 2 (52). Valid from index 51.
    # Displaced by 26. Valid from 51 + 26 = 77.
    expected_span_b_idx = (lagging_span2_length - 1) + displacement

    first_valid_span_b = -1
    for i in range(n):
        if not np.isnan(span_b[i]):
            first_valid_span_b = i
            break

    print(f"First valid Span B index: {first_valid_span_b}")

    assert first_valid_span_b == expected_span_b_idx, \
        f"Expected valid Span B data from index {expected_span_b_idx}, but got {first_valid_span_b}"
    print("✅ PASS: Span B starts at correct index")

if __name__ == "__main__":
    test_ichimoku_correctness()
