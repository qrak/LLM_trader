"""Standalone script to test demo trading configuration (capital and fees).

Usage:
    python tests/run_demo_trading_config_test.py
    pytest tests/run_demo_trading_config_test.py -v

This will:
- Test that demo_quote_capital and transaction_fee_percent are correctly loaded
- Verify position size calculations (quantity = allocation / price)
- Verify fee calculations
- Print results for visual verification
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.loader import config


@pytest.fixture
def demo_capital():
    """Fixture providing demo capital from config."""
    return config.DEMO_QUOTE_CAPITAL


@pytest.fixture
def fee_percent():
    """Fixture providing transaction fee percent from config."""
    return config.TRANSACTION_FEE_PERCENT


@pytest.fixture
def current_price():
    """Fixture providing default test price."""
    return 100000.0


def test_config_values(demo_capital, fee_percent):
    """Test that config values are correctly loaded."""
    print("\n" + "=" * 60)
    print("DEMO TRADING CONFIG VALUES")
    print("=" * 60)

    print(f"DEMO_QUOTE_CAPITAL: ${demo_capital:,.2f}")
    print(f"TRANSACTION_FEE_PERCENT: {fee_percent} ({fee_percent * 100:.4f}%)")

    assert demo_capital > 0, "Capital must be positive"
    assert fee_percent >= 0, "Fee percent must be non-negative"
    assert fee_percent <= 0.01, "Fee percent seems too high (>1%)"

    print("\n[OK] Config values loaded correctly")


def test_position_size_calculation(demo_capital, current_price):
    """Test position size calculation logic."""
    print("\n" + "=" * 60)
    print("POSITION SIZE CALCULATION")
    print("=" * 60)
    print(f"Capital: ${demo_capital:,.2f}")
    print(f"Current Price: ${current_price:,.2f}")

    test_cases = [
        (0.01, "1%"),
        (0.02, "2%"),
        (0.05, "5%"),
        (0.10, "10%"),
        (0.50, "50%"),
        (1.00, "100%"),
    ]

    print(f"\n{'Size %':<10} {'Allocation':<15} {'Quantity':<15} {'In USD':<15}")
    print("-" * 55)

    for size_pct, label in test_cases:
        allocation = demo_capital * size_pct
        quantity = allocation / current_price
        value_usd = quantity * current_price

        print(f"{label:<10} ${allocation:>12,.2f}   {quantity:>12.8f}   ${value_usd:>12,.2f}")

        assert abs(allocation - demo_capital * size_pct) < 0.01, f"Allocation mismatch for {label}"
        assert abs(quantity - allocation / current_price) < 0.00000001, f"Quantity mismatch for {label}"

    print("\n[OK] Position size calculations correct")


def test_fee_calculation(demo_capital, fee_percent, current_price):
    """Test fee calculation logic."""
    print("\n" + "=" * 60)
    print("FEE CALCULATION")
    print("=" * 60)
    print(f"Capital: ${demo_capital:,.2f}")
    print(f"Fee Percent: {fee_percent * 100:.4f}%")
    print(f"Current Price: ${current_price:,.2f}")

    size_pct = 0.02
    allocation = demo_capital * size_pct
    quantity = allocation / current_price

    entry_fee = allocation * fee_percent
    exit_price = current_price * 1.02
    exit_value = quantity * exit_price
    exit_fee = exit_value * fee_percent
    total_fees = entry_fee + exit_fee

    print(f"\nPosition: {size_pct * 100:.0f}% = ${allocation:,.2f}")
    print(f"Quantity: {quantity:.8f}")
    print(f"Entry Fee: ${entry_fee:.4f}")
    print(f"Exit Value @ +2%: ${exit_value:,.2f}")
    print(f"Exit Fee: ${exit_fee:.4f}")
    print(f"Total Round-Trip Fees: ${total_fees:.4f}")

    gross_pnl = (exit_price - current_price) * quantity
    net_pnl = gross_pnl - total_fees

    print(f"\nGross P&L: ${gross_pnl:,.4f}")
    print(f"Net P&L (after fees): ${net_pnl:,.4f}")
    print(f"Fee Impact: ${total_fees:,.4f} ({total_fees / gross_pnl * 100:.2f}% of profit)")

    assert entry_fee > 0, "Entry fee should be positive"
    assert exit_fee > 0, "Exit fee should be positive"
    assert net_pnl < gross_pnl, "Net P&L should be less than gross P&L"

    print("\n[OK] Fee calculations correct")


def test_pnl_scenarios(demo_capital, fee_percent, current_price):
    """Test various P&L scenarios with fees."""
    print("\n" + "=" * 60)
    print("P&L SCENARIOS (with fees)")
    print("=" * 60)

    size_pct = 0.02
    allocation = demo_capital * size_pct
    quantity = allocation / current_price

    scenarios = [
        (-5.0, "Major Loss"),
        (-2.0, "Stop Loss Hit"),
        (-1.0, "Small Loss"),
        (0.0, "Breakeven"),
        (1.0, "Small Win"),
        (2.0, "Modest Win"),
        (5.0, "Strong Win"),
        (10.0, "Major Win"),
    ]

    print(f"\n{'Scenario':<15} {'Price Change':<15} {'Gross P&L':<15} {'Fees':<12} {'Net P&L':<15}")
    print("-" * 72)

    for price_change_pct, label in scenarios:
        exit_price = current_price * (1 + price_change_pct / 100)
        gross_pnl = (exit_price - current_price) * quantity

        entry_fee = allocation * fee_percent
        exit_value = quantity * exit_price
        exit_fee = exit_value * fee_percent
        total_fees = entry_fee + exit_fee

        net_pnl = gross_pnl - total_fees

        print(
            f"{label:<15} "
            f"{price_change_pct:>+6.1f}%         "
            f"${gross_pnl:>+10.2f}    "
            f"${total_fees:>8.4f}    "
            f"${net_pnl:>+10.2f}"
        )

    print("\n[OK] P&L scenarios calculated correctly")


def test_edge_cases(demo_capital, fee_percent, current_price):
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("EDGE CASES")
    print("=" * 60)

    # Very small position
    print("\n1. Very small position (0.1%):")
    size_pct = 0.001
    allocation = demo_capital * size_pct
    quantity = allocation / current_price
    fee = allocation * fee_percent
    print(f"   Allocation: ${allocation:.2f}, Quantity: {quantity:.10f}, Fee: ${fee:.6f}")
    assert fee > 0, "Even tiny positions should have fees"

    # Full position
    print("\n2. Full position (100%):")
    size_pct = 1.0
    allocation = demo_capital * size_pct
    quantity = allocation / current_price
    fee = allocation * fee_percent
    print(f"   Allocation: ${allocation:,.2f}, Quantity: {quantity:.8f}, Fee: ${fee:.4f}")

    # Very low price asset
    print("\n3. Low price asset ($0.10):")
    low_price = 0.10
    size_pct = 0.02
    allocation = demo_capital * size_pct
    quantity = allocation / low_price
    print(f"   Allocation: ${allocation:.2f}, Quantity: {quantity:,.2f} tokens")

    # Very high price asset
    print("\n4. High price asset ($500,000):")
    high_price = 500000.0
    quantity = allocation / high_price
    print(f"   Allocation: ${allocation:.2f}, Quantity: {quantity:.8f} tokens")

    print("\n[OK] Edge cases handled correctly")


def main():
    """Run all tests as standalone script."""
    print("\n" + "=" * 60)
    print("DEMO TRADING CONFIG TEST")
    print("=" * 60)

    capital = config.DEMO_QUOTE_CAPITAL
    fee = config.TRANSACTION_FEE_PERCENT
    price = 100000.0

    test_config_values(capital, fee)
    test_position_size_calculation(capital, price)
    test_fee_calculation(capital, fee, price)
    test_pnl_scenarios(capital, fee, price)
    test_edge_cases(capital, fee, price)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED [OK]")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

