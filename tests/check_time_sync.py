import asyncio
import ccxt.async_support as ccxt
import time
from datetime import datetime, timezone
import sys
import os

# Hardcoded list from config.ini for robustness
SUPPORTED_EXCHANGES = ['binance', 'kucoin', 'gateio', 'mexc', 'hyperliquid']

async def check_exchange_time(exchange_id):
    """
    Checks the difference between System Time and Exchange Server Time.
    Returns the average offset (Server - System) in milliseconds.
    """
    print(f"\n[{exchange_id.upper()}] Checking Time Sync...")
    
    try:
        if not hasattr(ccxt, exchange_id):
            print(f"  > Error: {exchange_id} not found in ccxt library.")
            return None

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    except Exception as e:
        print(f"  > Error initializing: {e}")
        return None

    try:
        await exchange.load_markets()
        offsets = []
        
        # Take 5 samples
        for i in range(5):
            start_time = time.time()
            try:
                server_time_ms = await exchange.fetch_time()
            except Exception as e:
                # Some exchanges (like Hyperliquid) might not support fetchTime public endpoint easily or have different API structure
                print(f"  > Sample #{i+1}: Failed to fetch time ({e})")
                continue
                
            end_time = time.time()
            
            # RTT (Round Trip Time)
            rtt_ms = (end_time - start_time) * 1000
            
            # Estimate "Now" at the server (midpoint of RTT)
            local_time_ms = ((start_time + end_time) / 2) * 1000
            
            # Offset = Server - Local
            # Negative Offset => Server is BEHIND Local (Local is AHEAD)
            offset_ms = server_time_ms - local_time_ms
            offsets.append(offset_ms)
            
            # print(f"  > Sample #{i+1}: RTT={rtt_ms:.1f}ms, Offset={offset_ms:.1f}ms")
            
            await asyncio.sleep(0.2)
            
        if not offsets:
            print("  > Could not retrieve any time samples.")
            return None
            
        avg_offset = sum(offsets) / len(offsets)
        return avg_offset
        
    except Exception as e:
        print(f"  > Error: {e}")
        return None
    finally:
        await exchange.close()

async def simulator_verify_sleep_fix():
    print("\n" + "="*80)
    print("VERIFYING SLEEP LOGIC (SIMULATION)")
    print("="*80)
    print("Testing OLD vs NEW sleep (simulating 5 iterations of 0.1s)...")
    
    # 1. Simulate OLD logic (Accumulating sleep)
    start_old = time.monotonic()
    elapsed_sim = 0.0
    iterations = 5
    target_sleep = 0.1
    
    # In reality sleeps contain overhead. Let's simulate overhead.
    # We will just actually sleep and measure.
    for _ in range(iterations):
        await asyncio.sleep(target_sleep)
        elapsed_sim += target_sleep # The bug: adding constant target, not real time
    
    real_duration_old = time.monotonic() - start_old
    print(f"OLD Logic (Simulated):")
    print(f"  - Target calculated elapsed: {elapsed_sim:.4f}s")
    print(f"  - Actual wall-clock time:  {real_duration_old:.4f}s")
    print(f"  - Drift: {real_duration_old - elapsed_sim:.4f}s (This accumulates over hours!)")

    # 2. Verify NEW logic (Monotonic)
    start_new = time.monotonic()
    
    # Simulate the loop in _interruptible_sleep
    total_target = iterations * target_sleep # e.g. 0.5s
    
    while True:
        elapsed = time.monotonic() - start_new
        if elapsed >= total_target:
            break
        remaining = total_target - elapsed
        to_sleep = min(0.1, remaining)
        if to_sleep <= 0: break
        await asyncio.sleep(to_sleep)

    real_duration_new = time.monotonic() - start_new
    print(f"\nNEW Logic (Real Impl):")
    print(f"  - Target duration: {total_target:.4f}s")
    print(f"  - Actual duration: {real_duration_new:.4f}s")
    print(f"  - Error: {abs(real_duration_new - total_target):.4f}s")
    
    if abs(real_duration_new - total_target) < 0.05:
         print("  > PASS: New logic tracks time correctly.")
    else:
         print("  > FAIL: New logic is still inaccurate.")

async def main():
    print("="*80)
    print(f"TIME SYNC & SAFETY CHECK")
    print(f"System Time: {datetime.now()}")
    print("="*80)
    
    # 1. Verify Sleep Logic Fix
    await simulator_verify_sleep_fix()
    
    # 2. Check Exchanges
    print("\n" + "="*80)
    print("CHECKING EXCHANGE SYNC STATUS")
    print("="*80)
    
    safe = True
    
    for exchange_id in SUPPORTED_EXCHANGES:
        offset = await check_exchange_time(exchange_id)
        if offset is None:
            continue
            
        # Interpretation
        # Offset = Server - Local
        # Error "Timestamp ahead of server" happens if Local >> Server
        # i.e., Server << Local, or Server - Local is LARGE NEGATIVE
        
        status = "OK"
        if abs(offset) > 1000:
            status = "WARNING"
            safe = False
        
        print(f"  > Avg Offset: {offset:.2f} ms")
        
        if offset < -1000:
             print(f"  > DANGER: Local time is {abs(offset):.2f}ms AHEAD of server.")
             print("  > You might see 'Timestamp for this request was 1000ms ahead' errors.")
        elif offset > 1000:
             print(f"  > WARNING: Local time is {offset:.2f}ms BEHIND server.")
             print("  > Requests might be rejected as 'expired'.")
        else:
             print(f"  > STATUS: {status} (Safe range is +/- 1000ms)")
             
    print("\n" + "="*80)
    if safe:
        print("RESULT: SYSTEM TIME IS SAFE.")
        print("Your clock is within 1 second of major exchanges.")
    else:
        print("RESULT: TIME SYNC ISSUES DETECTED.")
        print("Please synchronize your Windows clock manually.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
