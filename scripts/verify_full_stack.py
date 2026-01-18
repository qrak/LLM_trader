import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.logger.logger import Logger
from src.platforms.defillama import DefiLlamaClient
from src.rag.market_data_manager import MarketDataManager
from src.parsing.unified_parser import UnifiedParser
from src.utils.format_utils import FormatUtils
from src.rag.file_handler import RagFileHandler

# Mocks
class MockConfig:
    DATA_DIR = "data"

class MockFileHandler:
    def __init__(self, *args, **kwargs): pass

async def main():
    logger = Logger("VerifyStack", logger_debug=True)
    format_utils = FormatUtils()
    unified_parser = UnifiedParser(logger, format_utils)
    
    # 1. Initialize Client
    print("Initializing DefiLlamaClient...")
    defillama = DefiLlamaClient(logger)
    
    # 2. Initialize Manager (injecting client, mocking others)
    print("Initializing MarketDataManager...")
    # Mocking file handler since we don't need to write to disk for this test
    # But MarketDataManager expects a RagFileHandler, so we'll mock it or pass None if allowed (it's not typed to allow None in init but we can try)
    # Actually, let's just make a dummy object
    mock_fh = MockFileHandler()
    
    manager = MarketDataManager(
        logger=logger, 
        file_handler=mock_fh,
        unified_parser=unified_parser,
        defillama_client=defillama,
        # Others can be None as they are optional
    )
    
    # 3. Fetch Overview
    print("Fetching Market Overview...")
    overview = await manager.fetch_market_overview()
    
    # 4. Verification
    
    # Check Macro (Backward Compatibility)
    if overview and "macro" in overview:
        macro = overview["macro"]
        print("\n[SUCCESS] Macro Data Fetched (Legacy Key):")
        print(f"  Stablecoin MC: ${macro.get('stablecoins_market_cap', 0):,.2f}")
        print(f"  Total TVL: ${macro.get('total_tvl', 0):,.2f}")
    else:
        print("\n[FAIL] Legacy 'macro' key missing.")

    # Check Fundamentals (New Aggregated Data)
    if overview and "fundamentals" in overview:
        fund = overview["fundamentals"]
        print("\n[SUCCESS] DeFi Fundamentals Fetched (New Key):")
        
        # Check DEX Volumes
        dex = fund.get("dex_volumes")
        if dex:
            print(f"  DEX Volume 24h: ${dex.get('total_24h', 0):,.2f}")
            top_dex = dex.get("top_protocols", [])
            if top_dex:
                print(f"  Top DEX: {top_dex[0].get('name')} (${top_dex[0].get('total24h', 0):,.2f})")
        else:
            print("  [WARN] DEX Volumes missing")

        # Check Fees
        fees = fund.get("fees")
        if fees:
            print(f"  Fees 24h: ${fees.get('total_24h_fees', 0):,.2f}")
            revenue = fees.get('total_24h_revenue', 0)
            if revenue > 0:
                print(f"  Revenue 24h: ${revenue:,.2f}")
        else:
            print("  [WARN] Fees/Revenue missing")

        # Check Options
        opts = fund.get("options")
        if opts:
            notional = opts.get('notional_volume_24h', 0)
            if notional > 0:
                print(f"  Options Notional 24h: ${notional:,.2f}")
        else:
            print("  [WARN] Options data missing")
            
    else:
        print("\n[FAIL] 'fundamentals' key missing from overview!")
        print(overview.keys() if overview else "Overview is None")

    await defillama.close()

if __name__ == "__main__":
    asyncio.run(main())
