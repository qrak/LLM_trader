import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.logger.logger import Logger
from src.rag.rag_engine import RagEngine
from src.rag.market_data_manager import MarketDataManager
from src.parsing.unified_parser import UnifiedParser
from src.utils.format_utils import FormatUtils
from src.platforms.defillama import DefiLlamaClient

class MockConfig:
    RAG_NEWS_LIMIT = 5
    RAG_UPDATE_INTERVAL_HOURS = 1

async def main():
    logger = Logger("VerifyRagEngine", logger_debug=False)
    format_utils = FormatUtils()
    unified_parser = UnifiedParser(logger, format_utils)
    
    # Setup dependencies
    defillama = DefiLlamaClient(logger)
    
    # Mock FileHandler
    class MockFileHandler:
        def __init__(self, *args, **kwargs): pass
        def load_json(self, *args, **kwargs): return {}
        def save_json(self, *args, **kwargs): pass

    mock_fh = MockFileHandler()

    # We need a proper MarketDataManager
    market_manager = MarketDataManager(
        logger=logger,
        file_handler=mock_fh,
        unified_parser=unified_parser,
        defillama_client=defillama,
        # We need coingecko_api for it to work properly
        coingecko_api=None 
    )
    
    # Mock RagEngine dependencies
    # We only care about get_market_overview
    rag = RagEngine(
        logger=logger,
        token_counter=None, # Will fail if token_counter needed in init
        config=MockConfig(),
        file_handler=object(), # mock
        news_manager=object(), # mock
        market_data_manager=market_manager,
        index_manager=object(), # mock
        category_fetcher=object(), # mock
        category_processor=object(), # mock
        ticker_manager=object(), # mock
        news_category_analyzer=object(), # mock
        context_builder=object(), # mock
    )
    
    # Inject coingecko api into rag to simulate previous state, 
    # but we want to verify it uses market_data_manager now
    # rag.coingecko_api = object() 

    print("Fetching Market Overview via RagEngine...")
    try:
        overview = await rag.get_market_overview()
        
        if overview and "fundamentals" in overview:
            print("\n[SUCCESS] Fundamentals found in RagEngine overview!")
            fund = overview["fundamentals"]
            print(f"  TVL: ${fund.get('macro', {}).get('total_tvl', 'N/A')}")
        else:
            print("\n[FAIL] 'fundamentals' key missing in RagEngine overview.")
            if overview:
                print("Keys found:", overview.keys())
            else:
                print("Overview is None")
                
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
