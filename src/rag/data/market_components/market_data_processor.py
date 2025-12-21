"""
Market Data Processing Utilities
Handles data normalization and processing operations.
"""
from typing import Dict, Optional, List

from src.logger.logger import Logger
from src.parsing.unified_parser import UnifiedParser


class MarketDataProcessor:
    """Handles processing and normalization of market data."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.parser = UnifiedParser(logger)
    
    def normalize_timestamp(self, timestamp_field) -> float:
        """Convert various timestamp formats to a float timestamp."""
        return self.parser.parse_timestamp(timestamp_field)
    
    def extract_top_coins(self, coingecko_data: Optional[Dict]) -> List[str]:
        """Extract top cryptocurrency symbols from CoinGecko data."""
        try:
            if not coingecko_data:
                self.logger.warning("No CoinGecko data provided")
                return []
            
            # Check for direct dominance data (current format)
            if 'dominance' in coingecko_data:
                dominance_data = coingecko_data['dominance']
            elif 'data' in coingecko_data and 'dominance' in coingecko_data['data']:
                dominance_data = coingecko_data['data']['dominance']
            else:
                self.logger.warning("No dominance data found in CoinGecko response")
                return []
            
            # Sort by dominance percentage and extract top symbols
            sorted_coins = sorted(
                dominance_data.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Get top 10 coins, excluding stablecoins from top positions
            stablecoins = {'usdt', 'usdc', 'busd', 'dai', 'tusd', 'usd', 'steth'}
            top_coins = []
            
            for symbol, dominance in sorted_coins:
                symbol_upper = symbol.upper()
                if symbol.lower() not in stablecoins:
                    top_coins.append(symbol_upper)
                if len(top_coins) >= 10:
                    break
            
            self.logger.debug(f"Extracted {len(top_coins)} top coins: {top_coins}")
            return top_coins
            
        except Exception as e:
            self.logger.error(f"Error extracting top coins: {e}")
            return []
    
    def process_coin_data(self, values: Dict) -> Optional[Dict]:
        """Process individual coin data from market sources."""
        try:
            processed_coin = {}
            
            # Process basic information
            if 'symbol' in values:
                processed_coin['symbol'] = values['symbol'].upper()
            
            # Process price information
            for price_key in ['close', 'last', 'price']:
                if price_key in values and values[price_key] is not None:
                    processed_coin['price'] = float(values[price_key])
                    break
            
            # Process volume information
            for volume_key in ['volume', 'baseVolume', 'quoteVolume']:
                if volume_key in values and values[volume_key] is not None:
                    processed_coin['volume'] = float(values[volume_key])
                    break
            
            # Process percentage change
            for change_key in ['percentage', 'change', 'percentage_change']:
                if change_key in values and values[change_key] is not None:
                    processed_coin['change_24h'] = float(values[change_key])
                    break
            
            return processed_coin if processed_coin else None
            
        except Exception as e:
            self.logger.error(f"Error processing coin data: {e}")
            return None
