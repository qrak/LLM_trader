"""
Market Overview Builder
Handles building and structuring market overview data.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from src.logger.logger import Logger


class MarketOverviewBuilder:
    """Handles building structured market overview data."""
    
    def __init__(self, logger: Logger, processor):
        self.logger = logger
        self.processor = processor
    
    def build_overview_structure(self, price_data: Optional[Dict], coingecko_data: Optional[Dict]) -> Dict[str, Any]:
        """Build the complete market overview structure."""
        overview = {
            "timestamp": datetime.now().isoformat(), 
            "summary": "CRYPTO MARKET OVERVIEW"
        }
        
        try:
            # Add CoinGecko global data if available
            if coingecko_data:
                # Handle both direct global data and wrapped data
                if 'data' in coingecko_data:
                    overview['global_data'] = coingecko_data['data']
                elif any(key in coingecko_data for key in ['market_cap', 'volume', 'dominance', 'stats']):
                    # Direct global data format
                    overview['global_data'] = coingecko_data
                else:
                    self.logger.warning(f"Unexpected CoinGecko data format: {list(coingecko_data.keys())}")
            
            # Add price data if available
            if price_data:
                overview['coin_data'] = {}
                for symbol, values in price_data.items():
                    processed_coin = self.processor.process_coin_data(values)
                    if processed_coin:
                        overview['coin_data'][symbol] = processed_coin
            
            return self._finalize_overview(overview)
            
        except Exception as e:
            self.logger.error(f"Error building overview structure: {e}")
            return overview
    
    def build_overview(self, coingecko_data: Optional[Dict], price_data: Optional[Dict], top_coins: Optional[list] = None) -> Dict[str, Any]:
        """Build market overview from fetched data - main entry point."""
        try:
            return self.build_overview_structure(price_data, coingecko_data)
        except Exception as e:
            self.logger.error(f"Error building market overview: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": "CRYPTO MARKET OVERVIEW - Error occurred",
                "published_on": datetime.now().timestamp()
            }
    
    def _finalize_overview(self, overview: Dict) -> Dict[str, Any]:
        """Finalize and validate the overview structure."""
        try:
            # Add metadata
            overview['published_on'] = datetime.now().timestamp()
            overview['data_sources'] = []
            
            # Track data sources
            if 'global_data' in overview:
                overview['data_sources'].append('coingecko_global')
            if 'coin_data' in overview:
                overview['data_sources'].append('price_data')
            
            # Add summary statistics
            if 'coin_data' in overview:
                coin_count = len(overview['coin_data'])
                overview['summary'] += f" - {coin_count} coins tracked"
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error finalizing overview: {e}")
            return overview
