import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.platforms.defillama import DefiLlamaClient, DexVolumeData, FeesData, OptionsData, DeFiFundamentalsData
from src.logger.logger import Logger

class TestDefiLlamaFundamentals(unittest.IsolatedAsyncioTestCase):
    
    async def asyncSetUp(self):
        self.mock_logger = MagicMock(spec=Logger)
        # Use a temp cache directory to avoid interference with real cache
        self.client = DefiLlamaClient(
            logger=self.mock_logger,
            cache_dir='temp_test_cache'
        )

    async def test_get_dex_volumes(self):
        mock_response = {
            "total24h": 5000000.0,
            "change_1d": 5.5,
            "protocols": [
                {"name": "Uniswap", "total24h": 3000000.0},
                {"name": "Curve", "total24h": 2000000.0}
            ]
        }
        
        # Patch the session.get context manager chain
        mock_get_ctx = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_response
        mock_get_ctx.__aenter__.return_value = mock_resp
        
        with patch("aiohttp.ClientSession.get", return_value=mock_get_ctx):
            # We also need to patch _get_session since it creates the session
            with patch.object(self.client, '_get_session', return_value=MagicMock()) as mock_session_getter:
                mock_session_getter.return_value.get.return_value = mock_get_ctx
                
                dex_data = await self.client.get_dex_volumes()
                
                self.assertIsInstance(dex_data, DexVolumeData)
                self.assertEqual(dex_data.total_24h, 5000000.0)
                self.assertEqual(dex_data.change_1d, 5.5)
                self.assertEqual(len(dex_data.top_protocols), 2)
                self.assertEqual(dex_data.top_protocols[0]["name"], "Uniswap")

    async def test_get_fees_data(self):
        mock_response = {
            "total24h": 100000.0,
            "protocols": [
                {"name": "Lido", "total24h": 60000.0, "total24hRevenue": 6000.0},
                {"name": "Aave", "total24h": 40000.0, "total24hRevenue": 0.0}
            ]
        }
        
        mock_get_ctx = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_response
        mock_get_ctx.__aenter__.return_value = mock_resp
        
        with patch.object(self.client, '_get_session', return_value=MagicMock()) as mock_session_getter:
            mock_session_getter.return_value.get.return_value = mock_get_ctx
            
            fees_data = await self.client.get_fees_data()
            
            self.assertIsInstance(fees_data, FeesData)
            self.assertEqual(fees_data.total_24h_fees, 100000.0)
            self.assertEqual(fees_data.top_earners[0]["name"], "Lido")

    async def test_get_defi_fundamentals_aggregation(self):
        """Test that get_defi_fundamentals correctly aggregates data from all sources."""
        from src.platforms.defillama import MacroMarketData
        
        # Force cache invalidation by clearing last_update
        self.client.last_update = None
        
        # Mock individual methods to return controlled test data
        self.client.get_macro_overview = AsyncMock(return_value=MacroMarketData(
            stablecoins_market_cap=100.0,
            stablecoins_24h_change=0.0,
            total_tvl=200.0,
            top_chains=[]
        ))
        
        self.client.get_dex_volumes = AsyncMock(return_value=DexVolumeData(total_24h=500.0))
        self.client.get_fees_data = AsyncMock(return_value=FeesData(total_24h_fees=50.0))
        self.client.get_options_data = AsyncMock(return_value=OptionsData(notional_volume_24h=1000.0))
        
        result = await self.client.get_defi_fundamentals()
        
        self.assertIsInstance(result, DeFiFundamentalsData)
        self.assertEqual(result.macro.total_tvl, 200.0)
        self.assertEqual(result.dex_volumes.total_24h, 500.0)
        self.assertEqual(result.fees.total_24h_fees, 50.0)
        self.assertEqual(result.options.notional_volume_24h, 1000.0)

if __name__ == '__main__':
    unittest.main()
