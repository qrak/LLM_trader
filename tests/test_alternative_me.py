import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from src.platforms.alternative_me import AlternativeMeAPI
from src.logger.logger import Logger

class TestAlternativeMeAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logger = MagicMock(spec=Logger)
        # Use a real directory or mock os.makedirs
        with patch('os.makedirs'):
            self.api = AlternativeMeAPI(logger=self.logger, data_dir="/tmp/test_data_altme")
            # We don't want to actually load cache file if not needed
            with patch('os.path.exists', return_value=False):
                pass

    async def test_session_reuse(self):
        # Mock ClientSession
        with patch('aiohttp.ClientSession') as MockSession:
            mock_session_instance = MagicMock()
            mock_session_instance.closed = False
            MockSession.return_value = mock_session_instance

            # Setup get return value
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                "data": [{
                    "value": "50",
                    "value_classification": "Neutral",
                    "timestamp": "1678886400"
                }]
            }

            # The context manager returned by get()
            mock_get_ctx = AsyncMock()
            mock_get_ctx.__aenter__.return_value = mock_resp

            # session.get() returns the context manager
            mock_session_instance.get = MagicMock(return_value=mock_get_ctx)

            # Initialize
            await self.api.initialize()

            # Verify session created once
            MockSession.assert_called_once()
            self.assertEqual(self.api.session, mock_session_instance)

            # Call get_fear_greed_index
            await self.api.get_fear_greed_index(force_refresh=True)

            # Verify session used
            mock_session_instance.get.assert_called_once()

            # Call again
            await self.api.get_fear_greed_index(force_refresh=True)

            # Verify session reused (get called twice, constructor still once)
            self.assertEqual(mock_session_instance.get.call_count, 2)
            MockSession.assert_called_once()

    async def test_session_recreation_if_closed(self):
        mock_session1 = MagicMock()
        mock_session1.closed = True  # It is closed

        mock_session2 = MagicMock()
        mock_session2.closed = False # It is open

        # Setup responses for session2
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {"data": [{"value": "50", "value_classification": "Neutral", "timestamp": "1678886400"}]}
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_resp

        mock_session2.get = MagicMock(return_value=mock_get_ctx)

        with patch('aiohttp.ClientSession', side_effect=[mock_session1, mock_session2]) as MockSession:
            # Initialize (creates session 1)
            await self.api.initialize()

            # Verify session 1 is set
            self.assertEqual(self.api.session, mock_session1)

            # Call get_fear_greed_index -> should detect closed session and create session 2
            await self.api.get_fear_greed_index(force_refresh=True)

            # Verify constructor called twice
            self.assertEqual(MockSession.call_count, 2)
            # Verify session is now session 2
            self.assertEqual(self.api.session, mock_session2)
            # Verify get called on session 2
            mock_session2.get.assert_called_once()

    async def test_close(self):
         with patch('aiohttp.ClientSession') as MockSession:
            mock_session_instance = MagicMock()
            mock_session_instance.closed = False
            mock_session_instance.close = AsyncMock()

            MockSession.return_value = mock_session_instance

            await self.api.initialize()

            await self.api.close()

            mock_session_instance.close.assert_called_once()

if __name__ == "__main__":
    unittest.main()
