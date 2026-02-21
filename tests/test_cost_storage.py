
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
import time
import tempfile
from src.utils.token_counter import CostStorage, ProviderCostStats

class TestCostStorage(unittest.TestCase):
    def setUp(self):
        self.fd, self.test_file = tempfile.mkstemp(suffix=".json")
        os.close(self.fd)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_initialization(self):
        storage = CostStorage(file_path=self.test_file)
        self.assertEqual(storage.file_path, self.test_file)
        self.assertEqual(len(storage._providers), 3) # openrouter, google, lmstudio
        # Verify lock and dirty flag
        self.assertFalse(storage._dirty)
        self.assertEqual(storage._last_save_time, 0.0)

    def test_record_usage_updates_memory(self):
        storage = CostStorage(file_path=self.test_file)

        # Ensure it doesn't save (and reset dirty) by setting last save time to far future
        storage._last_save_time = time.time() + 10000

        storage.record_usage("openrouter", 100, 50, 0.01)

        stats = storage.get_provider_costs("openrouter")
        self.assertEqual(stats.total_input_tokens, 100)
        self.assertEqual(stats.total_output_tokens, 50)
        self.assertAlmostEqual(stats.total_cost, 0.01)
        self.assertTrue(storage._dirty)

    @patch('src.utils.token_counter.open', new_callable=mock_open)
    @patch('src.utils.token_counter.json.dump')
    @patch('src.utils.token_counter.tempfile.NamedTemporaryFile')
    def test_record_usage_does_not_save_immediately(self, mock_temp_file, mock_json_dump, mock_file):
        storage = CostStorage(file_path=self.test_file)

        # Mock time to be 1000.0
        with patch('src.utils.token_counter.time.time', return_value=1000.0):
            storage._last_save_time = 1000.0 # Just saved

            # Record usage
            storage.record_usage("openrouter", 10, 10, 0.001)

            # Should not save because not enough time passed (0 seconds passed)
            mock_json_dump.assert_not_called()
            # self._dirty should be True
            self.assertTrue(storage._dirty)

    @patch('src.utils.token_counter.open', new_callable=mock_open)
    @patch('src.utils.token_counter.json.dump')
    @patch('src.utils.token_counter.tempfile.NamedTemporaryFile')
    @patch('src.utils.token_counter.os.replace')
    @patch('src.utils.token_counter.os.fsync')
    def test_record_usage_saves_after_interval(self, mock_fsync, mock_replace, mock_temp_file_cls, mock_json_dump, mock_file):
        storage = CostStorage(file_path=self.test_file)

        # Setup mock for temp file
        mock_temp = MagicMock()
        mock_temp_file_cls.return_value.__enter__.return_value = mock_temp
        mock_temp.name = "temp_file_path"

        # Mock time sequence
        # 1. record_usage calls time.time() -> 1006.0
        # 2. save calls time.time() -> 1007.0 (update _last_save_time)
        with patch('src.utils.token_counter.time.time') as mock_time:
            mock_time.side_effect = [1006.0, 1007.0]

            storage._last_save_time = 1000.0

            # record_usage calls time.time() -> returns 1006.0. 1006-1000 = 6 > 5.
            storage.record_usage("openrouter", 10, 10, 0.001)

            # Should save
            mock_json_dump.assert_called_once()

            # _dirty should be False (reset by save)
            self.assertFalse(storage._dirty)
            self.assertEqual(storage._last_save_time, 1007.0)

    @patch('src.utils.token_counter.tempfile.NamedTemporaryFile')
    @patch('src.utils.token_counter.os.replace')
    @patch('src.utils.token_counter.os.fsync')
    def test_save_is_atomic(self, mock_fsync, mock_replace, mock_temp_file_cls):
        storage = CostStorage(file_path=self.test_file)
        storage._dirty = True

        # Setup mock for temp file
        mock_temp = MagicMock()
        # Mock the context manager __enter__ return value
        mock_temp_file_cls.return_value.__enter__.return_value = mock_temp
        # Mock the name attribute of the file object returned by __enter__
        mock_temp.name = "temp_file_path"

        # Need to mock os.path.exists and os.remove for cleanup block, but they are not called if no exception.

        with patch('src.utils.token_counter.json.dump'):
             storage.save()

             mock_temp_file_cls.assert_called_once()
             mock_fsync.assert_called()
             mock_replace.assert_called_with("temp_file_path", self.test_file)

    def test_load_corrupted_file_resets_defaults(self):
        # Write corrupted JSON to the test file
        with open(self.test_file, 'w') as f:
            f.write("{ invalid json")

        # Initialize storage
        storage = CostStorage(file_path=self.test_file)

        # Verify it reset to defaults
        self.assertEqual(storage.get_total_openrouter_cost(), 0.0)
        self.assertIsNone(storage._last_reset)

    @patch('src.utils.token_counter.CostStorage.save')
    def test_reset_clears_costs(self, mock_save):
        storage = CostStorage(file_path=self.test_file)
        # Set last save time to future to prevent auto-save during record_usage
        storage._last_save_time = time.time() + 10000

        storage.record_usage("openrouter", 100, 100, 1.0)

        storage.reset()

        self.assertEqual(storage.get_total_openrouter_cost(), 0.0)
        self.assertIsNotNone(storage._last_reset)
        mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()
