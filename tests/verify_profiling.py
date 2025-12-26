
import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch
import asyncio

# Adjust path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the decorator
from src.utils.profiler import profile_performance

class TestProfiler(unittest.TestCase):
    
    @patch('src.utils.profiler.config')
    def test_profiler_decorator_sync(self, mock_config):
        # Setup mock config to have LOGGER_DEBUG = True
        mock_config.LOGGER_DEBUG = True
        
        # Create a mock logger
        mock_logger = MagicMock()
        
        class TestClass:
            def __init__(self):
                self.logger = mock_logger
                
            @profile_performance
            def heavy_method(self):
                time.sleep(0.01)
                return "done"
                
        obj = TestClass()
        result = obj.heavy_method()
        
        self.assertEqual(result, "done")
        
        # Verify logger was called
        mock_logger.debug.assert_called()
        args = mock_logger.debug.call_args[0][0]
        print(f"Log message caught: {args}")
        self.assertIn("Performance: TestClass.heavy_method took", args)

    @patch('src.utils.profiler.config')
    def test_profiler_decorator_async(self, mock_config):
        # Setup mock config to have LOGGER_DEBUG = True
        mock_config.LOGGER_DEBUG = True
        
        # Create a mock logger
        mock_logger = MagicMock()
        
        class TestClass:
            def __init__(self):
                self.logger = mock_logger
                
            @profile_performance
            async def heavy_method_async(self):
                await asyncio.sleep(0.01)
                return "done_async"
                
        obj = TestClass()
        
        # Run async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(obj.heavy_method_async())
        loop.close()
        
        self.assertEqual(result, "done_async")
        
        # Verify logger was called
        mock_logger.debug.assert_called()
        args = mock_logger.debug.call_args[0][0]
        print(f"Log message caught: {args}")
        self.assertIn("Performance: TestClass.heavy_method_async took", args)

    @patch('src.utils.profiler.config')
    def test_profiler_disabled(self, mock_config):
        # Setup mock config to have LOGGER_DEBUG = False
        mock_config.LOGGER_DEBUG = False
        
        mock_logger = MagicMock()
        
        class TestClass:
            def __init__(self):
                self.logger = mock_logger
                
            @profile_performance
            def method(self):
                return "done"
                
        obj = TestClass()
        result = obj.method()
        
        self.assertEqual(result, "done")
        mock_logger.debug.assert_not_called()

if __name__ == '__main__':
    unittest.main()
