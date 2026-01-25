from src.logger.logger import Logger
from src.analyzer.data_fetcher import DataFetcher

class DataFetcherFactory:
    """Factory for creating DataFetcher instances following DI pattern."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        
    def create(self, exchange) -> DataFetcher:
        """Create a new DataFetcher for the given exchange."""
        return DataFetcher(exchange, self.logger)
