import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING

import ccxt.async_support as ccxt
import aiohttp

from src.logger.logger import Logger
from src.utils.decorators import retry_async

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol


class ExchangeManager:
    def __init__(self, logger: Logger, config: "ConfigProtocol"):
        """Initialize ExchangeManager with logger and self.config.
        
        Args:
            logger: Logger instance
            config: ConfigProtocol instance for exchange settings
            
        Raises:
            ValueError: If config is None
        """
        if config is None:
            raise ValueError("config is a required parameter and cannot be None")
        
        self.logger = logger
        self.config = config
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.symbols_by_exchange: Dict[str, Set[str]] = {}
        self.exchange_last_loaded: Dict[str, datetime] = {}
        self._update_task: Optional[asyncio.Task] = None
        self._shutdown_in_progress = False
        self.exchange_config: Dict[str, Any] = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        self.exchange_names = self.config.SUPPORTED_EXCHANGES
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize the session for exchanges - no longer loads all exchanges upfront"""
        self.logger.info("Initializing ExchangeManager with lazy loading")
        # Create a single session for all exchanges to share
        self.session = aiohttp.ClientSession()
        self.exchange_config['session'] = self.session
        
        # Start periodic update task, but it will only update already loaded exchanges
        self._update_task = asyncio.create_task(self._periodic_update())
        self._update_task.add_done_callback(self._handle_update_task_done)
    
    def _handle_update_task_done(self, task):
        if task.exception() and not self._shutdown_in_progress:
            self.logger.error(f"Periodic update task failed: {task.exception()}")
    
    async def shutdown(self) -> None:
        """Close all exchanges and stop periodic updates"""
        self._shutdown_in_progress = True
        
        if self._update_task:
            self.logger.info("Cancelling periodic update task")
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.exception(f"Error during update task cancellation: {e}")
            finally:
                self._update_task = None
        
        self.logger.info("Closing exchange connections")
        for exchange_id, exchange in list(self.exchanges.items()):
            try:
                await exchange.close()
                self.logger.debug(f"Closed {exchange_id} connection")
            except Exception as e:
                self.logger.error(f"Error closing {exchange_id} connection: {e}")
        
        # Close the shared session last, after all exchanges have been closed
        if self.session:
            try:
                self.logger.debug("Closing shared aiohttp session")
                await self.session.close()
            except Exception as e:
                self.logger.error(f"Error closing shared aiohttp session: {e}")
            finally:
                self.session = None
        
        self.exchanges.clear()
        self.symbols_by_exchange.clear()
        self.exchange_last_loaded.clear()
        self.logger.info("ExchangeManager shutdown complete")
    
    @retry_async()
    async def _load_exchange(self, exchange_id: str) -> Optional[ccxt.Exchange]:
        """Load a single exchange and its markets"""
        self.logger.info(f"Loading {exchange_id} markets")
        try:
            # Create exchange instance with the shared session
            exchange_class = getattr(ccxt, exchange_id)
            exchange_config = self.exchange_config.copy()
            
            # Add the session to the config
            if self.session:
                exchange_config['session'] = self.session
                
            exchange = exchange_class(exchange_config)
            
            # Load markets
            await exchange.load_markets()
            
            # Update tracking
            self.exchanges[exchange_id] = exchange
            self.symbols_by_exchange[exchange_id] = set(exchange.symbols)
            self.exchange_last_loaded[exchange_id] = datetime.now()
            self.logger.info(f"Loaded {exchange_id} with {len(exchange.symbols)} symbols")
            
            return exchange
        except Exception as e:
            self.logger.error(f"Failed to load {exchange_id} markets: {e}")
            return None

    async def _ensure_exchange_loaded(self, exchange_id: str) -> Optional[ccxt.Exchange]:
        """Ensure exchange is loaded and markets are up to date"""
        now = datetime.now()
        
        # Check if exchange is already loaded
        if exchange_id in self.exchanges:
            last_loaded = self.exchange_last_loaded.get(exchange_id)
            
            # Check if refresh is needed (based on MARKET_REFRESH_HOURS)
            if last_loaded and (now - last_loaded).total_seconds() < self.config.MARKET_REFRESH_HOURS * 3600:
                self.logger.debug(f"Using cached {exchange_id} markets")
                return self.exchanges[exchange_id]
            else:
                self.logger.info(f"Refreshing {exchange_id} markets (last loaded: {last_loaded})")
                await self._refresh_exchange_markets(exchange_id)
                return self.exchanges.get(exchange_id)
        else:
            # Load exchange for the first time
            return await self._load_exchange(exchange_id)
    
    async def _refresh_exchange_markets(self, exchange_id: str) -> None:
        """Refresh markets for a single exchange"""
        if exchange_id not in self.exchanges:
            return
            
        exchange = self.exchanges[exchange_id]
        try:
            self.logger.debug(f"Refreshing {exchange_id} markets")
            await exchange.load_markets(reload=True)
            self.symbols_by_exchange[exchange_id] = set(exchange.symbols)
            self.exchange_last_loaded[exchange_id] = datetime.now()
            self.logger.info(f"Refreshed {exchange_id} with {len(exchange.symbols)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to refresh {exchange_id} markets: {e}")
            # Try to reconnect if refresh fails
            try:
                # Close old exchange connection first
                try:
                    await exchange.close()
                except Exception as e_close:
                    self.logger.warning(f"Error closing old {exchange_id} connection: {e_close}")
                
                # Create new exchange instance
                new_exchange = await self._load_exchange(exchange_id)
                if not new_exchange:
                    # Remove failed exchange from dicts to avoid using a dead instance
                    self.exchanges.pop(exchange_id, None)
                    self.symbols_by_exchange.pop(exchange_id, None)
                    self.exchange_last_loaded.pop(exchange_id, None)
            except Exception as reconnect_err:
                self.logger.error(f"Failed to reconnect to {exchange_id}: {reconnect_err}")
                # Remove failed exchange from dicts to avoid using a dead instance
                self.exchanges.pop(exchange_id, None)
                self.symbols_by_exchange.pop(exchange_id, None)
                self.exchange_last_loaded.pop(exchange_id, None)
    
    async def _periodic_update(self) -> None:
        """Periodically refresh markets for loaded exchanges only"""
        while not self._shutdown_in_progress:
            try:
                # Only refresh exchanges that are already loaded
                loaded_exchanges = list(self.exchanges.keys())
                if loaded_exchanges:
                    self.logger.info(f"Checking {len(loaded_exchanges)} loaded exchanges for periodic refresh")
                    
                    for exchange_id in loaded_exchanges:
                        await self._refresh_exchange_markets(exchange_id)
                else:
                    self.logger.debug("No exchanges loaded yet, skipping periodic refresh")
                
                # Wait for next update cycle
                sleep_hours = self.config.MARKET_REFRESH_HOURS
                self.logger.info(f"Next periodic update in {sleep_hours} hours")
                await asyncio.sleep(sleep_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic update: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying on error
    
    async def find_symbol_exchange(self, symbol: str) -> Tuple[Optional[ccxt.Exchange], Optional[str]]:
        """Find the first exchange that supports the given symbol using lazy loading"""
        self.logger.debug(f"Looking for symbol {symbol} across exchanges")
        
        for exchange_id in self.exchange_names:
            try:
                # Check if we already have this exchange loaded and symbol cached
                if exchange_id in self.symbols_by_exchange and symbol in self.symbols_by_exchange[exchange_id]:
                    exchange = self.exchanges.get(exchange_id)
                    if exchange:
                        self.logger.debug(f"Found {symbol} in cached {exchange_id} markets")
                        return exchange, exchange_id
                
                # Try to load/refresh the exchange to check for the symbol
                exchange = await self._ensure_exchange_loaded(exchange_id)
                if exchange and exchange_id in self.symbols_by_exchange:
                    if symbol in self.symbols_by_exchange[exchange_id]:
                        self.logger.info(f"Found {symbol} on {exchange_id}")
                        return exchange, exchange_id
                    else:
                        self.logger.debug(f"Symbol {symbol} not found on {exchange_id}")
                
            except Exception as e:
                self.logger.error(f"Error checking {exchange_id} for symbol {symbol}: {e}")
                continue
        
        self.logger.warning(f"Symbol {symbol} not found on any supported exchange")
        return None, None
    
    def get_all_symbols(self) -> Set[str]:
        """Get all unique symbols across all loaded exchanges"""
        all_symbols = set()
        for symbols in self.symbols_by_exchange.values():
            all_symbols.update(symbols)
        return all_symbols