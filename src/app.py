import asyncio

from src.logger.logger import Logger
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.coingecko import CoinGeckoAPI
from src.platforms.cryptocompare import CryptoCompareAPI
from src.platforms.exchange_manager import ExchangeManager
from src.analyzer.core.analysis_engine import AnalysisEngine
from src.discord_interface.notifier import DiscordNotifier
from src.rag import RagEngine
from src.utils.token_counter import TokenCounter
from src.utils.keyboard_handler import KeyboardHandler
from src.utils.loader import config
from src.utils.format_utils import FormatUtils
from src.analyzer.data.data_processor import DataProcessor
from src.models.manager import ModelManager


class DiscordCryptoBot:
    def __init__(self, logger: Logger):
        self.market_analyzer = None
        self.coingecko_api = None
        self.cryptocompare_api = None
        self.alternative_me_api = None
        self.rag_engine = None
        self.logger = logger
        self.discord_notifier = None
        self.symbol_manager = None
        self.token_counter = None
        self.keyboard_handler = None
        self.format_utils = None
        self.data_processor = None
        self.model_manager = None
        self.tasks = []
        self.running = False
        self._active_tasks = set()

    async def initialize(self):
        self.logger.info("Initializing Discord Crypto Bot...")

        # Initialize TokenCounter early
        self.token_counter = TokenCounter()
        self.logger.debug("TokenCounter initialized")

        # Initialize DataProcessor and FormatUtils
        self.data_processor = DataProcessor()
        self.format_utils = FormatUtils(self.data_processor)
        self.logger.debug("DataProcessor and FormatUtils initialized")

        self.symbol_manager = ExchangeManager(self.logger, config)
        await self.symbol_manager.initialize()
        self.logger.debug("SymbolManager initialized")

        # Initialize API clients
        self.coingecko_api = CoinGeckoAPI(logger=self.logger)
        await self.coingecko_api.initialize()
        self.logger.debug("CoinGeckoAPI initialized")
        
        self.cryptocompare_api = CryptoCompareAPI(logger=self.logger, config=config)
        await self.cryptocompare_api.initialize()
        self.logger.debug("CryptoCompareAPI initialized")
        
        self.alternative_me_api = AlternativeMeAPI(logger=self.logger)
        await self.alternative_me_api.initialize()
        self.logger.debug("AlternativeMeAPI initialized")

        # Pass token_counter and initialized API clients to RagEngine to avoid double initialization
        self.rag_engine = RagEngine(
            self.logger, 
            self.token_counter,
            config,
            coingecko_api=self.coingecko_api,
            cryptocompare_api=self.cryptocompare_api,
            format_utils=self.format_utils
        )
        await self.rag_engine.initialize()
        self.logger.debug("RagEngine initialized")
        
        self.rag_engine.set_symbol_manager(self.symbol_manager)
        self.logger.debug("Passed SymbolManager to RagEngine")

        # Initialize ModelManager before AnalysisEngine (required dependency)
        self.model_manager = ModelManager(self.logger, config)
        self.logger.debug("ModelManager initialized")

        self.market_analyzer = AnalysisEngine(
            logger=self.logger,
            rag_engine=self.rag_engine,
            coingecko_api=self.coingecko_api,
            model_manager=self.model_manager,
            alternative_me_api=self.alternative_me_api,
            cryptocompare_api=self.cryptocompare_api,
            format_utils=self.format_utils,
            data_processor=self.data_processor,
            config=config
        )

        self.logger.debug("AnalysisEngine initialized")

        self.discord_notifier = DiscordNotifier(
            logger=self.logger,
            symbol_manager=self.symbol_manager,
            market_analyzer=self.market_analyzer,
            config=config,
            format_utils=self.format_utils
        )
        discord_task = asyncio.create_task(
            self.discord_notifier.start(),
            name="Discord-Bot"
        )
        self._active_tasks.add(discord_task)
        discord_task.add_done_callback(self._active_tasks.discard)
        self.tasks.append(discord_task)

        # Wait for Discord notifier to be fully ready
        self.logger.debug("Waiting for Discord notifier to fully initialize...")
        await self.discord_notifier.wait_until_ready()
        self.logger.debug("DiscordNotifier initialized")

        self.market_analyzer.set_discord_notifier(self.discord_notifier)
        self.logger.debug("AnalysisEngine set DiscordNotifier")

        await self.rag_engine.start_periodic_updates()
        
        # Initialize keyboard handler
        self.keyboard_handler = KeyboardHandler(logger=self.logger)
        self.keyboard_handler.register_command('r', self.refresh_crypto_news, "Refresh crypto news")
        self.keyboard_handler.register_command('o', self.refresh_market_overview, "Refresh market overview data")
        self.keyboard_handler.register_command('h', self.show_help, "Show this help message")
        self.keyboard_handler.register_command('q', self.request_shutdown, "Quit the application")
        self.keyboard_handler.register_command('R', self.reload_configuration, "Reload config.ini and keys.env (SHIFT+R)")
        
        # Start keyboard handler
        keyboard_task = asyncio.create_task(
            self.keyboard_handler.start_listening(),
            name="Keyboard-Handler"
        )
        self._active_tasks.add(keyboard_task)
        keyboard_task.add_done_callback(self._active_tasks.discard)
        self.tasks.append(keyboard_task)
        
        self.logger.info("Discord Crypto Bot initialized successfully")
        self.logger.info("Keyboard commands available: Press 'h' for help")

    async def start(self):
        """Start the bot and its components"""
        self.logger.info("Bot components running...")
        self.running = True
    
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            self.logger.info("Discord bot received cancellation request...")
        except Exception as e:
            self.logger.error(f"Error in bot: {e}")

    async def refresh_crypto_news(self):
        """Refresh crypto news data"""
        self.logger.info("Manually refreshing crypto news...")
        try:
            await self.rag_engine.refresh_market_data()
            self.logger.info("Crypto news refreshed successfully")
        except Exception as e:
            self.logger.error(f"Error refreshing crypto news: {e}")

    async def refresh_market_overview(self):
        """Refresh market overview data"""
        self.logger.info("Manually refreshing market overview data...")
        try:
            market_overview = await self.rag_engine.market_data_manager.fetch_market_overview()
            if market_overview:
                self.rag_engine.current_market_overview = market_overview
                self.logger.info("Market overview data refreshed successfully")
            else:
                self.logger.warning("Failed to fetch market overview data")
        except Exception as e:
            self.logger.error(f"Error refreshing market overview: {e}")

    async def reload_configuration(self):
        """Reload config.ini and keys.env without restarting"""
        self.logger.info("Reloading configuration files (config.ini and keys.env)...")
        try:
            config.reload()
            self.logger.info("Configuration files reloaded successfully!")
            self.logger.info("Note: Some components may require manual refresh to use new settings")
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")

    async def show_help(self):
        """Show help information about available commands"""
        self.logger.info("\n=== Available Keyboard Commands ===")
        self.keyboard_handler.display_help()
        self.logger.info("=================================")

    async def request_shutdown(self):
        """Request application shutdown"""
        self.logger.info("Shutdown requested via keyboard command")
        self.running = False
        # Cancel all tasks to trigger shutdown
        for task in self.tasks:
            if not task.done():
                task.cancel()

    async def shutdown(self):
        self.logger.info("Shutting down gracefully...")
        self.running = False

        # Step 1: Cancel and wait for active tasks
        await self._shutdown_active_tasks()
        
        # Step 2: Close keyboard handler
        await self._shutdown_keyboard_handler()
        
        # Step 3: Close components in reverse initialization order
        await self._shutdown_discord_notifier()
        await self._shutdown_market_analyzer()
        await self._shutdown_rag_engine()
        await self._shutdown_api_clients()
        await self._shutdown_symbol_manager()
        
        self._cleanup_references()
        self.logger.info("Shutdown complete")

    async def _shutdown_active_tasks(self):
        """Cancel and wait for active tasks to complete."""
        pending_tasks = list(self._active_tasks)
        if not pending_tasks:
            return
            
        self.logger.info(f"Cancelling {len(pending_tasks)} active tasks...")
        for task in pending_tasks:
            if not task.done():
                task.cancel()

        try:
            await asyncio.wait(pending_tasks, timeout=3.0)
        except asyncio.TimeoutError:
            self.logger.warning("Some tasks didn't complete in time")

    async def _shutdown_keyboard_handler(self):
        """Close keyboard handler safely."""
        if not self.keyboard_handler:
            return
            
        try:
            self.logger.info("Closing keyboard handler...")
            await self.keyboard_handler.stop_listening()
            self.keyboard_handler = None
        except Exception as e:
            self.logger.warning(f"Error closing keyboard handler: {e}")

    async def _shutdown_discord_notifier(self):
        """Close Discord notifier safely."""
        if not self.discord_notifier:
            return
            
        try:
            self.logger.info("Closing Discord notifier...")
            await asyncio.wait_for(self.discord_notifier.__aexit__(None, None, None), timeout=5.0)
            self.discord_notifier = None
        except asyncio.TimeoutError:
            self.logger.warning("Discord notifier shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing Discord notifier: {e}")

    async def _shutdown_market_analyzer(self):
        """Close market analyzer and model manager safely."""
        # Close model_manager first (part of market_analyzer dependencies)
        if hasattr(self, 'model_manager') and self.model_manager:
            try:
                self.logger.info("Closing ModelManager...")
                await asyncio.wait_for(self.model_manager.close(), timeout=3.0)
                self.model_manager = None
            except asyncio.TimeoutError:
                self.logger.warning("ModelManager shutdown timed out")
            except Exception as e:
                self.logger.warning(f"Error closing ModelManager: {e}")
        
        # Then close market analyzer
        if not self.market_analyzer:
            return
            
        try:
            self.logger.info("Closing market analyzer...")
            await asyncio.wait_for(self.market_analyzer.close(), timeout=3.0)
            self.market_analyzer = None
        except asyncio.TimeoutError:
            self.logger.warning("Market analyzer shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing market analyzer: {e}")

    async def _shutdown_rag_engine(self):
        """Close RAG engine safely."""
        if not hasattr(self, 'rag_engine') or not self.rag_engine:
            return
            
        try:
            self.logger.info("Closing RAG engine...")
            await asyncio.wait_for(self.rag_engine.close(), timeout=3.0)
            self.rag_engine = None
        except asyncio.TimeoutError:
            self.logger.warning("RAG engine shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing RAG engine: {e}")

    async def _shutdown_api_clients(self):
        """Close all API clients safely."""
        api_clients = [
            ("AlternativeMeAPI", self.alternative_me_api),
            ("CryptoCompareAPI", self.cryptocompare_api),
            ("CoinGeckoAPI", self.coingecko_api)
        ]

        for client_name, client in api_clients:
            await self._shutdown_single_api_client(client_name, client)

    async def _shutdown_single_api_client(self, client_name: str, client):
        """Close a single API client safely."""
        if not client or not hasattr(client, 'close'):
            return
            
        try:
            self.logger.info(f"Closing {client_name}...")
            await asyncio.wait_for(client.close(), timeout=3.0)
        except asyncio.TimeoutError:
            self.logger.warning(f"{client_name} shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing {client_name}: {e}")

    async def _shutdown_symbol_manager(self):
        """Close symbol manager safely."""
        if not self.symbol_manager:
            return
            
        try:
            self.logger.info("Closing SymbolManager...")
            await asyncio.wait_for(self.symbol_manager.shutdown(), timeout=3.0)
            self.symbol_manager = None
        except asyncio.TimeoutError:
            self.logger.warning("SymbolManager shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing SymbolManager: {e}")

    def _cleanup_references(self):
        """Set all component references to None to help garbage collection."""
        self.alternative_me_api = None
        self.cryptocompare_api = None
        self.coingecko_api = None