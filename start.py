"""
Crypto Trading Bot - Entry Point
Automated trading with AI-powered decisions.
"""
import asyncio
import sys
import warnings
import time
import os
import aiohttp
import signal
import atexit
from pathlib import Path
from typing import Optional

if sys.platform != "win32":
    import fcntl

from src.config.loader import config
from src.app import CryptoTradingBot
from src.logger.logger import Logger
from src.utils.graceful_shutdown_manager import GracefulShutdownManager
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.coingecko import CoinGeckoAPI
from src.platforms.cryptocompare.news_api import CryptoCompareNewsAPI
from src.platforms.cryptocompare.market_api import CryptoCompareMarketAPI
from src.platforms.cryptocompare.categories_api import CryptoCompareCategoriesAPI
from src.platforms.exchange_manager import ExchangeManager
from src.analyzer.analysis_engine import AnalysisEngine
from src.rag import RagEngine
from src.utils.token_counter import TokenCounter
from src.utils.format_utils import FormatUtils
from src.analyzer.data_processor import DataProcessor
from src.contracts.manager import ModelManager
from src.trading import (
    TradingStrategy, TradingPersistence, TradingBrainService,
    TradingStatisticsService, TradingMemoryService, PositionExtractor
)
from src.notifiers import DiscordNotifier, ConsoleNotifier
from src.utils.keyboard_handler import KeyboardHandler
from src.rag.text_splitting import SentenceSplitter
from src.parsing.unified_parser import UnifiedParser
from src.factories import TechnicalIndicatorsFactory
from src.rag.article_processor import ArticleProcessor
from src.analyzer.formatters import (
    MarketOverviewFormatter,
    LongTermFormatter,
    MarketFormatter,
    MarketPeriodFormatter
)
from src.rag import (
    RagFileHandler, NewsManager, MarketDataManager,
    IndexManager, ContextBuilder, CategoryFetcher,
    CategoryProcessor, TickerManager, NewsCategoryAnalyzer
)
from src.analyzer import (
    TechnicalCalculator, PatternAnalyzer, MarketDataCollector,
    MarketMetricsCalculator, AnalysisResultProcessor,
    TechnicalFormatter
)
from src.analyzer.prompts import PromptBuilder
from src.analyzer.pattern_engine import ChartGenerator

warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

lock_file_handle: Optional[int] = None


def check_single_instance() -> bool:
    """
    Check if another instance is already running using a lock file.
    Works cross-platform (Windows, Linux, macOS).
    
    Returns:
        True if this is the only instance, False if another instance is running.
    """
    global lock_file_handle
    
    lock_file_path = Path.home() / ".llm_trader.lock"
    
    try:
        lock_file_handle = os.open(str(lock_file_path), os.O_CREAT | os.O_RDWR)
        
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(lock_file_handle, msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
        else:
            try:
                fcntl.flock(lock_file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        
        atexit.register(_cleanup_lock_file)
        return True
        
    except Exception as e:
        print(f"Warning: Could not create lock file: {e}")
        return True


def _cleanup_lock_file() -> None:
    """Release the lock file on exit."""
    global lock_file_handle
    
    if lock_file_handle is not None:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(lock_file_handle, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file_handle, fcntl.LOCK_UN)
            os.close(lock_file_handle)
        except Exception:
            pass
        
        lock_file_path = Path.home() / ".llm_trader.lock"
        try:
            lock_file_path.unlink(missing_ok=True)
        except Exception:
            pass


class CompositionRoot:
    """Composition Root for the trading bot application.
    
    Responsible for building and wiring all dependencies following the
    Dependency Injection pattern before injecting them into CryptoTradingBot.
    """
    
    def __init__(self):
        self.config = config
        self.logger = Logger(logger_name="Bot", logger_debug=config.LOGGER_DEBUG)
        self.loop = None
        self.shutdown_manager = None
    
    async def build_dependencies(self) -> dict:
        """Build all dependencies for the trading bot.
        
        Returns:
            Dictionary of all initialized components ready for injection.
        """
        start_time = time.perf_counter()
        
        self.logger.info("Initializing Crypto Trading Bot...")
        
        # Initialize data directory
        data_dir = self.config.DATA_DIR
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "news_cache"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "trading"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "charts"), exist_ok=True)
        
        # Initialize ExchangeManager
        exchange_manager = ExchangeManager(logger=self.logger, config=self.config)
        await exchange_manager.initialize()
        self.logger.debug("ExchangeManager initialized")
        
        # Initialize utilities
        token_counter = TokenCounter()
        data_processor = DataProcessor()
        format_utils = FormatUtils(data_processor=data_processor)
        sentence_splitter = SentenceSplitter(logger=self.logger)
        ti_factory = TechnicalIndicatorsFactory()
        unified_parser = UnifiedParser(self.logger)
        
        # Initialize ArticleProcessor
        article_processor = ArticleProcessor(
            logger=self.logger,
            unified_parser=unified_parser,
            format_utils=format_utils,
            sentence_splitter=sentence_splitter
        )
        
        # Initialize API Clients
        coingecko_api = CoinGeckoAPI(
            logger=self.logger,
            api_key=self.config.COINGECKO_API_KEY,
            cache_dir='cache',
            update_interval_hours=self.config.RAG_COINGECKO_UPDATE_INTERVAL_HOURS,
            global_api_url=self.config.RAG_COINGECKO_GLOBAL_API_URL
        )
        await coingecko_api.initialize()
        self.logger.debug("CoinGeckoAPI initialized")
        
        # Initialize CryptoCompare components
        cryptocompare_session = aiohttp.ClientSession()
        
        news_api = CryptoCompareNewsAPI(
            logger=self.logger,
            config=self.config,
            cache_dir='data/news_cache',
            update_interval_hours=self.config.RAG_UPDATE_INTERVAL_HOURS
        )
        await news_api.initialize()
        
        categories_api = CryptoCompareCategoriesAPI(
            logger=self.logger,
            config=self.config,
            data_dir='data',
            categories_update_interval_hours=self.config.RAG_CATEGORIES_UPDATE_INTERVAL_HOURS
        )
        await categories_api.initialize()
        
        market_api = CryptoCompareMarketAPI(logger=self.logger, config=self.config)
        self.logger.debug("CryptoCompare components initialized")
        
        alternative_me_api = AlternativeMeAPI(logger=self.logger)
        await alternative_me_api.initialize()
        self.logger.debug("AlternativeMeAPI initialized")
        
        # Create RAG component managers
        rag_file_handler = RagFileHandler(
            logger=self.logger,
            config=self.config,
            unified_parser=unified_parser
        )
        
        news_manager = NewsManager(
            logger=self.logger,
            file_handler=rag_file_handler,
            news_api=news_api,
            categories_api=categories_api,
            session=cryptocompare_session,
            article_processor=article_processor
        )
        
        market_data_manager = MarketDataManager(
            logger=self.logger,
            file_handler=rag_file_handler,
            coingecko_api=coingecko_api,
            market_api=market_api,
            exchange_manager=exchange_manager,
            unified_parser=unified_parser
        )
        
        index_manager = IndexManager(
            logger=self.logger,
            article_processor=article_processor
        )
        
        category_fetcher = CategoryFetcher(
            logger=self.logger,
            categories_api=categories_api
        )
        
        category_processor = CategoryProcessor(
            logger=self.logger,
            file_handler=rag_file_handler
        )
        
        ticker_manager = TickerManager(
            logger=self.logger,
            file_handler=rag_file_handler,
            exchange_manager=exchange_manager
        )
        
        news_category_analyzer = NewsCategoryAnalyzer(
            logger=self.logger,
            category_processor=category_processor,
            unified_parser=unified_parser
        )
        
        context_builder = ContextBuilder(
            logger=self.logger,
            token_counter=token_counter,
            article_processor=article_processor,
            sentence_splitter=sentence_splitter
        )
        context_builder.config = self.config
        
        self.logger.debug("RAG components created")
        
        # Initialize RagEngine
        rag_engine = RagEngine(
            logger=self.logger,
            token_counter=token_counter,
            config=self.config,
            coingecko_api=coingecko_api,
            exchange_manager=exchange_manager,
            file_handler=rag_file_handler,
            news_manager=news_manager,
            market_data_manager=market_data_manager,
            index_manager=index_manager,
            category_fetcher=category_fetcher,
            category_processor=category_processor,
            ticker_manager=ticker_manager,
            news_category_analyzer=news_category_analyzer,
            context_builder=context_builder
        )
        await rag_engine.initialize()
        self.logger.debug("RagEngine initialized")
        
        # Initialize ModelManager
        model_manager = ModelManager(self.logger, self.config, unified_parser=unified_parser)
        self.logger.debug("ModelManager initialized")
        
        # Create formatters for dependency injection
        overview_formatter = MarketOverviewFormatter(self.logger, format_utils)
        long_term_formatter = LongTermFormatter(self.logger, format_utils)
        market_formatter = MarketFormatter(self.logger, format_utils)
        period_formatter = MarketPeriodFormatter(self.logger, format_utils)
        
        # Create Analyzer components
        technical_calculator = TechnicalCalculator(
            logger=self.logger,
            format_utils=format_utils,
            ti_factory=ti_factory
        )
        
        pattern_analyzer = PatternAnalyzer(logger=self.logger)
        try:
            pattern_analyzer.warmup()
        except Exception as warmup_error:
            self.logger.warning(f"Pattern analyzer warm-up could not run: {warmup_error}")
        
        technical_formatter = TechnicalFormatter(
            technical_calculator,
            self.logger,
            format_utils
        )
        
        prompt_builder = PromptBuilder(
            timeframe=self.config.TIMEFRAME,
            logger=self.logger,
            technical_calculator=technical_calculator,
            config=self.config,
            format_utils=format_utils,
            data_processor=data_processor,
            overview_formatter=overview_formatter,
            long_term_formatter=long_term_formatter,
            technical_formatter=technical_formatter,
            market_formatter=market_formatter,
            period_formatter=period_formatter
        )
        
        market_data_collector = MarketDataCollector(
            logger=self.logger,
            rag_engine=rag_engine,
            alternative_me_api=alternative_me_api
        )
        
        metrics_calculator = MarketMetricsCalculator(logger=self.logger)
        
        result_processor = AnalysisResultProcessor(
            model_manager=model_manager,
            logger=self.logger,
            unified_parser=unified_parser
        )
        
        chart_generator = ChartGenerator(
            logger=self.logger,
            config=self.config,
            format_utils=format_utils
        )
        
        self.logger.debug("Analyzer components created")
        
        # Initialize AnalysisEngine
        market_analyzer = AnalysisEngine(
            logger=self.logger,
            rag_engine=rag_engine,
            coingecko_api=coingecko_api,
            model_manager=model_manager,
            alternative_me_api=alternative_me_api,
            market_api=market_api,
            config=self.config,
            technical_calculator=technical_calculator,
            pattern_analyzer=pattern_analyzer,
            prompt_builder=prompt_builder,
            data_collector=market_data_collector,
            metrics_calculator=metrics_calculator,
            result_processor=result_processor,
            chart_generator=chart_generator
        )
        self.logger.debug("AnalysisEngine initialized")
        
        # Initialize trading services
        position_extractor = PositionExtractor(self.logger, unified_parser=unified_parser)
        persistence = TradingPersistence(self.logger, data_dir="data/trading")
        brain_service = TradingBrainService(self.logger, persistence)
        memory_service = TradingMemoryService(self.logger, persistence, max_memory=10)
        statistics_service = TradingStatisticsService(self.logger, persistence)
        
        trading_strategy = TradingStrategy(
            self.logger,
            persistence=persistence,
            brain_service=brain_service,
            statistics_service=statistics_service,
            memory_service=memory_service,
            config=self.config,
            position_extractor=position_extractor
        )
        self.logger.debug("Trading components initialized")
        
        # Initialize notifier - Discord if enabled, otherwise Console fallback
        discord_notifier = None
        discord_task = None
        
        if self.config.DISCORD_BOT_ENABLED and hasattr(self.config, 'BOT_TOKEN_DISCORD') and self.config.BOT_TOKEN_DISCORD:
            try:
                discord_notifier = DiscordNotifier(
                    logger=self.logger,
                    config=self.config,
                    unified_parser=unified_parser,
                    formatter=format_utils
                )
                discord_task = asyncio.create_task(discord_notifier.start())
                await discord_notifier.wait_until_ready()
                self.logger.debug("Discord notifier initialized and ready")
            except Exception as e:
                self.logger.warning(f"Discord initialization failed: {e}. Falling back to console output.")
                discord_notifier = ConsoleNotifier(
                    logger=self.logger,
                    config=self.config,
                    unified_parser=unified_parser,
                    formatter=format_utils
                )
        else:
            discord_notifier = ConsoleNotifier(
                logger=self.logger,
                config=self.config,
                unified_parser=unified_parser,
                formatter=format_utils
            )
            self.logger.info("Discord disabled, using console notifications")
        
        # Start periodic RAG updates
        await rag_engine.start_periodic_updates()
        
        # Initialize keyboard handler
        keyboard_handler = KeyboardHandler(logger=self.logger)
        
        end_time = time.perf_counter()
        init_duration = end_time - start_time
        self.logger.info(f"All dependencies initialized successfully in {init_duration:.2f} seconds")
        
        return {
            'exchange_manager': exchange_manager,
            'market_analyzer': market_analyzer,
            'trading_strategy': trading_strategy,
            'discord_notifier': discord_notifier,
            'discord_task': discord_task,
            'keyboard_handler': keyboard_handler,
            'rag_engine': rag_engine,
            'coingecko_api': coingecko_api,
            'news_api': news_api,
            'market_api': market_api,
            'categories_api': categories_api,
            'alternative_me_api': alternative_me_api,
            'cryptocompare_session': cryptocompare_session,
            'persistence': persistence,
            'brain_service': brain_service,
            'statistics_service': statistics_service,
            'memory_service': memory_service,
        }
    
    async def run_async(self):
        """Async entry point for the application."""
        dependencies = await self.build_dependencies()
        
        bot = CryptoTradingBot(
            logger=self.logger,
            config=self.config,
            shutdown_manager=self.shutdown_manager,
            **dependencies
        )
        
        try:
            await bot.initialize()
            symbol = self.config.CRYPTO_PAIR
            timeframe = self.config.TIMEFRAME
            await bot.run(symbol, timeframe)
        except asyncio.CancelledError:
            self.logger.info("Trading cancelled, shutting down...")
        finally:
            pass
    
    def start(self):
        """Main entry point with clean shutdown delegation."""
        if not check_single_instance():
            if PYQT_AVAILABLE:
                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)
                    QApplication.setHighDpiScaleFactorRoundingPolicy(
                        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
                    )
                QMessageBox.critical(
                    None,
                    "Crypto Trading Bot",
                    "Another instance of Crypto Trading Bot is already running.",
                    QMessageBox.StandardButton.Ok
                )
            else:
                print("Another instance of Crypto Trading Bot is already running.")
            sys.exit(1)
        
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.shutdown_manager = GracefulShutdownManager(
            self.loop,
            logger=self.logger,
            confirmation_callback=GracefulShutdownManager.show_exit_confirmation
        )
        self.shutdown_manager.setup_signal_handlers()
        
        try:
            self.loop.run_until_complete(self.run_async())
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received - initiating graceful shutdown...")
            self.loop.run_until_complete(self.shutdown_manager.shutdown_gracefully())
        except Exception as e:
            print(f"Unhandled exception: {e}")
            import traceback
            traceback.print_exc()
            self.loop.run_until_complete(self.shutdown_manager.shutdown_gracefully())
        finally:
            self.loop.close()


if __name__ == "__main__":
    CompositionRoot().start()
