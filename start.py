"""
Crypto Trading Bot - Entry Point
Automated trading with AI-powered decisions.
"""
import torch # needed for SentenceSplitter (NLP model)
import asyncio
import sys
import warnings
# Suppress library warnings - MUST be done before other imports
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="discord")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.genai")
import time
import os
import aiohttp
import signal
import atexit
import chromadb
from pathlib import Path
from typing import Optional

if sys.platform != "win32":
    import fcntl

from src.config.loader import config
from src.app import CryptoTradingBot
from src.logger.logger import Logger
from src.utils.graceful_shutdown_manager import GracefulShutdownManager
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.defillama import DefiLlamaClient
from src.platforms.coingecko import CoinGeckoAPI
from src.platforms.cryptocompare.news_api import CryptoCompareNewsAPI
from src.platforms.cryptocompare.news_components import (
    CryptoCompareNewsClient,
    NewsCache,
    NewsProcessor,
    NewsFilter
)
from src.platforms.cryptocompare.market_api import CryptoCompareMarketAPI
from src.platforms.cryptocompare.categories_api import CryptoCompareCategoriesAPI
from src.platforms.cryptocompare.data_processor import CryptoCompareDataProcessor
from src.platforms.exchange_manager import ExchangeManager
from src.analyzer.analysis_engine import AnalysisEngine
from src.rag import RagEngine
from src.utils.token_counter import TokenCounter, CostStorage, ModelPricing
from src.utils.format_utils import FormatUtils
from src.contracts.model_contract import ModelManagerProtocol
from src.managers.model_manager import ModelManager, ProviderClients, ProviderOrchestrator
from src.factories import ProviderFactory
from src.managers.persistence_manager import PersistenceManager
from src.managers.risk_manager import RiskManager
from src.trading import (
    TradingStrategy, TradingBrainService,
    TradingStatisticsService, TradingMemoryService, PositionExtractor
)
from src.trading.vector_memory import VectorMemoryService
from src.dashboard.server import DashboardServer
from src.notifiers import DiscordNotifier, ConsoleNotifier
from src.utils.keyboard_handler import KeyboardHandler
from src.rag.text_splitting import SentenceSplitter
from src.parsing.unified_parser import UnifiedParser
from src.factories import TechnicalIndicatorsFactory, DataFetcherFactory
from src.rag.article_processor import ArticleProcessor
from src.rag.collision_resolver import CategoryCollisionResolver
from src.analyzer.pattern_engine import PatternEngine
from src.analyzer.pattern_engine.indicator_patterns import IndicatorPatternEngine
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
from src.rag.market_components import (
    MarketDataFetcher,
    MarketDataProcessor, 
    MarketDataCache,
    MarketOverviewBuilder
)
from src.analyzer import (
    TechnicalCalculator, PatternAnalyzer, MarketDataCollector,
    MarketMetricsCalculator, AnalysisResultProcessor,
    TechnicalFormatter
)
from src.analyzer.prompts import PromptBuilder
from src.analyzer.prompts.template_manager import TemplateManager
from src.analyzer.prompts.context_builder import ContextBuilder as AnalyzerContextBuilder
from src.analyzer.pattern_engine import ChartGenerator
from src.utils.timeframe_validator import TimeframeValidator

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
        """Build all dependencies for the trading bot via segmented provisions."""
        start_time = time.perf_counter()
        self.logger.info("Initializing Crypto Trading Bot...")
        
        self._init_directories()
        infra = await self._provision_infrastructure()
        utils = self._provision_utilities(infra)
        apis = await self._provision_platforms(infra, utils)
        rag = await self._provision_rag_layer(infra, apis, utils)
        models = self._provision_model_layer(utils)
        analyzer = await self._provision_analyzer_layer(infra, apis, utils, rag, models)
        trading = self._provision_trading_layer(infra, utils, rag, models)
        notifiers = await self._provision_notifiers(utils)
        
        # Start periodic RAG updates
        await rag.start_periodic_updates()
        
        end_time = time.perf_counter()
        init_duration = end_time - start_time
        self.logger.info(f"All dependencies initialized successfully in {init_duration:.2f} seconds")
        
        # Combine everything for the bot and dashboard
        deps = {
            'exchange_manager': infra['exchange_manager'],
            'market_analyzer': analyzer['engine'],
            'trading_strategy': trading['strategy'],
            'discord_notifier': notifiers['notifier'],
            'discord_task': notifiers['task'],
            'keyboard_handler': infra['keyboard_handler'],
            'rag_engine': rag,
            'coingecko_api': apis['coingecko'],
            'news_api': apis['news'],
            'market_api': apis['market'],
            'categories_api': apis['categories'],
            'alternative_me_api': apis['alternative_me'],
            'cryptocompare_session': infra['session'],
            'persistence': trading['persistence'],
            'model_manager': models['manager'],
            'brain_service': trading['brain_service'],
            'statistics_service': trading['statistics_service'],
            'memory_service': trading['memory_service'],
        }
        
        # Initialize Dashboard Server
        dashboard_server = DashboardServer(
            brain_service=trading['brain_service'],
            vector_memory=trading['brain_service'].vector_memory if trading['brain_service'] and hasattr(trading['brain_service'], 'vector_memory') else None,
            analysis_engine=analyzer['engine'],
            config=self.config,
            logger=self.logger,
            unified_parser=utils['parser'],
            persistence=trading['persistence'],
            exchange_manager=infra['exchange_manager'],
            host=self.config.DASHBOARD_HOST,
            port=self.config.DASHBOARD_PORT
        )
        
        deps['dashboard_server'] = dashboard_server
        deps['dashboard_state'] = dashboard_server.dashboard_state
        
        return deps

    def _init_directories(self):
        """Ensure all required directories exist."""
        data_dir = self.config.DATA_DIR
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "news_cache"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "trading"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "charts"), exist_ok=True)
        
        # Calculate symbol-specific brain dir
        safe_symbol = self.config.CRYPTO_PAIR.replace("/", "_").replace("-", "_")
        brain_dir = os.path.join(data_dir, "trading", f"brain_{safe_symbol}_{self.config.TIMEFRAME}")
        os.makedirs(brain_dir, exist_ok=True)

    async def _provision_infrastructure(self) -> dict:
        """Provision base infrastructure components."""
        exchange_manager = ExchangeManager(logger=self.logger, config=self.config)
        await exchange_manager.initialize()
        
        session = aiohttp.ClientSession()
        keyboard_handler = KeyboardHandler(logger=self.logger)
        
        return {
            'exchange_manager': exchange_manager,
            'session': session,
            'keyboard_handler': keyboard_handler
        }

    def _provision_utilities(self, infra: dict) -> dict:
        """Provision utility singletons."""
        format_utils = FormatUtils()
        parser = UnifiedParser(self.logger, format_utils=format_utils)
        token_counter = TokenCounter()
        sentence_splitter = SentenceSplitter(logger=self.logger)
        ti_factory = TechnicalIndicatorsFactory()
        timeframe_validator = TimeframeValidator()
        data_fetcher_factory = DataFetcherFactory(self.logger)
        collision_resolver = CategoryCollisionResolver()
        
        return {
            'format_utils': format_utils,
            'parser': parser,
            'token_counter': token_counter,
            'sentence_splitter': sentence_splitter,
            'ti_factory': ti_factory,
            'timeframe_validator': timeframe_validator,
            'data_fetcher_factory': data_fetcher_factory,
            'collision_resolver': collision_resolver
        }

    async def _provision_platforms(self, infra: dict, utils: dict) -> dict:
        """Provision external API clients."""
        from aiohttp_client_cache import SQLiteBackend
        coingecko_backend = SQLiteBackend(cache_name='cache/coingecko_cache.db', expire_after=-1)
        
        coingecko = CoinGeckoAPI(
            logger=self.logger,
            cache_backend=coingecko_backend,
            cache_dir='data/market_data',
            api_key=self.config.COINGECKO_API_KEY,
            update_interval_hours=24,
            global_api_url=self.config.RAG_COINGECKO_GLOBAL_API_URL
        )
        await coingecko.initialize()
        
        news_client = CryptoCompareNewsClient(self.logger, self.config)
        news_cache = NewsCache('data/news_cache', self.logger)
        news_api = CryptoCompareNewsAPI(
            self.logger, self.config, client=news_client, 
            cache=news_cache, processor=NewsProcessor(self.logger), 
            news_filter=NewsFilter(self.logger)
        )
        await news_api.initialize()
        
        cc_data_processor = CryptoCompareDataProcessor(self.logger)
        categories = CryptoCompareCategoriesAPI(
            logger=self.logger, config=self.config, data_processor=cc_data_processor,
            collision_resolver=utils['collision_resolver'],
            data_dir='data', categories_update_interval_hours=self.config.RAG_CATEGORIES_UPDATE_INTERVAL_HOURS
        )
        await categories.initialize()
        
        defillama = DefiLlamaClient(
            logger=self.logger, session=infra['session'], cache_dir='cache',
            update_interval_hours=self.config.RAG_DEFILLAMA_UPDATE_INTERVAL_HOURS
        )
        
        alternative_me = AlternativeMeAPI(logger=self.logger)
        await alternative_me.initialize()
        
        return {
            'coingecko': coingecko,
            'news': news_api,
            'market': CryptoCompareMarketAPI(logger=self.logger, config=self.config),
            'categories': categories,
            'defillama': defillama,
            'alternative_me': alternative_me
        }

    async def _provision_rag_layer(self, infra: dict, apis: dict, utils: dict) -> RagEngine:
        """Provision the RAG (Retrieval Augmented Generation) engine."""
        article_processor = ArticleProcessor(
            logger=self.logger, unified_parser=utils['parser'],
            format_utils=utils['format_utils'], sentence_splitter=utils['sentence_splitter']
        )
        
        file_handler = RagFileHandler(logger=self.logger, config=self.config, unified_parser=utils['parser'])
        news_manager = NewsManager(
            logger=self.logger, file_handler=file_handler, news_api=apis['news'],
            categories_api=apis['categories'], session=infra['session'], article_processor=article_processor
        )
        
        marker_fetcher = MarketDataFetcher(self.logger, apis['coingecko'], infra['exchange_manager'], apis['market'], apis['defillama'])
        market_processor = MarketDataProcessor(self.logger, utils['parser'])
        data_manager = MarketDataManager(
            self.logger, file_handler, apis['coingecko'], apis['market'], 
            infra['exchange_manager'], unified_parser=utils['parser'], defillama_client=apis['defillama'],
            fetcher=marker_fetcher, processor=market_processor,
            cache=MarketDataCache(self.logger, file_handler), 
            overview_builder=MarketOverviewBuilder(self.logger, market_processor)
        )
        
        category_processor = CategoryProcessor(self.logger, utils['collision_resolver'], file_handler)
        engine = RagEngine(
            logger=self.logger, token_counter=utils['token_counter'], config=self.config,
            coingecko_api=apis['coingecko'], exchange_manager=infra['exchange_manager'],
            file_handler=file_handler, news_manager=news_manager, market_data_manager=data_manager,
            index_manager=IndexManager(self.logger, article_processor),
            category_fetcher=CategoryFetcher(self.logger, apis['categories']),
            category_processor=category_processor,
            ticker_manager=TickerManager(self.logger, file_handler, infra['exchange_manager']),
            news_category_analyzer=NewsCategoryAnalyzer(self.logger, category_processor, utils['parser']),
            context_builder=ContextBuilder(self.logger, utils['token_counter'], article_processor, utils['sentence_splitter'])
        )
        await engine.initialize()
        return engine

    def _provision_model_layer(self, utils: dict) -> dict:
        """Provision AI model managers and providers."""
        provider_factory = ProviderFactory(self.logger, self.config)
        provider_clients = ProviderClients.from_factory_dict(provider_factory.create_all_clients())
        orchestrator = ProviderOrchestrator(self.logger, self.config, provider_clients)
        
        manager = ModelManager(
            logger=self.logger, config=self.config, unified_parser=utils['parser'],
            token_counter=utils['token_counter'], cost_storage=CostStorage(),
            model_pricing=ModelPricing(), orchestrator=orchestrator, provider_clients=provider_clients
        )
        
        return {'manager': manager}

    async def _provision_analyzer_layer(self, infra: dict, apis: dict, utils: dict, rag: RagEngine, models: dict) -> dict:
        """Provision the market analysis engine."""
        overview_fmt = MarketOverviewFormatter(self.logger, utils['format_utils'])
        long_term_fmt = LongTermFormatter(self.logger, utils['format_utils'])
        period_fmt = MarketPeriodFormatter(self.logger, utils['format_utils'])
        
        market_fmt = MarketFormatter(
            self.logger, utils['format_utils'], self.config, utils['token_counter'],
            overview_fmt, period_fmt, long_term_fmt
        )
        
        tech_calc = TechnicalCalculator(self.logger, utils['format_utils'], utils['ti_factory'])
        pattern_analyzer = PatternAnalyzer(
            pattern_engine=PatternEngine(lookback=5, lookahead=5),
            indicator_pattern_engine=IndicatorPatternEngine(),
            logger=self.logger
        )
        try:
            pattern_analyzer.warmup()
        except Exception as warmup_error:
            self.logger.warning(f"Pattern analyzer warm-up could not run: {warmup_error}")
            
        ctx_builder = AnalyzerContextBuilder(
            self.config.TIMEFRAME, self.logger, utils['format_utils'],
            market_fmt, period_fmt, long_term_fmt, utils['timeframe_validator']
        )
        
        prompt_builder = PromptBuilder(
            self.config.TIMEFRAME, self.logger, tech_calc, self.config, utils['format_utils'],
            overview_fmt, long_term_fmt, TechnicalFormatter(tech_calc, self.logger, utils['format_utils']),
            market_fmt, period_fmt, utils['timeframe_validator'],
            TemplateManager(self.config, self.logger, utils['timeframe_validator']), ctx_builder
        )
        
        engine = AnalysisEngine(
            self.logger, rag, apis['coingecko'], models['manager'], apis['alternative_me'],
            apis['market'], self.config, tech_calc, pattern_analyzer, prompt_builder,
            MarketDataCollector(self.logger, rag, apis['alternative_me']),
            MarketMetricsCalculator(self.logger),
            AnalysisResultProcessor(models['manager'], self.logger, utils['parser']),
            ChartGenerator(self.logger, self.config, formatter=utils['format_utils'].fmt, format_utils=utils['format_utils']),
            data_fetcher_factory=utils['data_fetcher_factory']
        )
        
        return {'engine': engine}

    def _provision_trading_layer(self, infra: dict, utils: dict, rag: RagEngine, models: dict) -> dict:
        """Provision trading strategy and memory services."""
        persistence = PersistenceManager(self.logger, data_dir="data/trading")
        risk_manager = RiskManager(self.logger, self.config)
        
        # Calculate specialized brain path
        safe_symbol = self.config.CRYPTO_PAIR.replace("/", "_").replace("-", "_")
        brain_path = os.path.join(self.config.DATA_DIR, "trading", f"brain_{safe_symbol}_{self.config.TIMEFRAME}")
        
        # Create symbol-specific chroma client
        import chromadb
        chroma_client = chromadb.PersistentClient(path=brain_path)
        
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        
        # Inject chroma_client into VectorMemoryService
        vector_memory = VectorMemoryService(self.logger, chroma_client, embedding_model=embedding_model)
        
        brain_service = TradingBrainService(
            self.logger, persistence, vector_memory, 
            symbol=self.config.CRYPTO_PAIR, timeframe=self.config.TIMEFRAME
        )
        
        memory_service = TradingMemoryService(self.logger, persistence, max_memory=10, vector_memory=vector_memory)
        statistics_service = TradingStatisticsService(self.logger, persistence)
        
        from src.factories.position_factory import PositionFactory
        strategy = TradingStrategy(
            self.logger, persistence, brain_service, statistics_service, memory_service,
            risk_manager, self.config, PositionExtractor(self.logger, utils['parser']),
            PositionFactory(self.logger)
        )
        
        return {
            'strategy': strategy,
            'persistence': persistence,
            'brain_service': brain_service,
            'memory_service': memory_service,
            'statistics_service': statistics_service
        }

    async def _provision_notifiers(self, utils: dict) -> dict:
        """Provision notification services."""
        notifier = None
        task = None
        
        if self.config.DISCORD_BOT_ENABLED and hasattr(self.config, 'BOT_TOKEN_DISCORD') and self.config.BOT_TOKEN_DISCORD:
            try:
                import discord
                from src.notifiers.filehandler_components import (
                    TrackingPersistence, MessageTracker, CleanupScheduler, MessageDeleter
                )
                from src.notifiers.filehandler import DiscordFileHandler
                
                intents = discord.Intents.default()
                intents.message_content = False
                intents.reactions = False
                intents.typing = False
                intents.presences = False
                
                bot = discord.Client(intents=intents)
                
                persistence = TrackingPersistence("data/tracked_messages.json", self.logger)
                tracker = MessageTracker(persistence, self.logger, self.config)
                scheduler = CleanupScheduler(7200, self.logger)
                deleter = MessageDeleter(bot, self.logger)
                
                file_handler = DiscordFileHandler(
                    bot=bot,
                    logger=self.logger,
                    config=self.config,
                    persistence=persistence,
                    tracker=tracker,
                    scheduler=scheduler,
                    deleter=deleter
                )
                
                notifier = DiscordNotifier(
                    self.logger, self.config, utils['parser'], 
                    utils['format_utils'], bot, file_handler
                )
                
                task = asyncio.create_task(notifier.start())
                await notifier.wait_until_ready()
            except Exception as e:
                self.logger.warning(f"Discord initialization failed: {e}. Falling back to console output.")
                notifier = ConsoleNotifier(self.logger, self.config, utils['parser'], utils['format_utils'])
        else:
            notifier = ConsoleNotifier(self.logger, self.config, utils['parser'], utils['format_utils'])
            
        return {'notifier': notifier, 'task': task}
    
    async def run_async(self):
        """Async entry point for the application."""
        dependencies = await self.build_dependencies()
        
        # Extract dashboard_server before passing to bot (bot doesn't accept it)
        dashboard_server = dependencies.pop('dashboard_server', None)
        
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
            
            # Create tasks
            bot_task = asyncio.create_task(bot.run(symbol, timeframe))
            
            # Start dashboard if available
            if dashboard_server:
                dashboard_task = await dashboard_server.start()
                await asyncio.gather(bot_task, dashboard_task, return_exceptions=True)
            else:
                await bot_task
                
        except asyncio.CancelledError:
            self.logger.info("Trading cancelled, shutting down...")
        finally:
            # Clean up dashboard server
            if dashboard_server:
                await dashboard_server.stop()
    
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
