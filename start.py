"""
Crypto Trading Bot - Entry Point
Automated trading with AI-powered decisions.
"""
# --- Standard Library ---
import asyncio
import atexit
import os
import sys
import time
import warnings
from pathlib import Path
import hashlib
from typing import Optional

# --- Third-party ---
import aiohttp
import torch  # noqa: F401  # needed to initialize PyTorch before sentence-transformers
import chromadb

# --- Local ---
from src.config.loader import config
from src.app import CryptoTradingBot, POSITION_UPDATE_INTERVAL
from sentence_transformers import SentenceTransformer
from src.logger.logger import Logger
from src.utils.graceful_shutdown_manager import GracefulShutdownManager
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.defillama import DefiLlamaClient
from src.platforms.coingecko import CoinGeckoAPI
from src.platforms.ccxt_market_api import CCXTMarketAPI
from src.rag.local_taxonomy import LocalTaxonomyProvider
from src.platforms.exchange_manager import ExchangeManager
from src.analyzer.analysis_engine import AnalysisEngine
from src.rag import RagEngine
from src.rag.scoring_policy import ArticleScoringPolicy
from src.rag.news_ingestion import RSSCrawl4AINewsProvider, Crawl4AIEnricher
from src.utils.token_counter import TokenCounter, CostStorage, ModelPricing
from src.utils.format_utils import FormatUtils
from src.managers.model_manager import ModelManager, ProviderClients, ProviderOrchestrator
from src.factories import ProviderFactory
from src.managers.persistence_manager import PersistenceManager
from src.managers.risk_manager import RiskManager
from src.trading import (
    TradingStrategy, TradingBrainService,
    TradingStatisticsService, TradingMemoryService, PositionExtractor,
    ExitMonitor, PositionStatusMonitor
)
from src.trading.vector_memory import VectorMemoryService
from src.dashboard.server import DashboardServer
from src.notifiers import DiscordNotifier, ConsoleNotifier
from src.utils.keyboard_handler import KeyboardHandler
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
    IndexManager, ContextBuilder,
    CategoryProcessor, TickerManager
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
from src.utils.indicator_classifier import build_exit_execution_context_from_config
from src.trading.stop_loss_tightening_policy import StopLossTighteningPolicy

# Suppress known deprecation warnings from third-party libraries at runtime
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="discord")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.genai")


def _configure_hf_hub_auth() -> None:
    """Expose optional Hugging Face token to libraries that read process env vars."""
    hf_token = config.get_env("HF_TOKEN")
    if not hf_token:
        return

    token = str(hf_token).strip()
    if not token:
        return

    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def _get_best_device() -> str:
    """Auto-detect best available hardware accelerator for embeddings.

    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

try:
    import tkinter as tk
    from tkinter import messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

class SingleInstanceLock:
    """Manages a single instance lock file to prevent multiple application instances."""

    def __init__(self, app_name: str = ".llm_trader.lock"):
        self.lock_file_path = Path.home() / app_name
        self._lock_handle: Optional[int] = None
        self._mutex_handle = None

    def _acquire_windows_mutex(self) -> bool:
        """Use a named mutex on Windows to guarantee single process instance."""
        try:
            import ctypes

            lock_key = hashlib.sha1(str(self.lock_file_path).encode("utf-8")).hexdigest()[:16]
            mutex_name = f"Local\\LLMTraderSingleInstance_{lock_key}"
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.CreateMutexW(None, False, mutex_name)
            if not handle:
                return True  # Fall back to file lock path below.

            self._mutex_handle = handle
            ERROR_ALREADY_EXISTS = 183
            if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
                kernel32.CloseHandle(handle)
                self._mutex_handle = None
                return False
            return True
        except Exception:
            return True  # Fall back to file lock path below.

    def _release_windows_mutex(self) -> None:
        if self._mutex_handle:
            try:
                import ctypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle(self._mutex_handle)
            except Exception:
                pass
            self._mutex_handle = None

    def acquire(self) -> bool:
        """Attempt to acquire the lock. Returns True if successful."""
        try:
            if sys.platform == "win32" and not self._acquire_windows_mutex():
                return False

            self._lock_handle = os.open(str(self.lock_file_path), os.O_CREAT | os.O_RDWR)

            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(self._lock_handle, msvcrt.LK_NBLCK, 1)
                except OSError:
                    self._release_windows_mutex()
                    return False
            else:
                import fcntl  # pylint: disable=import-error
                try:
                    fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False

            atexit.register(self.release)
            return True

        except Exception as e:
            self._release_windows_mutex()
            print(f"Warning: Could not create lock file: {e}")
            return True

    def release(self) -> None:
        """Release the lock and cleanup."""
        if self._lock_handle is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(self._lock_handle, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl  # pylint: disable=import-error
                    fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
                os.close(self._lock_handle)
            except Exception:
                pass
            self._lock_handle = None

            try:
                self.lock_file_path.unlink(missing_ok=True)
            except Exception:
                pass

        self._release_windows_mutex()


def _show_error_dialog(title: str, message: str) -> bool:
    """Show a best-effort GUI error dialog and return True if displayed."""
    if not TKINTER_AVAILABLE:
        return False

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(title, message, parent=root)
        return True
    except Exception:
        return False
    finally:
        if root is not None:
            try:
                root.destroy()
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
        self.logger.install_crash_handler()
        self.loop = None
        self.shutdown_manager = None

    async def build_dependencies(self) -> dict:
        """Build all dependencies for the trading bot via segmented provisions."""
        start_time = time.perf_counter()
        self.logger.info("Initializing Crypto Trading Bot...")

        self._init_directories()
        infra = await self._provision_infrastructure()
        utils = self._provision_utilities()
        apis = await self._provision_platforms(infra, utils)
        rag = await self._provision_rag_layer(infra, apis, utils)
        models = self._provision_model_layer(utils)
        analyzer = await self._provision_analyzer_layer(infra, apis, utils, rag, models)
        trading = self._provision_trading_layer(utils)
        notifiers = await self._provision_notifiers(utils)

        end_time = time.perf_counter()
        init_duration = end_time - start_time
        self.logger.info("All dependencies initialized successfully in %.2f seconds", init_duration)

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
            'market_api': apis['market'],
            'alternative_me_api': apis['alternative_me'],
            'http_session': infra['session'],
            'persistence': trading['persistence'],
            'model_manager': models['manager'],
            'brain_service': trading['brain_service'],
            'statistics_service': trading['statistics_service'],
            'memory_service': trading['memory_service'],
            'exit_monitor': trading['exit_monitor'],
        }

        # Always instantiate DashboardServer so the 'd' keyboard toggle can start/stop it at runtime.
        # The server socket is NOT opened until start() is called, so this is safe even when disabled.
        dashboard_server = DashboardServer(
            brain_service=trading['brain_service'],
            vector_memory=trading['brain_service'].vector_memory if trading['brain_service'] else None,
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
        trading['strategy'].set_dashboard_state(dashboard_server.dashboard_state)

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

    def _provision_utilities(self) -> dict:
        """Provision utility singletons."""
        format_utils = FormatUtils()
        parser = UnifiedParser(self.logger, format_utils=format_utils)
        token_counter = TokenCounter()
        # SentenceSplitter removed (simplified NLP)
        ti_factory = TechnicalIndicatorsFactory()
        timeframe_validator = TimeframeValidator()
        data_fetcher_factory = DataFetcherFactory(self.logger)
        collision_resolver = CategoryCollisionResolver()

        return {
            'format_utils': format_utils,
            'parser': parser,
            'token_counter': token_counter,
            # 'sentence_splitter': sentence_splitter, # Removed
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

        news_client = RSSCrawl4AINewsProvider(
            self.logger, self.config,
            enricher=Crawl4AIEnricher(
                concurrency=self.config.RAG_NEWS_CRAWL_CONCURRENCY,
                timeout=float(self.config.RAG_NEWS_CRAWL_TIMEOUT),
                min_chars=self.config.RAG_NEWS_ENRICH_MIN_CHARS,
                use_crawl4ai=self.config.RAG_NEWS_CRAWL4AI_ENABLED,
            ),
        )

        defillama = DefiLlamaClient(
            logger=self.logger, session=infra['session'], cache_dir='cache',
            update_interval_hours=self.config.RAG_DEFILLAMA_UPDATE_INTERVAL_HOURS
        )

        alternative_me = AlternativeMeAPI(logger=self.logger)
        await alternative_me.initialize()

        return {
            'coingecko': coingecko,
            'news': news_client,
            'market': CCXTMarketAPI(
                logger=self.logger,
                exchange_manager=infra['exchange_manager'],
            ),
            'defillama': defillama,
            'alternative_me': alternative_me
        }

    async def _provision_rag_layer(self, infra: dict, apis: dict, utils: dict) -> RagEngine:
        """Provision the RAG (Retrieval Augmented Generation) engine."""
        file_handler = RagFileHandler(logger=self.logger, config=self.config, unified_parser=utils['parser'])
        symbol_name_map = file_handler.load_symbol_name_map()

        article_processor = ArticleProcessor(
            logger=self.logger, unified_parser=utils['parser'],
            format_utils=utils['format_utils'],
            symbol_name_map=symbol_name_map,
        )
        news_manager = NewsManager(
            logger=self.logger, file_handler=file_handler, news_client=apis['news'],
            session=infra['session'], article_processor=article_processor
        )

        marker_fetcher = MarketDataFetcher(
            self.logger, apis['coingecko'], infra['exchange_manager'], apis['market'], apis['defillama']
        )
        market_processor = MarketDataProcessor(self.logger, utils['parser'])
        data_manager = MarketDataManager(
            self.logger, file_handler, apis['coingecko'], apis['market'],
            infra['exchange_manager'], unified_parser=utils['parser'],
            fetcher=marker_fetcher, processor=market_processor,
            cache=MarketDataCache(self.logger, file_handler),
            overview_builder=MarketOverviewBuilder(self.logger, market_processor)
        )

        category_processor = CategoryProcessor(self.logger, utils['collision_resolver'], file_handler)
        engine = RagEngine(
            logger=self.logger, token_counter=utils['token_counter'], config=self.config,
            coingecko_api=apis['coingecko'],
            file_handler=file_handler, news_manager=news_manager, market_data_manager=data_manager,
            index_manager=IndexManager(self.logger, article_processor),
            category_fetcher=LocalTaxonomyProvider(self.logger),
            category_processor=category_processor,
            ticker_manager=TickerManager(self.logger, file_handler, infra['exchange_manager']),
            context_builder=ContextBuilder(
                self.logger,
                utils['token_counter'],
                self.config,
                ArticleScoringPolicy(config=self.config),
                article_processor,
                symbol_name_map=symbol_name_map,
            )
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

    async def _provision_analyzer_layer(
        self, infra: dict, apis: dict, utils: dict, rag: RagEngine, models: dict
    ) -> dict:
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
            self.logger.warning("Pattern analyzer warm-up could not run: %s", warmup_error)

        ctx_builder = AnalyzerContextBuilder(
            self.config.TIMEFRAME, self.logger, utils['format_utils'],
            market_fmt, period_fmt, long_term_fmt, utils['timeframe_validator']
        )
        
        prompt_builder = PromptBuilder(
            self.config.TIMEFRAME, self.logger, tech_calc, self.config, utils['format_utils'],
            overview_fmt, long_term_fmt, TechnicalFormatter(tech_calc, self.logger, utils['format_utils']),
            market_fmt, utils['timeframe_validator'],
            TemplateManager(self.config, self.logger, utils['timeframe_validator']), ctx_builder
        )
        
        engine = AnalysisEngine(
            self.logger, rag, apis['coingecko'], models['manager'], apis['alternative_me'],
            apis['market'], self.config, tech_calc, pattern_analyzer, prompt_builder,
            MarketDataCollector(self.logger, rag, apis['alternative_me'], session=infra['session']),
            MarketMetricsCalculator(self.logger),
            AnalysisResultProcessor(models['manager'], self.logger, utils['parser']),
            ChartGenerator(
                self.logger, self.config, formatter=utils['format_utils'].fmt, format_utils=utils['format_utils']
            ),
            data_fetcher_factory=utils['data_fetcher_factory']
        )
        
        return {'engine': engine}

    def _provision_trading_layer(self, utils: dict) -> dict:
        """Provision trading strategy and memory services."""
        persistence = PersistenceManager(self.logger, data_dir="data/trading")
        risk_manager = RiskManager(self.logger, self.config)

        _configure_hf_hub_auth()

        # Calculate specialized brain path
        safe_symbol = self.config.CRYPTO_PAIR.replace("/", "_").replace("-", "_")
        brain_path = os.path.join(self.config.DATA_DIR, "trading", f"brain_{safe_symbol}_{self.config.TIMEFRAME}")

        # Create symbol-specific chroma client
        chroma_client = chromadb.PersistentClient(path=brain_path)

        # Auto-detect best hardware accelerator; log for observability
        embed_device = _get_best_device()
        self.logger.info("Embedding device: %s", embed_device)
        embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=embed_device)
        timeframe = TimeframeValidator.validate_and_normalize(self.config.TIMEFRAME)
        timeframe_minutes = TimeframeValidator.to_minutes(timeframe)

        # Inject chroma_client into VectorMemoryService
        vector_memory = VectorMemoryService(
            self.logger,
            chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=timeframe_minutes,
        )
        exit_execution_context = build_exit_execution_context_from_config(self.config, timeframe)
        tightening_policy = StopLossTighteningPolicy.from_config(self.config)

        brain_service = TradingBrainService(
            self.logger,
            persistence,
            vector_memory,
            exit_execution_context=exit_execution_context,
            timeframe_minutes=timeframe_minutes,
            tightening_policy=tightening_policy,
        )
        brain_service.refresh_semantic_rules_if_stale()
        
        memory_service = TradingMemoryService(
            self.logger,
            persistence,
            max_memory=10,
            vector_memory=vector_memory,
            initial_capital=self.config.DEMO_QUOTE_CAPITAL,
        )
        statistics_service = TradingStatisticsService(self.logger, persistence)
        exit_monitor = ExitMonitor(self.config, timeframe, POSITION_UPDATE_INTERVAL)
        exit_monitor.validate()
        
        from src.factories.position_factory import PositionFactory
        strategy = TradingStrategy(
            self.logger, persistence, brain_service, statistics_service, memory_service,
            risk_manager, self.config, PositionExtractor(self.logger, utils['parser']),
            PositionFactory(self.logger), tightening_policy=tightening_policy
        )
        
        return {
            'strategy': strategy,
            'persistence': persistence,
            'brain_service': brain_service,
            'memory_service': memory_service,
            'statistics_service': statistics_service,
            'exit_monitor': exit_monitor
        }

    async def _provision_notifiers(self, utils: dict) -> dict:
        """Provision notification services."""
        notifier = None
        task = None
        
        if self.config.DISCORD_BOT_ENABLED and self.config.BOT_TOKEN_DISCORD:
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
                self.logger.warning("Discord initialization failed: %s. Falling back to console output.", e)
                notifier = ConsoleNotifier(self.logger, self.config, utils['parser'], utils['format_utils'])
        else:
            notifier = ConsoleNotifier(self.logger, self.config, utils['parser'], utils['format_utils'])
            
        return {'notifier': notifier, 'task': task}
    
    async def run_async(self):
        """Async entry point for the application."""
        def _asyncio_exception_handler(loop, context):
            exc = context.get("exception")
            msg = context.get("message", "Unknown asyncio error")
            if exc is not None:
                if isinstance(exc, KeyboardInterrupt):
                    if self.shutdown_manager and self.shutdown_manager.is_shutting_down:
                        return
                    self.logger.debug("Asyncio task KeyboardInterrupt during shutdown: %s", msg)
                    return
                self.logger.error("Asyncio unhandled exception: %s", msg, exc_info=exc)
            else:
                self.logger.error("Asyncio error: %s", msg)

        if self.loop:
            self.loop.set_exception_handler(_asyncio_exception_handler)

        dependencies = await self.build_dependencies()

        # Extract dashboard_server before passing to bot (bot doesn't accept it)
        dashboard_server = dependencies.pop('dashboard_server', None)

        bot = CryptoTradingBot(
            logger=self.logger,
            config=self.config,
            shutdown_manager=self.shutdown_manager,
            **dependencies
        )
        bot.set_position_monitor(PositionStatusMonitor(
            logger=self.logger,
            config=self.config,
            persistence=dependencies['persistence'],
            trading_strategy=dependencies['trading_strategy'],
            exit_monitor=dependencies['exit_monitor'],
            notifier=dependencies['discord_notifier'],
            active_tasks=bot.active_tasks,
            is_running=lambda: bot.running,
            fetch_current_ticker=bot._fetch_current_ticker,
            interruptible_sleep=bot._interruptible_sleep,
            get_symbol=lambda: bot.current_symbol,
        ))

        try:
            await bot.initialize()
            symbol = self.config.CRYPTO_PAIR
            timeframe = self.config.TIMEFRAME

            # Track whether dashboard is currently running
            dashboard_running = False

            async def _toggle_dashboard():
                nonlocal dashboard_running
                if not dashboard_server:
                    return
                if dashboard_running:
                    self.logger.info("Dashboard: stopping (kill switch)...")
                    await dashboard_server.stop()
                    dashboard_running = False
                    self.logger.info("Dashboard stopped. Press 'd' to restart.")
                else:
                    self.logger.info("Dashboard: starting...")
                    await dashboard_server.start()
                    dashboard_running = True
                    self.logger.info("Dashboard live at http://localhost:%s", self.config.DASHBOARD_PORT)

            bot.keyboard_handler.register_command('d', _toggle_dashboard, "Toggle dashboard on/off")

            self.logger.info("Keyboard commands: 'a' = force analysis, 'd' = toggle dashboard, 'h' = help, 'q' = quit")

            # Auto-start dashboard if enabled in config (fire-and-forget task)
            if dashboard_server and self.config.DASHBOARD_ENABLED:
                await dashboard_server.start()
                dashboard_running = True
            elif not self.config.DASHBOARD_ENABLED:
                self.logger.info("Dashboard disabled (config). Press 'd' to start it.")

            # Bot runs in the foreground here; dashboard lifecycle is managed by _toggle_dashboard.
            await bot.run(symbol, timeframe)

        except asyncio.CancelledError:
            self.logger.info("Trading cancelled, shutting down...")
        finally:
            # Clean up dashboard server
            if dashboard_server:
                await dashboard_server.stop()
    
    def start(self):
        """Main entry point with clean shutdown delegation."""
        # Initialize lock manager
        single_instance_lock = SingleInstanceLock()
        
        if not single_instance_lock.acquire():
            shown = _show_error_dialog(
                "Crypto Trading Bot",
                "Another instance of Crypto Trading Bot is already running."
            )
            if not shown:
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
            while True:
                try:
                    self.loop.run_until_complete(self.run_async())
                    break
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt received.")
                    if GracefulShutdownManager.show_exit_confirmation():
                        self.loop.run_until_complete(self.shutdown_manager.shutdown_gracefully())
                        break
                    print("Shutdown cancelled. Continuing operation...")
                except Exception:
                    self.logger.exception("Unhandled exception in main loop — shutting down")
                    self.loop.run_until_complete(self.shutdown_manager.shutdown_gracefully())
                    break
        finally:
            # Give any remaining threads time to clean up before closing the loop
            # This prevents RuntimeError when Discord or other background threads try to access the closed loop
            try:
                if not self.loop.is_closed():
                    self.loop.close()
            except Exception as e:
                self.logger.error("Error closing event loop: %s", e)


if __name__ == "__main__":
    CompositionRoot().start()
