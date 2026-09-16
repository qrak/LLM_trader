"""
Provisioning stages for the bot's composition root.

Each stage builds one slice of the dependency graph (infrastructure, platforms, RAG,
models, analyzer, trading, notifiers, dashboard). They are mixed into CompositionRoot in
start.py, which owns the lifecycle (event loop, shutdown, entry point).
"""

import asyncio
import os
from pathlib import Path
from typing import Any

import aiohttp
import chromadb
from aiohttp_client_cache import SQLiteBackend
from sentence_transformers import SentenceTransformer

from src.analyzer import (
    AnalysisResultProcessor,
    MarketDataCollector,
    MarketMetricsCalculator,
    PatternAnalyzer,
    TechnicalCalculator,
    TechnicalFormatter,
)
from src.analyzer.analysis_engine import AnalysisEngine
from src.analyzer.formatters import (
    EVFrameworkFormatter,
    LongTermFormatter,
    MarketFormatter,
    MarketOverviewFormatter,
)
from src.analyzer.pattern_engine import ChartGenerator
from src.analyzer.pattern_engine.indicator_patterns import IndicatorPatternEngine
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.prompts.template_manager import TemplateManager
from src.analyzer.trend_validator import TrendValidator
from src.app import POSITION_UPDATE_INTERVAL
from src.composition.startup_support import (
    cleanup_legacy_embedding_cache,
    configure_hf_hub_auth,
    get_best_device,
    print_summary_table,
)
from src.dashboard.routers.ws_router import ConnectionManager
from src.dashboard.server import DashboardServer
from src.logger.logger import Logger
from src.managers.model_manager import (
    ModelManager,
    ProviderClients,
    ProviderOrchestrator,
)
from src.managers.persistence_manager import PersistenceManager
from src.managers.post_mortem_repository import PostMortemRepository
from src.managers.risk_manager import RiskManager
from src.notifiers import ConsoleNotifier, DiscordNotifier
from src.parsing.unified_parser import UnifiedParser
from src.platforms.ai_providers import (
    BlockRunClient,
    DeepSeekClient,
    GoogleAIClient,
    LMStudioClient,
    OpenRouterClient,
)
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.ccxt_market_api import CCXTMarketAPI
from src.platforms.coingecko import CoinGeckoAPI
from src.platforms.defillama import DefiLlamaClient
from src.platforms.exchange_manager import ExchangeManager
from src.rag import (
    CategoryProcessor,
    ContextBuilder,
    IndexManager,
    MarketDataManager,
    NewsManager,
    RagEngine,
    RagFileHandler,
    TickerManager,
)
from src.rag.article_processor import ArticleProcessor
from src.rag.collision_resolver import CategoryCollisionResolver
from src.rag.local_taxonomy import LocalTaxonomyProvider
from src.rag.market_components import (
    MarketDataCache,
    MarketDataFetcher,
    MarketDataProcessor,
    MarketOverviewBuilder,
)
from src.rag.news_ingestion import Crawl4AIEnricher, RSSCrawl4AINewsProvider
from src.rag.scoring_policy import ArticleScoringPolicy
from src.trading import (
    ExitMonitor,
    MarketConditionsExtractor,
    PositionExtractor,
    TradingBrainService,
    TradingMemoryService,
    TradingStatisticsService,
    TradingStrategy,
)
from src.trading.guards.configured_symbol import ConfiguredSymbolGuard
from src.trading.guards.cooldown_window import CooldownWindowGuard
from src.trading.guards.max_position_size import MaxPositionSizeGuard
from src.trading.guards.pipeline import GuardPipeline
from src.trading.post_mortem import PostMortemService
from src.trading.stop_loss_tightening_policy import StopLossTighteningPolicy
from src.trading.vector_memory import VectorMemoryService
from src.utils.format_utils import FormatUtils
from src.utils.indicator_classifier import build_exit_execution_context_from_config

# pylint: disable=wrong-import-position
from src.utils.journal_rotator import JournalRotator
from src.utils.keyboard_handler import KeyboardHandler
from src.utils.timeframe_validator import TimeframeValidator
from src.utils.token_counter import CostStorage, ModelPricing, TokenCounter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProvisioningMixin:
    """Builds and wires every service slice; mixed into CompositionRoot."""

    config: Any
    console: Any
    logger: Logger

    def _init_directories(self):
        """Ensure all required directories exist."""
        data_dir = self.config.DATA_DIR
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "news_cache"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "trading"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "charts"), exist_ok=True)

        safe_symbol = self.config.CRYPTO_PAIR.replace("/", "_").replace("-", "_")
        brain_dir = os.path.join(
            data_dir, "trading", f"brain_{safe_symbol}_{self.config.TIMEFRAME}"
        )
        os.makedirs(brain_dir, exist_ok=True)

    async def _run_maintenance_tasks(self, brain_service: TradingBrainService) -> None:
        """Run post-provisioning maintenance: journal rotation."""
        try:
            rotated_count = JournalRotator().rotate_all_journals()
            if rotated_count > 0:
                self.logger.info(
                    "  -> Journal maintenance: rotated %d journal file(s) to .ai/archive/",
                    rotated_count,
                )
            else:
                self.logger.info(
                    "  -> Journal maintenance: all journal files within size limits"
                )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Journal rotation maintenance skipped: %s", e)

    async def _provision_infrastructure(self) -> dict:
        """Provision base infrastructure components."""
        exchange_manager = ExchangeManager(logger=self.logger, config=self.config)
        await exchange_manager.initialize()

        session = aiohttp.ClientSession()
        keyboard_handler = KeyboardHandler(logger=self.logger)

        return {
            "exchange_manager": exchange_manager,
            "session": session,
            "keyboard_handler": keyboard_handler,
        }

    def _provision_utilities(self) -> dict:
        """Provision utility singletons."""
        format_utils = FormatUtils()
        parser = UnifiedParser(self.logger, format_utils=format_utils)
        token_counter = TokenCounter()
        timeframe_validator = TimeframeValidator()
        collision_resolver = CategoryCollisionResolver()

        return {
            "format_utils": format_utils,
            "parser": parser,
            "token_counter": token_counter,
            "timeframe_validator": timeframe_validator,
            "collision_resolver": collision_resolver,
        }

    async def _provision_platforms(self, infra: dict) -> dict:
        """Provision external API clients."""
        self.logger.info(
            "  -> Fetching CoinGecko coin catalog & initializing market APIs..."
        )
        coingecko_cache_ttl_seconds = int(
            self.config.RAG_COINGECKO_UPDATE_INTERVAL_HOURS * 3600
        )
        coingecko_backend = SQLiteBackend(
            cache_name="cache/coingecko_cache.db",
            expire_after=coingecko_cache_ttl_seconds,
        )

        coingecko = CoinGeckoAPI(
            logger=self.logger,
            cache_backend=coingecko_backend,
            cache_dir="data/market_data",
            api_key=self.config.COINGECKO_API_KEY,
            update_interval_hours=24,
            global_api_url=self.config.RAG_COINGECKO_GLOBAL_API_URL,
        )
        await coingecko.initialize()
        self.logger.info(
            "  -> CoinGecko API ready (%d unique symbols mapped)",
            len(coingecko.symbol_to_id_map),
        )

        news_client = RSSCrawl4AINewsProvider(
            self.logger,
            self.config,
            enricher=Crawl4AIEnricher(
                logger=self.logger,
                config=self.config,
            ),
        )

        defillama = DefiLlamaClient(
            logger=self.logger,
            session=infra["session"],
            cache_dir="cache",
            update_interval_hours=self.config.RAG_DEFILLAMA_UPDATE_INTERVAL_HOURS,
        )

        alternative_me = AlternativeMeAPI(logger=self.logger)
        await alternative_me.initialize()

        return {
            "coingecko": coingecko,
            "news": news_client,
            "market": CCXTMarketAPI(
                logger=self.logger,
                exchange_manager=infra["exchange_manager"],
            ),
            "defillama": defillama,
            "alternative_me": alternative_me,
        }

    async def _provision_rag_layer(
        self, infra: dict, apis: dict, utils: dict
    ) -> RagEngine:
        """Provision the RAG (Retrieval Augmented Generation) engine."""
        self.logger.info(
            "  -> Loading news cache, taxonomy & building RAG search index..."
        )
        file_handler = RagFileHandler(
            logger=self.logger, config=self.config, unified_parser=utils["parser"]
        )
        symbol_name_map = file_handler.load_symbol_name_map()

        article_processor = ArticleProcessor(
            logger=self.logger,
            unified_parser=utils["parser"],
            format_utils=utils["format_utils"],
            symbol_name_map=symbol_name_map,
        )
        news_manager = NewsManager(
            logger=self.logger,
            file_handler=file_handler,
            news_client=apis["news"],
            session=infra["session"],
            article_processor=article_processor,
        )

        marker_fetcher = MarketDataFetcher(
            self.logger,
            apis["coingecko"],
            infra["exchange_manager"],
            apis["market"],
            apis["defillama"],
        )
        market_processor = MarketDataProcessor(self.logger, utils["parser"])
        data_manager = MarketDataManager(
            self.logger,
            file_handler,
            apis["coingecko"],
            apis["market"],
            infra["exchange_manager"],
            unified_parser=utils["parser"],
            fetcher=marker_fetcher,
            processor=market_processor,
            cache=MarketDataCache(self.logger, file_handler),
            overview_builder=MarketOverviewBuilder(self.logger, market_processor),
        )

        category_processor = CategoryProcessor(
            self.logger, utils["collision_resolver"], utils["parser"], file_handler
        )
        engine = RagEngine(
            logger=self.logger,
            config=self.config,
            coingecko_api=apis["coingecko"],
            news_manager=news_manager,
            market_data_manager=data_manager,
            index_manager=IndexManager(self.logger, article_processor),
            category_fetcher=LocalTaxonomyProvider(self.logger),
            category_processor=category_processor,
            ticker_manager=TickerManager(
                self.logger, file_handler, infra["exchange_manager"]
            ),
            context_builder=ContextBuilder(
                self.logger,
                utils["token_counter"],
                self.config,
                ArticleScoringPolicy(config=self.config),
                article_processor,
                symbol_name_map=symbol_name_map,
            ),
        )
        await engine.initialize()
        db_size = engine.news_manager.get_database_size() if engine.news_manager is not None else 0
        self.logger.info(
            "  -> RAG engine ready (%d news articles indexed)",
            db_size,
        )
        return engine

    def _provision_model_layer(self, utils: dict) -> dict:
        """Provision AI model managers and providers."""
        google_client: GoogleAIClient | None = None
        google_paid_client: GoogleAIClient | None = None
        if self.config.GOOGLE_STUDIO_API_KEY:
            google_client = GoogleAIClient(
                api_key=self.config.GOOGLE_STUDIO_API_KEY,
                model=self.config.GOOGLE_STUDIO_MODEL,
                logger=self.logger,
            )
            self.logger.debug("Google AI client initialized")
            if self.config.GOOGLE_STUDIO_PAID_API_KEY:
                google_paid_client = GoogleAIClient(
                    api_key=self.config.GOOGLE_STUDIO_PAID_API_KEY,
                    model=self.config.GOOGLE_STUDIO_MODEL,
                    logger=self.logger,
                )
                self.logger.debug(
                    "Google AI paid client initialized as fallback for overloaded free tier"
                )
        openrouter_client: OpenRouterClient | None = None
        if self.config.OPENROUTER_API_KEY:
            openrouter_client = OpenRouterClient(
                api_key=self.config.OPENROUTER_API_KEY,
                base_url=self.config.OPENROUTER_BASE_URL,
                logger=self.logger,
            )
            self.logger.debug("OpenRouter client initialized")
        deepseek_client: DeepSeekClient | None = None
        if self.config.DEEPSEEK_API_KEY:
            deepseek_client = DeepSeekClient(
                api_key=self.config.DEEPSEEK_API_KEY,
                base_url=self.config.DEEPSEEK_BASE_URL,
                logger=self.logger,
            )
            self.logger.debug("DeepSeek client initialized")
        lmstudio_client: LMStudioClient | None = None
        if self.config.LM_STUDIO_BASE_URL:
            lmstudio_client = LMStudioClient(
                base_url=self.config.LM_STUDIO_BASE_URL,
                logger=self.logger,
            )
            self.logger.debug(
                "LM Studio client initialized for URL: %s",
                self.config.LM_STUDIO_BASE_URL,
            )
        blockrun_client: BlockRunClient | None = None
        if self.config.BLOCKRUN_WALLET_KEY:
            blockrun_client = BlockRunClient(
                wallet_key=self.config.BLOCKRUN_WALLET_KEY,
                base_url=self.config.BLOCKRUN_BASE_URL,
                logger=self.logger,
            )
            self.logger.debug("BlockRun client initialized")
        provider_clients = ProviderClients(
            google=google_client,
            google_paid=google_paid_client,
            openrouter=openrouter_client,
            lmstudio=lmstudio_client,
            blockrun=blockrun_client,
            deepseek=deepseek_client,
        )
        orchestrator = ProviderOrchestrator(self.logger, self.config, provider_clients)

        manager = ModelManager(
            logger=self.logger,
            config=self.config,
            unified_parser=utils["parser"],
            token_counter=utils["token_counter"],
            cost_storage=CostStorage(),
            model_pricing=ModelPricing(),
            orchestrator=orchestrator,
            provider_clients=provider_clients,
        )
        primary_provider = self.config.PROVIDER
        self.logger.info(
            "  -> AI Provider fallback chain ready (Primary provider: %s)",
            primary_provider,
        )

        return {"manager": manager}

    async def _provision_analyzer_layer(
        self, infra: dict, apis: dict, utils: dict, rag: RagEngine, models: dict
    ) -> dict:
        """Provision the market analysis engine."""
        overview_fmt = MarketOverviewFormatter(self.logger, utils["format_utils"])
        long_term_fmt = LongTermFormatter(self.logger, utils["format_utils"])

        market_fmt = MarketFormatter(
            self.logger,
            utils["format_utils"],
            self.config,
            utils["token_counter"],
            overview_fmt,
            long_term_fmt,
        )

        tech_calc = TechnicalCalculator(self.logger, utils["format_utils"])
        pattern_analyzer = PatternAnalyzer(
            indicator_pattern_engine=IndicatorPatternEngine(), logger=self.logger
        )
        try:
            self.logger.info(
                "  -> Warming up Numba JIT pattern engine (compiling 50+ indicator kernels)..."
            )
            pattern_analyzer.warmup()
            self.logger.info("  -> Numba JIT pattern engine compiled & warm")
        except Exception as warmup_error:  # noqa: BLE001
            self.logger.warning(
                "Pattern analyzer warm-up could not run: %s", warmup_error
            )

        ev_fmt = EVFrameworkFormatter(self.config)

        prompt_builder = PromptBuilder(
            self.config.TIMEFRAME,
            self.logger,
            self.config,
            utils["format_utils"],
            overview_fmt,
            long_term_fmt,
            TechnicalFormatter(tech_calc, self.logger, utils["format_utils"]),
            market_fmt,
            ev_formatter=ev_fmt,
            timeframe_validator=utils["timeframe_validator"],
            template_manager=TemplateManager(self.config, self.logger, utils["timeframe_validator"]),
        )

        engine = AnalysisEngine(
            self.logger,
            rag,
            models["manager"],
            apis["market"],
            self.config,
            tech_calc,
            pattern_analyzer,
            prompt_builder,
            MarketDataCollector(
                self.logger, rag, apis["alternative_me"], session=infra["session"]
            ),
            MarketMetricsCalculator(self.logger),
            AnalysisResultProcessor(
                models["manager"],
                self.logger,
                utils["parser"],
                TrendValidator(),
                PatternQualityScorer(),
            ),
            ChartGenerator(
                self.logger,
                self.config,
                formatter=utils["format_utils"].fmt,
                format_utils=utils["format_utils"],
            ),
        )

        return {"engine": engine, "ev_formatter": ev_fmt}

    def _provision_trading_layer(self, utils: dict, models: dict) -> dict:
        """Provision trading strategy and memory services."""
        persistence = PersistenceManager(self.logger, data_dir="data/trading")

        trade_db_path = os.path.join(
            self.config.DATA_DIR, "trading", "trade_history.db"
        )
        post_mortem_repo = PostMortemRepository(
            logger=self.logger, db_path=trade_db_path
        )

        risk_manager = RiskManager(self.logger, self.config)

        configure_hf_hub_auth()
        cleanup_legacy_embedding_cache(self.logger)

        safe_symbol = self.config.CRYPTO_PAIR.replace("/", "_").replace("-", "_")
        brain_path = os.path.join(
            self.config.DATA_DIR,
            "trading",
            f"brain_{safe_symbol}_{self.config.TIMEFRAME}",
        )

        self.logger.info(
            "  -> Connecting to ChromaDB vector memory at %s...", brain_path
        )
        chroma_client = chromadb.PersistentClient(path=brain_path)

        embed_device = get_best_device()
        self.logger.info(
            "  -> Loading SentenceTransformer embedding model ('BAAI/bge-base-en-v1.5') on %s...",
            embed_device,
        )
        embedding_model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5", device=embed_device
        )
        self.logger.info("  -> SentenceTransformer embedding model loaded successfully")
        timeframe = TimeframeValidator.validate_and_normalize(self.config.TIMEFRAME)
        timeframe_minutes = TimeframeValidator.to_minutes(timeframe)

        vector_memory = VectorMemoryService(
            self.logger,
            chroma_client,
            embedding_model=embedding_model,
            timeframe_minutes=timeframe_minutes,
        )
        exit_execution_context = build_exit_execution_context_from_config(
            self.config, timeframe
        )
        tightening_policy = StopLossTighteningPolicy.from_config(self.config)

        brain_service = TradingBrainService(
            self.logger,
            persistence,
            vector_memory,
            exit_execution_context=exit_execution_context,
            timeframe_minutes=timeframe_minutes,
            tightening_policy=tightening_policy,
        )

        try:
            prune_results = vector_memory.prune_aged_documents()
            for coll_name, count in prune_results.items():
                if count > 0:
                    self.logger.info(
                        "ChromaDB maintenance: removed %d documents from %s",
                        count,
                        coll_name,
                    )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("ChromaDB startup maintenance failed: %s", e)

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
        guard_pipeline = GuardPipeline(
            [
                ConfiguredSymbolGuard(),
                MaxPositionSizeGuard(),
                CooldownWindowGuard(persistence=persistence),
            ]
        )
        self.logger.info(
            "Order guard pipeline active: %s", ", ".join(guard_pipeline.guard_names)
        )

        post_mortem_service = PostMortemService(
            logger=self.logger,
            model_manager=models["manager"],
            unified_parser=utils["parser"],
            repository=post_mortem_repo,
        )

        strategy = TradingStrategy(
            self.logger,
            persistence,
            brain_service,
            statistics_service,
            memory_service,
            risk_manager,
            self.config,
            PositionExtractor(),
            conditions_extractor=MarketConditionsExtractor(self.logger),
            tightening_policy=tightening_policy,
            guard_pipeline=guard_pipeline,
            post_mortem_service=post_mortem_service,
        )

        return {
            "strategy": strategy,
            "persistence": persistence,
            "brain_service": brain_service,
            "memory_service": memory_service,
            "statistics_service": statistics_service,
            "exit_monitor": exit_monitor,
            "post_mortem_repo": post_mortem_repo,
        }

    async def _provision_notifiers(self, utils: dict) -> dict:
        """Provision notification services."""
        notifier = None
        task = None

        if self.config.DISCORD_BOT_ENABLED and self.config.BOT_TOKEN_DISCORD:
            try:
                import discord

                from src.notifiers.filehandler import DiscordFileHandler

                intents = discord.Intents.default()
                intents.message_content = False
                intents.reactions = False
                intents.typing = False
                intents.presences = False

                bot = discord.Client(intents=intents)

                file_handler = DiscordFileHandler(
                    bot=bot,
                    logger=self.logger,
                    config=self.config,
                    tracking_file="data/tracked_messages.json",
                    cleanup_interval=7200,
                )

                notifier = DiscordNotifier(
                    self.logger,
                    self.config,
                    utils["parser"],
                    utils["format_utils"],
                    bot,
                    file_handler,
                )

                task = asyncio.create_task(notifier.start())
                await notifier.wait_until_ready()
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    "Discord initialization failed: %s. Falling back to console output.",
                    e,
                )
                notifier = ConsoleNotifier(
                    self.logger, self.config, utils["parser"], utils["format_utils"]
                )
        else:
            notifier = ConsoleNotifier(
                self.logger, self.config, utils["parser"], utils["format_utils"]
            )

        return {"notifier": notifier, "task": task}

    def _display_startup_summary(
        self, apis: dict, rag: RagEngine, trading: dict, init_duration: float
    ) -> None:
        """Display the initialization summary panel and control footer."""
        symbols_count = (
            len(apis["coingecko"].symbol_to_id_map)
            if apis.get("coingecko") and hasattr(apis["coingecko"], "symbol_to_id_map")
            else 0
        )
        news_count = (
            rag.news_manager.get_database_size()  # type: ignore[reportOptionalMemberAccess]
            if rag and hasattr(rag, "news_manager")
            else 0
        )
        guards_count = (
            len(trading["strategy"].guard_pipeline.guard_names)
            if trading.get("strategy")
            and hasattr(trading["strategy"], "guard_pipeline")
            else 0
        )

        summary_stats = {
            "Symbols mapped": f"{symbols_count:,}",
            "News articles indexed": f"{news_count:,}",
            "Primary AI provider": str(self.config.PROVIDER),
            "Vector memory active": "ChromaDB (bge-base-en-v1.5)",
            "Order guard rules": f"{guards_count}",
            "Trading pair / TF": f"{self.config.CRYPTO_PAIR} ({self.config.TIMEFRAME})",
        }
        print_summary_table(self.console, init_duration, summary_stats)

        dashboard_url = f"http://localhost:{self.config.DASHBOARD_PORT}"
        self.console.print()
        self.console.print(
            "  [dim]Keyboard commands:[/] [bold]'a'[/] force analysis  "
            "[bold]'d'[/] toggle dashboard  [bold]'h'[/] help  [bold]'q'[/] quit  "
            "[bold]'Shift+R'[/] reload",
        )
        self.console.print(
            f"  [bold green]Dashboard →[/] [link={dashboard_url}]{dashboard_url}[/]"
        )
        self.console.print()

    def _provision_dashboard_layer(
        self, infra: dict, utils: dict, analyzer: dict, trading: dict
    ) -> dict:
        """Provision dashboard server and admin interface."""
        force_analysis_event = asyncio.Event()
        connection_manager = ConnectionManager()

        config_path = str(PROJECT_ROOT / "config" / "config.ini")
        admin_credentials = {
            "username": self.config.ADMIN_USERNAME,
            "password_hash": self.config.ADMIN_PASSWORD_HASH,
            "signing_key": self.config.ADMIN_SIGNING_KEY,
        }
        dashboard_server = DashboardServer(
            brain_service=trading["brain_service"],
            vector_memory=trading["brain_service"].vector_memory
            if trading["brain_service"]
            else None,
            analysis_engine=analyzer["engine"],
            config=self.config,
            logger=self.logger,
            unified_parser=utils["parser"],
            persistence=trading["persistence"],
            exchange_manager=infra["exchange_manager"],
            host=self.config.DASHBOARD_HOST,
            port=self.config.DASHBOARD_PORT,
            force_analysis_event=force_analysis_event,
            config_path=config_path,
            admin_credentials=admin_credentials,
            post_mortem_repo=trading.get("post_mortem_repo"),
            connection_manager=connection_manager,
        )
        trading["strategy"].set_dashboard_state(dashboard_server.dashboard_state)

        return {
            "dashboard_server": dashboard_server,
            "dashboard_state": dashboard_server.dashboard_state,
            "force_analysis_event": force_analysis_event,
        }

