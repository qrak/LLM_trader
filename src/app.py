"""Main entry point for the Crypto Trading Bot application.

This module defines the `CryptoTradingBot` class, which orchestrates the interaction
between various components like the market analyzer, trading strategy, and external APIs.
"""
import asyncio
import io
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from src.logger.logger import Logger
from src.utils.timeframe_validator import TimeframeValidator
from src.managers.persistence_manager import PersistenceManager
from src.contracts.model_contract import ModelManagerProtocol
from src.trading import (
    ExitMonitor,
    PositionStatusMonitor,
    TradingBrainService,
    TradingStatisticsService,
    TradingMemoryService
)


# Configuration Constants
POSITION_UPDATE_INTERVAL = 3600  # 1 hour
SLEEP_CHUNK_SIZE = 1.0  # Check for interruptions every second
CANDLE_BUFFER_SECONDS = 2  # Seconds to wait after candle start
ERROR_WAIT_SHORT = 60   # Seconds to wait after minor error
ERROR_WAIT_LONG = 300   # Seconds to wait after major error


class CryptoTradingBot:
    """Automated crypto trading bot - TRADING MODE ONLY."""

<<<<<<< HEAD
=======
    @staticmethod
    def _format_utc_and_local(dt: datetime, local_tz=None) -> str:
        """Format a timestamp with both UTC and local time for operator-facing logs."""
        utc_dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        local_dt = utc_dt.astimezone(local_tz)
        local_zone = local_dt.tzname() or "local"
        return (
            f"{utc_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC "
            f"/ {local_dt.strftime('%Y-%m-%d %H:%M:%S')} {local_zone}"
        )

>>>>>>> main
    def __init__(
        self,
        logger: Logger,
        config,
        shutdown_manager: Optional[Any],
        exchange_manager,
        market_analyzer,
        trading_strategy,
        discord_notifier,
        keyboard_handler,
        rag_engine,
        coingecko_api,
<<<<<<< HEAD
        news_client,
=======
>>>>>>> main
        market_api,
        alternative_me_api,
        http_session,
        persistence: PersistenceManager,
        model_manager: ModelManagerProtocol,
        brain_service: TradingBrainService,
        statistics_service: TradingStatisticsService,
        memory_service: TradingMemoryService,
        exit_monitor: ExitMonitor,
        dashboard_state = None,
        discord_task: Optional[asyncio.Task] = None
    ):
        # pylint: disable=too-many-arguments, too-many-locals
        # Reason: Dependency Injection pattern requires all components to be injected.
        """Initialize bot with all dependencies injected.

        All components are injected via constructor following the Dependency
        Injection pattern, with start.py acting as the composition root.
        """
        self.logger = logger
        self.config = config
        self.shutdown_manager = shutdown_manager

        # Injected core components
        self.exchange_manager = exchange_manager
        self.market_analyzer = market_analyzer
        self.trading_strategy = trading_strategy
        self.discord_notifier = discord_notifier
        self.keyboard_handler = keyboard_handler
        self.rag_engine = rag_engine

        # Injected API clients
        self.coingecko_api = coingecko_api
<<<<<<< HEAD
        self.news_client = news_client
=======
>>>>>>> main
        self.market_api = market_api
        self.alternative_me_api = alternative_me_api
<<<<<<< HEAD
        self.cryptocompare_session = cryptocompare_session
=======
        self.http_session = http_session
>>>>>>> main

        # Injected trading services
        self.persistence = persistence
        self.model_manager = model_manager
        self.brain_service = brain_service
        self.statistics_service = statistics_service
        self.memory_service = memory_service
        self.exit_monitor = exit_monitor
        self.dashboard_state = dashboard_state

        # Runtime state
        self.tasks = []
        self.running = False
        self._active_tasks = set()
        self._force_analysis = asyncio.Event()
        self._discord_task = discord_task
<<<<<<< HEAD
        self._position_status_task: Optional[asyncio.Task] = None
=======
        self.position_monitor: Optional[PositionStatusMonitor] = None
>>>>>>> main

        # Trading state
        self.current_exchange = None
        self.current_symbol: Optional[str] = None
        self.current_timeframe: Optional[str] = None

    @property
    def active_tasks(self) -> set[asyncio.Task]:
        """Return bot-managed background tasks for composition-root wiring."""
        return self._active_tasks

    def set_position_monitor(self, position_monitor: PositionStatusMonitor) -> None:
        """Inject the position monitor after bot-bound callbacks are available."""
        self.position_monitor = position_monitor

    def _require_position_monitor(self) -> PositionStatusMonitor:
        if self.position_monitor is None:
            raise RuntimeError("Position monitor dependency is not configured")
        return self.position_monitor

    async def initialize(self):
        """Initialize all components."""
        if self.shutdown_manager:
            self.shutdown_manager.register_shutdown_callback(self.shutdown)

            # Register components for shutdown
            if self.keyboard_handler:
                self.shutdown_manager.register_shutdown_callback(self.keyboard_handler.stop_listening)

            if self.model_manager:
                self.shutdown_manager.register_shutdown_callback(self.model_manager.close)

            if self.market_analyzer:
                self.shutdown_manager.register_shutdown_callback(self.market_analyzer.close)

            if self.rag_engine:
                self.shutdown_manager.register_shutdown_callback(self.rag_engine.close)

            if self.exchange_manager:
                self.shutdown_manager.register_shutdown_callback(self.exchange_manager.shutdown)

<<<<<<< HEAD
            if self.cryptocompare_session:
                self.shutdown_manager.register_shutdown_callback(self.cryptocompare_session.close)

            # API clients (if they have close method)
            for client in [self.alternative_me_api, self.coingecko_api, self.news_client, self.market_api, self.categories_api]:
=======
            if self.http_session:
                self.shutdown_manager.register_shutdown_callback(self.http_session.close)

            # API clients (if they have close method)
            for client in [self.alternative_me_api, self.coingecko_api, self.market_api]:
>>>>>>> main
                if client:
                    try:
                        self.shutdown_manager.register_shutdown_callback(client.close)
                    except AttributeError:
                        pass

        # Register keyboard commands
        self.keyboard_handler.register_command('a', self._force_analysis_now, "Force immediate analysis")
        self.keyboard_handler.register_command('h', self._show_help, "Show available keyboard commands")
        self.keyboard_handler.register_command('q', self._request_shutdown, "Quit the application")

        # Start keyboard handler task
        keyboard_task = asyncio.create_task(
            self.keyboard_handler.start_listening(),
            name="Keyboard-Handler"
        )
        self._active_tasks.add(keyboard_task)
        keyboard_task.add_done_callback(self._active_tasks.discard)
        self.tasks.append(keyboard_task)

        self.logger.info("Crypto Trading Bot ready")
    async def shutdown(self):
        """Callback for graceful shutdown."""
        self.logger.info("Signaling trading loops to stop...")
        self.running = False

        # Cancel active tasks managed by bot
        pending_tasks = list(self._active_tasks)
        if pending_tasks:
            self.logger.info("Cancelling %s bot-specific tasks...", len(pending_tasks))
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.wait(pending_tasks, timeout=3.0)
            except asyncio.TimeoutError:
                self.logger.warning("Bot tasks shutdown timed out")

        # Discord notifier cleanup (properly closes the bot)
        if self.discord_notifier:
            try:
<<<<<<< HEAD
                await asyncio.wait_for(self._discord_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
=======
                await self.discord_notifier.shutdown()
            except Exception as e:
                self.logger.warning("Error shutting down Discord notifier: %s", e)

        # Let the Discord start task exit naturally after bot.close().
        if self._discord_task and not self._discord_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._discord_task), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Discord task did not exit after bot shutdown; cancelling it")
                self._discord_task.cancel()
                try:
                    await self._discord_task
                except asyncio.CancelledError:
                    pass
>>>>>>> main

        self.logger.info("Bot shutdown signaling complete.")


    async def run(self, symbol: str, timeframe: Optional[str] = None):
        """Run the trading bot in continuous mode.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Optional timeframe override
        """
        self.current_symbol = symbol
<<<<<<< HEAD
        self.current_timeframe = timeframe or self.config.TIMEFRAME
=======
        requested_timeframe = timeframe or self.config.TIMEFRAME
        try:
            self.current_timeframe = TimeframeValidator.validate_and_normalize(requested_timeframe)
            self.exit_monitor.timeframe = self.current_timeframe
            self.exit_monitor.validate()
        except ValueError as e:
            self.logger.error("Invalid timeframe or exit monitoring config '%s': %s", requested_timeframe, e)
            return
>>>>>>> main

        # Find exchange that supports the symbol
        exchange, exchange_id = await self.exchange_manager.find_symbol_exchange(symbol)
        if not exchange:
            self.logger.error("Symbol %s not found on any configured exchange", symbol)
            return

        self.current_exchange = exchange
        self.logger.info("Starting trading for %s on %s", symbol, exchange_id)
        self.logger.info("Timeframe: %s", self.current_timeframe)

        # Initialize analyzer for this symbol
        self.market_analyzer.initialize_for_symbol(
            symbol=symbol,
            exchange=exchange,
            timeframe=self.current_timeframe
        )

        # Enable running state before starting any async loops
        self.running = True
        check_count = 0

        # Log current position if any
        if self.trading_strategy.current_position:
            position = self.trading_strategy.current_position
            self.logger.info("Existing position: %s @ $%s", position.direction, f"{position.entry_price:,.2f}")
<<<<<<< HEAD
            # Start hourly position updates for existing position
            if self.discord_notifier:
                await self._start_position_status_updates()
=======
            await self._require_position_monitor().start()
>>>>>>> main
        else:
            self.logger.info("No existing position")

        # Fetch initial price for dashboard (one-time startup call)
        await self._fetch_current_ticker()

        # Check if resuming from previous session (regardless of position status)
        last_analysis_time = self.persistence.get_last_analysis_time()
        if last_analysis_time:
<<<<<<< HEAD
            self.logger.info("Resuming from last analysis at %s UTC", last_analysis_time.strftime('%Y-%m-%d %H:%M:%S'))
=======
            self.logger.info(
                "Resuming from last analysis at %s",
                self._format_utc_and_local(last_analysis_time)
            )
>>>>>>> main
            await self._wait_until_next_timeframe_after(last_analysis_time)
        self.logger.info("Ready for next analysis after wait")

        # Initial run is considered regular (unless we want to skipping update on restart, but safer to update)
        is_regular_run = True

        while self.running:
            try:
                check_count += 1
                await self._execute_trading_check(check_count, force_news_update=is_regular_run, is_candle_close=is_regular_run)

                # Check if still running before waiting
                if not self.running:
                    break

                # Wait for next timeframe
                # Returns True if forced (interrupted), False if waited full duration (regular)
                was_forced_wait = await self._wait_for_next_timeframe()
                is_regular_run = not was_forced_wait

            except asyncio.CancelledError:
                self.logger.info("Trading cancelled")
                self.running = False
                break
            except Exception as e:
                self.logger.error("Error in trading loop: %s", e)
                await self._interruptible_sleep(ERROR_WAIT_SHORT)

    async def _execute_trading_check(self, check_count: int, force_news_update: bool = True, is_candle_close: bool = True):
        """Execute a single trading check iteration."""
        self._log_check_header(check_count)
        
        current_ticker, current_price = await self._fetch_ticker_data()
        await self._check_position_status(current_price, is_candle_close=is_candle_close)
        await self._execute_market_knowledge_update(force_news_update)
        
        self.logger.info("Running market analysis...")
        context_data = await self._build_analysis_context(current_price, current_ticker)
        result = await self.market_analyzer.analyze_market(**context_data)
        
        if "error" in result:
            self.logger.error("Analysis failed: %s", result['error'])
            return
        
        self.persistence.save_last_analysis_time()
        decision = await self.trading_strategy.process_analysis(result, self.current_symbol)
        
        if decision:
            await self.discord_notifier.send_trading_decision(decision, self.config.MAIN_CHANNEL_ID)
            await self._handle_new_position(decision, current_price)
        else:
            self.logger.info("No trading action taken")

        await self._send_discord_notification(result)
        self._save_analysis_data(result)
    
    def _log_check_header(self, check_count: int):
        """Log trading check header"""
        current_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("Trading Check #%s at %s", check_count, current_time.strftime('%Y-%m-%d %H:%M:%S'))
        self.logger.info("=" * 60)
    
    async def _fetch_ticker_data(self):
        """Fetch current ticker and price"""
        try:
            current_ticker = await self._fetch_current_ticker()
            if current_ticker:
                current_price = float(current_ticker.get('last', current_ticker.get('close', 0)))
                return current_ticker, current_price
        except Exception as e:
            self.logger.warning("Could not fetch current ticker: %s", e)
        return None, None
    
    async def _check_position_status(self, current_price: Optional[float], *, is_candle_close: bool = True):
        """Check if existing position hit stop/target.

<<<<<<< HEAD
        Soft stop mode: SL/TP evaluation only runs on candle close.
        Forced analysis (keyboard 'a') skips automated stop checks
        but the AI can still consciously signal CLOSE.
        """
        if not (self.trading_strategy.current_position and current_price is not None):
            return

        if not is_candle_close:
            self.logger.info("Intra-candle check: skipping SL/TP evaluation (soft stop mode)")
            return

        try:
            close_reason = await self.trading_strategy.check_position(current_price)
            if close_reason:
                self.logger.info("Position closed: %s", close_reason)
                await self._stop_position_status_updates()
                if self.discord_notifier:
                    history = self.persistence.load_trade_history()
                    await self.discord_notifier.send_performance_stats(
                        trade_history=history,
                        symbol=self.current_symbol,
                        channel_id=self.config.MAIN_CHANNEL_ID
                    )
        except Exception as e:
            self.logger.error("Error checking position: %s", e)
=======
        Soft exit mode: SL/TP evaluation only runs on candle close.
        Forced analysis (keyboard 'a') skips automated stop checks
        but the AI can still consciously signal CLOSE.
        """
        await self._require_position_monitor().check_soft_exit_status(current_price, is_candle_close=is_candle_close)
>>>>>>> main
    
    async def _execute_market_knowledge_update(self, force_news_update: bool):
        """Update market knowledge based on analysis type"""
        if force_news_update:
            self.logger.info("Updating market knowledge (Regular Analysis)...")
            await self.rag_engine.update_if_needed(force_update=True)
        else:
            self.logger.info("Skipping forced market knowledge update (Forced Analysis)")
            await self.rag_engine.update_if_needed(force_update=False)
    
    async def _build_analysis_context(self, current_price: Optional[float], current_ticker) -> Dict[str, Any]:
        """Build context data for market analysis"""
        position_context = self.trading_strategy.get_position_context(current_price)
        memory_context = self.memory_service.get_context_summary()
        statistics_context = self.statistics_service.get_context()
        
        if statistics_context:
            position_context = f"{position_context}\n\n{statistics_context}"
        
        previous_data = await self.persistence.async_load_previous_response()
        previous_response = previous_data.get("response") if previous_data else None
        previous_indicators = previous_data.get("technical_indicators") if previous_data else None
        
        last_analysis_time_str = self._get_formatted_last_analysis_time()
        dynamic_thresholds = self.brain_service.get_dynamic_thresholds()
        
        return {
            "previous_response": previous_response,
            "previous_indicators": previous_indicators,
            "position_context": position_context,
            "performance_context": memory_context,
            "brain_service": self.brain_service,
            "last_analysis_time": last_analysis_time_str,
            "current_ticker": current_ticker,
            "dynamic_thresholds": dynamic_thresholds
        }
    
    def _get_formatted_last_analysis_time(self) -> Optional[str]:
        """Get last analysis time formatted as UTC string"""
        last_analysis_time_obj = self.persistence.get_last_analysis_time()
        if not last_analysis_time_obj:
            return None
<<<<<<< HEAD
        
        if last_analysis_time_obj.tzinfo is None:
            last_analysis_time_obj = last_analysis_time_obj.astimezone(timezone.utc)
        else:
            last_analysis_time_obj = last_analysis_time_obj.astimezone(timezone.utc)
        
=======

        if last_analysis_time_obj.tzinfo is None:
            last_analysis_time_obj = last_analysis_time_obj.replace(tzinfo=timezone.utc)
        else:
            last_analysis_time_obj = last_analysis_time_obj.astimezone(timezone.utc)

>>>>>>> main
        return last_analysis_time_obj.strftime('%Y-%m-%d %H:%M:%S')
    
    async def _handle_new_position(self, decision, current_price: Optional[float]):
        """Handle new position creation and status updates"""
        if decision.action not in ('BUY', 'SELL') or not self.trading_strategy.current_position:
            return
<<<<<<< HEAD
        
        if not self.discord_notifier:
            return
        
        try:
            if current_price is None:
                ticker = await self._fetch_current_ticker()
                current_price = float(ticker.get('last', ticker.get('close', 0))) if ticker else 0.0
            
            await self.discord_notifier.send_position_status(
                position=self.trading_strategy.current_position,
                current_price=current_price,
                channel_id=self.config.MAIN_CHANNEL_ID
            )
        except Exception as e:
            self.logger.warning("Error sending initial position status: %s", e)
        
        await self._start_position_status_updates()
=======

        await self._require_position_monitor().handle_new_position(current_price)
>>>>>>> main
    
    async def _send_discord_notification(self, result: Dict[str, Any]):
        """Send Discord notification with analysis results"""
        if self.discord_notifier:
            chart_image = None
            last_chart_buffer = self.market_analyzer.last_chart_buffer if self.market_analyzer else None
            if last_chart_buffer is not None:
                try:
                    last_chart_buffer.seek(0)
                    chart_image = io.BytesIO(last_chart_buffer.getvalue())
                    chart_image.seek(0)
                except Exception as e:
                    self.logger.warning("Failed to prepare chart image for Discord notification: %s", e)

            await self.discord_notifier.send_analysis_notification(
                result=result,
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                channel_id=self.config.MAIN_CHANNEL_ID,
                chart_image=chart_image
            )
    
    def _save_analysis_data(self, result: Dict[str, Any]):
        """Save analysis response and technical data"""
        raw_response = result.get("raw_response", "")
        if raw_response:
            technical_data = result.get("technical_data")
            generated_prompt = result.get("generated_prompt")
            self.persistence.save_previous_response(raw_response, technical_data, generated_prompt)

    async def _fetch_current_ticker(self) -> Optional[Dict[str, Any]]:
        """Fetch current ticker from exchange."""
        try:
            if self.current_exchange is None or self.current_symbol is None:
                return None
            ticker = await self.current_exchange.fetch_ticker(self.current_symbol)
            if ticker and self.dashboard_state:
                price = float(ticker.get('last', ticker.get('close', 0)))
                if price > 0:
                    await self.dashboard_state.update_price(price)
            return ticker
        except Exception as e:
            self.logger.error("Error fetching current ticker: %s", e)
            return None

    async def _wait_for_next_timeframe(self):
        """Wait until the next timeframe candle starts."""
        try:
            if self.current_timeframe is None:
                raise ValueError("current timeframe is not set")
            timeframe = self.current_timeframe
            current_time_ms = int(time.time() * 1000)

            # Calculate next candle start using validator (handles alignment)
<<<<<<< HEAD
            next_candle_ms = TimeframeValidator.calculate_next_candle_time(current_time_ms, self.current_timeframe)
=======
            next_candle_ms = TimeframeValidator.calculate_next_candle_time(current_time_ms, timeframe)
>>>>>>> main
            delay_ms = next_candle_ms - current_time_ms + (CANDLE_BUFFER_SECONDS * 1000)
            delay_seconds = max(0, delay_ms / 1000)

            next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)
<<<<<<< HEAD
            self.logger.info("Next check at %s UTC (in %.0fs)", next_check_time.strftime('%Y-%m-%d %H:%M:%S'), delay_seconds)
=======
            self.logger.info(
                "Next check at %s (in %.0fs)",
                self._format_utc_and_local(next_check_time),
                delay_seconds
            )
>>>>>>> main
            if self.dashboard_state:
                await self.dashboard_state.update_next_check(next_check_time)
            return await self._interruptible_sleep(delay_seconds)

        except Exception as e:
            self.logger.error("Error calculating next timeframe: %s", e)
            await self._interruptible_sleep(ERROR_WAIT_LONG)
            return False

    async def _wait_until_next_timeframe_after(self, last_time: datetime):
        """Wait until the next timeframe candle after a specific timestamp.

        Args:
            last_time: Timestamp of last analysis
        """
        try:
            if self.current_timeframe is None:
                raise ValueError("current timeframe is not set")
            timeframe = self.current_timeframe
            # Ensure last_time is timezone-aware (assume UTC if naive)
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)

            current_time_ms = int(time.time() * 1000)
            last_time_ms = int(last_time.timestamp() * 1000)

            # Calculate next candle after last analysis using validator (handles alignment)
<<<<<<< HEAD
            next_candle_ms = TimeframeValidator.calculate_next_candle_time(last_time_ms, self.current_timeframe)

            # Check if we're past the next candle boundary
            if current_time_ms >= next_candle_ms:
                self.logger.info("Resuming from last check at %s. Next candle already passed - proceeding immediately", last_time.strftime('%Y-%m-%d %H:%M:%S'))
=======
            next_candle_ms = TimeframeValidator.calculate_next_candle_time(last_time_ms, timeframe)

            # Check if we're past the next candle boundary
            if current_time_ms >= next_candle_ms:
                self.logger.info(
                    "Resuming from last check at %s. Next candle already passed - proceeding immediately",
                    self._format_utc_and_local(last_time)
                )
>>>>>>> main
                return

            # Wait for next candle
            # Use buffer to ensure we are safely into the next candle
            delay_ms = next_candle_ms - current_time_ms + (CANDLE_BUFFER_SECONDS * 1000)
            delay_seconds = max(0, delay_ms / 1000)
            next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)

            # Check if we're still within the same candle as the last analysis (for logging context)
<<<<<<< HEAD
            is_same = TimeframeValidator.is_same_candle(current_time_ms, last_time_ms, self.current_timeframe)

            context_msg = "Still in same candle" if is_same else "Resuming wait"
            self.logger.info("Resuming from last check at %s. %s - next check at %s UTC (in %.0fs)", last_time.strftime('%Y-%m-%d %H:%M:%S'), context_msg, next_check_time.strftime('%Y-%m-%d %H:%M:%S'), delay_seconds)
=======
            is_same = TimeframeValidator.is_same_candle(current_time_ms, last_time_ms, timeframe)

            context_msg = "Still in same candle" if is_same else "Resuming wait"
            self.logger.info(
                "Resuming from last check at %s. %s - next check at %s (in %.0fs)",
                self._format_utc_and_local(last_time),
                context_msg,
                self._format_utc_and_local(next_check_time),
                delay_seconds
            )
>>>>>>> main

            if self.dashboard_state:
                await self.dashboard_state.update_next_check(next_check_time)
            await self._interruptible_sleep(delay_seconds)

        except Exception as e:
            self.logger.error("Error calculating wait time: %s", e)
            await self._interruptible_sleep(ERROR_WAIT_SHORT)

    async def _interruptible_sleep(self, seconds: float, respect_force_analysis: bool = True):
        """Sleep in small chunks to allow responsive shutdown and force analysis.

        Uses SLEEP_CHUNK_SIZE to check for interruptions periodically.
<<<<<<< HEAD
=======
        Properly handles cancellation for graceful shutdown.
>>>>>>> main

        Args:
            seconds: Duration to sleep
            respect_force_analysis: If True, wake early on force analysis event (main loop only)

        Returns:
            bool: True if sleep was interrupted by force_analysis, False otherwise
        """
        start_time = time.monotonic()  # Use monotonic clock to track real elapsed time

        # Only clear force analysis flag for main loop sleeps
        if respect_force_analysis:
            self._force_analysis.clear()

<<<<<<< HEAD
        while self.running:
            elapsed = time.monotonic() - start_time
            if elapsed >= seconds:
                break

            # Check for force analysis (only if this sleep respects it)
            if respect_force_analysis and self._force_analysis.is_set():
                self._force_analysis.clear()
                self.logger.info("Force analysis triggered - interrupting wait")
                return True

            remaining = seconds - elapsed
            sleep_time = min(SLEEP_CHUNK_SIZE, remaining)
            await asyncio.sleep(sleep_time)
=======
        try:
            while self.running:
                elapsed = time.monotonic() - start_time
                if elapsed >= seconds:
                    break

                # Check for force analysis (only if this sleep respects it)
                if respect_force_analysis and self._force_analysis.is_set():
                    self._force_analysis.clear()
                    self.logger.info("Force analysis triggered - interrupting wait")
                    return True

                remaining = seconds - elapsed
                sleep_time = min(SLEEP_CHUNK_SIZE, remaining)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            # Exit immediately on cancellation (graceful shutdown)
            self.logger.debug("Interruptible sleep cancelled during shutdown")
            return False
>>>>>>> main

        return False

    async def _force_analysis_now(self):
        """Force immediate analysis by interrupting the wait."""
        self.logger.info("Forcing immediate analysis...")
        self._force_analysis.set()

<<<<<<< HEAD
    async def _start_position_status_updates(self):
        """Start periodic position status updates to Discord (every hour)."""
        if self._position_status_task and not self._position_status_task.done():
            return  # Already running

        self._position_status_task = asyncio.create_task(
            self._position_status_loop(),
            name="Position-Status-Updates"
        )
        self._active_tasks.add(self._position_status_task)
        self._position_status_task.add_done_callback(self._active_tasks.discard)
        self.logger.debug("Started hourly position status updates")

    async def _stop_position_status_updates(self):
        """Stop periodic position status updates."""
        if self._position_status_task and not self._position_status_task.done():
            self._position_status_task.cancel()
            try:
                await self._position_status_task
            except asyncio.CancelledError:
                pass
            self._position_status_task = None
            self.logger.debug("Stopped position status updates")

    async def _position_status_loop(self):
        """Send position status updates every hour while position is open."""
        try:
            while self.running:
                # Wait for interval (in chunks for responsiveness)
                await self._interruptible_sleep(POSITION_UPDATE_INTERVAL, respect_force_analysis=False)

                if not self.running:
                    break

                # Check if position is still open
                if not self.trading_strategy.current_position:
                    self.logger.debug("Position closed, stopping status updates")
                    break

                # Send position status update
                if self.discord_notifier:
                    try:
                        ticker = await self._fetch_current_ticker()
                        current_price = float(ticker.get('last', ticker.get('close', 0))) if ticker else 0.0

                        await self.discord_notifier.send_position_status(
                            position=self.trading_strategy.current_position,
                            current_price=current_price,
                            channel_id=self.config.MAIN_CHANNEL_ID
                        )
                        self.logger.debug("Sent hourly position status update to Discord")
                    except Exception as e:
                        self.logger.warning("Error sending position status update: %s", e)
        except asyncio.CancelledError:
            self.logger.debug("Position status loop cancelled")
            raise

=======
>>>>>>> main
    async def _show_help(self):
        """Show help information about available commands."""
        self.keyboard_handler.display_help()

    async def _request_shutdown(self):
        """Request application shutdown via keyboard."""
        self.logger.info("Shutdown requested via keyboard command")
        if self.shutdown_manager:
            await self.shutdown_manager.shutdown_gracefully()
        else:
            self.running = False
            for task in self.tasks:
                if not task.done():
                    task.cancel()
