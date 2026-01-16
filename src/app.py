import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from src.logger.logger import Logger
from src.utils.timeframe_validator import TimeframeValidator
from src.managers.persistence_manager import PersistenceManager
from src.contracts.model_contract import ModelManagerProtocol
from src.trading import (
    TradingBrainService,
    TradingStatisticsService,
    TradingMemoryService
)



class CryptoTradingBot:
    """Automated crypto trading bot - TRADING MODE ONLY."""
    
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
        news_api,
        market_api,
        categories_api,
        alternative_me_api,
        cryptocompare_session,
        persistence: PersistenceManager,
        model_manager: ModelManagerProtocol,
        brain_service: TradingBrainService,
        statistics_service: TradingStatisticsService,
        memory_service: TradingMemoryService,
        dashboard_state = None,
        discord_task: Optional[asyncio.Task] = None
    ):
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
        self.news_api = news_api
        self.market_api = market_api
        self.categories_api = categories_api
        self.alternative_me_api = alternative_me_api
        self.cryptocompare_session = cryptocompare_session
        
        # Injected trading services
        self.persistence = persistence
        self.model_manager = model_manager
        self.brain_service = brain_service
        self.statistics_service = statistics_service
        self.memory_service = memory_service
        self.dashboard_state = dashboard_state
        
        # Runtime state
        self.tasks = []
        self.running = False
        self._active_tasks = set()
        self._force_analysis = asyncio.Event()
        self._discord_task = discord_task
        self._position_status_task: Optional[asyncio.Task] = None
        
        # Trading state
        self.current_exchange = None
        self.current_symbol: Optional[str] = None
        self.current_timeframe: Optional[str] = None

    async def initialize(self):
        """Initialize all components."""
        if self.shutdown_manager:
            self.shutdown_manager.register_shutdown_callback(self.shutdown)
        
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
        self.logger.info("Keyboard commands: 'a' = force analysis, 'h' = help, 'q' = quit")


    async def run(self, symbol: str, timeframe: str = None):
        """Run the trading bot in continuous mode.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Optional timeframe override
        """
        self.current_symbol = symbol
        self.current_timeframe = timeframe or self.config.TIMEFRAME
        
        # Find exchange that supports the symbol
        exchange, exchange_id = await self.exchange_manager.find_symbol_exchange(symbol)
        if not exchange:
            self.logger.error(f"Symbol {symbol} not found on any configured exchange")
            return
        
        self.current_exchange = exchange
        self.logger.info(f"Starting trading for {symbol} on {exchange_id}")
        self.logger.info(f"Timeframe: {self.current_timeframe}")
        
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
            pos = self.trading_strategy.current_position
            self.logger.info(f"Existing position: {pos.direction} @ ${pos.entry_price:,.2f}")
            # Start hourly position updates for existing position
            if self.discord_notifier:
                await self._start_position_status_updates()
        else:
            self.logger.info("No existing position")

        # Fetch initial price for dashboard (one-time startup call)
        await self._fetch_current_ticker()
        
        # Check if resuming from previous session (regardless of position status)
        last_analysis_time = self.persistence.get_last_analysis_time()
        if last_analysis_time:
            self.logger.info(f"Resuming from last analysis at {last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            await self._wait_until_next_timeframe_after(last_analysis_time)
        self.logger.info("Ready for next analysis after wait")
        
        # Initial run is considered regular (unless we want to skipping update on restart, but safer to update)
        is_regular_run = True

        while self.running:
            try:
                check_count += 1
                await self._execute_trading_check(check_count, force_news_update=is_regular_run)
                
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
                self.logger.error(f"Error in trading loop: {e}")
                # Wait 60 seconds before retrying on error
                await self._interruptible_sleep(60)
    
    async def _execute_trading_check(self, check_count: int, force_news_update: bool = True):
        """Execute a single trading check iteration."""
        current_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info(f"Trading Check #{check_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
        
        # Fetch current ticker exactly ONCE for this entire check cycle to avoid redundant API calls
        current_ticker = None
        current_price = None
        try:
            current_ticker = await self._fetch_current_ticker()
            if current_ticker:
                current_price = float(current_ticker.get('last', current_ticker.get('close', 0)))
        except Exception as e:
            self.logger.warning(f"Could not fetch current ticker: {e}")

        # Check if existing position hit stop/target
        if self.trading_strategy.current_position and current_price is not None:
            try:
                close_reason = await self.trading_strategy.check_position(current_price)
                if close_reason:
                    self.logger.info(f"Position closed: {close_reason}")
                    # Stop hourly position updates
                    await self._stop_position_status_updates()
                    # Send performance stats to Discord
                    if self.discord_notifier:
                        history = self.persistence.load_trade_history()
                        await self.discord_notifier.send_performance_stats(
                            trade_history=history,
                            symbol=self.current_symbol,
                            channel_id=self.config.MAIN_CHANNEL_ID
                        )
            except Exception as e:
                self.logger.error(f"Error checking position: {e}")
        
        # Run market analysis
        if force_news_update:
            self.logger.info("Updating market knowledge (Regular Analysis)...")
            await self.rag_engine.update_if_needed(force_update=True)
        else:
            self.logger.info("Skipping forced market knowledge update (Forced Analysis)")
            # Still check if update is needed by interval
            await self.rag_engine.update_if_needed(force_update=False)

        self.logger.info("Running market analysis...")
        
        # Build trading context with P&L data (separate for system prompt)
        # Use the SAME current_price fetched above
        position_context = self.trading_strategy.get_position_context(current_price)
        memory_context = self.memory_service.get_context_summary(current_price)
        statistics_context = self.statistics_service.get_context()
        # Combine position context with statistics for unified trading context
        if statistics_context:
            position_context = f"{position_context}\n\n{statistics_context}"

        # Load previous response for AI continuity
        previous_data = self.persistence.load_previous_response()
        previous_response = None
        previous_indicators = None

        if previous_data:
            previous_response = previous_data.get("response")
            previous_indicators = previous_data.get("technical_indicators")

        # Get last analysis time for temporal context
        last_analysis_time_obj = self.persistence.get_last_analysis_time()
        last_analysis_time_str = None
        if last_analysis_time_obj:
            # Ensure we are working with UTC
            if last_analysis_time_obj.tzinfo is None:
                # Legacy: Naive timestamps were local time. Convert to UTC.
                # astimezone(timezone.utc) assumes naive is local system time
                last_analysis_time_obj = last_analysis_time_obj.astimezone(timezone.utc)
            else:
                # Convert aware timestamp to UTC to match prompt expectation
                last_analysis_time_obj = last_analysis_time_obj.astimezone(timezone.utc)
            
            last_analysis_time_str = last_analysis_time_obj.strftime('%Y-%m-%d %H:%M:%S')

        # Get dynamic thresholds for prompt template
        dynamic_thresholds = self.brain_service.get_dynamic_thresholds()

        result = await self.market_analyzer.analyze_market(
            previous_response=previous_response,
            previous_indicators=previous_indicators,
            position_context=position_context,
            performance_context=memory_context,
            brain_service=self.brain_service,  # Pass service, not context
            last_analysis_time=last_analysis_time_str,
            current_ticker=current_ticker,
            dynamic_thresholds=dynamic_thresholds
        )
        
        if "error" in result:
            self.logger.error(f"Analysis failed: {result['error']}")
            return
        
        # Save the timestamp of successful analysis
        self.persistence.save_last_analysis_time()
        
        # Process the analysis for trading decision
        decision = await self.trading_strategy.process_analysis(result, self.current_symbol)
        
        if decision:
            await self.discord_notifier.send_trading_decision(decision, self.config.MAIN_CHANNEL_ID)
            # Send initial position status and start hourly updates if a new position was opened
            if decision.action in ('BUY', 'SELL') and self.trading_strategy.current_position:
                if self.discord_notifier:
                    # Send initial position status
                    try:

                        if current_price is None:
                             # Fallback if initial fetch failed
                             ticker = await self._fetch_current_ticker()
                             current_price = float(ticker.get('last', ticker.get('close', 0))) if ticker else 0.0

                        await self.discord_notifier.send_position_status(
                            position=self.trading_strategy.current_position,
                            current_price=current_price,
                            channel_id=self.config.MAIN_CHANNEL_ID
                        )
                    except Exception as e:
                        self.logger.warning(f"Error sending initial position status: {e}")
                    # Start hourly updates
                    await self._start_position_status_updates()
        else:
            self.logger.info("No trading action taken")
        
        # Send Discord notification
        if self.discord_notifier:
            await self.discord_notifier.send_analysis_notification(
                result=result,
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                channel_id=self.config.MAIN_CHANNEL_ID
            )
        
        # Save the response and technical indicators for context
        raw_response = result.get("raw_response", "")
        technical_data = result.get("technical_data")  # Get technical indicators from result
        generated_prompt = result.get("generated_prompt")  # Get prompt for dashboard
        
        if raw_response:
            self.persistence.save_previous_response(raw_response, technical_data, generated_prompt)
    
    async def _fetch_current_ticker(self) -> Optional[Dict[str, Any]]:
        """Fetch current ticker from exchange."""
        try:
            ticker = await self.current_exchange.fetch_ticker(self.current_symbol)
            if ticker and self.dashboard_state:
                price = float(ticker.get('last', ticker.get('close', 0)))
                if price > 0:
                    await self.dashboard_state.update_price(price)
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching current ticker: {e}")
            return None
    
    async def _wait_for_next_timeframe(self):
        """Wait until the next timeframe candle starts."""
        try:
            # Get current time from exchange if possible
            try:
                current_time_ms = await self.current_exchange.fetch_time()
            except Exception:
                current_time_ms = int(time.time() * 1000)
            
            # Calculate interval in milliseconds
            interval_seconds = TimeframeValidator.to_minutes(self.current_timeframe) * 60
            interval_ms = interval_seconds * 1000
            
            # Calculate next candle start
            next_candle_ms = ((current_time_ms // interval_ms) + 1) * interval_ms
            delay_ms = next_candle_ms - current_time_ms + 2000  # Add 2 second buffer
            delay_seconds = delay_ms / 1000
            
            next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)
            self.logger.info(f"Next check at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {delay_seconds:.0f}s)")
            if self.dashboard_state:
                await self.dashboard_state.update_next_check(next_check_time)
            return await self._interruptible_sleep(delay_seconds)
            
        except Exception as e:
            self.logger.error(f"Error calculating next timeframe: {e}")
            # Default to 5 minute wait on error
            await self._interruptible_sleep(300)
            return False
    
    async def _wait_until_next_timeframe_after(self, last_time: datetime):
        """Wait until the next timeframe candle after a specific timestamp.
        
        Args:
            last_time: Timestamp of last analysis
        """
        try:
            # Ensure last_time is timezone-aware (assume UTC if naive)
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            
            # Get current time
            try:
                current_time_ms = await self.current_exchange.fetch_time()
            except Exception:
                current_time_ms = int(time.time() * 1000)
            

            # Calculate interval
            interval_seconds = TimeframeValidator.to_minutes(self.current_timeframe) * 60
            interval_ms = interval_seconds * 1000
            
            # Get last analysis time in ms
            last_time_ms = int(last_time.timestamp() * 1000)
            
            # Calculate next candle after last analysis
            next_candle_ms = ((last_time_ms // interval_ms) + 1) * interval_ms
            
            # Check if we're still within the same candle as the last analysis
            current_candle_ms = (current_time_ms // interval_ms) * interval_ms
            last_candle_ms = (last_time_ms // interval_ms) * interval_ms
            
            if current_candle_ms == last_candle_ms:
                # Still in same candle - wait for next one
                delay_ms = next_candle_ms - current_time_ms + 1000
                delay_seconds = max(0, delay_ms / 1000)
                next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)
                
                self.logger.info(
                    f"Resuming from last check at {last_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Still in same candle - next check at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {delay_seconds:.0f}s)"
                )
                if self.dashboard_state:
                    await self.dashboard_state.update_next_check(next_check_time)
                await self._interruptible_sleep(delay_seconds)
            elif current_time_ms >= next_candle_ms:
                # Already past next candle - run immediately
                self.logger.info(
                    f"Resuming from last check at {last_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Next candle already passed - proceeding immediately"
                )
                return
            else:
                # In a different candle but not yet at next_candle_ms (edge case)
                delay_ms = next_candle_ms - current_time_ms + 1000
                delay_seconds = max(0, delay_ms / 1000)
                next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)
                
                self.logger.info(
                    f"Resuming from last check at {last_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Next check at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {delay_seconds:.0f}s)"
                )
                if self.dashboard_state:
                    await self.dashboard_state.update_next_check(next_check_time)
                await self._interruptible_sleep(delay_seconds)
            
        except Exception as e:
            self.logger.error(f"Error calculating wait time: {e}")
            # Default to 1 minute wait on error
            await self._interruptible_sleep(60)
    
    async def _interruptible_sleep(self, seconds: float, respect_force_analysis: bool = True):
        """Sleep in small chunks to allow responsive shutdown and force analysis.
        
        Args:
            seconds: Duration to sleep
            respect_force_analysis: If True, wake early on force analysis event (main loop only)
        """
        chunk_size = 1.0  # Check every second
        elapsed = 0.0
        
        # Only clear force analysis flag for main loop sleeps
        if respect_force_analysis:
            self._force_analysis.clear()
        
        try:
            while elapsed < seconds and self.running:
                # Check for force analysis (only if this sleep respects it)
                if respect_force_analysis and self._force_analysis.is_set():
                    self._force_analysis.clear()
                    self.logger.info("Force analysis triggered - interrupting wait")
                    return True
                
                sleep_time = min(chunk_size, seconds - elapsed)
                await asyncio.sleep(sleep_time)
                elapsed += sleep_time
            
            return False
        except asyncio.CancelledError:
            # Allow immediate cancellation
            raise
    
    async def _force_analysis_now(self):
        """Force immediate analysis by interrupting the wait."""
        self.logger.info("Forcing immediate analysis...")
        self._force_analysis.set()
    
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
        update_interval = 3600  # 1 hour in seconds
        
        try:
            while self.running:
                # Wait for 1 hour (in chunks for responsiveness)
                await self._interruptible_sleep(update_interval, respect_force_analysis=False)
                
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
                        self.logger.warning(f"Error sending position status update: {e}")
        except asyncio.CancelledError:
            self.logger.debug("Position status loop cancelled")
            raise
    
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
    
    async def shutdown(self):
        self.logger.info("Shutting down gracefully...")
        self.running = False
        
        # Give a moment for loops to see running=False
        await asyncio.sleep(0.1)

        if self.shutdown_manager:
            await self.shutdown_manager.cleanup_bot_resources(self)
        
        self.logger.info("Shutdown complete")
        
