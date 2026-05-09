"""
Discord Notifier - Send-only notification service with message expiration.
Sends AI trading analysis to Discord with automatic message cleanup.
"""
import asyncio
import io
from typing import Optional, TYPE_CHECKING, List, Dict, Any, Callable, Awaitable

import discord
from aiohttp import ClientSession

from src.utils.decorators import retry_async
from .base_notifier import BaseNotifier
from .filehandler import DiscordFileHandler

ENTRY_ACTIONS = {'BUY', 'SELL'}

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.parsing.unified_parser import UnifiedParser
    from src.utils.format_utils import FormatUtils



class DiscordNotifier(BaseNotifier):
    """Send-only Discord notifier with message expiration tracking."""

    _DISCORD_TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
    _DISCORD_SEND_MAX_ATTEMPTS = 3
    _DISCORD_SEND_INITIAL_BACKOFF_SECONDS = 1.0

    def __init__(self, logger, config: "ConfigProtocol", unified_parser: "UnifiedParser", formatter: "FormatUtils", bot: discord.Client, file_handler: DiscordFileHandler) -> None:
        """Initialize DiscordNotifier.

        Args:
            logger: Logger instance
            config: ConfigProtocol instance for Discord settings
            unified_parser: UnifiedParser for JSON extraction (DRY)
            formatter: FormatUtils instance for value formatting
            bot: Injected Discord client instance
            file_handler: Injected DiscordFileHandler instance
        """


        super().__init__(logger, config, unified_parser, formatter)
        self.session: Optional[ClientSession] = None
        self._ready_event = asyncio.Event()
        self._shutdown_started = False

        self.bot = bot
        self.bot.discord_notifier = self
        self.bot.event(self.on_ready)
        self.file_handler = file_handler

        # Small pacing gap between Discord API sends to reduce burst rate-limit pressure.
        self._send_lock = asyncio.Lock()
        self._last_send_timestamp = 0.0
        self._discord_send_interval_seconds = 0.4

    async def _send_with_spacing(self, send_operation: Callable[[], Awaitable[discord.Message]]) -> discord.Message:
        """Serialize and pace Discord sends with a minimal delay between requests."""
        async with self._send_lock:
            now = asyncio.get_running_loop().time()
            elapsed = now - self._last_send_timestamp
            if elapsed < self._discord_send_interval_seconds:
                await asyncio.sleep(self._discord_send_interval_seconds - elapsed)

            sent_message = await send_operation()
            self._last_send_timestamp = asyncio.get_running_loop().time()
            return sent_message

    def _is_transient_discord_error(self, exc: Exception) -> bool:
        """Return True for transient Discord 5xx errors worth retrying."""
        status = getattr(exc, "status", None)
        return isinstance(status, int) and status in self._DISCORD_TRANSIENT_STATUS_CODES

    async def _send_with_transient_retry(
        self,
        send_operation: Callable[[], Awaitable[discord.Message]],
        *,
        operation_name: str,
    ) -> discord.Message:
        """Retry transient Discord upstream failures with bounded backoff."""
        delay = self._DISCORD_SEND_INITIAL_BACKOFF_SECONDS
        for attempt in range(1, self._DISCORD_SEND_MAX_ATTEMPTS + 1):
            try:
                return await self._send_with_spacing(send_operation)
            except Exception as exc:  # noqa: BLE001
                if not self._is_transient_discord_error(exc) or attempt >= self._DISCORD_SEND_MAX_ATTEMPTS:
                    raise

                self.logger.warning(
                    "Transient Discord error while %s (status=%s, attempt=%s/%s): %s",
                    operation_name,
                    getattr(exc, "status", "unknown"),
                    attempt,
                    self._DISCORD_SEND_MAX_ATTEMPTS,
                    exc,
                )
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError("Unreachable: Discord transient retry loop exhausted unexpectedly")


    async def on_ready(self):
        """Called when bot is ready."""
        try:
            self.logger.info("DiscordNotifier: Logged in as %s", self.bot.user.name)
            self.file_handler.initialize()
            self.logger.debug("FileHandler initialized")
            self.is_initialized = True
            self._ready_event.set()
            self.logger.debug("DiscordNotifier ready")
        except Exception as e:
            self.logger.error("Error in on_ready: %s", e, exc_info=True)

    async def __aenter__(self):
        if self.session is None:
            self.session = ClientSession()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        if self._shutdown_started:
            return
        self._shutdown_started = True

        if self.session:
            try:
                await self.session.close()
                self.session = None
            except Exception as e:
                self.logger.warning("Session close error: %s", e)

        if self.file_handler:
            try:
                await self.file_handler.shutdown()
            except Exception as e:
                self.logger.warning("Error during file handler shutdown: %s", e)

        if self.bot:
            try:
                if not self.bot.is_closed():
                    self.logger.info("Closing Discord bot connection...")
                    await self.bot.close()
                    # Allow discord.py's keep-alive thread to observe the closed websocket.
                    await asyncio.sleep(0.25)
            except Exception as e:
                self.logger.warning("Error closing Discord bot: %s", e)

        self.logger.info("Discord notifier resources released")

    async def start(self) -> None:
        """Start the Discord bot."""
        if not self.bot:
            self.logger.error("Discord bot is not initialized.")
            return
        token = self.config.BOT_TOKEN_DISCORD
        if not token:
            self.logger.error("BOT_TOKEN_DISCORD is not configured.")
            return
        try:
            await self.bot.start(token)
        except discord.LoginFailure as e:
            self.logger.error("Discord Login Failure: %s. Check your BOT_TOKEN_DISCORD.", e, exc_info=True)
        except Exception as e:
            self.logger.error("Failed to start Discord bot: %s", e, exc_info=True)

    async def wait_until_ready(self) -> None:
        """Wait for the bot to fully initialize."""
        await self._ready_event.wait()

    async def shutdown(self) -> None:
        """Shutdown the Discord notifier and cleanup resources."""
        try:
            await self.__aexit__(None, None, None)
        except Exception as e:
            self.logger.warning("Error during Discord notifier shutdown: %s", e)

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def send_message(
            self,
            message: str,
            channel_id: int,
            expire_after: Optional[int] = None
    ) -> Optional[discord.Message]:
        """Send a text message to Discord with automatic expiration.

        Args:
            message: Message text
            channel_id: Discord channel ID
            expire_after: Message expiry time in seconds (defaults to FILE_MESSAGE_EXPIRY)

        Returns:
            The sent message or None on failure
        """
        if expire_after is None:
            expire_after = self.config.FILE_MESSAGE_EXPIRY

        await self.wait_until_ready()
        channel = self.bot.get_channel(channel_id)
        if not channel:
            self.logger.error("Channel with ID %s not found.", channel_id)
            return None

        try:
            # Discord limit: 2000 chars per message. Set reasonable max total length.
            MAX_TOTAL_LENGTH = 20000  # 10 chunks max

            if len(message) > MAX_TOTAL_LENGTH:
                self.logger.warning("Message length (%s) exceeds maximum (%s). Truncating.", len(message), MAX_TOTAL_LENGTH)
                content = message[:MAX_TOTAL_LENGTH]
            else:
                content = message

            chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]

            sent_message = None
            for i, chunk in enumerate(chunks):
                if i > 0:
                    await asyncio.sleep(1)

                delete_after = float(expire_after) if expire_after is not None else None
                sent_message = await self._send_with_transient_retry(
                    lambda: channel.send(
                        content=chunk,
                        delete_after=delete_after
                    ),
                    operation_name="sending message chunk",
                )
                await self.file_handler.track_message(
                    message_id=sent_message.id,
                    channel_id=channel_id,
                    user_id=None,
                    message_type="message",
                    expire_after=expire_after
                )

            self.logger.debug("Sent %s message chunk(s) (Last ID: %s)", len(chunks), sent_message.id if sent_message else 'None')
            return sent_message
        except discord.HTTPException as e:
            self.logger.error("Discord HTTPException when sending message: %s", e, exc_info=True)
        except Exception as e:
            self.logger.error("Unexpected error when sending message: %s", e, exc_info=True)
        return None

    def _get_discord_color(self, color_key: str) -> discord.Color:
        """Convert color key to discord.Color."""
        color_map = {
            'green': discord.Color.green(),
            'red': discord.Color.red(),
            'grey': discord.Color.light_grey(),
            'orange': discord.Color.orange(),
            'blue': discord.Color.blue(),
        }
        return color_map.get(color_key, discord.Color.light_grey())

    async def _send_embed(
        self,
        embed: discord.Embed,
        channel_id: int,
        expire_after: Optional[float] = None
    ) -> Optional[discord.Message]:
        """Send a Discord embed to a channel with expiration.

        Args:
            embed: Discord embed to send
            channel_id: Discord channel ID
            expire_after: Message expiry time in seconds (defaults to FILE_MESSAGE_EXPIRY)

        Returns:
            The sent message or None on failure
        """
        if expire_after is None:
            expire_after = float(self.config.FILE_MESSAGE_EXPIRY)

        channel = self.bot.get_channel(channel_id)
        if not channel:
            self.logger.error("Channel with ID %s not found.", channel_id)
            return None

        try:
            sent_message = await self._send_with_transient_retry(
                lambda: channel.send(embed=embed, delete_after=expire_after),
                operation_name="sending embed",
            )

            # Track message for persistent deletion
            await self.file_handler.track_message(
                message_id=sent_message.id,
                channel_id=channel_id,
                user_id=None,
                message_type="embed",
                expire_after=int(expire_after)
            )

            return sent_message
        except Exception as e:
            self.logger.error("Error sending embed: %s", e)
            return None

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def send_trading_decision(self, decision: Any, channel_id: int) -> None:
        """Send a trading decision as Discord embed.

        Args:
            decision: TradingDecision dataclass
            channel_id: Discord channel ID
        """
        try:
            await self.wait_until_ready()
            channel = self.bot.get_channel(channel_id)
            if not channel:
                self.logger.error("Channel with ID %s not found.", channel_id)
                return

            color_key, emoji = self.get_action_styling(decision.action)
            color = self._get_discord_color(color_key)

            embed = discord.Embed(
                title=f"{emoji} TRADING DECISION: {decision.action}",
                description=decision.reasoning[:4096] if decision.reasoning else "No reasoning provided",
                color=color
            )

            embed.add_field(name="Symbol", value=decision.symbol, inline=True)
            embed.add_field(name="Price", value=f"${decision.price:,.2f}", inline=True)
            embed.add_field(name="Confidence", value=decision.confidence, inline=True)

            if decision.stop_loss:
                embed.add_field(name="Stop Loss", value=f"${decision.stop_loss:,.2f}", inline=True)
            if decision.take_profit:
                embed.add_field(name="Take Profit", value=f"${decision.take_profit:,.2f}", inline=True)
            if decision.position_size:
                embed.add_field(name="Position Size", value=f"{decision.position_size * 100:.2f}%", inline=True)
            if decision.quote_amount:
                embed.add_field(name="Invested", value=f"${decision.quote_amount:,.2f}", inline=True)
            if decision.quantity:
                embed.add_field(name="Quantity", value=self.formatter.fmt(decision.quantity), inline=True)
            if decision.action in ENTRY_ACTIONS and decision.quantity:
                entry_fee = decision.price * decision.quantity * self.config.TRANSACTION_FEE_PERCENT
                embed.add_field(name="Entry Fee", value=f"${entry_fee:.4f}", inline=True)

            embed.set_footer(text=f"Time: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            await self._send_embed(embed, channel_id)
        except Exception as e:
            self.logger.error("Error sending trading decision: %s", e)

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def send_analysis_notification(
            self,
            result: dict,
            symbol: str,
            timeframe: str,
            channel_id: int,
            chart_image: Optional[io.BytesIO] = None
    ) -> None:
        """Send full analysis notification with reasoning and JSON embed.

        Args:
            result: Analysis result dict with corrected analysis and raw_response
            symbol: Trading symbol
            timeframe: Trading timeframe
            channel_id: Discord channel ID
            chart_image: Optional PNG chart image buffer to attach
        """
        try:
            # Get the corrected analysis dict (has R/R correction and other validations applied)
            analysis = result.get("analysis")
            if not analysis:
                return

            # Get reasoning text from raw_response (narrative text, not data)
            raw_response = result.get("raw_response", "")
            reasoning = self.unified_parser.extract_text_before_json(raw_response) if raw_response else ""

            if reasoning:
                await self.send_message(
                    message=f"**{symbol} Analysis**\n\n{reasoning}",
                    channel_id=channel_id
                )

            # Use the corrected analysis dict for the embed (not re-parsed raw JSON)
            embed = self._create_analysis_embed(analysis, symbol, timeframe)
            if embed:
                await self._send_embed(embed, channel_id)

            if chart_image is not None:
                await self._send_analysis_chart(
                    chart_image=chart_image,
                    symbol=symbol,
                    timeframe=timeframe,
                    channel_id=channel_id
                )
        except Exception as e:
            self.logger.error("Error sending analysis notification: %s", e)

    async def _send_analysis_chart(
            self,
            chart_image: io.BytesIO,
            symbol: str,
            timeframe: str,
            channel_id: int,
            expire_after: Optional[float] = None
    ) -> Optional[discord.Message]:
        """Send analysis chart image to Discord and track it for cleanup."""
        if expire_after is None:
            expire_after = float(self.config.FILE_MESSAGE_EXPIRY)

        channel = self.bot.get_channel(channel_id)
        if not channel:
            self.logger.error("Channel with ID %s not found for chart upload.", channel_id)
            return None

        try:
            chart_image.seek(0)
            chart_bytes = chart_image.getvalue()
            chart_image.seek(0)

            if not chart_bytes:
                self.logger.warning("Chart image buffer is empty; skipping Discord chart upload")
                return None

            safe_symbol = symbol.replace('/', '_')
            filename = f"{safe_symbol}_{timeframe}_analysis_chart.png"

            sent_message = await self._send_with_transient_retry(
                lambda: channel.send(
                    content=f"📈 {symbol} {timeframe} chart snapshot",
                    file=discord.File(io.BytesIO(chart_bytes), filename=filename),
                    delete_after=expire_after,
                ),
                operation_name="sending analysis chart",
            )

            await self.file_handler.track_message(
                message_id=sent_message.id,
                channel_id=channel_id,
                user_id=None,
                message_type="chart",
                expire_after=int(expire_after)
            )
            return sent_message
        except Exception as e:
            self.logger.error("Error sending analysis chart: %s", e)
            return None

    async def send_position_status(
            self,
            position: Any,
            current_price: float,
            channel_id: int
    ) -> None:
        """Send current open position status embed.

        Args:
            position: Current Position object
            current_price: Current market price
            channel_id: Discord channel ID
        """
        try:
            pnl_pct, pnl_quote = self.calculate_position_pnl(position, current_price)
            stop_distance_pct, target_distance_pct = self.calculate_stop_target_distances(position, current_price)
            hours_held = self.calculate_time_held(position.entry_time)

            color_key, emoji = self.get_pnl_styling(pnl_pct)
            color = self._get_discord_color(color_key)

            embed = discord.Embed(
                title=f"{emoji} Open {position.direction} Position - {position.symbol}",
                description="Current position monitoring",
                color=color
            )

            embed.add_field(name="Entry Price", value=f"${position.entry_price:,.2f}", inline=True)
            embed.add_field(name="Current Price", value=f"${current_price:,.2f}", inline=True)
            embed.add_field(name="Quantity", value=self.formatter.fmt(position.size), inline=True)
            try:
                if position.quote_amount > 0:
                    embed.add_field(name="Invested", value=f"${position.quote_amount:,.2f}", inline=True)
            except AttributeError:
                pass

            embed.add_field(name="Unrealized P&L", value=f"{pnl_pct:+.2f}%", inline=True)
            embed.add_field(name=f"P&L ({self.config.QUOTE_CURRENCY})", value=f"${pnl_quote:+,.2f}", inline=True)
            embed.add_field(name="Confidence", value=position.confidence, inline=True)
            embed.add_field(name="Position Size %", value=f"{position.size_pct * 100:.2f}%", inline=True)
            embed.add_field(name="Stop Loss", value=f"${position.stop_loss:,.2f} ({stop_distance_pct:+.2f}%)", inline=True)
            embed.add_field(name="Take Profit", value=f"${position.take_profit:,.2f} ({target_distance_pct:+.2f}%)", inline=True)
            embed.add_field(name="Exit Monitoring", value=self.format_exit_monitoring(), inline=False)
            embed.add_field(name="Entry Fee", value=f"${position.entry_fee:.4f}", inline=True)
            embed.add_field(name="Time Held", value=f"{hours_held:.1f}h", inline=True)
            embed.set_footer(text=f"Entry Time: {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            await self._send_embed(embed, channel_id)
        except Exception as e:
            self.logger.error("Error sending position status: %s", e)

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def send_performance_stats(
            self,
            trade_history: List[Dict[str, Any]],
            symbol: str,
            channel_id: int
    ) -> None:
        """Send overall performance statistics embed.

        Args:
            trade_history: Full trade history list
            symbol: Trading symbol
            channel_id: Discord channel ID
        """
        try:
            stats = self.calculate_performance_stats(trade_history)
            if not stats:
                return

            embed = discord.Embed(
                title="📈 Trading Performance Summary",
                description=f"Overall performance after {stats['closed_trades']} closed trades",
                color=discord.Color.blue() if stats['net_pnl'] > 0 else discord.Color.red()
            )

            embed.add_field(name=f"Total P&L ({self.config.QUOTE_CURRENCY})", value=f"${stats['total_pnl_quote']:+,.2f}", inline=True)
            embed.add_field(name="Total P&L (%)", value=f"{stats['total_pnl_pct']:+.2f}%", inline=True)
            embed.add_field(name="Avg P&L/Trade", value=f"{stats['avg_pnl_pct']:+.2f}%", inline=True)
            embed.add_field(name="Win Rate", value=f"{stats['win_rate']:.1f}% ({stats['winning_trades']}/{stats['closed_trades']})", inline=True)
            embed.add_field(name="Total Trades", value=str(stats['closed_trades']), inline=True)
            embed.add_field(name="Total Fees", value=f"${stats['total_fees']:.4f}", inline=True)
            embed.add_field(name=f"Net P&L ({self.config.QUOTE_CURRENCY})", value=f"${stats['net_pnl']:+,.2f}", inline=True)

            last_closed_trade = stats.get('last_closed_trade')
            if last_closed_trade:
                outcome = last_closed_trade.get('outcome', 'UNKNOWN')
                close_reason = last_closed_trade.get('close_reason')
                outcome_value = f"{outcome}\n{close_reason}" if close_reason else outcome
                embed.add_field(name="Last Outcome", value=outcome_value, inline=True)
                embed.add_field(
                    name=f"Last Trade P&L ({self.config.QUOTE_CURRENCY})",
                    value=f"${last_closed_trade['pnl_quote']:+,.2f} ({last_closed_trade['pnl_pct']:+.2f}%)",
                    inline=True
                )

            embed.set_footer(text=f"Symbol: {symbol}")
            await self._send_embed(embed, channel_id)
        except Exception as e:
            self.logger.error("Error sending performance stats: %s", e)

    def _create_analysis_embed(self, analysis: dict, symbol: str, timeframe: str) -> Optional[discord.Embed]:
        """Create Discord embed from analysis JSON."""
        try:
            fields = self.extract_analysis_fields(analysis)
            color_key, _ = self.get_action_styling(fields['signal'])
            color = self._get_discord_color(color_key)

            embed = discord.Embed(
                title=f"📊 {symbol} - {fields['signal']}",
                description=fields['reasoning'][:4096],
                color=color
            )

            if fields['entry_price']:
                embed.add_field(name="Entry", value=f"${fields['entry_price']:,.2f}", inline=True)
            if fields['stop_loss']:
                embed.add_field(name="Stop Loss", value=f"${fields['stop_loss']:,.2f}", inline=True)
            if fields['take_profit']:
                embed.add_field(name="Take Profit", value=f"${fields['take_profit']:,.2f}", inline=True)

            embed.add_field(name="Confidence", value=f"{fields['confidence']}%", inline=True)

            if fields['risk_reward_ratio']:
                embed.add_field(name="R:R", value=f"{fields['risk_reward_ratio']:.2f}", inline=True)

            trend = fields['trend']
            if trend:
                direction = trend.get('direction', 'N/A')
                # Try legacy 'strength' field first, then prefer daily (macro), fall back to 4h
                strength = trend.get('strength')
                if strength is None:
                    strength = trend.get('strength_daily', trend.get('strength_4h', 0))
                embed.add_field(name="Trend", value=f"{direction} ({strength}%)", inline=True)

            key_levels = fields['key_levels']
            if key_levels:
                supports = key_levels.get('support', [])
                resistances = key_levels.get('resistance', [])
                if supports:
                    support_str = ", ".join([f"${s:,.2f}" for s in supports[:3]])
                    embed.add_field(name="Support", value=support_str, inline=False)
                if resistances:
                    resistance_str = ", ".join([f"${r:,.2f}" for r in resistances[:3]])
                    embed.add_field(name="Resistance", value=resistance_str, inline=False)

            embed.set_footer(text=f"Timeframe: {timeframe}")
            return embed
        except Exception as e:
            self.logger.error("Error creating embed: %s", e)
            return None
