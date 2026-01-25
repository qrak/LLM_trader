"""
Discord Notifier - Send-only notification service with message expiration.
Sends AI trading analysis to Discord with automatic message cleanup.
"""
import asyncio

from typing import Optional, TYPE_CHECKING, List, Dict, Any

import discord
from aiohttp import ClientSession

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.parsing.unified_parser import UnifiedParser
    from src.utils.format_utils import FormatUtils

from .base_notifier import BaseNotifier
from .filehandler import DiscordFileHandler
from src.utils.decorators import retry_async



class DiscordNotifier(BaseNotifier):
    """Send-only Discord notifier with message expiration tracking."""

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

        self.bot = bot
        self.bot.discord_notifier = self
        self.bot.event(self.on_ready)
        self.file_handler = file_handler


    async def on_ready(self):
        """Called when bot is ready."""
        try:
            self.logger.info(f"DiscordNotifier: Logged in as {self.bot.user.name}")
            self.file_handler.initialize()
            self.logger.debug("FileHandler initialized")
            self.is_initialized = True
            self._ready_event.set()
            self.logger.debug("DiscordNotifier ready")
        except Exception as e:
            self.logger.error(f"Error in on_ready: {e}", exc_info=True)

    async def __aenter__(self):
        if self.session is None:
            self.session = ClientSession()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        if self.session:
            try:
                await self.session.close()
                self.session = None
            except Exception as e:
                self.logger.warning(f"Session close error: {e}")

        if self.file_handler:
            try:
                await self.file_handler.shutdown()
            except Exception as e:
                self.logger.warning(f"Error during file handler shutdown: {e}")

        if self.bot:
            try:
                self.logger.info("Closing Discord bot connection...")
                await asyncio.sleep(0.5)
                try:
                    await asyncio.wait_for(self.bot.close(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Bot close operation timed out")
            except Exception as e:
                self.logger.warning(f"Error closing Discord bot: {e}")

        self.logger.info("Discord notifier resources released")

    async def start(self) -> None:
        """Start the Discord bot."""
        if not self.bot:
            self.logger.error("Discord bot is not initialized.")
            return
        try:
            await self.bot.start(self.config.BOT_TOKEN_DISCORD)
        except discord.LoginFailure as e:
            self.logger.error(f"Discord Login Failure: {e}. Check your BOT_TOKEN_DISCORD.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Failed to start Discord bot: {e}", exc_info=True)

    async def wait_until_ready(self) -> None:
        """Wait for the bot to fully initialize."""
        await self._ready_event.wait()

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
            self.logger.error(f"Channel with ID {channel_id} not found.")
            return None

        try:
            # Hard limit at 4000 characters (2 chunks max)
            content = message[:4000]
            chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]

            sent_message = None
            for i, chunk in enumerate(chunks):
                if i > 0:
                    await asyncio.sleep(1)
                
                delete_after = float(expire_after) if expire_after is not None else None
                sent_message = await channel.send(
                    content=chunk,
                    delete_after=delete_after
                )
                await self.file_handler.track_message(
                    message_id=sent_message.id,
                    channel_id=channel_id,
                    user_id=None,
                    message_type="message",
                    expire_after=expire_after
                )
            
            self.logger.debug(f"Sent {len(chunks)} message chunk(s) (Last ID: {sent_message.id if sent_message else 'None'})")
            return sent_message
        except discord.HTTPException as e:
            self.logger.error(f"Discord HTTPException when sending message: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error when sending message: {e}", exc_info=True)
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
            self.logger.error(f"Channel with ID {channel_id} not found.")
            return None
        
        try:
            sent_message = await channel.send(embed=embed, delete_after=expire_after)
            
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
            self.logger.error(f"Error sending embed: {e}")
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
                self.logger.error(f"Channel with ID {channel_id} not found.")
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
            if decision.action in ['BUY', 'SELL'] and decision.quantity:
                entry_fee = decision.price * decision.quantity * self.config.TRANSACTION_FEE_PERCENT
                embed.add_field(name="Entry Fee", value=f"${entry_fee:.4f}", inline=True)

            embed.set_footer(text=f"Time: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            await self._send_embed(embed, channel_id)
        except Exception as e:
            self.logger.error(f"Error sending trading decision: {e}")

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def send_analysis_notification(
            self,
            result: dict,
            symbol: str,
            timeframe: str,
            channel_id: int
    ) -> None:
        """Send full analysis notification with reasoning and JSON embed.

        Args:
            result: Analysis result dict with raw_response
            symbol: Trading symbol
            timeframe: Trading timeframe
            channel_id: Discord channel ID
        """
        try:
            raw_response = result.get("raw_response", "")
            if not raw_response:
                return

            reasoning, analysis_json = self.parse_analysis_response(raw_response)

            if reasoning:
                await self.send_message(
                    message=f"**{symbol} Analysis**\n\n{reasoning}",
                    channel_id=channel_id
                )

            if analysis_json:
                embed = self._create_analysis_embed(analysis_json, symbol, timeframe)
                if embed:
                    await self._send_embed(embed, channel_id)
        except Exception as e:
            self.logger.error(f"Error sending analysis notification: {e}")

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
            if hasattr(position, 'quote_amount') and position.quote_amount > 0:
                 embed.add_field(name="Invested", value=f"${position.quote_amount:,.2f}", inline=True)
            
            embed.add_field(name="Unrealized P&L", value=f"{pnl_pct:+.2f}%", inline=True)
            embed.add_field(name=f"P&L ({self.config.QUOTE_CURRENCY})", value=f"${pnl_quote:+,.2f}", inline=True)
            embed.add_field(name="Confidence", value=position.confidence, inline=True)
            embed.add_field(name="Position Size %", value=f"{position.size_pct * 100:.2f}%", inline=True)
            embed.add_field(name="Stop Loss", value=f"${position.stop_loss:,.2f} ({stop_distance_pct:+.2f}%)", inline=True)
            embed.add_field(name="Take Profit", value=f"${position.take_profit:,.2f} ({target_distance_pct:+.2f}%)", inline=True)
            embed.add_field(name="Entry Fee", value=f"${position.entry_fee:.4f}", inline=True)
            embed.add_field(name="Time Held", value=f"{hours_held:.1f}h", inline=True)
            embed.set_footer(text=f"Entry Time: {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            await self._send_embed(embed, channel_id)
        except Exception as e:
            self.logger.error(f"Error sending position status: {e}")

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
            embed.set_footer(text=f"Symbol: {symbol}")
            await self._send_embed(embed, channel_id)
        except Exception as e:
            self.logger.error(f"Error sending performance stats: {e}")

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
                strength = trend.get('strength', 0)
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
            self.logger.error(f"Error creating embed: {e}")
            return None
