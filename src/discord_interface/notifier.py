"""
Discord Notifier - Send-only notification service with message expiration.
Sends AI trading analysis to Discord with automatic message cleanup.
"""
import asyncio
import re
import json
from datetime import datetime
from typing import Optional, TYPE_CHECKING, List, Dict, Any

import discord
from aiohttp import ClientSession

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.parsing.unified_parser import UnifiedParser

from .filehandler import DiscordFileHandler
from src.utils.decorators import retry_async


class DiscordNotifier:
    """Send-only Discord notifier with message expiration tracking."""

    def __init__(self, logger, config: "ConfigProtocol", unified_parser: "UnifiedParser") -> None:
        """Initialize DiscordNotifier.
        
        Args:
            logger: Logger instance
            config: ConfigProtocol instance for Discord settings
            unified_parser: UnifiedParser for JSON extraction (DRY)
        """
        if config is None:
            raise ValueError("config is a required parameter and cannot be None")
        
        self.logger = logger
        self.config = config
        self.unified_parser = unified_parser
        self.session: Optional[ClientSession] = None
        self.is_initialized = False
        self._ready_event = asyncio.Event()

        intents = discord.Intents.default()
        intents.message_content = False
        intents.reactions = False
        intents.typing = False
        intents.presences = False

        self.bot = discord.Client(intents=intents)
        self.bot.discord_notifier = self

        # Register event handler
        self.bot.event(self.on_ready)

        # Initialize the file handler for message tracking/expiration
        self.file_handler = DiscordFileHandler(self.bot, self.logger, self.config)

    async def on_ready(self):
        """Called when bot is ready."""
        try:
            self.logger.info(f"DiscordNotifier: Logged in as {self.bot.user.name}")

            # Initialize file handler for message expiration
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
            # Use Discord's built-in timed deletion
            delete_after = float(expire_after) if expire_after is not None else None
            sent_message = await channel.send(
                content=message[:2000],
                delete_after=delete_after
            )
            
            # Track message for backup expiration (belt-and-suspenders)
            await self.file_handler.track_message(
                message_id=sent_message.id,
                channel_id=channel_id,
                user_id=None,
                message_type="message",
                expire_after=expire_after
            )
            self.logger.debug(f"Sent and tracking message (ID: {sent_message.id})")
            
            return sent_message
        except discord.HTTPException as e:
            self.logger.error(f"Discord HTTPException when sending message: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error when sending message: {e}", exc_info=True)
        
        return None
    
    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def send_trading_decision(
            self,
            decision: Any,
            channel_id: int
    ) -> None:
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

            # Determine color based on action
            color_map = {
                'BUY': discord.Color.green(),
                'SELL': discord.Color.red(),
                'HOLD': discord.Color.light_grey(),
                'CLOSE': discord.Color.orange(),
                'CLOSE_LONG': discord.Color.orange(),
                'CLOSE_SHORT': discord.Color.orange(),
                'UPDATE': discord.Color.blue(),
            }
            color = color_map.get(decision.action, discord.Color.light_grey())
            
            emoji_map = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '⚪',
                'CLOSE': '🟠',
                'CLOSE_LONG': '🟠',
                'CLOSE_SHORT': '🟠',
                'UPDATE': '🔵',
            }
            emoji = emoji_map.get(decision.action, '📊')
            
            embed = discord.Embed(
                title=f"{emoji} TRADING DECISION: {decision.action}",
                description=decision.reasoning[:1024] if decision.reasoning else "No reasoning provided",
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
                embed.add_field(name="Position Size", value=f"{decision.position_size * 100:.1f}%", inline=True)
                # Calculate and display entry fee for BUY/SELL actions
                if decision.action in ['BUY', 'SELL']:
                    entry_fee = decision.price * decision.position_size * self.config.TRANSACTION_FEE_PERCENT
                    embed.add_field(name="Entry Fee", value=f"${entry_fee:.4f}", inline=True)
            embed.set_footer(text=f"Time: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            await channel.send(embed=embed, delete_after=float(self.config.FILE_MESSAGE_EXPIRY))
            
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
            
            # Extract reasoning (text before JSON block) and JSON analysis using UnifiedParser
            reasoning = self.unified_parser.extract_text_before_json(raw_response)
            analysis_json = self.unified_parser.extract_json_block(raw_response, unwrap_key='analysis')
            
            # Send reasoning as regular message
            if reasoning:
                await self.send_message(
                    message=f"**{symbol} Analysis**\n\n{reasoning}",
                    channel_id=channel_id
                )
            
            # Send JSON data as embed if present
            if analysis_json:
                embed = self._create_analysis_embed(analysis_json, symbol, timeframe)
                if embed:
                    channel = self.bot.get_channel(channel_id)
                    if channel:
                        await channel.send(embed=embed, delete_after=float(self.config.FILE_MESSAGE_EXPIRY))
                        
        except Exception as e:
            self.logger.error(f"Error sending analysis notification: {e}")
    
    async def send_position_status(
            self,
            position: Any,  # Position dataclass
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
            # Calculate unrealized P&L
            pnl_pct = position.calculate_pnl(current_price)
            pnl_usdt = (current_price - position.entry_price) * position.size if position.direction == 'LONG' else (position.entry_price - current_price) * position.size
            
            # Calculate distance to stop and target
            if position.direction == 'LONG':
                stop_distance_pct = ((position.stop_loss - current_price) / current_price) * 100
                target_distance_pct = ((position.take_profit - current_price) / current_price) * 100
            else:  # SHORT
                stop_distance_pct = ((current_price - position.stop_loss) / current_price) * 100
                target_distance_pct = ((current_price - position.take_profit) / current_price) * 100
            
            # Determine embed color based on P&L
            if pnl_pct > 0:
                color = discord.Color.green()
                emoji = "📈"
            elif pnl_pct < 0:
                color = discord.Color.red()
                emoji = "📉"
            else:
                color = discord.Color.light_grey()
                emoji = "➡️"
            
            # Create position embed
            embed = discord.Embed(
                title=f"{emoji} Open {position.direction} Position - {position.symbol}",
                description=f"Current position monitoring",
                color=color
            )
            
            # Position details
            embed.add_field(name="Entry Price", value=f"${position.entry_price:,.2f}", inline=True)
            embed.add_field(name="Current Price", value=f"${current_price:,.2f}", inline=True)
            embed.add_field(name="Position Size", value=f"{position.size:.4f}", inline=True)
            
            # P&L
            embed.add_field(name="Unrealized P&L", value=f"{pnl_pct:+.2f}%", inline=True)
            embed.add_field(name=f"P&L ({self.config.QUOTE_CURRENCY})", value=f"${pnl_usdt:+,.2f}", inline=True)
            embed.add_field(name="Confidence", value=position.confidence, inline=True)
            
            # Stop Loss and Take Profit
            embed.add_field(name="Stop Loss", value=f"${position.stop_loss:,.2f} ({stop_distance_pct:+.2f}%)", inline=True)
            embed.add_field(name="Take Profit", value=f"${position.take_profit:,.2f} ({target_distance_pct:+.2f}%)", inline=True)
            embed.add_field(name="Entry Fee", value=f"${position.entry_fee:.4f}", inline=True)
            # Entry time
            time_held = datetime.now() - position.entry_time
            hours_held = time_held.total_seconds() / 3600
            embed.add_field(name="Time Held", value=f"{hours_held:.1f}h", inline=True)
            
            embed.set_footer(text=f"Entry Time: {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            channel = self.bot.get_channel(channel_id)
            if channel:
                await channel.send(embed=embed, delete_after=float(self.config.FILE_MESSAGE_EXPIRY))
                
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
            if not trade_history:
                return
            # Calculate overall performance
            total_pnl_usdt = 0.0
            total_pnl_pct = 0.0
            total_fees = 0.0
            closed_trades = 0
            winning_trades = 0
            open_position = None
            for decision_dict in trade_history:
                action = decision_dict.get('action', '')
                price = decision_dict.get('price', 0)
                position_size = decision_dict.get('position_size', 1.0)
                if action in ['BUY', 'SELL']:
                    open_position = decision_dict
                elif action in ['CLOSE', 'CLOSE_LONG', 'CLOSE_SHORT'] and open_position:
                    open_action = open_position.get('action', '')
                    open_price = open_position.get('price', 0)
                    open_size = open_position.get('position_size', 1.0)
                    if open_action == 'BUY':
                        pnl_pct = ((price - open_price) / open_price) * 100
                        pnl_usdt = (price - open_price) * position_size
                    else:  # SELL
                        pnl_pct = ((open_price - price) / open_price) * 100
                        pnl_usdt = (open_price - price) * position_size
                    # Calculate fees for this trade (entry + exit)
                    entry_fee = open_price * open_size * self.config.TRANSACTION_FEE_PERCENT
                    exit_fee = price * position_size * self.config.TRANSACTION_FEE_PERCENT
                    total_fees += entry_fee + exit_fee
                    total_pnl_usdt += pnl_usdt
                    total_pnl_pct += pnl_pct
                    closed_trades += 1
                    if pnl_pct > 0:
                        winning_trades += 1
                    open_position = None
            if closed_trades == 0:
                return
            avg_pnl_pct = total_pnl_pct / closed_trades
            win_rate = (winning_trades / closed_trades) * 100
            net_pnl = total_pnl_usdt - total_fees
            # Create performance embed
            embed = discord.Embed(
                title="📈 Trading Performance Summary",
                description=f"Overall performance after {closed_trades} closed trades",
                color=discord.Color.blue() if net_pnl > 0 else discord.Color.red()
            )
            embed.add_field(name=f"Total P&L ({self.config.QUOTE_CURRENCY})", value=f"${total_pnl_usdt:+,.2f}", inline=True)
            embed.add_field(name="Total P&L (%)", value=f"{total_pnl_pct:+.2f}%", inline=True)
            embed.add_field(name="Avg P&L/Trade", value=f"{avg_pnl_pct:+.2f}%", inline=True)
            embed.add_field(name="Win Rate", value=f"{win_rate:.1f}% ({winning_trades}/{closed_trades})", inline=True)
            embed.add_field(name="Total Trades", value=str(closed_trades), inline=True)
            embed.add_field(name="Total Fees", value=f"${total_fees:.4f}", inline=True)
            embed.add_field(name=f"Net P&L ({self.config.QUOTE_CURRENCY})", value=f"${net_pnl:+,.2f}", inline=True)
            embed.set_footer(text=f"Symbol: {symbol}")
            channel = self.bot.get_channel(channel_id)
            if channel:
                await channel.send(embed=embed, delete_after=float(self.config.FILE_MESSAGE_EXPIRY))
        except Exception as e:
            self.logger.error(f"Error sending performance stats: {e}")
    

    
    def _create_analysis_embed(self, analysis: dict, symbol: str, timeframe: str) -> Optional[discord.Embed]:
        """Create Discord embed from analysis JSON."""
        try:
            signal = analysis.get('signal', 'UNKNOWN')
            confidence = analysis.get('confidence', 0)
            reasoning = analysis.get('reasoning', 'No reasoning provided')
            
            # Determine color based on signal
            color_map = {
                'BUY': discord.Color.green(),
                'SELL': discord.Color.red(),
                'HOLD': discord.Color.light_grey(),
                'UPDATE': discord.Color.blue(),
            }
            color = color_map.get(signal, discord.Color.light_grey())
            
            embed = discord.Embed(
                title=f"📊 {symbol} - {signal}",
                description=reasoning[:1024],
                color=color
            )
            
            # Add key fields
            if 'entry_price' in analysis:
                embed.add_field(name="Entry", value=f"${analysis['entry_price']:,.2f}", inline=True)
            if 'stop_loss' in analysis:
                embed.add_field(name="Stop Loss", value=f"${analysis['stop_loss']:,.2f}", inline=True)
            if 'take_profit' in analysis:
                embed.add_field(name="Take Profit", value=f"${analysis['take_profit']:,.2f}", inline=True)
            
            embed.add_field(name="Confidence", value=f"{confidence}%", inline=True)
            
            if 'risk_reward_ratio' in analysis:
                embed.add_field(name="R:R", value=f"{analysis['risk_reward_ratio']:.2f}", inline=True)
            
            # Add trend info
            trend = analysis.get('trend', {})
            if trend:
                direction = trend.get('direction', 'N/A')
                strength = trend.get('strength', 0)
                embed.add_field(name="Trend", value=f"{direction} ({strength}%)", inline=True)
            
            # Add key levels
            key_levels = analysis.get('key_levels', {})
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

