"""
Consolidated Market Analysis Formatter - Main Coordinator.
Delegates to specialized formatters.
"""
from typing import Dict, Any, Optional

from src.logger.logger import Logger
from .market_overview_formatter import MarketOverviewFormatter
from .market_period_formatter import MarketPeriodFormatter
from .long_term_formatter import LongTermFormatter


class MarketFormatter:
    """Main coordinator for market analysis formatting."""
    
    def __init__(self, logger: Optional[Logger] = None, format_utils=None, config=None, token_counter=None):
        """Initialize the market formatter and its specialized components.
        
        Args:
            logger: Optional logger instance
            format_utils: Format utilities for value formatting
            config: Configuration instance (ConfigProtocol)
            token_counter: Utility for counting tokens
        """
        self.logger = logger
        self.format_utils = format_utils
        self.config = config
        self.token_counter = token_counter
        self.overview_formatter = MarketOverviewFormatter(logger, format_utils)
        self.period_formatter = MarketPeriodFormatter(logger, format_utils)
        self.long_term_formatter = LongTermFormatter(logger, format_utils)
    

    
    def format_coin_details_section(self, coin_details: Dict[str, Any], max_description_tokens: int = 256) -> str:
        """Format coin details into a compressed section (removed low-trading-value data)
        
        Args:
            coin_details: Dictionary containing coin details from CryptoCompare API
            max_description_tokens: Maximum tokens allowed for description (default: 256)
            
        Returns:
            str: Compressed coin details section
        """
        if not coin_details:
            return ""
        
        # Only include high-value trading data (removed: Algorithm, Proof Type, Regulatory Classifications, Weiss Ratings)
        # These rarely change and don't impact immediate trading decisions
        
        section = "## Cryptocurrency Details\n"
        
        # Basic information only
        if coin_details.get("full_name"):
            section += f"- {coin_details['full_name']}"
            if coin_details.get("coin_name"):
                section += f" ({coin_details['coin_name']} Project)"
            section += "\n"
        
        # Project description (optional - based on config)
        include_description = self.config.INCLUDE_COIN_DESCRIPTION if self.config else False
        if include_description:
            description = coin_details.get("description", "")
            if description:
                # Use token-based truncation instead of character-based
                description_tokens = self.token_counter.count_tokens(description)
                
                if description_tokens > max_description_tokens:
                    # Truncate by sentences to maintain readability
                    description = self._truncate_description_by_tokens(description, max_description_tokens)
                    
                section += f"\nProject Description:\n{description}\n"
        
        return section
    
    def _truncate_description_by_tokens(self, description: str, max_tokens: int) -> str:
        """Truncate description by tokens while preserving sentence boundaries
        
        Args:
            description: The original description text
            max_tokens: Maximum tokens allowed
            
        Returns:
            str: Truncated description ending with complete sentences
        """
        # Split by sentences (simple approach)
        sentences = description.split('. ')
        truncated = ""
        
        for i, sentence in enumerate(sentences):
            # Add sentence with proper punctuation
            test_text = truncated + (sentence if sentence.endswith('.') else sentence + '.')
            if i < len(sentences) - 1:
                test_text += ' '
            
            # Check if adding this sentence would exceed token limit
            if self.token_counter.count_tokens(test_text) > max_tokens:
                # If even the first sentence is too long, truncate it directly
                if not truncated:
                    words = sentence.split()
                    for j, _ in enumerate(words):
                        test_word_text = ' '.join(words[:j+1]) + '...'
                        if self.token_counter.count_tokens(test_word_text) > max_tokens:
                            if j == 0:  # Even first word is too long
                                return sentence[:50] + '...'
                            return ' '.join(words[:j]) + '...'
                    return sentence + '...'
                else:
                    # Add ellipsis to indicate truncation
                    return truncated.rstrip() + '...'
            
            truncated = test_text
        
        return truncated
    
    def format_ticker_data(self, ticker_data: dict, symbol: str) -> str:
        """
        Format ticker data for the analyzed coin.
        Shows VWAP, bid/ask spreads, and volume metrics from CCXT ticker.
        
        Args:
            ticker_data: Ticker data dict with VWAP, bid, ask, volumes
            symbol: Trading pair symbol (e.g., "BTC/USDT")
        
        Returns:
            Formatted ticker string
        """
        if not ticker_data:
            return ""
        
        lines = [f"## {symbol} Real-Time Ticker Data:"]
        
        # Price metrics
        vwap = ticker_data.get("VWAP")
        last = ticker_data.get("LAST")
        if vwap:
            lines.append(f"  • VWAP (24h): ${self.format_utils.fmt(vwap, precision=2)}")
        if last and vwap:
            vwap_diff = ((last - vwap) / vwap * 100)
            direction = "above" if vwap_diff >= 0 else "below"
            lines.append(f"  • Current Price vs VWAP: {vwap_diff:+.2f}% ({direction})")
        
        # Bid/Ask spread
        bid = ticker_data.get("BID")
        ask = ticker_data.get("ASK")
        if bid and ask:
            spread = ask - bid
            spread_pct = (spread / bid * 100) if bid > 0 else 0
            lines.append(f"  • Bid/Ask Spread: ${spread:.2f} ({spread_pct:.3f}%)")
            lines.append(f"    - Best Bid: ${self.format_utils.fmt(bid, precision=2)} | Best Ask: ${self.format_utils.fmt(ask, precision=2)}")
        
        # Volume metrics
        volume = ticker_data.get("VOLUME24HOUR")
        quote_volume = ticker_data.get("QUOTEVOLUME24HOUR")
        if volume:
            lines.append(f"  • 24h Volume: {self.format_utils.fmt(volume, precision=2)} {symbol.split('/')[0]}")
        if quote_volume:
            lines.append(f"  • 24h Quote Volume: ${self.format_utils.fmt(quote_volume)}")
        
        # Bid/Ask volume (liquidity at best levels)
        bid_volume = ticker_data.get("BIDVOLUME")
        ask_volume = ticker_data.get("ASKVOLUME")
        if bid_volume and ask_volume:
            total_depth = bid_volume + ask_volume
            bid_pct = (bid_volume / total_depth * 100) if total_depth > 0 else 0
            lines.append(f"  • Liquidity at Best Levels: {self.format_utils.fmt(bid_volume + ask_volume, precision=2)} ({bid_pct:.1f}% bid / {100-bid_pct:.1f}% ask)")
        
        # 24h Range
        high = ticker_data.get("HIGH24HOUR")
        low = ticker_data.get("LOW24HOUR")
        if high and low and last:
            range_size = high - low
            range_pct = (range_size / low * 100) if low > 0 else 0
            position_in_range = ((last - low) / range_size * 100) if range_size > 0 else 50
            lines.append(f"  • 24h Range: ${self.format_utils.fmt(low, precision=2)} - ${self.format_utils.fmt(high, precision=2)} ({range_pct:.2f}% range)")
            lines.append(f"  • Current Position in Range: {position_in_range:.1f}%")
        
        return "\n".join(lines)

    
    def format_order_book_depth(self, order_book: dict, symbol: str) -> str:
        """
        Format order book depth data.
        Shows bid/ask imbalance, liquidity depth, and spread metrics.
        
        Args:
            order_book: Order book dict from fetch_order_book_depth
            symbol: Trading pair symbol
        
        Returns:
            Formatted order book string
        """
        if not order_book or "error" in order_book:
            return ""
        
        base_currency = symbol.split('/')[0] if '/' in symbol else ""
        quote_currency = symbol.split('/')[1] if '/' in symbol else ""
        
        lines = [f"## {symbol} Order Book Depth:"]
        
        # Spread metrics
        spread = order_book.get("spread")
        spread_pct = order_book.get("spread_percent")
        if spread is not None and spread_pct is not None:
            # Spread is in quote currency (price difference)
            currency_symbol = "$" if quote_currency in ["USD", "USDT", "USDC"] else f"{quote_currency} "
            lines.append(f"  • Spread: {currency_symbol}{spread:.2f} ({spread_pct:.3f}%)")
        
        # Liquidity depth
        bid_depth = order_book.get("bid_depth", 0)
        ask_depth = order_book.get("ask_depth", 0)
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            lines.append(f"  • Total Liquidity (Top 20 levels): {self.format_utils.fmt(total_depth)} {base_currency}")
            lines.append(f"    - Bid Depth: {self.format_utils.fmt(bid_depth)} {base_currency}")
            lines.append(f"    - Ask Depth: {self.format_utils.fmt(ask_depth)} {base_currency}")
        
        # Imbalance (-1 to +1, positive = more bids)
        imbalance = order_book.get("imbalance")
        if imbalance is not None:
            if imbalance > 0.3:
                sentiment = "Strong Buy Pressure"
            elif imbalance > 0.1:
                sentiment = "Moderate Buy Pressure"
            elif imbalance < -0.3:
                sentiment = "Strong Sell Pressure"
            elif imbalance < -0.1:
                sentiment = "Moderate Sell Pressure"
            else:
                sentiment = "Balanced"
            lines.append(f"  • Order Book Imbalance: {imbalance:+.3f} ({sentiment})")
        
        return "\n".join(lines)
    
    def format_trade_flow(self, trades: dict, symbol: str) -> str:
        """
        Format recent trade flow analysis.
        Shows buy/sell pressure, trade velocity, and order flow metrics.
        
        Args:
            trades: Trade flow dict from fetch_recent_trades
            symbol: Trading pair symbol
        
        Returns:
            Formatted trade flow string
        """
        if not trades or "error" in trades:
            return ""
        
        base_currency = symbol.split('/')[0] if '/' in symbol else ""
        
        lines = [f"## {symbol} Recent Trade Flow:"]
        
        # Trade count and velocity (real-time data, independent of analysis timeframe)
        total_trades = trades.get("total_trades", 0)
        velocity = trades.get("trade_velocity")
        time_span = trades.get("time_span_minutes")
        if total_trades:
            time_context = f" (last {time_span:.1f} min)" if time_span else ""
            lines.append(f"  • Total Recent Trades: {total_trades}{time_context}")
        if velocity:
            lines.append(f"  • Trade Velocity: {velocity:.2f} trades/minute")
        
        # Buy/Sell pressure
        buy_volume = trades.get("buy_volume", 0)
        sell_volume = trades.get("sell_volume", 0)
        buy_sell_ratio = trades.get("buy_sell_ratio")
        buy_pressure = trades.get("buy_pressure_percent")
        
        if buy_volume and sell_volume:
            lines.append(f"  • Buy Volume: {self.format_utils.fmt(buy_volume)} {base_currency}")
            lines.append(f"  • Sell Volume: {self.format_utils.fmt(sell_volume)} {base_currency}")
        
        if buy_sell_ratio:
            lines.append(f"  • Buy/Sell Ratio: {buy_sell_ratio:.2f}")
        
        if buy_pressure is not None:
            if buy_pressure > 60:
                sentiment = "Strong Buying"
            elif buy_pressure > 52:
                sentiment = "Moderate Buying"
            elif buy_pressure < 40:
                sentiment = "Strong Selling"
            elif buy_pressure < 48:
                sentiment = "Moderate Selling"
            else:
                sentiment = "Balanced"
            lines.append(f"  • Buy Pressure: {buy_pressure:.1f}% ({sentiment})")
        
        # Average trade size
        avg_trade_size = trades.get("avg_trade_size")
        if avg_trade_size:
            lines.append(f"  • Average Trade Size: {self.format_utils.fmt(avg_trade_size)} {base_currency}")
        
        return "\n".join(lines)
    
    def format_funding_rate(self, funding: dict, symbol: str) -> str:
        """
        Format funding rate data for futures/perpetual contracts.
        Shows funding rate, annualized rate, and sentiment.
        
        Args:
            funding: Funding rate dict from fetch_funding_rate
            symbol: Trading pair symbol
        
        Returns:
            Formatted funding rate string
        """
        if not funding or "error" in funding:
            return f"## {symbol} Funding Rate (Futures):\n  • Status: Not Available (Spot Market or Data Unavailable)"
        
        lines = [f"## {symbol} Funding Rate (Futures):"]
        
        # Funding rate
        rate_pct = funding.get("funding_rate_percent")
        if rate_pct is not None:
            lines.append(f"  • Current Funding Rate: {rate_pct:.4f}%")
        
        # Annualized rate
        annualized = funding.get("annualized_rate")
        if annualized is not None:
            lines.append(f"  • Annualized Rate: {annualized:.2f}%")
        
        # Sentiment interpretation
        sentiment = funding.get("sentiment", "Unknown")
        lines.append(f"  • Market Sentiment: {sentiment}")
        
        # Explanation
        if rate_pct is not None:
            if rate_pct > 0.01:
                lines.append("  • Interpretation: Longs pay shorts (bullish positioning)")
            elif rate_pct < -0.01:
                lines.append("  • Interpretation: Shorts pay longs (bearish positioning)")
            else:
                lines.append("  • Interpretation: Neutral positioning")
        
        return "\n".join(lines)
