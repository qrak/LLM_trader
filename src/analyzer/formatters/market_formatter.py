"""
Consolidated Market Analysis Formatter - Main Coordinator.
Delegates to specialized formatters.
"""
from datetime import datetime, timezone
from typing import Any

from src.logger.logger import Logger
from .market_overview_formatter import MarketOverviewFormatter
from .market_period_formatter import MarketPeriodFormatter
from .long_term_formatter import LongTermFormatter


class MarketFormatter:
    """Main coordinator for market analysis formatting."""

    def __init__(
        self,
        logger: Logger | None = None,
        format_utils=None,
        config=None,
        token_counter=None,
        overview_formatter: MarketOverviewFormatter | None = None,
        period_formatter: MarketPeriodFormatter | None = None,
        long_term_formatter: LongTermFormatter | None = None
    ):
        """Initialize the market formatter and its specialized components.

        Args:
            logger: Optional logger instance
            format_utils: Format utilities for value formatting
            config: Configuration instance (ConfigProtocol)
            token_counter: Utility for counting tokens
            overview_formatter: MarketOverviewFormatter instance
            period_formatter: MarketPeriodFormatter instance
            long_term_formatter: LongTermFormatter instance
        """
        self.logger = logger
        self.format_utils = format_utils
        self.config = config
        self.token_counter = token_counter

        self.overview_formatter = overview_formatter
        self.period_formatter = period_formatter
        self.long_term_formatter = long_term_formatter

    def _format_snapshot_timestamp(self, timestamp_ms: int | None) -> str:
        """Format a millisecond timestamp for prompt display."""
        if not timestamp_ms:
            return "Unavailable"

        snapshot_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return snapshot_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    def _format_signed_number(self, value: float | None, precision: int = 3) -> str:
        """Format signed numeric deltas consistently for prompts."""
        if value is None:
            return "N/A"
        return f"{value:+.{precision}f}"

    def _format_order_book_sentiment(self, imbalance: float) -> str:
        """Translate imbalance into a short pressure label."""
        if imbalance > 0.3:
            return "Strong Buy Pressure"
        if imbalance > 0.1:
            return "Moderate Buy Pressure"
        if imbalance < -0.3:
            return "Strong Sell Pressure"
        if imbalance < -0.1:
            return "Moderate Sell Pressure"
        return "Balanced"

    def _format_quote_value(self, value: float, quote_currency: str) -> str:
        """Format a quote-denominated value using the symbol quote convention."""
        currency_symbol = "$" if quote_currency in ["USD", "USDT", "USDC"] else f"{quote_currency} "
        return f"{currency_symbol}{self.format_utils.fmt(value, precision=2)}"

    def format_microstructure_snapshot_notice(self, symbol: str, timeframe: str, microstructure: dict[str, Any]) -> str:
        """Explain that microstructure data is a live snapshot and not timeframe aggregation."""
        snapshot_context = microstructure.get('snapshot_context', {})
        if not snapshot_context.get('is_live_snapshot'):
            return ""

        lines = [f"## {symbol} Live Microstructure Snapshot Notice:"]
        lines.append("  • Scope: Point-in-time exchange snapshot captured during this analysis cycle")
        lines.append(f"  • Timeframe Guardrail: Not aggregated over the configured {timeframe} timeframe")
        lines.append("  • Delta Basis: Changes compare only against the immediately previous analysis snapshot for this symbol")
        return "\n".join(lines)



    def format_coin_details_section(self, coin_details: dict[str, Any], max_description_tokens: int = 256) -> str:
        """Format coin details into a compressed section (removed low-trading-value data)

        Args:
            coin_details: Dictionary containing coin details from market metadata provider
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


    def format_order_book_depth(self, order_book: dict, symbol: str, timeframe: str) -> str:
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

        lines = [f"## {symbol} Live Order Book Snapshot:"]
        lines.append(f"  • Snapshot Timestamp: {self._format_snapshot_timestamp(order_book.get('timestamp'))}")
        levels_analyzed = order_book.get('levels_analyzed')
        if levels_analyzed:
            lines.append(f"  • Visible Levels Analyzed: {levels_analyzed} per side (live snapshot, not {timeframe} aggregation)")

        # Spread metrics
        spread = order_book.get("spread")
        spread_pct = order_book.get("spread_percent")
        if spread is not None and spread_pct is not None:
            lines.append(f"  • Spread: {self._format_quote_value(spread, quote_currency)} ({spread_pct:.3f}%)")

        best_bid = order_book.get("best_bid")
        best_ask = order_book.get("best_ask")
        best_bid_size = order_book.get("best_bid_size")
        best_ask_size = order_book.get("best_ask_size")
        if best_bid is not None and best_ask is not None and best_bid_size is not None and best_ask_size is not None:
            lines.append(
                f"  • Top Of Book: Bid {self._format_quote_value(best_bid, quote_currency)} x {self.format_utils.fmt(best_bid_size)} {base_currency} | "
                f"Ask {self._format_quote_value(best_ask, quote_currency)} x {self.format_utils.fmt(best_ask_size)} {base_currency}"
            )

        # Liquidity depth
        bid_depth = order_book.get("bid_depth", 0)
        ask_depth = order_book.get("ask_depth", 0)
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            lines.append(f"  • Total Visible Liquidity: {self.format_utils.fmt(total_depth)} {base_currency}")
            lines.append(f"    - Bid Depth: {self.format_utils.fmt(bid_depth)} {base_currency}")
            lines.append(f"    - Ask Depth: {self.format_utils.fmt(ask_depth)} {base_currency}")

        # Imbalance (-1 to +1, positive = more bids)
        imbalance = order_book.get("imbalance")
        if imbalance is not None:
            lines.append(f"  • Order Book Imbalance: {imbalance:+.3f} ({self._format_order_book_sentiment(imbalance)})")

        depth_by_level = order_book.get("depth_by_level", {})
        top_10 = depth_by_level.get("10")
        top_20 = depth_by_level.get("20")
        if top_10:
            lines.append(
                f"  • Top 10 Level Imbalance: {top_10.get('imbalance', 0):+.3f} "
                f"({self.format_utils.fmt(top_10.get('bid_depth', 0))} bid / {self.format_utils.fmt(top_10.get('ask_depth', 0))} ask {base_currency})"
            )
        if top_20:
            lines.append(
                f"  • Top 20 Level Imbalance: {top_20.get('imbalance', 0):+.3f} "
                f"({self.format_utils.fmt(top_20.get('bid_depth', 0))} bid / {self.format_utils.fmt(top_20.get('ask_depth', 0))} ask {base_currency})"
            )

        near_mid = order_book.get("liquidity_near_mid", {}).get("10bps")
        if near_mid:
            lines.append(
                f"  • Near-Mid Liquidity (10 bps): {self.format_utils.fmt(near_mid.get('bid_depth', 0))} bid / "
                f"{self.format_utils.fmt(near_mid.get('ask_depth', 0))} ask {base_currency} "
                f"(imbalance {near_mid.get('imbalance', 0):+.3f})"
            )

        largest_bid_wall = order_book.get("largest_bid_wall")
        if largest_bid_wall:
            lines.append(
                f"  • Largest Bid Wall: {self.format_utils.fmt(largest_bid_wall.get('amount', 0))} {base_currency} at "
                f"{self._format_quote_value(largest_bid_wall.get('price', 0), quote_currency)} "
                f"({largest_bid_wall.get('distance_bps', 0):.1f} bps from mid)"
            )

        largest_ask_wall = order_book.get("largest_ask_wall")
        if largest_ask_wall:
            lines.append(
                f"  • Largest Ask Wall: {self.format_utils.fmt(largest_ask_wall.get('amount', 0))} {base_currency} at "
                f"{self._format_quote_value(largest_ask_wall.get('price', 0), quote_currency)} "
                f"({largest_ask_wall.get('distance_bps', 0):.1f} bps from mid)"
            )

        delta = order_book.get("delta_from_previous_snapshot")
        if delta:
            interval_seconds = delta.get("snapshot_interval_seconds")
            interval_text = f" after {interval_seconds:.0f}s" if interval_seconds is not None else ""
            lines.append(f"  • Delta vs Previous Snapshot{interval_text}:")
            lines.append(f"    - Spread: {self._format_signed_number(delta.get('spread'), precision=4)}")
            lines.append(f"    - Spread %: {self._format_signed_number(delta.get('spread_percent'))}%")
            lines.append(f"    - Bid Depth: {self._format_signed_number(delta.get('bid_depth'))} {base_currency}")
            lines.append(f"    - Ask Depth: {self._format_signed_number(delta.get('ask_depth'))} {base_currency}")
            lines.append(f"    - Imbalance: {self._format_signed_number(delta.get('imbalance'))}")

            top_10_delta = delta.get("top_10", {})
            if top_10_delta:
                lines.append(
                    f"    - Top 10 Imbalance: {self._format_signed_number(top_10_delta.get('imbalance'))} "
                    f"({self._format_signed_number(top_10_delta.get('bid_depth'))} bid / "
                    f"{self._format_signed_number(top_10_delta.get('ask_depth'))} ask {base_currency})"
                )

            near_mid_delta = delta.get("near_mid_10bps", {})
            if near_mid_delta:
                lines.append(
                    f"    - Near-Mid 10 bps Imbalance: {self._format_signed_number(near_mid_delta.get('imbalance'))} "
                    f"({self._format_signed_number(near_mid_delta.get('bid_depth'))} bid / "
                    f"{self._format_signed_number(near_mid_delta.get('ask_depth'))} ask {base_currency})"
                )

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
