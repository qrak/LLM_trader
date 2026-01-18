"""
Market Overview Formatter - Formats global market overview data.
Handles market overview, top coins, and DeFi statistics.
"""
from typing import Optional
from datetime import datetime

from src.logger.logger import Logger


class MarketOverviewFormatter:
    """Formatter for market overview data including global metrics, top coins, and DeFi."""
    
    def __init__(self, logger: Optional[Logger] = None, format_utils=None):
        """Initialize the market overview formatter.
        
        Args:
            logger: Optional logger instance
            format_utils: Format utilities for value formatting
        """
        self.logger = logger
        self.format_utils = format_utils
    
    def format_market_overview(self, market_overview: dict, analyzed_symbol: str = None) -> str:
        """
        Format market overview data with top coins and DeFi metrics.
        
        Args:
            market_overview: Market overview data from CoinGecko
            analyzed_symbol: Trading pair being analyzed (e.g., "BTC/USDT")
                           Used to provide comparison context and exclude from top coins list
        
        Returns:
            Formatted market overview string
        """
        if not market_overview:
            return ""
        
        # Extract base symbol from trading pair (BTC/USDT -> BTC)
        analyzed_coin_symbol = None
        if analyzed_symbol:
            analyzed_coin_symbol = analyzed_symbol.split('/')[0].lower()
        
        sections = []
        
        # Market cap and dominance
        market_cap_data = market_overview.get("market_cap", {})
        if 'total_usd' in market_cap_data:
            market_cap = market_cap_data['total_usd']
            sections.append(f"Total Market Cap: ${self.format_utils.fmt(market_cap)}")
        
        dominance_data = market_overview.get("dominance", {})
        if 'btc' in dominance_data:
            btc_dom = dominance_data['btc']
            sections.append(f"Bitcoin Dominance: {self.format_utils.fmt(btc_dom)}%")
        
        if 'eth' in dominance_data:
            eth_dom = dominance_data['eth']
            sections.append(f"Ethereum Dominance: {self.format_utils.fmt(eth_dom)}%")
        
        # Market metrics
        volume_data = market_overview.get("volume", {})
        total_volume = volume_data.get('total_usd', 0)
        if total_volume:
            sections.append(f"Total Market 24h Volume: ${self.format_utils.fmt(total_volume)}")
        
        if 'change_24h' in market_cap_data:
            change = market_cap_data['change_24h']
            direction = "UP" if change >= 0 else "DOWN"
            sections.append(f"Total Market Cap Change 24h ({direction}): {self.format_utils.fmt(change)}%")
        
        # Find analyzed coin in top_coins if present
        top_coins = market_overview.get("top_coins", [])
        analyzed_coin_data = None
        other_top_coins = []
        
        if top_coins and analyzed_coin_symbol:
            for coin in top_coins:
                if coin.get("symbol", "").lower() == analyzed_coin_symbol:
                    analyzed_coin_data = coin
                else:
                    other_top_coins.append(coin)
        else:
            other_top_coins = top_coins
        
        # Show analyzed coin position if it's in top coins
        if analyzed_coin_data:
            position_summary = self._format_analyzed_coin_position(
                analyzed_coin_data, 
                market_cap_data.get('total_usd', 0),
                total_volume
            )
            if position_summary:
                sections.append(position_summary)
        
        # Top coins summary (excluding analyzed coin)
        if other_top_coins:
            top_coins_summary = self._format_top_coins_summary(
                other_top_coins[:5],
                total_volume
            )
            if top_coins_summary:
                sections.append(top_coins_summary)
        
        # DeFi metrics
        defi_data = market_overview.get("defi", {})
        if defi_data:
            total_market_cap = market_cap_data.get('total_usd', 0)
            defi_summary = self._format_defi_summary(defi_data, total_market_cap)
            if defi_summary:
                sections.append(defi_summary)
        
        if sections:
            # Format on-chain fundamentals if available
            fundamentals = market_overview.get("fundamentals")
            if fundamentals:
                fundamentals_summary = self._format_onchain_fundamentals(fundamentals)
                if fundamentals_summary:
                    sections.append(fundamentals_summary)
            
            return "## Market Overview:\n" + "\n".join([f"- {section}" for section in sections])
        
        return ""
    
    def _format_onchain_fundamentals(self, fundamentals: dict) -> str:
        """
        Format detailed on-chain fundamentals from DefiLlama.
        Includes TVL, Stablecoins, DEX Volumes, Fees, and Options.
        """
        if not fundamentals:
            return ""
            
        lines = ["## On-Chain Fundamentals (DefiLlama):"]
        
        # Macro Data (TVL & Stablecoins)
        macro = fundamentals.get("macro", {})
        if macro:
            tvl = macro.get("total_tvl", 0)
            stables = macro.get("stablecoins_market_cap", 0)
            stable_change = macro.get("stablecoins_24h_change", 0)
            
            if tvl:
                lines.append(f"  • TVL: ${self.format_utils.fmt(tvl)}")
            if stables:
                lines.append(f"  • Dry Powder (Stablecoins): ${self.format_utils.fmt(stables)} ({stable_change:+.1f}%)")
        
        # Activity (DEX & Fees)
        activity_lines = []
        dex_data = fundamentals.get("dex_volumes", {})
        if dex_data and dex_data.get("total_24h"):
            dex_vol = dex_data.get("total_24h", 0)
            dex_change = dex_data.get("change_1d", 0)
            top_dex = dex_data.get("top_protocols", [])
            top_dex_str = f" (Top: {top_dex[0].get('name')} ${self.format_utils.fmt(top_dex[0].get('total24h', 0))})" if top_dex else ""
            activity_lines.append(f"    - DEX Volume: ${self.format_utils.fmt(dex_vol)}{top_dex_str} | 24h Change: {dex_change:+.1f}%")
            
        fees_data = fundamentals.get("fees", {})
        if fees_data and fees_data.get("total_24h_fees"):
            fees = fees_data.get("total_24h_fees", 0)
            top_earner = fees_data.get("top_earners", [])
            top_earner_str = f" (Top: {top_earner[0].get('name')} ${self.format_utils.fmt(top_earner[0].get('total24h', 0))})" if top_earner else ""
            activity_lines.append(f"    - Protocol Fees: ${self.format_utils.fmt(fees)}{top_earner_str}")
            
        if activity_lines:
            lines.append("  • Activity (24h):")
            lines.extend(activity_lines)
            
        # Options (Smart Money)
        options_data = fundamentals.get("options", {})
        if options_data and options_data.get("notional_volume_24h"):
            notional = options_data.get("notional_volume_24h", 0)
            premium = options_data.get("premium_volume_24h", 0)
            lines.append("  • Smart Money (Options):")
            lines.append(f"    - Notional Volume: ${self.format_utils.fmt(notional)}")
            if premium:
                lines.append(f"    - Premium Volume: ${self.format_utils.fmt(premium)}")
                
        return "\n".join(lines)
    
    def _format_analyzed_coin_position(self, coin_data: dict, total_market_cap: float, total_volume: float) -> str:
        """
        Format position summary for the coin being analyzed.
        Shows its rank, market share, volume share, and supply metrics.
        """
        if not coin_data:
            return ""
        
        symbol = coin_data.get("symbol", "?").upper()
        name = coin_data.get("name", symbol)
        rank = coin_data.get("market_cap_rank", "?")
        market_cap = coin_data.get("market_cap", 0)
        coin_volume = coin_data.get("total_volume", 0)
        circ_supply = coin_data.get("circulating_supply", 0)
        max_supply = coin_data.get("max_supply")
        
        lines = [f"## {symbol} ({name}) Market Position:"]
        
        # Rank and market share
        if rank and market_cap and total_market_cap:
            market_share = (market_cap / total_market_cap * 100) if total_market_cap > 0 else 0
            lines.append(f"  • Rank: #{rank} | Market Cap: ${self.format_utils.fmt(market_cap)} ({market_share:.2f}% of total)")
        
        # Volume share and liquidity
        if coin_volume and total_volume:
            volume_share = (coin_volume / total_volume * 100) if total_volume > 0 else 0
            lines.append(f"  • 24h Volume: ${self.format_utils.fmt(coin_volume)} ({volume_share:.2f}% of total market volume)")
        
        # Supply metrics
        if circ_supply:
            if max_supply:
                supply_pct = (circ_supply / max_supply * 100)
                lines.append(f"  • Supply: {self.format_utils.fmt(circ_supply)} circulating / {self.format_utils.fmt(max_supply)} max ({supply_pct:.1f}%)")
            else:
                lines.append(f"  • Circulating Supply: {self.format_utils.fmt(circ_supply)} (no max supply)")
        
        # Market cap change
        mcap_change_24h = coin_data.get("market_cap_change_percentage_24h", 0)
        if mcap_change_24h:
            direction = "UP" if mcap_change_24h >= 0 else "DOWN"
            lines.append(f"  • Market Cap 24h Change ({direction}): {mcap_change_24h:+.2f}%")
        
        return "\n".join(lines)
    
    def _format_top_coins_summary(self, top_coins: list, total_volume: float = 0) -> str:
        """
        Format top coins with ATH, price changes, and volume metrics.
        Shows comparison data useful for relative analysis.
        """
        if not top_coins:
            return ""
        
        lines = ["## Top Coins Status (Market Leaders):"]
        for coin in top_coins:
            symbol = coin.get("symbol", "?").upper()
            name = coin.get("name", symbol)
            rank = coin.get("market_cap_rank", "?")
            price = coin.get("current_price", 0)
            change_1h = coin.get("price_change_percentage_1h_in_currency", 0)
            change_24h = coin.get("price_change_percentage_24h", 0)
            change_7d = coin.get("price_change_percentage_7d_in_currency", 0)
            ath = coin.get("ath", 0)
            ath_pct = coin.get("ath_change_percentage", 0)
            ath_date = coin.get("ath_date", "")
            coin_volume = coin.get("total_volume", 0)
            
            # Parse ATH date if available
            ath_date_str = ""
            if ath_date:
                try:
                    dt = datetime.fromisoformat(ath_date.replace('Z', '+00:00'))
                    ath_date_str = dt.strftime("%b %d, %Y")
                except (ValueError, TypeError):
                    pass
            
            # Format: Rank #X Symbol (Name): $price (momentum data) | volume | ATH context
            line_parts = [f"  • #{rank} {symbol} ({name}): ${price:,.2f}"]
            
            # Momentum: 1h, 24h, 7d
            momentum_parts = []
            if change_1h:
                momentum_parts.append(f"{change_1h:+.1f}% 1h")
            if change_24h is not None:
                momentum_parts.append(f"{change_24h:+.2f}% 24h")
            if change_7d:
                momentum_parts.append(f"{change_7d:+.1f}% 7d")
            
            if momentum_parts:
                line_parts.append(f"({', '.join(momentum_parts)})")
            
            # Volume share (if total volume provided)
            if coin_volume and total_volume:
                volume_share = (coin_volume / total_volume * 100)
                line_parts.append(f"| Vol: {volume_share:.1f}% of market")
            
            # ATH context
            if ath and ath_pct:
                ath_info = f"| ATH: ${ath:,.2f}"
                if ath_date_str:
                    ath_info += f" ({ath_date_str})"
                ath_info += f", now {ath_pct:+.1f}%"
                line_parts.append(ath_info)
            
            lines.append(" ".join(line_parts))
        
        return "\n".join(lines)
    
    def _format_defi_summary(self, defi_data: dict, total_market_cap: float) -> str:
        """Format DeFi metrics."""
        if not defi_data:
            return ""
        
        try:
            defi_mcap = float(defi_data.get("defi_market_cap", 0))
            defi_dom = float(defi_data.get("defi_dominance", 0))
            defi_vol = float(defi_data.get("trading_volume_24h", 0))
            
            lines = [
                "## DeFi Market:",
                f"  • DeFi Market Cap: ${self.format_utils.fmt(defi_mcap)}",
                f"  • DeFi Dominance: {defi_dom:.2f}%"
            ]
            
            if total_market_cap > 0:
                defi_pct_of_total = (defi_mcap / total_market_cap * 100)
                lines.append(f"  • DeFi % of Total Market: {defi_pct_of_total:.2f}%")
            
            if defi_vol > 0:
                lines.append(f"  • 24h DeFi Volume: ${self.format_utils.fmt(defi_vol)}")
            
            top_coin = defi_data.get("top_coin_name")
            top_coin_dom = defi_data.get("top_coin_defi_dominance")
            if top_coin and top_coin_dom:
                lines.append(f"  • Top DeFi Asset: {top_coin} ({top_coin_dom:.1f}% of DeFi)")
            
            return "\n".join(lines)
        except (ValueError, TypeError) as e:
            if self.logger:
                self.logger.warning("Error formatting DeFi summary: %s", e)
            return ""
