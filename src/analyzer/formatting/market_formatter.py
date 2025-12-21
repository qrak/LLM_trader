"""
Consolidated Market Analysis Formatter.
Handles all market analysis formatting in a single comprehensive class.
"""
from typing import Dict, List, Optional, Any
from src.logger.logger import Logger
from src.utils.token_counter import TokenCounter
from src.indicators.constants import INDICATOR_THRESHOLDS


class MarketFormatter:
    """Consolidated formatter for all market analysis sections."""
    
    def __init__(self, logger: Optional[Logger] = None, format_utils=None):
        """Initialize the market formatter."""
        self.logger = logger
        self.token_counter = TokenCounter()
        self.format_utils = format_utils
        # Reference the global indicator thresholds constant
        self.INDICATOR_THRESHOLDS = INDICATOR_THRESHOLDS
    
    def format_market_period_metrics(self, market_metrics: dict) -> str:
        """Format market metrics for different periods."""
        if not market_metrics:
            return ""
        
        sections = []
        
        for period, period_data in market_metrics.items():
            if not period_data:
                continue
            
            # Extract metrics from nested structure
            metrics = period_data.get('metrics', {})
            if not metrics:
                continue
                
            period_sections = []
            period_sections.extend(self._format_period_price_section(metrics))
            period_sections.extend(self._format_period_volume_section(metrics))
            
            # Add indicator changes if available
            if 'indicator_changes' in period_data:
                period_sections.extend(self._format_indicator_changes_section(
                    period_data['indicator_changes'], period
                ))
            
            if period_sections:
                sections.append(f"\n{period.upper()} Analysis:")
                sections.extend(period_sections)
        
        return "\n".join(sections)
    
    def format_long_term_analysis(self, long_term_data: dict, current_price: float = None) -> str:
        """Format comprehensive long-term analysis from historical data."""
        if not long_term_data:
            return ""
        
        sections = []
        
        # Simple Moving Averages
        sma_section = self._format_sma_section(long_term_data)
        if sma_section:
            sections.append(sma_section)
        
        # Volume SMAs
        volume_sma_section = self._format_volume_sma_section(long_term_data)
        if volume_sma_section:
            sections.append(volume_sma_section)
        
        # Price position analysis
        if current_price:
            price_position_section = self._format_price_position_section(long_term_data, current_price)
            if price_position_section:
                sections.append(price_position_section)
        
        # Daily indicators
        if current_price:
            daily_indicators_section = self._format_daily_indicators_section(long_term_data, current_price)
            if daily_indicators_section:
                sections.append(daily_indicators_section)
        
        # ADX analysis
        adx_section = self._format_adx_section(long_term_data)
        if adx_section:
            sections.append(adx_section)
        
        # Ichimoku analysis
        if current_price:
            ichimoku_section = self._format_ichimoku_section(long_term_data, current_price)
            if ichimoku_section:
                sections.append(ichimoku_section)
        
        # Macro trend analysis (365-day SMA context)
        if 'macro_trend' in long_term_data:
            macro_trend_section = self._format_macro_trend_section(long_term_data['macro_trend'])
            if macro_trend_section:
                sections.append(macro_trend_section)
        
        if sections:
            return "\n\n".join(sections)
        
        return ""
    
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
            sections.append(f"📊 Total Market Cap: ${self.format_utils.fmt(market_cap)}")
        
        dominance_data = market_overview.get("dominance", {})
        if 'btc' in dominance_data:
            btc_dom = dominance_data['btc']
            sections.append(f"₿ Bitcoin Dominance: {self.format_utils.fmt(btc_dom)}%")
        
        if 'eth' in dominance_data:
            eth_dom = dominance_data['eth']
            sections.append(f"Ξ Ethereum Dominance: {self.format_utils.fmt(eth_dom)}%")
        
        # Market metrics
        volume_data = market_overview.get("volume", {})
        total_volume = volume_data.get('total_usd', 0)
        if total_volume:
            sections.append(f"📈 Total Market 24h Volume: ${self.format_utils.fmt(total_volume)}")
        
        if 'change_24h' in market_cap_data:
            change = market_cap_data['change_24h']
            direction = "📈" if change >= 0 else "📉"
            sections.append(f"{direction} Total Market Cap Change (24h): {self.format_utils.fmt(change)}%")
        
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
            return "## Market Overview:\n" + "\n".join([f"- {section}" for section in sections])
        
        return ""
    
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
            direction = "📈" if mcap_change_24h >= 0 else "📉"
            lines.append(f"  • Market Cap 24h Change: {direction} {mcap_change_24h:+.2f}%")
        
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
            market_cap = coin.get("market_cap", 0)
            coin_volume = coin.get("total_volume", 0)
            
            direction = "📈" if change_24h >= 0 else "📉"
            
            # Parse ATH date if available
            ath_date_str = ""
            if ath_date:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(ath_date.replace('Z', '+00:00'))
                    ath_date_str = dt.strftime("%b %d, %Y")
                except:
                    pass
            
            # Format: Rank #X Symbol (Name): $price (momentum data) | volume | ATH context
            line_parts = [f"  • #{rank} {symbol} ({name}): ${price:,.2f}"]
            
            # Momentum: 1h, 24h, 7d
            momentum_parts = []
            if change_1h:
                mom_dir = "📈" if change_1h >= 0 else "📉"
                momentum_parts.append(f"{mom_dir}{change_1h:+.1f}% 1h")
            if change_24h is not None:
                momentum_parts.append(f"{direction}{change_24h:+.2f}% 24h")
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
            self.logger.warning(f"Error formatting DeFi summary: {e}") if self.logger else None
            return ""
    
    def _format_period_price_section(self, metrics: dict) -> List[str]:
        """Format price-related metrics for a period."""
        price_sections = []
        
        # Get basic metrics (from _calculate_basic_metrics structure)
        highest_price = metrics.get('highest_price')
        lowest_price = metrics.get('lowest_price')
        price_change = metrics.get('price_change')
        price_change_percent = metrics.get('price_change_percent')
        avg_price = metrics.get('avg_price')
        
        if avg_price is not None:
            price_sections.append(f"  💰 Average Price: ${self.format_utils.fmt(avg_price)}")
        
        if highest_price and lowest_price:
            price_sections.append(f"  📈 Range: ${self.format_utils.fmt(lowest_price)} - ${self.format_utils.fmt(highest_price)}")
        
        if price_change is not None and price_change_percent is not None:
            direction = "📈" if price_change >= 0 else "📉"
            price_sections.append(f"  {direction} Change: ${self.format_utils.fmt(price_change)} ({self.format_utils.fmt(price_change_percent)}%)")
        
        return price_sections
    
    def _format_period_volume_section(self, metrics: dict) -> List[str]:
        """Format volume-related metrics for a period."""
        volume_sections = []
        
        total_volume = metrics.get('total_volume')
        avg_volume = metrics.get('avg_volume')
        
        if total_volume is not None:
            volume_sections.append(f"  📊 Total Volume: {self.format_utils.fmt(total_volume)}")
        
        if avg_volume is not None:
            volume_sections.append(f"  📊 Average Volume: {self.format_utils.fmt(avg_volume)}")
        
        return volume_sections
    
    def _format_indicator_changes_section(self, indicator_changes: dict, period_name: str) -> List[str]:
        """Format indicator changes for a period."""
        if not indicator_changes:
            return []
        
        changes_sections = [f"  📊 {period_name.capitalize()} Indicator Changes:"]
        
        # RSI changes
        rsi_change = indicator_changes.get('rsi_change')
        if rsi_change is not None:
            rsi_direction = "📈" if rsi_change >= 0 else "📉"
            changes_sections.append(f"    • RSI: {rsi_direction} {self.format_utils.fmt(abs(rsi_change))} value change")
        
        # MACD changes (use macd_line which is the main MACD indicator)
        macd_change = indicator_changes.get('macd_line_change')
        if macd_change is not None:
            macd_direction = "📈" if macd_change >= 0 else "📉"
            changes_sections.append(f"    • MACD Line: {macd_direction} {self.format_utils.fmt(abs(macd_change))}")
        
        # MACD Histogram changes
        macd_hist_change = indicator_changes.get('macd_hist_change')
        if macd_hist_change is not None:
            macd_hist_direction = "📈" if macd_hist_change >= 0 else "📉"
            changes_sections.append(f"    • MACD Histogram: {macd_hist_direction} {self.format_utils.fmt(abs(macd_hist_change))}")
        
        # ADX changes
        adx_change = indicator_changes.get('adx_change')
        if adx_change is not None:
            adx_direction = "📈" if adx_change >= 0 else "📉"
            changes_sections.append(f"    • ADX: {adx_direction} {self.format_utils.fmt(abs(adx_change))} value change")
        
        # Stochastic %K changes
        stoch_k_change = indicator_changes.get('stoch_k_change')
        if stoch_k_change is not None:
            stoch_direction = "📈" if stoch_k_change >= 0 else "📉"
            changes_sections.append(f"    • Stochastic %K: {stoch_direction} {self.format_utils.fmt(abs(stoch_k_change))} value change")
        
        # Bollinger Bands position changes
        bb_position_change = indicator_changes.get('bb_position_change')
        if bb_position_change is not None:
            bb_direction = "📈" if bb_position_change >= 0 else "📉"
            changes_sections.append(f"    • BB Position: {bb_direction} {self.format_utils.fmt(abs(bb_position_change))}")
        
        return changes_sections
    
    def _format_sma_section(self, long_term_data: dict) -> str:
        """Format Simple Moving Averages section."""
        sma_items = []
        for period in [20, 50, 100, 200]:
            key = f'sma_{period}'
            if key in long_term_data:
                sma_items.append(f"SMA{period}: {self.format_utils.format_value(long_term_data[key])}")
        
        if sma_items:
            return "## Simple Moving Averages:\n" + " | ".join(sma_items)
        return ""
    
    def _format_volume_sma_section(self, long_term_data: dict) -> str:
        """Format Volume SMA section."""
        volume_sma_items = []
        for period in [20, 50]:
            key = f'volume_sma_{period}'
            if key in long_term_data:
                volume_sma_items.append(f"Vol SMA{period}: {self.format_utils.format_value(long_term_data[key])}")
        
        if volume_sma_items:
            return "## Volume Moving Averages:\n" + " | ".join(volume_sma_items)
        return ""
    
    def _format_price_position_section(self, long_term_data: dict, current_price: float) -> str:
        """Format price position relative to moving averages."""
        position_items = []
        
        for period in [20, 50, 100, 200]:
            key = f'sma_{period}'
            if key in long_term_data and long_term_data[key]:
                sma_value = long_term_data[key]
                percentage = ((current_price - sma_value) / sma_value) * 100
                direction = "above" if percentage > 0 else "below"
                position_items.append(f"SMA{period}: {self.format_utils.fmt(abs(percentage))}% {direction}")
        
        if position_items:
            return "## Price Position vs SMAs:\n" + " | ".join(position_items)
        return ""
    
    def _format_daily_indicators_section(self, long_term_data: dict, current_price: float) -> str:
        """Format daily timeframe indicators."""
        indicator_items = []
        
        # RSI
        if 'daily_rsi' in long_term_data:
            rsi_val = long_term_data['daily_rsi']
            rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
            indicator_items.append(f"Daily RSI: {self.format_utils.format_value(rsi_val)} ({rsi_status})")
        
        # MACD
        if 'daily_macd_line' in long_term_data and 'daily_macd_signal' in long_term_data:
            macd_line = long_term_data['daily_macd_line']
            macd_signal = long_term_data['daily_macd_signal']
            macd_status = "Bullish" if macd_line > macd_signal else "Bearish"
            indicator_items.append(f"Daily MACD: {macd_status}")
        
        # Stochastic
        if 'daily_stoch_k' in long_term_data:
            stoch_val = long_term_data['daily_stoch_k']
            stoch_status = "Overbought" if stoch_val > 80 else "Oversold" if stoch_val < 20 else "Neutral"
            indicator_items.append(f"Daily Stoch: {self.format_utils.format_value(stoch_val)} ({stoch_status})")
        
        if indicator_items:
            return "## Daily Indicators:\n" + " | ".join(indicator_items)
        return ""
    
    def _format_adx_section(self, long_term_data: dict) -> str:
        """Format ADX trend strength analysis."""
        if 'daily_adx' not in long_term_data:
            return ""
        
        adx_val = long_term_data['daily_adx']
        if adx_val < 25:
            strength = "Weak/No Trend"
        elif adx_val < 50:
            strength = "Strong Trend"
        elif adx_val < 75:
            strength = "Very Strong Trend"
        else:
            strength = "Extremely Strong Trend"
        
        return f"## Trend Strength (Daily ADX): {self.format_utils.format_value(adx_val)} ({strength})"
    
    def _format_ichimoku_section(self, long_term_data: dict, current_price: float) -> str:
        """Format Ichimoku cloud analysis."""
        ichimoku_items = []
        
        # Tenkan and Kijun
        if 'ichimoku_tenkan' in long_term_data:
            tenkan = long_term_data['ichimoku_tenkan']
            ichimoku_items.append(f"Tenkan: {self.format_utils.format_value(tenkan)}")
        
        if 'ichimoku_kijun' in long_term_data:
            kijun = long_term_data['ichimoku_kijun']
            ichimoku_items.append(f"Kijun: {self.format_utils.format_value(kijun)}")
        
        # Cloud analysis
        if 'ichimoku_span_a' in long_term_data and 'ichimoku_span_b' in long_term_data:
            span_a = long_term_data['ichimoku_span_a']
            span_b = long_term_data['ichimoku_span_b']
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)
            
            if current_price > cloud_top:
                cloud_position = "Above Cloud (Bullish)"
            elif current_price < cloud_bottom:
                cloud_position = "Below Cloud (Bearish)"
            else:
                cloud_position = "Inside Cloud (Neutral)"
            
            ichimoku_items.append(f"Cloud Position: {cloud_position}")
        
        if ichimoku_items:
            return "## Ichimoku Analysis:\n" + " | ".join(ichimoku_items)
        return ""
    
    def _format_macro_trend_section(self, macro_trend: dict) -> str:
        """Format 365-day macro trend analysis based on SMA relationships."""
        if not macro_trend:
            return ""
            
        trend_direction = macro_trend.get('trend_direction', 'Neutral')
        sma_alignment = macro_trend.get('sma_alignment', 'Mixed')
        sma_50_vs_200 = macro_trend.get('sma_50_vs_200', 'Neutral')
        price_above_200sma = macro_trend.get('price_above_200sma', False)
        golden_cross = macro_trend.get('golden_cross', False)
        death_cross = macro_trend.get('death_cross', False)
        long_term_price_change_pct = macro_trend.get('long_term_price_change_pct')
        
        # Build status indicators
        status_parts = []
        status_parts.append(f"Trend: {trend_direction}")
        
        # Add price change if available
        if long_term_price_change_pct is not None:
            change_sign = "+" if long_term_price_change_pct >= 0 else ""
            status_parts.append(f"365D Change: {change_sign}{self.format_utils.fmt(long_term_price_change_pct)}%")
        
        status_parts.append(f"SMA Alignment: {sma_alignment}")
        status_parts.append(f"50vs200 SMA: {sma_50_vs_200}")
        status_parts.append(f"Price>200SMA: {'✓' if price_above_200sma else '✗'}")
        
        if golden_cross:
            status_parts.append("Golden Cross Detected")
        elif death_cross:
            status_parts.append("Death Cross Detected")
            
        return f"## Macro Trend Analysis (365D):\n{' | '.join(status_parts)}"
    
    def _format_weekly_macro_section(self, weekly_macro: dict) -> str:
        """Format weekly macro trend (200W SMA methodology)."""
        if not weekly_macro:
            return ""
        
        trend = weekly_macro.get('trend_direction', 'Neutral')
        confidence = weekly_macro.get('confidence_score', 0)
        cycle_phase = weekly_macro.get('cycle_phase')
        distance = weekly_macro.get('distance_from_200w_sma_pct')
        
        lines = ["📊 WEEKLY MACRO TREND (200W SMA Analysis):"]
        lines.append(f"  • Overall Trend: **{trend}** (Confidence: {confidence}%)")
        
        if cycle_phase:
            lines.append(f"  • Market Cycle Phase: {cycle_phase}")
        if distance is not None:
            lines.append(f"  • Distance from 200W SMA: {distance:+.1f}%")
        if weekly_macro.get('price_above_200w_sma') is not None:
            status = "✅ Above" if weekly_macro['price_above_200w_sma'] else "⚠️ Below"
            lines.append(f"  • Price vs 200W SMA: {status}")
        
        # Golden/Death Cross with dates
        if weekly_macro.get('golden_cross'):
            weeks = weekly_macro.get('golden_cross_weeks_ago', 0)
            date = weekly_macro.get('golden_cross_date', 'N/A')
            lines.append(f"  • 🌟 Golden Cross: {weeks} weeks ago ({date})")
        if weekly_macro.get('death_cross'):
            weeks = weekly_macro.get('death_cross_weeks_ago', 0)
            date = weekly_macro.get('death_cross_date', 'N/A')
            lines.append(f"  • ⚠️ Death Cross: {weeks} weeks ago ({date})")
        
        # Multi-year trend
        if weekly_macro.get('multi_year_trend'):
            mt = weekly_macro['multi_year_trend']
            years = mt.get('years_analyzed', 0)
            pct = mt.get('price_change_pct', 0)
            start = mt.get('start_date', 'N/A')
            end = mt.get('end_date', 'N/A')
            lines.append(f"  • {years:.1f}-Year Trend: {pct:+.1f}% ({start} to {end})")
        
        return "\n".join(lines)
    
    def format_coin_details_section(self, coin_details: Dict[str, Any], max_description_tokens: int = 256) -> str:
        """Format coin details into a structured section for prompt building
        
        Args:
            coin_details: Dictionary containing coin details from CryptoCompare API
            max_description_tokens: Maximum tokens allowed for description (default: 150)
            
        Returns:
            str: Formatted coin details section
        """
        if not coin_details:
            return ""
        
        section = "CRYPTOCURRENCY DETAILS:\n"
        
        # Basic information
        if coin_details.get("full_name"):
            section += f"- Full Name: {coin_details['full_name']}\n"
        if coin_details.get("coin_name"):
            section += f"- Project: {coin_details['coin_name']}\n"
        
        # Technical details
        algorithm = coin_details.get("algorithm", "N/A")
        proof_type = coin_details.get("proof_type", "N/A")
        if algorithm != "N/A" or proof_type != "N/A":
            section += f"- Algorithm: {algorithm}\n"
            section += f"- Proof Type: {proof_type}\n"
        
        # Taxonomy classifications
        taxonomy = coin_details.get("taxonomy", {})
        if taxonomy:
            section += "\nRegulatory Classifications:\n"
            if taxonomy.get("Access"):
                section += f"- Access Model: {taxonomy['Access']}\n"
            if taxonomy.get("FCA"):
                section += f"- UK FCA Classification: {taxonomy['FCA']}\n"
            if taxonomy.get("FINMA"):
                section += f"- Swiss FINMA Classification: {taxonomy['FINMA']}\n"
            if taxonomy.get("Industry"):
                section += f"- Industry Category: {taxonomy['Industry']}\n"
            if taxonomy.get("CollateralizedAsset"):
                collateral_text = "Yes" if taxonomy["CollateralizedAsset"] == "Yes" else "No"
                section += f"- Collateralized Asset: {collateral_text}\n"
        
        # Weiss ratings
        rating = coin_details.get("rating", {})
        if rating:
            weiss = rating.get("Weiss", {})
            if weiss:
                section += "\nWeiss Cryptocurrency Ratings:\n"
                section += "- Independent Rating System: Weiss Ratings is a US-based independent agency (since 1971)\n"
                section += "- Scale: A=Excellent (strong buy), B=Good (buy), C=Fair (hold/avoid), D=Weak (sell), E=Very weak (sell)\n"
                section += "- Modifiers: + indicates upper third of grade, - indicates lower third of grade\n"
                section += "- Two Components: Tech/Adoption (long-term potential) + Market Performance (short-term price patterns)\n"
                
                overall_rating = weiss.get("Rating")
                if overall_rating:
                    section += f"- Overall Rating: {overall_rating}\n"
                
                tech_rating = weiss.get("TechnologyAdoptionRating")
                if tech_rating:
                    section += f"- Technology/Adoption Grade: {tech_rating}\n"
                
                market_rating = weiss.get("MarketPerformanceRating")
                if market_rating:
                    section += f"- Market Performance Grade: {market_rating}\n"

        
        # Project description (keep last as it can be long)
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
                    for j, word in enumerate(words):
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
        
        lines = [f"## {symbol} Order Book Depth:"]
        
        # Spread metrics
        spread = order_book.get("spread")
        spread_pct = order_book.get("spread_percent")
        if spread is not None and spread_pct is not None:
            lines.append(f"  • Spread: ${spread:.2f} ({spread_pct:.3f}%)")
        
        # Liquidity depth
        bid_depth = order_book.get("bid_depth", 0)
        ask_depth = order_book.get("ask_depth", 0)
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            lines.append(f"  • Total Liquidity (Top 20 levels): ${self.format_utils.fmt(total_depth)}")
            lines.append(f"    - Bid Depth: ${self.format_utils.fmt(bid_depth)}")
            lines.append(f"    - Ask Depth: ${self.format_utils.fmt(ask_depth)}")
        
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
        
        lines = [f"## {symbol} Recent Trade Flow:"]
        
        # Trade count and velocity
        total_trades = trades.get("total_trades", 0)
        velocity = trades.get("trade_velocity")
        if total_trades:
            lines.append(f"  • Total Recent Trades: {total_trades}")
        if velocity:
            lines.append(f"  • Trade Velocity: {velocity:.2f} trades/minute")
        
        # Buy/Sell pressure
        buy_volume = trades.get("buy_volume", 0)
        sell_volume = trades.get("sell_volume", 0)
        buy_sell_ratio = trades.get("buy_sell_ratio")
        buy_pressure = trades.get("buy_pressure_percent")
        
        if buy_volume and sell_volume:
            lines.append(f"  • Buy Volume: ${self.format_utils.fmt(buy_volume)}")
            lines.append(f"  • Sell Volume: ${self.format_utils.fmt(sell_volume)}")
        
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
            lines.append(f"  • Average Trade Size: ${self.format_utils.fmt(avg_trade_size)}")
        
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
            return ""
        
        lines = [f"## {symbol} Funding Rate (Futures):"]
        
        # Funding rate
        rate = funding.get("funding_rate")
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
                lines.append(f"  • Interpretation: Longs pay shorts (bullish positioning)")
            elif rate_pct < -0.01:
                lines.append(f"  • Interpretation: Shorts pay longs (bearish positioning)")
            else:
                lines.append(f"  • Interpretation: Neutral positioning")
        
        return "\n".join(lines)

