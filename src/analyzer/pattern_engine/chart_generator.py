"""
Chart generator utility for creating static PNG images optimized for AI pattern analysis.
Generates candlestick charts with OHLC data and annotations for visual pattern recognition.
"""
import io
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.logger.logger import Logger
from src.utils.profiler import profile_performance


class ChartGenerator:
    """Generates interactive charts and static images for market data with OHLCV and RSI."""
    
    def __init__(self, logger: Optional[Logger] = None, config: Optional[Any] = None, formatter: Optional[Callable] = None, format_utils=None):
        """Initialize the chart generator.
        
        Args:
            logger: Optional logger instance for debugging
            config: Optional config instance to avoid circular imports
            formatter: Optional formatting function for price formatting (e.g., fmt from format_utils)
        """
        self.logger = logger
        self.config = config
        self.formatter = formatter
        self.format_utils = format_utils

        self.formatter = formatter or self._default_formatter
        
        # AI-optimized colors for better pattern recognition
        self.ai_colors = {
            'background': '#000000',  # Pure black for maximum contrast
            'grid': '#404040',        # Lighter grid for visibility
            'text': '#ffffff',        # Pure white text
            'candle_up': '#00ff00',   # Bright green for bullish candles
            'candle_down': '#ff0000', # Bright red for bearish candles
            'volume': '#0080ff',      # Bright blue for volume
            'volume_up': '#00aa00',   # Green for bullish volume bars
            'volume_down': '#aa0000', # Red for bearish volume bars
            'rsi': '#ffff00',         # Bright yellow for RSI
            'sma_50': '#ff8c00',      # Orange for SMA 50 (short-term)
            'sma_200': '#9932cc',     # Purple for SMA 200 (long-term)
            'cmf': '#00ffff',         # Cyan for CMF
            'obv': '#ff00ff',         # Magenta for OBV
            'rsi_oversold': '#00ff00',  # Green for oversold line (30)
            'rsi_overbought': '#ff0000' # Red for overbought line (70)
        }
        # AI chart candle limit from config
        self.ai_candle_limit = config.AI_CHART_CANDLE_LIMIT if config is not None else 200
        
    def _default_formatter(self, val, precision=8):
        """Default formatter for price values when no formatter is provided."""
        if isinstance(val, (int, float)) and not np.isnan(val):
            if abs(val) < 0.00001:
                return f"{val:.8f}"
            elif abs(val) < 0.01:
                return f"{val:.6f}"
            elif abs(val) < 10:
                return f"{val:.4f}"
            else:
                return f"{val:.2f}"
        return "N/A"
    
    def _image_export_with_timeout(self, fig: go.Figure, format: str, width: int, height: int, scale: int, timeout: int = 30) -> bytes:
        """Execute image export with a timeout to prevent indefinite hangs.
        
        Args:
            fig: Plotly figure to export
            format: Image format (e.g., "png")
            width: Image width
            height: Image height
            scale: Image scale factor
            timeout: Timeout in seconds (default: 30)
            
        Returns:
            Image bytes
            
        Raises:
            TimeoutError: If export takes longer than timeout
            Exception: If export fails for other reasons
        """
        result = {'img_bytes': None, 'exception': None}
        
        def export_worker():
            try:
                result['img_bytes'] = fig.to_image(format=format, width=width, height=height, scale=scale)
            except Exception as e:
                result['exception'] = e
        
        thread = threading.Thread(target=export_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Image export timed out after {timeout} seconds (kaleido may be hanging)")
        
        if result['exception']:
            raise result['exception']
        
        return result['img_bytes']
    
    def _retry_image_export(self, fig: go.Figure, format: str, width: int, height: int, scale: int, max_retries: int = 3, timeout: int = 30) -> bytes:
        """Retry image export with exponential backoff to handle kaleido/choreographer issues.
        
        Args:
            fig: Plotly figure to export
            format: Image format (e.g., "png")
            width: Image width
            height: Image height
            scale: Image scale factor
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for each attempt (default: 30)
            
        Returns:
            Image bytes
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if self.logger and attempt > 0:
                    self.logger.debug(f"Retry attempt {attempt + 1}/{max_retries} for image export")
                
                img_bytes = self._image_export_with_timeout(fig, format, width, height, scale, timeout)
                
                if self.logger and attempt > 0:
                    self.logger.info(f"Image export succeeded on retry attempt {attempt + 1}")
                
                return img_bytes
                
            except Exception as e:
                last_exception = e
                if self.logger:
                    self.logger.warning(f"Image export attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    if self.logger:
                        self.logger.debug(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        raise last_exception
    
    @profile_performance
    def create_chart_image(
        self,
        ohlcv: np.ndarray,
        technical_history: Optional[Dict[str, np.ndarray]] = None,
        pair_symbol: str = "",
        timeframe: str = "1h",
        height: int = 2160,  # Updated to 2160p height
        width: int = 3840,   # Updated to 3840p width
        save_to_disk: bool = False,
        output_path: Optional[str] = None,
        simple_mode: bool = True,  # Default to simple mode for AI analysis
        timestamps: Optional[List] = None
    ) -> Union[io.BytesIO, str]:
        """Create a PNG chart image optimized for AI pattern analysis.
        
        Args:
            ohlcv: OHLCV data array with columns [timestamp, open, high, low, close, volume]
            technical_history: Optional technical indicators data
            pair_symbol: Trading pair symbol (e.g., "BTCUSDT")
            timeframe: Chart timeframe (e.g., "1h", "4h")
            height: Image height in pixels
            width: Image width in pixels
            save_to_disk: If True, saves image to disk for testing
            output_path: Optional custom output path for disk save
            simple_mode: If True, creates simplified chart with only price data (recommended for AI)
            timestamps: Optional pre-computed timestamps to avoid redundant conversion
            
        Returns:
            BytesIO object containing PNG image data, or file path if saved to disk
        """
        try:
            fig = self._create_simple_candlestick_chart(ohlcv, pair_symbol, timeframe, height, width, timestamps, technical_history)
            
            # Generate the image with retry logic to handle kaleido/choreographer issues
            img_bytes = self._retry_image_export(fig, format="png", width=width, height=height, scale=1)
            
            if save_to_disk:
                # Save to disk for testing purposes
                if output_path is None:
                    # Generate timestamp
                    if self.format_utils:
                        timestamp = self.format_utils.format_current_time("%Y%m%d_%H%M%S")
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    chart_type = "simple" if simple_mode else "full"
                    # Use config path if available
                    base_path = self.config.DEBUG_CHART_SAVE_PATH if self.config else "test_images"
                    
                    filename = f"{pair_symbol.replace('/', '')}_{timeframe}_{chart_type}_AI_analysis_{timestamp}.png"
                    output_path = os.path.join(os.getcwd(), base_path, filename)
                

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(img_bytes)
                
                if self.logger:
                    if len(ohlcv) > 0:
                        # Parse timestamps with or without format_utils
                        if self.format_utils:
                            first_time_dt = self.format_utils.parse_timestamp_ms(ohlcv[0][0])
                            last_time_dt = self.format_utils.parse_timestamp_ms(ohlcv[-1][0])
                        else:
                            first_time_dt = datetime.fromtimestamp(ohlcv[0][0] / 1000)
                            last_time_dt = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                        
                        first_time = first_time_dt.strftime('%Y-%m-%d %H:%M') if first_time_dt else 'N/A'
                        last_time = last_time_dt.strftime('%Y-%m-%d %H:%M') if last_time_dt else 'N/A'
                        current_price = ohlcv[-1][4]  # Close price of last candle
                        self.logger.info(f"📈 Data range: {first_time} to {last_time} | Current price: {current_price}")
                
                return output_path
            else:
                # Return BytesIO for memory efficiency
                img_buffer = io.BytesIO(img_bytes)
                img_buffer.seek(0)
                
                if self.logger:
                    self.logger.debug(f"Generated chart image for {pair_symbol} ({len(img_bytes)} bytes)")
                
                return img_buffer
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating chart image: {str(e)}")
            raise
    
    def _create_simple_candlestick_chart(
        self,
        ohlcv: np.ndarray,
        pair_symbol: str,
        timeframe: str,
        height: int,
        width: int,
        timestamps: Optional[List] = None,
        technical_history: Optional[Dict[str, np.ndarray]] = None
    ) -> go.Figure:
        """Create a multi-panel chart with price, RSI, Volume, and CMF/OBV for AI pattern analysis.
        
        Args:
            ohlcv: OHLCV data array
            pair_symbol: Trading pair symbol
            timeframe: Chart timeframe
            height: Chart height
            width: Chart width
            timestamps: Optional pre-computed timestamps to avoid redundant conversion
            technical_history: Optional dict containing indicator arrays (rsi, sma_50, sma_200, cmf, obv)
            
        Returns:
            Plotly figure object with 4 subplots
        """
        chosen_limit = int(self.ai_candle_limit)
        original_len = len(ohlcv)
        if chosen_limit and original_len > chosen_limit:
            ohlcv = ohlcv[-chosen_limit:]
            if timestamps is not None and len(timestamps) > chosen_limit:
                timestamps = timestamps[-chosen_limit:]

        if timestamps is not None:
            timestamps_py = timestamps
        else:
            timestamps = pd.to_datetime(ohlcv[:, 0], unit='ms')
            timestamps_py = timestamps.to_pydatetime().tolist()

        opens = ohlcv[:, 1].astype(float)
        highs = ohlcv[:, 2].astype(float)
        lows = ohlcv[:, 3].astype(float)
        closes = ohlcv[:, 4].astype(float)
        volumes = ohlcv[:, 5].astype(float) if ohlcv.shape[1] > 5 else np.zeros(len(ohlcv))

        # Slice indicator arrays to match displayed candle count
        slice_start = original_len - len(ohlcv) if chosen_limit and original_len > chosen_limit else 0
        def slice_indicator(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None or len(arr) == 0:
                return None
            if len(arr) > len(ohlcv):
                return arr[-len(ohlcv):]
            return arr

        rsi_data = slice_indicator(technical_history.get('rsi')) if technical_history else None
        sma_50_data = slice_indicator(technical_history.get('sma_50')) if technical_history else None
        sma_200_data = slice_indicator(technical_history.get('sma_200')) if technical_history else None
        cmf_data = slice_indicator(technical_history.get('cmf')) if technical_history else None
        obv_data = slice_indicator(technical_history.get('obv')) if technical_history else None

        # Create 4-row subplot: Price (55%), RSI (15%), Volume (15%), CMF+OBV (15%)
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.15, 0.15],
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": True}]
            ]
        )

        # ROW 1: Candlestick + SMA overlays
        candle = go.Candlestick(
            x=timestamps_py,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price",
            increasing_line_color=self.ai_colors['candle_up'],
            decreasing_line_color=self.ai_colors['candle_down'],
            increasing_line_width=1.2,
            decreasing_line_width=1.2,
            line=dict(width=0.8)
        )
        fig.add_trace(candle, row=1, col=1)

        # Add SMA 50 overlay (orange) - Short-term trend
        if sma_50_data is not None and len(sma_50_data) == len(timestamps_py):
            fig.add_trace(go.Scatter(
                x=timestamps_py,
                y=sma_50_data,
                mode='lines',
                name='SMA 50',
                line=dict(color=self.ai_colors['sma_50'], width=1.5),
                hoverinfo='name+y'
            ), row=1, col=1)

        # Add SMA 200 overlay (purple) - Long-term trend
        if sma_200_data is not None and len(sma_200_data) == len(timestamps_py):
            fig.add_trace(go.Scatter(
                x=timestamps_py,
                y=sma_200_data,
                mode='lines',
                name='SMA 200',
                line=dict(color=self.ai_colors['sma_200'], width=1.5),
                hoverinfo='name+y'
            ), row=1, col=1)

        # ROW 2: RSI indicator
        if rsi_data is not None and len(rsi_data) == len(timestamps_py):
            fig.add_trace(go.Scatter(
                x=timestamps_py,
                y=rsi_data,
                mode='lines',
                name='RSI (14)',
                line=dict(color=self.ai_colors['rsi'], width=1.5),
                hoverinfo='name+y'
            ), row=2, col=1)
            # Overbought line (70)
            fig.add_hline(y=70, row=2, col=1, line=dict(color=self.ai_colors['rsi_overbought'], width=1, dash='dash'))
            # Oversold line (30)
            fig.add_hline(y=30, row=2, col=1, line=dict(color=self.ai_colors['rsi_oversold'], width=1, dash='dash'))
            # Neutral line (50)
            fig.add_hline(y=50, row=2, col=1, line=dict(color='#666666', width=0.5, dash='dot'))

        # ROW 3: Volume bars (colored by candle direction)
        volume_colors = [self.ai_colors['volume_up'] if closes[i] >= opens[i] else self.ai_colors['volume_down'] for i in range(len(closes))]
        fig.add_trace(go.Bar(
            x=timestamps_py,
            y=volumes,
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7
        ), row=3, col=1)

        # ROW 4: CMF (left y-axis, area) + OBV (right y-axis, line)
        if cmf_data is not None and len(cmf_data) == len(timestamps_py):
            fig.add_trace(go.Scatter(
                x=timestamps_py,
                y=cmf_data,
                mode='lines',
                name='CMF (20)',
                fill='tozeroy',
                line=dict(color=self.ai_colors['cmf'], width=1),
                fillcolor='rgba(0, 255, 255, 0.3)',
                hoverinfo='name+y'
            ), row=4, col=1, secondary_y=False)
            # CMF zero line
            fig.add_hline(y=0, row=4, col=1, line=dict(color='#888888', width=1, dash='dash'))

        if obv_data is not None and len(obv_data) == len(timestamps_py):
            fig.add_trace(go.Scatter(
                x=timestamps_py,
                y=obv_data,
                mode='lines',
                name='OBV',
                line=dict(color=self.ai_colors['obv'], width=1.5),
                hoverinfo='name+y'
            ), row=4, col=1, secondary_y=True)

        # Layout configuration
        current_price = float(closes[-1])
        current_price_formatted = self.formatter(current_price)

        abs_price = abs(current_price) if current_price != 0 else 0.0
        if abs_price == 0:
            decimals = 2
        elif abs_price < 1e-6:
            decimals = 9
        elif abs_price < 1e-5:
            decimals = 8
        elif abs_price < 1e-4:
            decimals = 7
        elif abs_price < 1e-3:
            decimals = 6
        elif abs_price < 1e-2:
            decimals = 5
        elif abs_price < 1e-1:
            decimals = 4
        elif abs_price < 1:
            decimals = 4
        elif abs_price < 10:
            decimals = 3
        else:
            decimals = 2
        y_tickformat = f".{decimals}f"

        fig.update_layout(
            title=dict(
                text=f"{pair_symbol} - {timeframe} (Last {chosen_limit} Closed Candles) | Price: {current_price_formatted}",
                font=dict(size=28)
            ),
            template="plotly_dark",
            height=height,
            width=width,
            font=dict(family="Arial, sans-serif", size=22, color=self.ai_colors['text']),
            paper_bgcolor=self.ai_colors['background'],
            plot_bgcolor=self.ai_colors['background'],
            margin=dict(l=80, r=120, t=80, b=80),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="right",
                x=1,
                font=dict(size=18),
                bgcolor='rgba(0,0,0,0.7)'
            ),
            xaxis_rangeslider_visible=False
        )

        # Calculate x-axis range with small padding (2 candles of future space)
        if len(timestamps_py) > 1:
            delta = timestamps_py[-1] - timestamps_py[-2]
            x_range = [timestamps_py[0], timestamps_py[-1] + (delta * 2)]
        else:
            x_range = None

        # Configure all 4 x-axes (shared, but show labels on indicator subplots)
        common_xaxis = dict(
            showgrid=True,
            gridwidth=0.6,
            gridcolor=self.ai_colors['grid'],
            zeroline=False,
            tickformat='%m/%d %H:%M',
            tickangle=-45,
            nticks=30,
            type='date',
            range=x_range,
            tickfont=dict(size=14),
            showline=True,
            linewidth=1,
            linecolor=self.ai_colors['grid']
        )
        # Price chart - no x-axis labels (too cluttered)
        fig.update_xaxes(**common_xaxis, showticklabels=False, row=1, col=1)
        # RSI, Volume, CMF - show x-axis labels for AI readability
        fig.update_xaxes(**common_xaxis, showticklabels=True, row=2, col=1)
        fig.update_xaxes(**common_xaxis, showticklabels=True, row=3, col=1)
        fig.update_xaxes(**common_xaxis, showticklabels=True, title_text="Date/Time", row=4, col=1)

        # Y-axis configurations
        common_yaxis = dict(
            showgrid=True,
            gridwidth=0.6,
            gridcolor=self.ai_colors['grid'],
            zeroline=False,
            side="right",
            tickfont=dict(size=16)
        )
        # Row 1: Price
        fig.update_yaxes(**common_yaxis, title_text="Price", tickformat=y_tickformat, nticks=15, row=1, col=1)
        # Row 2: RSI (0-100 range)
        fig.update_yaxes(**common_yaxis, title_text="RSI", range=[0, 100], nticks=5, row=2, col=1)
        # Row 3: Volume
        fig.update_yaxes(**common_yaxis, title_text="Vol", nticks=4, row=3, col=1)
        # Row 4: CMF (primary y-axis, left), OBV (secondary y-axis, right)
        fig.update_yaxes(
            showgrid=True, gridwidth=0.6, gridcolor=self.ai_colors['grid'],
            zeroline=False, tickfont=dict(size=16),
            title_text="CMF", nticks=4, side="left", row=4, col=1, secondary_y=False
        )
        fig.update_yaxes(**common_yaxis, title_text="OBV", nticks=4, row=4, col=1, secondary_y=True)

        # Add current price horizontal line on price chart
        fig.add_hline(y=float(closes[-1]), row=1, col=1, line=dict(color='#666666', width=1, dash='dot'))

        # Day separators (vertical lines across all rows)
        if len(timestamps_py) > 1:
            for i in range(1, len(timestamps_py)):
                if timestamps_py[i].day != timestamps_py[i-1].day:
                    fig.add_vline(
                        x=timestamps_py[i],
                        line=dict(color="rgba(80, 80, 80, 0.5)", width=1, dash="longdash")
                    )

        # Local swing points (pivot highs/lows) on price chart
        window = 8
        if len(highs) > window * 2:
            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window+1]):
                    fig.add_annotation(
                        x=timestamps_py[i], y=highs[i],
                        text=self.formatter(highs[i]),
                        showarrow=True, arrowhead=2, arrowsize=0.6, arrowwidth=1,
                        ax=0, ay=-30,
                        font=dict(size=16, color='#aaaaaa'),
                        row=1, col=1
                    )
                if lows[i] == min(lows[i-window:i+window+1]):
                    fig.add_annotation(
                        x=timestamps_py[i], y=lows[i],
                        text=self.formatter(lows[i]),
                        showarrow=True, arrowhead=2, arrowsize=0.6, arrowwidth=1,
                        ax=0, ay=30,
                        font=dict(size=16, color='#aaaaaa'),
                        row=1, col=1
                    )

        # Global MAX/MIN annotations
        idx_high = int(np.argmax(highs))
        idx_low = int(np.argmin(lows))
        fig.add_annotation(
            x=timestamps_py[idx_high], y=float(highs[idx_high]),
            text=f"MAX: {self.formatter(float(highs[idx_high]))}",
            showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.0,
            ax=0, ay=-50,
            font=dict(size=20, color=self.ai_colors['text'], weight='bold'),
            bgcolor='rgba(0,0,0,0.5)', bordercolor=self.ai_colors['candle_up'], borderwidth=1,
            row=1, col=1
        )
        fig.add_annotation(
            x=timestamps_py[idx_low], y=float(lows[idx_low]),
            text=f"MIN: {self.formatter(float(lows[idx_low]))}",
            showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.0,
            ax=0, ay=50,
            font=dict(size=20, color=self.ai_colors['text'], weight='bold'),
            bgcolor='rgba(0,0,0,0.5)', bordercolor=self.ai_colors['candle_down'], borderwidth=1,
            row=1, col=1
        )

        # SMA legend annotation for AI comprehension (on price chart)
        sma_legend = []
        if sma_50_data is not None:
            sma_legend.append(f"<span style='color:{self.ai_colors['sma_50']}'>━</span> SMA 50 (Short-term trend)")
        if sma_200_data is not None:
            sma_legend.append(f"<span style='color:{self.ai_colors['sma_200']}'>━</span> SMA 200 (Long-term trend)")
        if sma_legend:
            # Add golden/death cross hint
            if sma_50_data is not None and sma_200_data is not None:
                sma_legend.append("<b>Golden Cross:</b> SMA50 crosses above SMA200 = Bullish")
                sma_legend.append("<b>Death Cross:</b> SMA50 crosses below SMA200 = Bearish")
            fig.add_annotation(
                xref='paper', yref='paper', x=0.01, y=0.99,
                xanchor='left', yanchor='top',
                text="<br>".join(sma_legend),
                showarrow=False,
                font=dict(size=18, family='Arial', color='#cccccc'),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='#444444',
                borderwidth=1,
                align='left'
            )

        # RSI interpretation annotation
        if rsi_data is not None:
            current_rsi = rsi_data[-1] if not np.isnan(rsi_data[-1]) else 0
            fig.add_annotation(
                xref='x2 domain', yref='y2 domain', x=0.01, y=0.95,
                xanchor='left', yanchor='top',
                text=f"RSI: {current_rsi:.1f} | <span style='color:{self.ai_colors['rsi_overbought']}'>70=Overbought</span> | <span style='color:{self.ai_colors['rsi_oversold']}'>30=Oversold</span>",
                showarrow=False,
                font=dict(size=16, color='#cccccc'),
                bgcolor='rgba(0,0,0,0.7)'
            )

        # CMF/OBV interpretation annotation
        cmf_obv_legend = []
        if cmf_data is not None:
            current_cmf = cmf_data[-1] if not np.isnan(cmf_data[-1]) else 0
            cmf_status = "Buying Pressure" if current_cmf > 0 else "Selling Pressure"
            cmf_obv_legend.append(f"<span style='color:{self.ai_colors['cmf']}'>CMF: {current_cmf:.3f}</span> ({cmf_status})")
        if obv_data is not None:
            cmf_obv_legend.append(f"<span style='color:{self.ai_colors['obv']}'>OBV</span>: Trend Confirmation (rising=accumulation, falling=distribution)")
        if cmf_obv_legend:
            fig.add_annotation(
                xref='x4 domain', yref='y4 domain', x=0.01, y=0.95,
                xanchor='left', yanchor='top',
                text=" | ".join(cmf_obv_legend),
                showarrow=False,
                font=dict(size=16, color='#cccccc'),
                bgcolor='rgba(0,0,0,0.7)'
            )

        return fig