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
        self.logger = logger
        self.config = config
        self.formatter = formatter or self._default_formatter
        
        # AI-optimized colors for better pattern recognition
        self.ai_colors = {
            'background': '#000000',  # Pure black for maximum contrast
            'grid': '#404040',        # Lighter grid for visibility
            'text': '#ffffff',        # Pure white text
            'candle_up': '#00ff00',   # Bright green for bullish candles
            'candle_down': '#ff0000', # Bright red for bearish candles
            'volume': '#0080ff',      # Bright blue for volume
            'rsi': '#ffff00'          # Bright yellow for RSI
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
        height: int = 600,  # Reduced from 600 for better aspect ratio
        width: int = 1600,   # Reduced from 1000 for better file size
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
            fig = self._create_simple_candlestick_chart(ohlcv, pair_symbol, timeframe, height, width, timestamps)
            
            # Generate the image with retry logic to handle kaleido/choreographer issues
            img_bytes = self._retry_image_export(fig, format="png", width=width, height=height, scale=2)
            
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
        timestamps: Optional[List] = None
    ) -> go.Figure:
        """Create a simple candlestick chart focused on price action patterns.
        
        Args:
            ohlcv: OHLCV data array
            pair_symbol: Trading pair symbol
            timeframe: Chart timeframe
            height: Chart height
            width: Chart width
            timestamps: Optional pre-computed timestamps to avoid redundant conversion
            
        Returns:
            Plotly figure object
        """
        # Determine limit: parameter > instance ai_candle_limit
        chosen_limit = int(self.ai_candle_limit)
        if chosen_limit and len(ohlcv) > chosen_limit:
            ohlcv = ohlcv[-chosen_limit:]
            # Limit timestamps if provided
            if timestamps is not None and len(timestamps) > chosen_limit:
                timestamps = timestamps[-chosen_limit:]

        # Use provided timestamps or convert from OHLCV data
        if timestamps is not None:
            timestamps_py = timestamps
        else:
            # Fallback: convert timestamps from OHLCV data
            timestamps = pd.to_datetime(ohlcv[:, 0], unit='ms')
            timestamps_py = timestamps.to_pydatetime().tolist()
        opens = ohlcv[:, 1].astype(float)
        highs = ohlcv[:, 2].astype(float)
        lows = ohlcv[:, 3].astype(float)
        closes = ohlcv[:, 4].astype(float)
        
        # Create single subplot for price only
        fig = go.Figure()
        
        # Add candlestick chart with AI-optimized colors and visibility
        candle = go.Candlestick(
            x=timestamps_py,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price",
            increasing_line_color=self.ai_colors['candle_up'],
            decreasing_line_color=self.ai_colors['candle_down'],
            increasing_line_width=1.2,  # thinner body edges
            decreasing_line_width=1.2,
            line=dict(width=0.8)  # thinner wick for clearer peaks
        )
        fig.add_trace(candle)
        
    # Layout for simple chart optimized for AI with price formatting
        current_price = float(closes[-1])  # Get current price for title
        current_price_formatted = self.formatter(current_price)

        # Determine dynamic decimal places for y-axis tick formatting
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
            title=f"{pair_symbol} Price Pattern - {timeframe} (Last {chosen_limit} Complete Closed Candles) - Current completed closed candle price: {current_price_formatted}",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=height,
            width=width,
            font=dict(family="Arial, sans-serif", size=16, color=self.ai_colors['text']),
            paper_bgcolor=self.ai_colors['background'],
            plot_bgcolor=self.ai_colors['background'],
            margin=dict(l=60, r=120, t=80, b=60),
            showlegend=False  # Hide legend for cleaner look
        )
        
        # Simple Y-axis configuration - denser grid on the RIGHT side
        fig.update_yaxes(
            side="right",
            title_text="Price",
            showgrid=True,
            gridwidth=0.8,
            gridcolor=self.ai_colors['grid'],
            zeroline=False,
            tickformat=y_tickformat,
            exponentformat='none',
            showexponent='none',
            nticks=30,  # Denser price grid
            tickfont=dict(size=9),  # Smaller font for dense labels
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(80, 80, 80, 0.4)'  # More visible minor grid
            )
        )
        
        # Calculate X-axis range with padding (5 empty candles)
        if len(timestamps_py) > 1:
            delta = timestamps_py[-1] - timestamps_py[-2]
            x_range = [timestamps_py[0], timestamps_py[-1] + (delta * 5)]
        else:
            x_range = None

        # Simple x-axis configuration for AI analysis - denser grid
        fig.update_xaxes(
            title_text="Date/Time",
            showgrid=True,
            gridwidth=0.8,
            gridcolor=self.ai_colors['grid'],
            zeroline=False,
            tickformat='%m/%d %H:%M',
            tickangle=-45,
            nticks=40,  # Denser time grid
            type='date',
            range=x_range,
            tickfont=dict(size=8),  # Smaller font to fit more labels
            automargin=True,
            showline=True,
            linewidth=1,
            linecolor=self.ai_colors['grid'],
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(80, 80, 80, 0.4)'  # More visible minor grid
            )
        )


        # Current Price Reference & Tag
        try:
            is_bullish = closes[-1] >= opens[-1]
            price_color = self.ai_colors['candle_up'] if is_bullish else self.ai_colors['candle_down']
            
            # Horizontal line
            fig.add_hline(y=float(closes[-1]), line=dict(color='#666666', width=1, dash='dot'))
        except Exception:
            pass

        # 1. Vertical Day Separators (Anchors for Time)
        day_changes = []
        if len(timestamps_py) > 1:
            for i in range(1, len(timestamps_py)):
                if timestamps_py[i].day != timestamps_py[i-1].day:
                    # Draw Line
                    fig.add_vline(
                        x=timestamps_py[i], 
                        line=dict(color="rgba(80, 80, 80, 0.5)", width=1, dash="longdash")
                    )
                    # Add Label Separately (Avoids Plotly internal datetime sum error)
                    fig.add_annotation(
                        x=timestamps_py[i], y=1.0, yref="paper",
                        text=timestamps_py[i].strftime("%a"),
                        showarrow=False,
                        font=dict(size=10, color="rgba(150, 150, 150, 0.8)"),
                        xanchor="left", yanchor="top"
                    )

        # 2. Local Swing Points (Pivot Highs/Lows)
        # Simple window-based check
        window = 8
        if len(highs) > window * 2:
            for i in range(window, len(highs) - window):
                # Pivot High
                if highs[i] == max(highs[i-window:i+window+1]):
                    # Only mark if significant (e.g. > 0.5% move) - optional, keeping simple for now
                    fig.add_annotation(
                        x=timestamps_py[i], y=highs[i],
                        text=self.formatter(highs[i]),
                        showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1,
                        ax=0, ay=-25,
                        font=dict(size=10, color='#aaaaaa')
                    )
                # Pivot Low
                if lows[i] == min(lows[i-window:i+window+1]):
                    fig.add_annotation(
                        x=timestamps_py[i], y=lows[i],
                        text=self.formatter(lows[i]),
                        showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1,
                        ax=0, ay=25,
                        font=dict(size=10, color='#aaaaaa')
                    )

        # 3. Global High/Low (Prominent)
        idx_high = int(np.argmax(highs))
        idx_low = int(np.argmin(lows))
        fig.add_annotation(
            x=timestamps_py[idx_high], y=float(highs[idx_high]),
            text=f"MAX: {self.formatter(float(highs[idx_high]))}",
            showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=1.0,
            ax=0, ay=-40,
            font=dict(size=12, color=self.ai_colors['text'], weight='bold'),
            bgcolor='rgba(0,0,0,0.5)', bordercolor=self.ai_colors['candle_up'], borderwidth=1
        )
        fig.add_annotation(
            x=timestamps_py[idx_low], y=float(lows[idx_low]),
            text=f"MIN: {self.formatter(float(lows[idx_low]))}",
            showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=1.0,
            ax=0, ay=40,
            font=dict(size=12, color=self.ai_colors['text'], weight='bold'),
            bgcolor='rgba(0,0,0,0.5)', bordercolor=self.ai_colors['candle_down'], borderwidth=1
        )
        
        # 4. DATA PANEL (Ground Truth for AI)
        # Lists the last 5 candles explicitly so AI can read text instead of guessing pixels
        recent_data = []
        count = min(5, len(closes))
        for i in range(count):
            idx = -1 * (count - i) # -5, -4, ... -1
            ts_str = timestamps_py[idx].strftime('%m/%d %H:%M')
            o = self.formatter(opens[idx])
            h = self.formatter(highs[idx])
            l = self.formatter(lows[idx])
            c = self.formatter(closes[idx])
            # Format: Date | O:.. H:.. L:.. C:..
            recent_data.append(f"{ts_str}| O:{o} H:{h} L:{l} C:{c}")
        
        data_panel_text = "<b>RECENT DATA (Ground Truth):</b><br>" + "<br>".join(recent_data)
        
        fig.add_annotation(
            xref='paper', yref='paper', x=0.01, y=0.01,
            xanchor='left', yanchor='bottom',
            text=data_panel_text,
            showarrow=False,
            font=dict(size=10, family='Courier New, monospace', color='#cccccc'),
            bgcolor='rgba(0,0,0,0.85)',
            bordercolor='#444444',
            borderwidth=1,
            align='left'
        )
        
        return fig