import numpy as np
from typing import Dict, List, Any
from datetime import datetime

from src.analyzer.pattern_engine.swing_detection import (
    detect_swing_highs_numba,
    detect_swing_lows_numba,
    classify_swings_numba
)
from src.analyzer.pattern_engine.pattern_matchers import (
    detect_head_shoulder_numba,
    detect_double_top_bottom_numba,
    detect_triangle_numba,
    detect_wedge_numba,
    detect_channel_numba,
    detect_multiple_tops_bottoms_numba
)


class PatternEngine:
    
    def __init__(self, lookback: int = 7, lookahead: int = 7, format_utils=None):
        self.lookback = lookback
        self.lookahead = lookahead
        self.format_utils = format_utils
    
    def detect_patterns(self, ohlcv: np.ndarray, timestamps: List[datetime] = None) -> Dict[str, List[Dict[str, Any]]]:
        if len(ohlcv) < self.lookback + self.lookahead:
            return {}
        
        high = ohlcv[:, 1].astype(np.float64)
        low = ohlcv[:, 2].astype(np.float64)
        close = ohlcv[:, 3].astype(np.float64)
        
        swing_highs = detect_swing_highs_numba(high, self.lookback, self.lookahead)
        swing_lows = detect_swing_lows_numba(low, self.lookback, self.lookahead)
        
        patterns = {}
        
        hs_patterns = detect_head_shoulder_numba(high, low, close, swing_highs, swing_lows)
        patterns['head_shoulder'] = self._extract_patterns(hs_patterns, 
            {1: 'Head and Shoulder', 2: 'Inverse Head and Shoulder'}, timestamps)
        
        dt_patterns = detect_double_top_bottom_numba(high, low, swing_highs, swing_lows)
        patterns['double_top_bottom'] = self._extract_patterns(dt_patterns,
            {1: 'Double Top', 2: 'Double Bottom'}, timestamps)
        
        tri_patterns = detect_triangle_numba(high, low, swing_highs, swing_lows)
        patterns['triangle'] = self._extract_patterns(tri_patterns,
            {1: 'Descending Triangle', 2: 'Ascending Triangle'}, timestamps)
        
        wedge_patterns = detect_wedge_numba(high, low, swing_highs, swing_lows)
        patterns['wedge'] = self._extract_patterns(wedge_patterns,
            {1: 'Wedge Up', 2: 'Wedge Down'}, timestamps)
        
        channel_patterns = detect_channel_numba(high, low, swing_highs, swing_lows)
        patterns['channel'] = self._extract_patterns(channel_patterns,
            {1: 'Channel Up', 2: 'Channel Down'}, timestamps)
        
        multiple_patterns = detect_multiple_tops_bottoms_numba(high, low, swing_highs, swing_lows)
        patterns['multiple_tops_bottoms'] = self._extract_patterns(multiple_patterns,
            {1: 'Multiple Top', 2: 'Multiple Bottom'}, timestamps)
        
        swing_classifications = classify_swings_numba(high, low, swing_highs, swing_lows)
        patterns['swing_points'] = self._extract_patterns(swing_classifications,
            {1: 'HH', 2: 'LH', 3: 'HL', 4: 'LL'}, timestamps)
        
        return patterns
    
    def _extract_patterns(self, pattern_array: np.ndarray, 
                         pattern_names: Dict[int, str],
                         timestamps: List[datetime] = None) -> List[Dict[str, Any]]:
        results = []
        indices = np.where(pattern_array > 0)[0]
        
        for idx in indices:
            pattern_type = pattern_array[idx]
            pattern_name = pattern_names.get(pattern_type, f'Unknown_{pattern_type}')
            
            # Build description with timestamp if available
            if timestamps and idx < len(timestamps):
                timestamp = timestamps[idx]
                if hasattr(timestamp, 'strftime'):
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(timestamp, (int, float)):
                    # Assume milliseconds timestamp
                    timestamp_str = self.format_utils.format_timestamp(timestamp)
                else:
                    timestamp_str = str(timestamp)
                description = f'{pattern_name} at {timestamp_str} (index {idx})'
            else:
                description = f'{pattern_name} at index {idx}'
            
            result = {
                'type': pattern_name,
                'index': int(idx),
                'description': description,
                'details': {
                    'pattern_code': int(pattern_type)
                }
            }
            
            if timestamps and idx < len(timestamps):
                result['timestamp'] = timestamps[idx]
            
            results.append(result)
        
        return results
