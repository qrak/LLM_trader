from typing import Union, List


class DataProcessor:
    """Handles processing, validation, and extraction of indicator values"""
    
    def __init__(self):
        """Initialize the indicator data processor"""
        pass
        
    def get_indicator_value(self, td: dict, key: str) -> Union[float, str]:
        """Get indicator value with proper type checking and error handling
        
        Args:
            td: Technical data dictionary
            key: Indicator key to retrieve
            
        Returns:
            float or str: Indicator value or 'N/A' if invalid
        """
        try:
            value = td[key]
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, (list, tuple)) and len(value) == 1:
                return float(value[0])
            if isinstance(value, (list, tuple)) and len(value) > 1:
                return float(value[-1])
            return 'N/A'
        except (KeyError, TypeError, ValueError, IndexError):
            return 'N/A'
