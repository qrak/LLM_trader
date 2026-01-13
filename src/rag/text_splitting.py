
"""
Text Splitting Utilities
Provides robust sentence splitting using wtpsplit with regex fallback.
"""
import re
import time
import logging
import logging
from typing import List, Optional
from functools import lru_cache

class SentenceSplitter:
    """
    Singleton class for sentence splitting using wtpsplit (SaT model).
    Ensures model is only loaded once.
    """
    _instance = None
    _model = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SentenceSplitter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, logger=None):
        if not self._initialized:
            if logger is None:
                raise ValueError("Logger must be provided to SentenceSplitter on first initialization")
            self.logger = logger
            self._initialize_model()
            self.__class__._initialized = True
            
    def _initialize_model(self):
        """Initialize the SaT model."""
        self.logger.info("Initializing SentenceSplitter (NLP model)...")
        start_time = time.perf_counter()
        try:
            from wtpsplit import SaT
            # Use the newer SaT model
            self._model = SaT("sat-3l-sm")
            duration = time.perf_counter() - start_time
            self.logger.info(f"SentenceSplitter initialized took {duration:.2f} seconds")
        except ImportError:
            self.logger.warning("wtpsplit or torch not available, utilizing regex fallback")
            self._model = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize wtpsplit: {e}, utilizing regex fallback")
            self._model = None

    @lru_cache(maxsize=1024)
    def split_text(self, text: str) -> List[str]:
        """
        Split text into sentences using wtpsplit if available, otherwise regex.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentence strings
        """
        if not text:
            return []
            
        # Clean up text slightly before splitting to avoid empty or whitespace-only sentences
        text = text.strip()
        
        if self._model:
            try:
                # SaT models don't use lang_code in split()
                return self._model.split(text)
            except Exception as e:
                self.logger.warning(f"wtpsplit failed: {e}, falling back to regex")
        
        # Fallback to regex splitting
        # Split on . ! ? followed by whitespace
        # This is a simple approximation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @property
    def is_available(self) -> bool:
        """Check if wtpsplit is successfully initialized."""
        return self._model is not None
