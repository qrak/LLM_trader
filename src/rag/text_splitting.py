
"""
Text Splitting Utilities
Provides robust sentence splitting using wtpsplit with regex fallback.
"""
import re
import logging
from typing import List, Optional

class SentenceSplitter:
    """
    Singleton class for sentence splitting using wtpsplit (SaT model).
    Ensures model is only loaded once.
    """
    _instance = None
    _model = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentenceSplitter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self._initialize_model()
            self.__class__._initialized = True
            
    def _initialize_model(self):
        """Initialize the SaT model."""
        try:
            from wtpsplit import SaT
            # Use the newer SaT model
            self._model = SaT("sat-3l-sm")
            self.logger.info("Initialized SaT sentence segmenter")
        except ImportError:
            self.logger.warning("wtpsplit or torch not available, utilizing regex fallback")
            self._model = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize wtpsplit: {e}, utilizing regex fallback")
            self._model = None

    def split_text(self, text: str, lang_code: str = "en") -> List[str]:
        """
        Split text into sentences using wtpsplit if available, otherwise regex.
        
        Args:
            text: Input text to split
            lang_code: Language code for the model (default: "en")
            
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
