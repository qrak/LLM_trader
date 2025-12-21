from typing import Dict

import tiktoken


class TokenCounter:
    """
    Handles token counting and tracking for AI model usage.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the TokenCounter.
        
        Args:
            encoding_name: The tokenizer encoding name to use (default: cl100k_base for OpenAI models)
        """
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.session_tokens = {
            "prompt": 0,
            "completion": 0,
            "system": 0,
            "total": 0
        }
    
    def count_tokens(self, text: str) -> int:
        """
        Count the tokens in the provided text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def track_prompt_tokens(self, text: str, message_type: str = "prompt") -> int:
        """
        Count tokens in prompt text and track them in session stats.
        
        Args:
            text: The prompt text
            message_type: Type of message ("prompt", "system", or "completion")
            
        Returns:
            Number of tokens
        """
        token_count = self.count_tokens(text)
        
        # Update session counts
        if message_type in self.session_tokens:
            self.session_tokens[message_type] += token_count
        else:
            self.session_tokens[message_type] = token_count
            
        self.session_tokens["total"] += token_count
        
        return token_count
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get current token usage statistics.
        
        Returns:
            Dictionary with token usage counts
        """
        return self.session_tokens.copy()
    
    def reset_session_stats(self) -> None:
        """Reset session token tracking statistics."""
        self.session_tokens = {
            "prompt": 0,
            "completion": 0,
            "system": 0,
            "total": 0
        }
    
    def estimate_cost(self, model_pricing: Dict[str, float] = None) -> Dict[str, float]:
        """
        Estimate the cost of tokens used based on model pricing.
        
        Args:
            model_pricing: Dictionary with "input" and "output" price per 1000 tokens
            
        Returns:
            Dictionary with cost estimates
        """
        if model_pricing is None:
            # Default pricing (approximate for GPT-4 equivalent models)
            model_pricing = {
                "input": 0.01,   # $0.01 per 1000 tokens
                "output": 0.03   # $0.03 per 1000 tokens
            }
        
        input_tokens = self.session_tokens["prompt"] + self.session_tokens["system"]
        output_tokens = self.session_tokens["completion"]
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }