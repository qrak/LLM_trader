import io
import json
import random
from typing import Optional, Dict, Any, List, Union

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel


class MockClient(BaseAIClient):
    """Mock AI provider used for local prompt testing.

    This client emulates provider APIs used by the project and returns
    deterministic, realistic responses based on hints included in the
    prompt messages (a `TEST_HINT: last_close=...` line) or otherwise
    returns a simple synthesized response.
    """

    def __init__(self, api_key: str = "", base_url: str = "", logger: Optional[Logger] = None):
        super().__init__(logger or Logger("mock_client", logger_debug=False))
        self.api_key = api_key
        self.base_url = base_url

    async def _initialize_client(self) -> None:
        """Mock client doesn't need initialization."""

    async def close(self) -> None:
        """Mock client doesn't need cleanup."""

    def _extract_last_close_hint(self, messages: List[Dict[str, Any]]) -> Optional[float]:
        """Try to find a TEST_HINT in messages with last_close value."""
        for m in reversed(messages):
            content = m.get("content", "")
            if "TEST_HINT:" in content:
                for part in content.splitlines():
                    if "TEST_HINT:" in part and "last_close" in part:
                        try:
                            kv = part.split("TEST_HINT:", 1)[1].strip()
                            for item in kv.split():
                                if item.startswith("last_close="):
                                    return float(item.split("=")[1])
                        except Exception:
                            continue
        return None

    def _synthesize_response(self, last_close: Optional[float] = None, has_chart: bool = False) -> str:
        """Create a human-readable analysis + JSON block following the project's template."""
        if last_close is None:
            last_close = 100.0
        rnd = (int(last_close) % 3)
        if rnd == 0:
            signal = "BUY"
            confidence = 85
            entry = round(last_close * 0.999, 2)
            sl = round(entry * 0.98, 2)
            tp = round(entry * 1.05, 2)
        elif rnd == 1:
            signal = "SELL"
            confidence = 80
            entry = round(last_close * 1.001, 2)
            sl = round(entry * 1.02, 2)
            tp = round(entry * 0.95, 2)
        else:
            signal = "HOLD"
            confidence = 55
            entry = round(last_close, 2)
            sl = round(last_close * 0.98, 2)
            tp = round(last_close * 1.02, 2)
        pos_size = round(min(0.05, max(0.001, random.random() * 0.03)), 4)
        rr = round(abs((tp - entry) / (entry - sl)) if entry != sl else 0.0, 2)
        if has_chart:
            analysis_text = (
                "Technical indicators point to a clear short-to-medium term bias. "
                "Chart Pattern Analysis: specific patterns detected in the provided image support this view. "
                "Volume and momentum signals have been considered in decision-making.\n\n"
            )
        else:
            analysis_text = (
                "Technical indicators point to a clear short-to-medium term bias. "
                "Volume and momentum signals have been considered in decision-making.\n\n"
            )
        payload = {
            "analysis": {
                "signal": signal,
                "confidence": confidence,
                "confluence_factors": {
                    "trend_alignment": random.randint(50, 95),
                    "momentum_strength": random.randint(50, 90),
                    "volume_support": random.randint(40, 85),
                    "pattern_quality": random.randint(45, 90),
                    "support_resistance_strength": random.randint(50, 95)
                },
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "position_size": pos_size,
                "reasoning": "Auto-generated test decision based on last_close hint",
                "key_levels": {"support": [round(entry * 0.98, 2)], "resistance": [round(entry * 1.02, 2)]},
                "trend": {"direction": "BULLISH" if signal == "BUY" else ("BEARISH" if signal == "SELL" else "NEUTRAL"),
                          "strength": confidence, "timeframe_alignment": "ALIGNED"},
                "risk_reward_ratio": rr
            }
        }
        response = f"{analysis_text}\n```json\n{json.dumps(payload, indent=4)}\n```"
        return response

    async def chat_completion(self, *args, **kwargs) -> Optional[ChatResponseModel]:
        """Emulate chat completion supporting both GoogleAI and OpenRouter signatures."""
        messages = []
        if len(args) > 0:
            if isinstance(args[0], str):
                if len(args) > 1:
                    messages = args[1]
            elif isinstance(args[0], list):
                messages = args[0]
        if not messages and "messages" in kwargs:
            messages = kwargs["messages"]
        last_close = self._extract_last_close_hint(messages)
        content = self._synthesize_response(last_close)
        return self.create_response(content)

    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """Emulate chart analysis with deterministic responses."""
        last_close = self._extract_last_close_hint(messages)
        content = self._synthesize_response(last_close, has_chart=True)
        return self.create_response(content)

    async def console_stream(self, model: str, messages: list, model_config: Dict[str, Any]) -> Optional[ChatResponseModel]:
        """Simulate streaming by returning the full content immediately."""
        return await self.chat_completion(model, messages, model_config)
