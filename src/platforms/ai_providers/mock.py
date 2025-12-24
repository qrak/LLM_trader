import json
import random
from typing import Optional, Dict, Any, List

from src.logger.logger import Logger
from src.platforms.ai_providers.openrouter import ResponseDict


class MockClient:
    """Mock AI provider used for local prompt testing.

    This client emulates provider APIs used by the project and returns
    deterministic, realistic responses based on hints included in the
    prompt messages (a `TEST_HINT: last_close=...` line) or otherwise
    returns a simple synthesized response.
    """

    def __init__(self, api_key: str = "", base_url: str = "", logger: Optional[Logger] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logger

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        await self.close()

    async def close(self) -> None:
        return None

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
            # Default safe response
            last_close = 100.0

        # Very small decision heuristic for deterministic tests
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

    async def chat_completion(self, model: str, messages: list, model_config: Dict[str, Any]) -> Optional[ResponseDict]:
        """Emulate chat completion; return a ResponseDict with choices/message/content."""
        last_close = self._extract_last_close_hint(messages)
        content = self._synthesize_response(last_close)

        # Return matching structure used by ModelManager._process_response
        return {
            "choices": [
                {
                    "message": {
                        "content": content
                    }
                }
            ]
        }

    async def chat_completion_with_chart_analysis(self, model: str, messages: list, chart_image, model_config: Dict[str, Any]) -> Optional[ResponseDict]:
        last_close = self._extract_last_close_hint(messages)
        # Pass has_chart=True to include chart specific text in the response
        content = self._synthesize_response(last_close, has_chart=True)

        return {
            "choices": [
                {
                    "message": {
                        "content": content
                    }
                }
            ]
        }

    async def console_stream(self, model: str, messages: list, model_config: Dict[str, Any]) -> Optional[ResponseDict]:
        # Simulate streaming by returning the full content immediately
        return await self.chat_completion(model, messages, model_config)
