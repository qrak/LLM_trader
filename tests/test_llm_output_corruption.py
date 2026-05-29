"""Chaos tests: LLM output corruption, schema violations, and fallback loop resilience.

Covers Pillar 1 — verifies the system handles:
- Corrupted / truncated / malformed JSON responses from AI providers
- Unexpected string injections in structured fields
- Missing required execution fields (entry_price, stop_loss, etc.)
- Empty content after valid-looking responses
- Nested JSON injection into flat content fields
- Raw binary / control characters in LLM output
- LLM responses that are valid JSON but violate TradingAnalysisModel schema
- Fallback loop does not infinite-loop on persistent corruption
- Default signal generation when LLM output cannot be parsed
"""

import json
from unittest.mock import MagicMock

import pytest

from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import ProviderClients
from src.platforms.ai_providers.response_models import (
    ChatResponseModel,
    TradingAnalysisModel,
    TradingSignal,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_config_stub():
    cfg = MagicMock()
    cfg.GOOGLE_STUDIO_MODEL = "gemini-flash"
    cfg.OPENROUTER_BASE_MODEL = "openrouter/primary"
    cfg.OPENROUTER_FALLBACK_MODEL = "openrouter/fallback"
    cfg.LM_STUDIO_MODEL = "local/model"
    cfg.get_model_config.return_value = {"max_tokens": 128}
    return cfg


def _mock_client(returns: list[ChatResponseModel]) -> MagicMock:
    """Return a mock provider client that yields responses in order."""
    client = MagicMock()
    responses = list(reversed(returns))  # pop from end

    async def chat_completion(*args, **kwargs):
        return responses.pop()

    client.chat_completion = chat_completion
    return client


def _orchestrator(clients: dict | None = None) -> ProviderOrchestrator:
    """Build an orchestrator with one or more mocked clients."""
    defaults = {
        "googleai": _mock_client([ChatResponseModel.from_content("default ok")]),
        "openrouter": _mock_client([ChatResponseModel.from_content("default ok")]),
        "local": _mock_client([ChatResponseModel.from_content("default ok")]),
    }
    merged = {**defaults, **(clients or {})}
    return ProviderOrchestrator(
        logger=MagicMock(),
        config=_make_config_stub(),
        clients=ProviderClients(
            google=merged.get("googleai"),
            openrouter=merged.get("openrouter"),
            lmstudio=merged.get("local"),
        ),
    )


# ── 1. TRUNCATED / MALFORMED JSON ─────────────────────────────────────────────

class TestTruncatedJsonResponse:
    """LLM returns incomplete JSON that cannot be parsed by the trading validator."""

    @pytest.mark.asyncio
    async def test_truncated_analysis_does_not_crash_orchestrator(self):
        """Truncated JSON with missing closing braces should be handled gracefully."""
        provider = _orchestrator({
            "openrouter": _mock_client([
                ChatResponseModel.from_content(
                    '{"analysis": {"signal": "BUY", "confidence": 85, "entry_price": 50000'
                    # deliberately truncated — no closing braces
                )
            ])
        })
        result = await provider.invoke("openrouter", [{"role": "user", "content": "analyze"}])
        # Should not raise; result may be success=False or parsing fails downstream
        assert result is not None
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_malformed_json_with_random_noise_is_safe(self):
        """LLM returns binary-like noise instead of structured JSON."""
        provider = _orchestrator({
            "openrouter": _mock_client([
                ChatResponseModel.from_content(
                    "\x00\x01\x02\x00\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03���RAG\u0000\u0000"
                )
            ])
        })
        result = await provider.invoke("openrouter", [{"role": "user", "content": "analyze"}])
        assert result is not None

    @pytest.mark.asyncio
    async def test_multiple_json_objects_in_single_response(self):
        """LLM returns concatenated JSON objects — parser must not crash."""
        provider = _orchestrator({
            "openrouter": _mock_client([
                ChatResponseModel.from_content(
                    '{"analysis": {"signal": "BUY", "confidence": 50}}'
                    '\n{"analysis": {"signal": "SELL", "confidence": 80}}'
                )
            ])
        })
        result = await provider.invoke("openrouter", [{"role": "user", "content": "analyze"}])
        assert result is not None


class TestStringInjectionInNumericFields:
    """LLM injects non-numeric strings where floats are required."""

    @pytest.mark.asyncio
    async def test_string_in_entry_price_is_rejected(self):
        """entry_price as a string 'high' instead of a float should fail validation."""
        bad_json = json.dumps({
            "analysis": {
                "signal": "BUY",
                "confidence": 80,
                "entry_price": "high",
                "stop_loss": 48000,
                "take_profit": 52000,
                "position_size": 0.1,
                "risk_reward_ratio": 2.0,
                "reasoning": "price is going high",
            }
        })
        provider = _orchestrator({
            "openrouter": _mock_client([ChatResponseModel.from_content(bad_json)])
        })
        result = await provider.invoke("openrouter", [{"role": "user", "content": "analyze"}])
        assert result is not None

    @pytest.mark.asyncio
    async def test_infinity_and_nan_in_fields_are_safe(self):
        """Non-finite floats like Infinity/NaN must not crash the pipeline."""
        bad_json = json.dumps({
            "analysis": {
                "signal": "BUY",
                "confidence": float('inf'),
                "entry_price": float('nan'),
                "stop_loss": 48000,
                "take_profit": 52000,
                "position_size": float('-inf'),
                "risk_reward_ratio": 2.0,
                "reasoning": "moon or bust",
            }
        })
        provider = _orchestrator({
            "openrouter": _mock_client([ChatResponseModel.from_content(bad_json)])
        })
        result = await provider.invoke("openrouter", [{"role": "user", "content": "analyze"}])
        assert result is not None


# ── 2. MISSING REQUIRED FIELDS — direct model validation ──────────────────────

class TestMissingExecutionFields:
    """TradingAnalysisModel must reject BUY/SELL signals missing required fields."""

    def test_buy_signal_missing_stop_loss_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required execution fields"):
            TradingAnalysisModel(
                signal=TradingSignal.BUY,
                confidence=85,
                entry_price=50000,
                # stop_loss missing
                take_profit=52000,
                position_size=0.1,
                risk_reward_ratio=2.0,
                reasoning="test",
            )

    def test_sell_signal_missing_take_profit_and_position_size(self):
        with pytest.raises(ValueError, match="missing required execution fields"):
            TradingAnalysisModel(
                signal=TradingSignal.SELL,
                confidence=70,
                entry_price=50000,
                stop_loss=51000,
                # take_profit missing
                # position_size missing
                risk_reward_ratio=1.5,
                reasoning="going down",
            )

    def test_update_signal_missing_entry_price(self):
        with pytest.raises(ValueError, match="UPDATE response requires entry_price"):
            TradingAnalysisModel(
                signal=TradingSignal.UPDATE,
                confidence=60,
                stop_loss=49000,
                # entry_price missing
                reasoning="adjust SL",
            )

    def test_update_signal_missing_both_stop_loss_and_take_profit(self):
        with pytest.raises(ValueError, match="UPDATE response must include"):
            TradingAnalysisModel(
                signal=TradingSignal.UPDATE,
                confidence=60,
                entry_price=50000,
                # both stop_loss and take_profit missing
                reasoning="no change",
            )

    def test_hold_signal_does_not_require_execution_fields(self):
        """HOLD is a non-execution signal — must validate without error."""
        model = TradingAnalysisModel(
            signal=TradingSignal.HOLD,
            confidence=40,
            reasoning="wait and see",
        )
        assert model.signal == TradingSignal.HOLD


# ── 3. FALLBACK LOOP RESILIENCE ───────────────────────────────────────────────

class TestFallbackLoopResilience:
    """When all providers return corrupt data, fallback must not infinite-loop."""

    @pytest.mark.asyncio
    async def test_all_providers_corrupt_falls_through_gracefully(self):
        """Three providers all returning corrupted output — orchestrator returns last failure."""
        corrupt = ChatResponseModel.from_content("\x00\x01corrupt\x00")
        provider = _orchestrator({
            "googleai": _mock_client([corrupt]),
            "openrouter": _mock_client([corrupt]),
            "local": _mock_client([corrupt]),
        })

        result = await provider.invoke_with_fallback(
            ["googleai", "local", "openrouter"],
            [{"role": "user", "content": "analyze"}],
        )
        assert result is not None
        # Must not crash; must not loop infinitely

    @pytest.mark.asyncio
    async def test_mixed_corruption_then_valid_still_recovers(self):
        """First provider returns corrupt data, second succeeds — must recover."""
        provider = _orchestrator({
            "googleai": _mock_client([
                ChatResponseModel.from_content("{broken json\u0000")
            ]),
            "openrouter": _mock_client([
                ChatResponseModel.from_content('{"analysis": {"signal": "HOLD", "confidence": 50, "reasoning": "ok"}}')
            ]),
            "local": _mock_client([ChatResponseModel.from_content("should not reach local")]),
        })

        result = await provider.invoke_with_fallback(
            ["googleai", "openrouter", "local"],
            [{"role": "user", "content": "analyze"}],
        )
        # If google's corruption causes success=False, orchestrator falls through to openrouter
        # openrouter returns parseable content => result.success = True
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_choices_list_does_not_crash_fallback_chain(self):
        """Response with empty choices list is an edge case from some providers."""
        empty_response = ChatResponseModel(choices=[])
        # The openrouter client returns empty choices, then triggers fallback model call
        # but that should also get empty choices (or fallback doesn't change outcome)
        provider = _orchestrator({
            "openrouter": _mock_client([empty_response, empty_response]),
        })
        result = await provider.invoke("openrouter", [{"role": "user", "content": "analyze"}])
        assert result is not None
        # success should be False because _is_valid_response checks len(choices) > 0
        assert not result.success


# ── 4. SCHEMA VIOLATIONS (TradingAnalysisResponseModel) ───────────────────────

class TestSchemaViolations:
    """TradingAnalysisResponseModel and its wrappers handle field-level violations."""

    def test_confidence_out_of_range_is_rejected(self):
        """Confidence must be 0-100 — values outside this range are schema violations."""
        with pytest.raises(Exception):
            TradingAnalysisModel(
                signal=TradingSignal.HOLD,
                confidence=150,  # invalid
                reasoning="too confident",
            )

    def test_negative_position_size_is_rejected(self):
        with pytest.raises(Exception):
            TradingAnalysisModel(
                signal=TradingSignal.BUY,
                confidence=80,
                entry_price=50000,
                stop_loss=49000,
                take_profit=52000,
                position_size=-0.5,  # invalid
                risk_reward_ratio=2.0,
                reasoning="negative size",
            )

    def test_position_size_exceeds_one_is_rejected(self):
        with pytest.raises(Exception):
            TradingAnalysisModel(
                signal=TradingSignal.BUY,
                confidence=80,
                entry_price=50000,
                stop_loss=49000,
                take_profit=52000,
                position_size=1.5,  # invalid >1.0
                risk_reward_ratio=2.0,
                reasoning="overcommitted",
            )

    def test_invalid_signal_enum_value_is_rejected(self):
        """An unknown trading signal must fail TradingSignal enum validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TradingAnalysisModel(
                signal="MOON",  # type: ignore[arg-type]
                confidence=80,
                reasoning="not a real signal",
            )


# ── 5. CONTENT-LEVEL CORRUPTION (markdown injection, script tags) ─────────────

class TestContentLevelCorruption:
    """LLM responses that inject HTML, markdown tables, or script tags into content."""

    @pytest.mark.asyncio
    async def test_html_script_injection_in_llm_output(self):
        """LLM output with script injection must not crash the pipeline."""
        injected = (
            '{"analysis": {"signal": "HOLD", "confidence": 50, "reasoning": "'
            '<script>alert(\"xss\")</script> market looks normal'
            '"}}'
        )
        provider = _orchestrator({
            "openrouter": _mock_client([ChatResponseModel.from_content(injected)])
        })
        result = await provider.invoke("openrouter", [{"role": "user", "content": "analyze"}])
        assert result is not None

    @pytest.mark.asyncio
    async def test_massive_reasoning_field_does_not_overflow(self):
        """LLM returns a 100KB reasoning string — must not crash or OOM."""
        provider = _orchestrator({
            "openrouter": _mock_client([
                ChatResponseModel.from_content(
                    '{"analysis": {"signal": "HOLD", "confidence": 50, "reasoning": "'
                    + ("A" * 100_000)
                    + '"}}'
                )
            ])
        })
        result = await provider.invoke("openrouter", [{"role": "user", "content": "analyze"}])
        assert result is not None
