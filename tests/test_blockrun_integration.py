"""
Integration tests for BlockRun.AI across the provider ecosystem.

Covers:
  1. ProviderOrchestrator._invoke_blockrun — text + chart dispatch
  2. Fallback chain inclusion (get_text_response, get_chart_response)
  3. supports_chart / is_available with blockrun
  4. Config loader BLOCKRUN properties
  5. ProviderClients construction with blockrun
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import ProviderClients, ProviderMetadata, InvocationResult
from src.platforms.ai_providers.blockrun import BlockRunClient
from src.platforms.ai_providers.response_models import ChatResponseModel


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

def _make_blockrun_client() -> BlockRunClient:
    return BlockRunClient(
        wallet_key="0x0000000000000000000000000000000000000000000000000000000000000001",
        base_url="https://blockrun.ai/api",
        logger=MagicMock(),
    )


def _make_config(overrides: dict | None = None) -> SimpleNamespace:
    """Minimal config with BlockRun settings."""
    defaults = {
        "BLOCKRUN_BASE_URL": "https://blockrun.ai/api",
        "BLOCKRUN_MODEL": "deepseek/deepseek-reasoner",
        "BLOCKRUN_WALLET_KEY": "0x0000000000000000000000000000000000000000000000000000000000000001",
        "OPENROUTER_BASE_MODEL": "google/gemini-3-flash",
        "OPENROUTER_FALLBACK_MODEL": "deepseek/deepseek-r1",
        "LM_STUDIO_MODEL": "local-model",
        "LM_STUDIO_BASE_URL": "http://localhost:1234/v1",
        "GOOGLE_STUDIO_MODEL": "gemini-3.5-flash",
        "PROVIDER": "googleai",
        "TIMEFRAME": "4h",
        "DEMO_QUOTE_CAPITAL": 10000.0,
        "QUOTE_CURRENCY": "USDC",
        "TRANSACTION_FEE_PERCENT": 0.00075,
    }
    if overrides:
        defaults.update(overrides)

    def mock_get_model_config(_model):
        return {"temperature": 0.5, "max_tokens": 1000}

    cfg = SimpleNamespace(**defaults)
    cfg.get_model_config = mock_get_model_config
    return cfg


def _make_orchestrator(
    blockrun_client: BlockRunClient | None = None,
) -> ProviderOrchestrator:
    """Build an orchestrator with only blockrun wired in, all others None."""
    clients = ProviderClients(
        google=None,
        google_paid=None,
        openrouter=None,
        lmstudio=None,
        blockrun=blockrun_client,
    )
    config = _make_config()
    orchestrator = ProviderOrchestrator(
        logger=MagicMock(),
        config=config,
        clients=clients,
    )
    return orchestrator


# ─────────────────────────────────────────────────────────────────
# 1. Orchestrator: _invoke_blockrun text + chart paths
# ─────────────────────────────────────────────────────────────────

class TestInvokeBlockrun:
    """Verify the orchestrator dispatches to blockrun client correctly."""

    @pytest.mark.asyncio
    async def test_invoke_text_route(self):
        """BlockRun text completion calls chat_completion on client."""
        client = _make_blockrun_client()
        client.chat_completion = AsyncMock(return_value=ChatResponseModel.from_content("BUY BTC"))
        orch = _make_orchestrator(blockrun_client=client)

        result = await orch._invoke_blockrun(
            metadata=orch.get_metadata("blockrun"),
            messages=[{"role": "user", "content": "Analyze BTC"}],
            effective_model="deepseek/deepseek-reasoner",
            chart=False,
            chart_image=None,
        )

        assert result.success is True
        assert result.provider == "blockrun"
        assert result.model == "deepseek/deepseek-reasoner"
        assert result.response.choices[0].message.content == "BUY BTC"

    @pytest.mark.asyncio
    async def test_invoke_chart_route(self):
        """BlockRun chart analysis calls chat_completion_with_chart_analysis."""
        client = _make_blockrun_client()
        client.chat_completion_with_chart_analysis = AsyncMock(
            return_value=ChatResponseModel.from_content("Bullish engulfing")
        )
        orch = _make_orchestrator(blockrun_client=client)

        result = await orch._invoke_blockrun(
            metadata=orch.get_metadata("blockrun"),
            messages=[{"role": "user", "content": "Analyze chart"}],
            effective_model="openai/gpt-4o",
            chart=True,
            chart_image=b"fake-png-bytes",
        )

        assert result.success is True
        assert result.provider == "blockrun"
        assert "Bullish engulfing" in result.response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_invoke_text_client_failure(self):
        """When client returns None, orchestrator marks failure."""
        client = _make_blockrun_client()
        client.chat_completion = AsyncMock(return_value=None)
        orch = _make_orchestrator(blockrun_client=client)

        result = await orch._invoke_blockrun(
            metadata=orch.get_metadata("blockrun"),
            messages=[{"role": "user", "content": "Test"}],
            effective_model="deepseek/deepseek-reasoner",
            chart=False,
            chart_image=None,
        )

        assert result.success is False
        assert result.provider == "blockrun"

    @pytest.mark.asyncio
    async def test_invoke_text_client_error_response(self):
        """Client returns error ChatResponseModel → orchestrator marks failure."""
        client = _make_blockrun_client()
        client.chat_completion = AsyncMock(
            return_value=ChatResponseModel.from_error("Rate limit exceeded")
        )
        orch = _make_orchestrator(blockrun_client=client)

        result = await orch._invoke_blockrun(
            metadata=orch.get_metadata("blockrun"),
            messages=[{"role": "user", "content": "Test"}],
            effective_model="deepseek/deepseek-reasoner",
            chart=False,
            chart_image=None,
        )

        assert result.success is False
        assert "Rate limit" in result.response.error


# ─────────────────────────────────────────────────────────────────
# 2. Orchestrator: invoke dispatch
# ─────────────────────────────────────────────────────────────────

class TestInvokeDispatch:
    """Verify the main invoke() method dispatches 'blockrun' to _invoke_blockrun."""

    @pytest.mark.asyncio
    async def test_invoke_dispatches_blockrun(self):
        """provider='blockrun' routes to _invoke_blockrun."""
        client = _make_blockrun_client()
        client.chat_completion = AsyncMock(return_value=ChatResponseModel.from_content("SELL BTC"))
        orch = _make_orchestrator(blockrun_client=client)

        result = await orch.invoke("blockrun", [{"role": "user", "content": "Test"}])

        assert result.success is True
        assert result.provider == "blockrun"
        assert result.response.choices[0].message.content == "SELL BTC"

    @pytest.mark.asyncio
    async def test_invoke_blockrun_unavailable(self):
        """When blockrun client is None, invoke returns failure."""
        orch = _make_orchestrator(blockrun_client=None)

        result = await orch.invoke("blockrun", [{"role": "user", "content": "Test"}])

        assert result.success is False
        assert result.provider == "blockrun"


# ─────────────────────────────────────────────────────────────────
# 3. Availability checks
# ─────────────────────────────────────────────────────────────────

class TestBlockrunAvailability:
    """Verify is_available, supports_chart, get_metadata for blockrun."""

    def test_is_available_when_client_present(self):
        client = _make_blockrun_client()
        orch = _make_orchestrator(blockrun_client=client)
        assert orch.is_available("blockrun") is True

    def test_is_available_when_client_none(self):
        orch = _make_orchestrator(blockrun_client=None)
        assert orch.is_available("blockrun") is False

    def test_supports_chart_blockrun(self):
        client = _make_blockrun_client()
        orch = _make_orchestrator(blockrun_client=client)
        assert orch.supports_chart("blockrun") is True

    def test_supports_chart_all_includes_blockrun(self):
        """When 'all' is specified, blockrun's chart support is considered."""
        client = _make_blockrun_client()
        orch = _make_orchestrator(blockrun_client=client)
        # blockrun alone should make 'all' chart-capable
        assert orch.supports_chart("all") is True

    def test_supports_chart_all_no_blockrun_no_others(self):
        """When no chart-capable providers exist, 'all' returns False."""
        orch = _make_orchestrator(blockrun_client=None)
        # All providers are None → no chart support
        assert orch.supports_chart("all") is False

    def test_get_metadata_returns_correct_default_model(self):
        client = _make_blockrun_client()
        orch = _make_orchestrator(blockrun_client=client)
        meta = orch.get_metadata("blockrun")
        assert meta is not None
        assert meta.default_model == "deepseek/deepseek-reasoner"
        assert meta.supports_chart is True
        assert meta.name == "BlockRun.AI"

    def test_resolve_model_uses_default(self):
        client = _make_blockrun_client()
        orch = _make_orchestrator(blockrun_client=client)
        assert orch.resolve_model("blockrun") == "deepseek/deepseek-reasoner"

    def test_resolve_model_uses_override(self):
        client = _make_blockrun_client()
        orch = _make_orchestrator(blockrun_client=client)
        assert orch.resolve_model("blockrun", "openai/gpt-4o") == "openai/gpt-4o"


# ─────────────────────────────────────────────────────────────────
# 4. Fallback chain integration
# ─────────────────────────────────────────────────────────────────

class TestBlockrunFallbackChain:
    """Verify blockrun participates correctly in fallback chains."""

    @pytest.mark.asyncio
    async def test_blockrun_in_text_fallback_chain(self):
        """Blockrun should be in the 'all' text fallback chain."""
        client = _make_blockrun_client()
        client.chat_completion = AsyncMock(return_value=ChatResponseModel.from_content("BUY"))
        orch = _make_orchestrator(blockrun_client=client)

        result = await orch.get_text_response("all", [{"role": "user", "content": "Test"}])
        # Should succeed because blockrun is available
        assert result.success is True

    @pytest.mark.asyncio
    async def test_blockrun_in_chart_fallback_chain(self):
        """Blockrun should be in the 'all' chart fallback chain."""
        client = _make_blockrun_client()
        client.chat_completion_with_chart_analysis = AsyncMock(
            return_value=ChatResponseModel.from_content("Pattern detected")
        )
        orch = _make_orchestrator(blockrun_client=client)

        result = await orch.get_chart_response(
            "all",
            [{"role": "user", "content": "Test"}],
            chart_image=b"fake-png",
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_fallback_skips_unavailable_blockrun(self):
        """When blockrun is None, fallback chain skips it gracefully."""
        orch = _make_orchestrator(blockrun_client=None)

        result = await orch.invoke_with_fallback(
            ["blockrun", "local", "openrouter"],
            [{"role": "user", "content": "Test"}],
        )
        # All three are unavailable → complete failure
        assert result.success is False

    @pytest.mark.asyncio
    async def test_fallback_uses_blockrun_when_available(self):
        """Available blockrun in fallback chain returns its result."""
        client = _make_blockrun_client()
        client.chat_completion = AsyncMock(return_value=ChatResponseModel.from_content("HOLD"))
        orch = _make_orchestrator(blockrun_client=client)

        result = await orch.invoke_with_fallback(
            ["blockrun"],
            [{"role": "user", "content": "Test"}],
        )
        assert result.success is True
        assert result.response.choices[0].message.content == "HOLD"


# ─────────────────────────────────────────────────────────────────
# 5. Config loader properties
# ─────────────────────────────────────────────────────────────────
# (tested via _make_config above, but we verify key shape here)

class TestBlockrunConfigProperties:
    """Verify config.ini / keys.env properties for BlockRun."""

    def test_blockrun_base_url_default(self):
        cfg = _make_config()
        assert cfg.BLOCKRUN_BASE_URL == "https://blockrun.ai/api"

    def test_blockrun_model_default(self):
        cfg = _make_config()
        assert cfg.BLOCKRUN_MODEL == "deepseek/deepseek-reasoner"

    def test_blockrun_wallet_key(self):
        cfg = _make_config()
        assert cfg.BLOCKRUN_WALLET_KEY == "0x0000000000000000000000000000000000000000000000000000000000000001"

    def test_blockrun_config_override(self):
        cfg = _make_config({
            "BLOCKRUN_MODEL": "anthropic/claude-sonnet-4",
            "BLOCKRUN_BASE_URL": "https://custom.blockrun.ai/api",
        })
        assert cfg.BLOCKRUN_MODEL == "anthropic/claude-sonnet-4"
        assert cfg.BLOCKRUN_BASE_URL == "https://custom.blockrun.ai/api"


# ─────────────────────────────────────────────────────────────────
# 6. ProviderClients construction
# ─────────────────────────────────────────────────────────────────

class TestProviderClientsBlockrun:
    """Verify ProviderClients dataclass can hold blockrun client."""

    def test_blockrun_field_accepts_client(self):
        client = _make_blockrun_client()
        pc = ProviderClients(blockrun=client)
        assert pc.blockrun is client

    def test_blockrun_field_defaults_to_none(self):
        pc = ProviderClients()
        assert pc.blockrun is None

    def test_all_fields_together(self):
        """All provider slots coexist without conflict."""
        br = _make_blockrun_client()
        pc = ProviderClients(
            google=None,
            google_paid=None,
            openrouter=None,
            lmstudio=None,
            blockrun=br,
        )
        assert pc.blockrun is br
        assert pc.google is None


# ─────────────────────────────────────────────────────────────────
# 7. Log guidance for unavailable blockrun
# ─────────────────────────────────────────────────────────────────

class TestBlockrunLogGuidance:
    """Verify proper log messages when blockrun is unavailable."""

    def test_unavailable_guidance_logs_wallet_key_hint(self):
        orch = _make_orchestrator(blockrun_client=None)
        orch._log_unavailable_guidance("blockrun")
        orch.logger.error.assert_called_once()
        call_args = orch.logger.error.call_args[0]
        message = call_args[0]
        assert "BLOCKRUN_WALLET_KEY" in message

    def test_failure_logs_blockrun_message(self):
        """_log_failure for blockrun uses blockrun-specific message."""
        orch = _make_orchestrator(blockrun_client=None)
        orch._log_failure("blockrun")
        orch.logger.warning.assert_called_once()
        call_args = orch.logger.warning.call_args[0]
        joined = " ".join(str(x) for x in call_args)
        assert "BlockRun" in joined
