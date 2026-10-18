"""Circuit breaker: consecutive failures open a provider, a probe closes it, and an
all-open chain still attempts the primary instead of turning an outage into silence."""

from unittest.mock import MagicMock

from src.managers.provider_health import ProviderHealth
from src.managers.provider_orchestrator import ProviderOrchestrator


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_threshold_must_be_reached_before_opening():
    clock = Clock()
    health = ProviderHealth(failure_threshold=3, cooldown_seconds=60, clock=clock)

    health.record_failure("deepseek")
    health.record_failure("deepseek")
    assert health.allows("deepseek") is True

    health.record_failure("deepseek")
    assert health.allows("deepseek") is False


def test_cooldown_elapsed_allows_one_probe_and_failure_backs_off_longer():
    clock = Clock()
    health = ProviderHealth(failure_threshold=2, cooldown_seconds=60, clock=clock)
    health.record_failure("googleai")
    health.record_failure("googleai")
    assert health.allows("googleai") is False

    clock.advance(59)
    assert health.allows("googleai") is False
    clock.advance(1)
    assert health.allows("googleai") is True

    health.record_failure("googleai")
    clock.advance(60)
    assert health.allows("googleai") is False, "failed probe must re-open with a longer cooldown"
    clock.advance(60)
    assert health.allows("googleai") is True


def test_success_closes_the_circuit_and_resets_backoff():
    clock = Clock()
    health = ProviderHealth(failure_threshold=2, cooldown_seconds=60, clock=clock)
    health.record_failure("openrouter")
    health.record_failure("openrouter")

    health.record_success("openrouter")

    assert health.allows("openrouter") is True
    assert health.snapshot()["openrouter"]["failures"] == 0


def test_rate_limit_does_not_open_the_circuit():
    clock = Clock()
    health = ProviderHealth(failure_threshold=1, cooldown_seconds=60, clock=clock)

    health.record_failure("googleai", category=ProviderHealth.RATE_LIMIT_CATEGORY)

    assert health.allows("googleai") is True
    assert health.snapshot()["googleai"]["failures"] == 0


def _orchestrator(health: ProviderHealth | None, availability: dict[str, bool]) -> ProviderOrchestrator:
    orchestrator = ProviderOrchestrator.__new__(ProviderOrchestrator)
    orchestrator.logger = MagicMock()
    orchestrator.provider_health = health
    orchestrator._providers = {}
    for provider, is_available in availability.items():
        metadata = MagicMock()
        metadata.is_available.return_value = is_available
        orchestrator._providers[provider] = metadata
    return orchestrator


def test_open_providers_are_skipped():
    clock = Clock()
    health = ProviderHealth(failure_threshold=1, cooldown_seconds=60, clock=clock)
    health.record_failure("deepseek")
    orchestrator = _orchestrator(health, {"deepseek": True, "openrouter": True})

    assert orchestrator._circuit_open_providers(["deepseek", "openrouter"]) == ["openrouter"]


def test_all_open_falls_back_to_the_primary_and_screams():
    clock = Clock()
    health = ProviderHealth(failure_threshold=1, cooldown_seconds=60, clock=clock)
    health.record_failure("deepseek")
    health.record_failure("openrouter")
    orchestrator = _orchestrator(health, {"deepseek": True, "openrouter": True})

    chosen = orchestrator._circuit_open_providers(["deepseek", "openrouter"])

    assert chosen == ["deepseek"]
    orchestrator.logger.critical.assert_called_once()


def test_unavailable_providers_are_never_chosen():
    orchestrator = _orchestrator(None, {"deepseek": False})

    assert orchestrator._circuit_open_providers(["deepseek"]) == []


def test_without_health_the_chain_is_unchanged():
    orchestrator = _orchestrator(None, {"deepseek": True, "openrouter": False})

    assert orchestrator._circuit_open_providers(["deepseek", "openrouter"]) == ["deepseek"]
