from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.market_data_manager import MarketDataManager
from src.rag.market_components import MarketDataCache


@pytest.fixture
def manager() -> MarketDataManager:
    logger = MagicMock()
    file_handler = MagicMock()
    unified_parser = SimpleNamespace(
        format_utils=SimpleNamespace(parse_timestamp=MagicMock())
    )
    cache = MarketDataCache(logger=logger, file_handler=file_handler)

    return MarketDataManager(
        logger=logger,
        file_handler=file_handler,
        unified_parser=unified_parser,
        fetcher=MagicMock(),
        processor=MagicMock(),
        cache=cache,
        overview_builder=MagicMock(),
    )


@pytest.mark.asyncio
async def test_update_fetches_when_overview_missing(manager: MarketDataManager):
    manager.fetch_market_overview = AsyncMock(return_value={"timestamp": "now", "published_on": 1.0})

    updated = await manager.update_market_overview_if_needed(max_age_hours=24)

    assert updated is True
    assert manager.get_current_overview() == {"timestamp": "now", "published_on": 1.0}
    manager.fetch_market_overview.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_skips_fetch_when_overview_is_fresh(manager: MarketDataManager):
    now_ts = datetime.now(timezone.utc).timestamp()
    manager.current_market_overview = {"published_on": now_ts}
    manager.unified_parser.format_utils.parse_timestamp.return_value = now_ts
    manager.fetch_market_overview = AsyncMock(return_value={"timestamp": "new", "published_on": now_ts + 5})

    updated = await manager.update_market_overview_if_needed(max_age_hours=24)

    assert updated is False
    manager.fetch_market_overview.assert_not_called()


@pytest.mark.asyncio
async def test_update_fetches_when_overview_is_stale(manager: MarketDataManager):
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).timestamp()
    manager.current_market_overview = {"published_on": stale_ts}
    manager.unified_parser.format_utils.parse_timestamp.return_value = stale_ts
    manager.fetch_market_overview = AsyncMock(return_value={"timestamp": "new", "published_on": 9999.0})

    updated = await manager.update_market_overview_if_needed(max_age_hours=24)

    assert updated is True
    manager.fetch_market_overview.assert_awaited_once()


def test_is_overview_stale_true_when_timestamp_unparseable(manager: MarketDataManager):
    manager.current_market_overview = {"published_on": "invalid"}
    manager.unified_parser.format_utils.parse_timestamp.return_value = None

    assert manager.is_overview_stale(max_age_hours=1) is True


def test_is_overview_stale_false_for_recent_data(manager: MarketDataManager):
    recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp()
    manager.current_market_overview = {"published_on": recent_ts}
    manager.unified_parser.format_utils.parse_timestamp.return_value = recent_ts

    assert manager.is_overview_stale(max_age_hours=1) is False
