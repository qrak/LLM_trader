

import json
from unittest.mock import call

import pytest

from src.platforms.defillama import DefiLlamaClient
from tests.conftest import null_logger

CHAINS_URL = "https://api.llama.fi/v2/chains"
STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoins"
DEX_URL = "https://api.llama.fi/overview/dexs"
FEES_URL = "https://api.llama.fi/overview/fees"
OPTIONS_URL = "https://api.llama.fi/overview/options"

CHAINS_BODY = json.dumps([
    {"gecko_id": "ethereum", "tvl": 60.0, "tokenSymbol": "ETH", "cmcId": "1027",
     "name": "Ethereum", "chainId": 1},
    {"gecko_id": "solana", "tvl": 40.0, "tokenSymbol": "SOL", "cmcId": "5426",
     "name": "Solana", "chainId": 101},
])
STABLECOINS_BODY = json.dumps({"peggedAssets": [{
    "id": "1", "name": "Tether", "symbol": "USDT", "gecko_id": "tether",
    "pegType": "peggedUSD", "circulating": {"peggedUSD": 1000.0},
    "circulatingPrevDay": {"peggedUSD": 900.0},
}]})
EMPTY_OVERVIEW_BODY = json.dumps({"protocols": [], "total24h": 1.0})


class FakeResponse:
    """Minimal aiohttp response carrying the exact bytes the server sent."""

    def __init__(self, status, body, content_type="application/json"):
        self.status = status
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = {"Content-Type": content_type}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def read(self):
        return self._body


class FakeSession:
    """Routes every URL to a canned response and records the requests."""

    def __init__(self, routes):
        self.routes = routes
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        return self.routes[url]


def make_client(tmp_path, routes, logger=None):

    return DefiLlamaClient(
        logger or null_logger(),
        session=FakeSession(routes),
        cache_dir=str(tmp_path),
    )


async def test_chain_tvl_is_read_from_a_text_html_json_body(tmp_path):
    logger = null_logger()
    client = make_client(
        tmp_path, {CHAINS_URL: FakeResponse(200, CHAINS_BODY, "text/html; charset=utf-8")}, logger
    )

    chains = await client.get_chains_tvl()

    assert [chain.name for chain in chains] == ["Ethereum", "Solana"]
    assert [chain.tvl for chain in chains] == [60.0, 40.0]
    assert not logger.error.called


async def test_macro_overview_survives_the_mislabeled_chain_endpoint(tmp_path):
    logger = null_logger()
    client = make_client(tmp_path, {
        STABLECOINS_URL: FakeResponse(200, STABLECOINS_BODY),
        CHAINS_URL: FakeResponse(200, CHAINS_BODY, "text/html; charset=utf-8"),
    }, logger)

    macro = await client.get_macro_overview()

    assert macro is not None
    assert macro.total_tvl == 100.0
    assert macro.stablecoins_market_cap == 1000.0
    assert macro.stablecoins_24h_change == pytest.approx(11.111111, rel=1e-5)
    assert [chain.name for chain in macro.top_chains] == ["Ethereum", "Solana"]
    assert not logger.error.called
    assert not logger.warning.called


async def test_defi_fundamentals_complete_instead_of_logging_a_bare_none(tmp_path):
    logger = null_logger()
    client = make_client(tmp_path, {
        STABLECOINS_URL: FakeResponse(200, STABLECOINS_BODY),
        CHAINS_URL: FakeResponse(200, CHAINS_BODY, "text/html; charset=utf-8"),
        DEX_URL: FakeResponse(200, EMPTY_OVERVIEW_BODY),
        FEES_URL: FakeResponse(200, EMPTY_OVERVIEW_BODY),
        OPTIONS_URL: FakeResponse(200, EMPTY_OVERVIEW_BODY),
    }, logger)

    fundamentals = await client.get_defi_fundamentals()

    assert fundamentals is not None
    assert fundamentals.macro.total_tvl == 100.0
    assert fundamentals.dex_volumes is not None
    assert not logger.error.called


async def test_an_html_body_is_named_instead_of_raising(tmp_path):
    logger = null_logger()
    client = make_client(
        tmp_path, {CHAINS_URL: FakeResponse(200, "<html>challenge</html>", "text/html")}, logger
    )

    assert await client.get_chains_tvl() == []
    non_json = [
        entry for entry in logger.error.call_args_list
        if entry.args[0] == "DefiLlama %s did not return JSON (%s): %s"
    ]
    assert len(non_json) == 1
    assert non_json[0].args[1] == CHAINS_URL
    assert "<html>challenge</html>" in non_json[0].args[3]


async def test_a_non_200_answer_reports_its_status_and_url(tmp_path):
    logger = null_logger()
    client = make_client(
        tmp_path, {CHAINS_URL: FakeResponse(503, "upstream down", "text/plain")}, logger
    )

    assert await client.get_chains_tvl() == []
    assert call("DefiLlama %s answered HTTP %s", CHAINS_URL, 503) in logger.warning.call_args_list


async def test_a_json_object_where_a_list_belongs_is_named_by_shape(tmp_path):
    logger = null_logger()
    client = make_client(
        tmp_path,
        {CHAINS_URL: FakeResponse(200, json.dumps({"error": "rate limited"}))},
        logger,
    )

    assert await client.get_chains_tvl() == []
    assert call(
        "DefiLlama %s returned %s, expected %s", CHAINS_URL, "dict", "list"
    ) in logger.error.call_args_list


async def test_a_mixed_chain_array_keeps_its_object_members(tmp_path):
    body = json.dumps([{"name": "Ethereum", "tvl": 60.0}, "not-a-chain", 7])
    client = make_client(tmp_path, {CHAINS_URL: FakeResponse(200, body, "text/html")})

    chains = await client.get_chains_tvl()

    assert [chain.name for chain in chains] == ["Ethereum"]


async def test_macro_failure_names_the_empty_source(tmp_path):
    logger = null_logger()
    client = make_client(tmp_path, {
        STABLECOINS_URL: FakeResponse(200, STABLECOINS_BODY),
        CHAINS_URL: FakeResponse(200, "<html>challenge</html>", "text/html"),
    }, logger)

    assert await client.get_macro_overview() is None
    assert call(
        "Macro overview unavailable: stablecoins=%s chains=%s", 1, 0
    ) in logger.warning.call_args_list
