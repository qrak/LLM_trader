"""
DefiLlama API Client
Handles fetching macro market data (Stablecoins, TVL) from DefiLlama.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import aiohttp
import asyncio
from src.logger.logger import Logger

# --- Pydantic Models ---

class StablecoinData(BaseModel):
    """Data model for a single stablecoin."""
    id: str = Field(default="")
    name: str = Field(default="")
    symbol: str = Field(default="")
    gecko_id: Optional[str] = None
    pegType: Optional[str] = None
    circulating: float = Field(default=0.0)
    circulatingPrevDay: float = Field(default=0.0)
    circulatingPrev1Week: float = Field(default=0.0)
    circulatingPrev1Month: float = Field(default=0.0)
    
    @property
    def price(self) -> float:
        return 1.0

class ChainTVLData(BaseModel):
    """Data model for a single chain's TVL."""
    gecko_id: Optional[str] = None
    tvl: float = Field(default=0.0)
    tokenSymbol: Optional[str] = None
    cmcId: Optional[Union[str, int]] = None
    name: str = Field(default="Unknown")
    chainId: Optional[Union[str, int]] = None

class MacroMarketData(BaseModel):
    """Aggregated macro market data."""
    stablecoins_market_cap: float
    stablecoins_24h_change: float
    total_tvl: float
    top_chains: List[ChainTVLData]

# --- Client Class ---

class DefiLlamaClient:
    """Client for DefiLlama API."""
    
    BASE_URL = "https://api.llama.fi"
    STABLECOINS_URL = "https://stablecoins.llama.fi"

    def __init__(self, logger: Logger, session: Optional[aiohttp.ClientSession] = None):
        self.logger = logger
        self._external_session = session is not None
        self.session = session

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_stablecoins(self) -> List[StablecoinData]:
        """Fetch list of all stablecoins."""
        url = f"{self.STABLECOINS_URL}/stablecoins"
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pegged = data.get("peggedAssets", [])
                    return [
                        StablecoinData(
                            id=item.get("id", ""),
                            name=item.get("name", ""),
                            symbol=item.get("symbol", ""),
                            gecko_id=item.get("gecko_id"),
                            pegType=item.get("pegType"),
                            circulating=float(item.get("circulating", {}).get("peggedUSD", 0) or 0),
                            circulatingPrevDay=float(item.get("circulatingPrevDay", {}).get("peggedUSD", 0) or 0),
                            circulatingPrev1Week=float(item.get("circulatingPrev1Week", {}).get("peggedUSD", 0) or 0),
                            circulatingPrev1Month=float(item.get("circulatingPrev1Month", {}).get("peggedUSD", 0) or 0)
                        )
                        for item in pegged
                        if item.get("circulating", {}).get("peggedUSD") is not None
                    ]
                else:
                    self.logger.error(f"DefiLlama Stablecoins API error: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error fetching stablecoins: {e}")
            return []

    async def get_chains_tvl(self) -> List[ChainTVLData]:
        """Fetch current TVL of all chains."""
        url = f"{self.BASE_URL}/v2/chains"
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return [ChainTVLData(**item) for item in data]
                else:
                    self.logger.error(f"DefiLlama Chains API error: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error fetching chain TVL: {e}")
            return []

    async def get_macro_overview(self) -> Optional[MacroMarketData]:
        """Get aggregated macro market data (Stablecoin MC + TVL)."""
        try:
            stables, chains = await asyncio.gather(
                self.get_stablecoins(),
                self.get_chains_tvl()
            )
            
            if not stables or not chains:
                return None

            # Calculate Stablecoin Metrics
            total_stable_mc = sum(s.circulating for s in stables)
            total_stable_mc_prev = sum(s.circulatingPrevDay for s in stables)
            stable_change = ((total_stable_mc - total_stable_mc_prev) / total_stable_mc_prev * 100) if total_stable_mc_prev else 0

            # Calculate TVL Metrics
            total_tvl = sum(c.tvl for c in chains)
            
            # Sort chains by TVL and take top 5
            top_chains = sorted(chains, key=lambda x: x.tvl, reverse=True)[:5]

            return MacroMarketData(
                stablecoins_market_cap=total_stable_mc,
                stablecoins_24h_change=stable_change,
                total_tvl=total_tvl,
                top_chains=top_chains
            )
        except Exception as e:
            self.logger.error(f"Error building macro overview: {e}")
            return None

    async def close(self):
        """Close the session if it was created internally."""
        if not self._external_session and self.session:
            await self.session.close()
            self.session = None
