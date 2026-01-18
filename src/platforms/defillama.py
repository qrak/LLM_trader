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

class DexVolumeData(BaseModel):
    """Data model for DEX volume overview."""
    total_24h: float = Field(default=0.0)
    change_1d: float = Field(default=0.0)
    top_protocols: List[Dict[str, Any]] = Field(default_factory=list)

class FeesData(BaseModel):
    """Data model for protocol fees/revenue."""
    total_24h_fees: float = Field(default=0.0)
    total_24h_revenue: float = Field(default=0.0)
    top_earners: List[Dict[str, Any]] = Field(default_factory=list)

class OptionsData(BaseModel):
    """Data model for options market overview."""
    notional_volume_24h: float = Field(default=0.0)
    premium_volume_24h: float = Field(default=0.0)
    top_protocols: List[Dict[str, Any]] = Field(default_factory=list)

class DeFiFundamentalsData(BaseModel):
    """Aggregated on-chain fundamentals."""
    macro: 'MacroMarketData'  # existing: stablecoins, TVL
    dex_volumes: Optional[DexVolumeData] = None
    fees: Optional[FeesData] = None
    options: Optional[OptionsData] = None

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

    async def get_dex_volumes(self) -> Optional[DexVolumeData]:
        """Fetch DEX volume overview."""
        url = f"{self.BASE_URL}/overview/dexs"
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    protocols = data.get("protocols", [])
                    # Sort by 24h volume
                    top_protocols = sorted(
                        [p for p in protocols if isinstance(p.get("total24h"), (int, float))],
                        key=lambda x: x.get("total24h", 0),
                        reverse=True
                    )[:5]
                    
                    return DexVolumeData(
                        total_24h=float(data.get("total24h") or 0),
                        change_1d=float(data.get("change_1d") or 0),
                        top_protocols=top_protocols
                    )
                else:
                    self.logger.warning(f"DefiLlama DEX API error: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching DEX volumes: {e}")
            return None

    async def get_fees_data(self) -> Optional[FeesData]:
        """Fetch fees and revenue overview."""
        url = f"{self.BASE_URL}/overview/fees"
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    protocols = data.get("protocols", [])
                    # Sort by 24h fees
                    top_earners = sorted(
                        [p for p in protocols if isinstance(p.get("total24h"), (int, float))],
                        key=lambda x: x.get("total24h", 0),
                        reverse=True
                    )[:5]
                    
                    # Calculate total revenue if not provided
                    total_revenue = sum(float(p.get("total24hRevenue") or 0) for p in protocols) if protocols else 0
                    
                    return FeesData(
                        total_24h_fees=float(data.get("total24h") or 0),
                        total_24h_revenue=total_revenue,
                        top_earners=top_earners
                    )
                else:
                    # Fees endpoint might return 404 or empty if not available/down
                    self.logger.warning(f"DefiLlama Fees API error: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching fees data: {e}")
            return None

    async def get_options_data(self) -> Optional[OptionsData]:
        """Fetch options market overview."""
        url = f"{self.BASE_URL}/overview/options"
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    protocols = data.get("protocols", [])
                    # Sort by notional volume
                    top_protocols = sorted(
                        [p for p in protocols if isinstance(p.get("totalNotionalVolume"), (int, float))],
                        key=lambda x: x.get("totalNotionalVolume", 0),
                        reverse=True
                    )[:5]
                    
                    return OptionsData(
                        notional_volume_24h=float(data.get("totalNotionalVolume") or 0),
                        premium_volume_24h=float(data.get("totalPremiumVolume") or 0),
                        top_protocols=top_protocols
                    )
                else:
                    # Options endpoint might return 404 or empty if not available/down
                    self.logger.warning(f"DefiLlama Options API error: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching options data: {e}")
            return None

    async def get_defi_fundamentals(self) -> Optional[DeFiFundamentalsData]:
        """Fetch all DeFi fundamentals (Macro + DEX + Fees + Options)."""
        try:
            # Run all requests in parallel
            macro_task = self.get_macro_overview()
            dex_task = self.get_dex_volumes()
            fees_task = self.get_fees_data()
            options_task = self.get_options_data()
            
            results = await asyncio.gather(
                macro_task, 
                dex_task, 
                fees_task, 
                options_task, 
                return_exceptions=True
            )
            
            macro, dex, fees, options = results
            
            # Handle exceptions in results
            if isinstance(macro, Exception) or not macro:
                self.logger.error(f"Failed to fetch macro data: {macro}")
                return None  # Macro is critical
                
            if isinstance(dex, Exception):
                self.logger.error(f"Failed to fetch DEX data: {dex}")
                dex = None
            if isinstance(fees, Exception):
                self.logger.error(f"Failed to fetch Fees data: {fees}")
                fees = None
            if isinstance(options, Exception):
                self.logger.error(f"Failed to fetch Options data: {options}")
                options = None
                
            return DeFiFundamentalsData(
                macro=macro,
                dex_volumes=dex,
                fees=fees,
                options=options
            )
            
        except Exception as e:
            self.logger.error(f"Error building DeFi fundamentals: {e}")
            return None
