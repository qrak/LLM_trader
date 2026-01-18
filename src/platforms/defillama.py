"""
DefiLlama API Client
Handles fetching macro market data (Stablecoins, TVL) from DefiLlama.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
from src.logger.logger import Logger


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


class DefiLlamaClient:
    """Client for DefiLlama API."""
    
    BASE_URL = "https://api.llama.fi"
    STABLECOINS_URL = "https://stablecoins.llama.fi"

    DEFILLAMA_CACHE_FILE = "defillama_fundamentals.json"
    
    def __init__(self, logger: Logger, session: Optional[aiohttp.ClientSession] = None, 
                 cache_dir: str = 'cache', update_interval_hours: float = 0.25):
        self.logger = logger
        self._external_session = session is not None
        self.session = session
        
        # Caching setup
        self.cache_dir = cache_dir
        self.update_interval = timedelta(hours=update_interval_hours)
        self.cache_file_path = f"{cache_dir}/{self.DEFILLAMA_CACHE_FILE}"
        self.last_update: Optional[datetime] = None
        
        # Ensure cache directory exists
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Try to load last update time from cache
        self._load_cache_metadata()

    def _load_cache_metadata(self):
        """Load metadata from cache file to set last_update without full parse."""
        import os
        import json
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "timestamp" in data:
                        self.last_update = datetime.fromisoformat(data["timestamp"])
        except Exception as e:
            self.logger.debug(f"Could not read DefiLlama cache metadata: {e}")

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

    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float."""
        try:
            if value is None: 
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0

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
                        [p for p in protocols if self._safe_float(p.get("total24h")) > 0],
                        key=lambda x: self._safe_float(x.get("total24h")),
                        reverse=True
                    )[:5]
                    
                    return DexVolumeData(
                        total_24h=self._safe_float(data.get("total24h")),
                        change_1d=self._safe_float(data.get("change_1d")),
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
                        [p for p in protocols if self._safe_float(p.get("total24h")) > 0],
                        key=lambda x: self._safe_float(x.get("total24h")),
                        reverse=True
                    )[:5]
                    
                    # Calculate total revenue if not provided
                    total_revenue = sum(self._safe_float(p.get("total24hRevenue")) for p in protocols) if protocols else 0
                    
                    return FeesData(
                        total_24h_fees=self._safe_float(data.get("total24h")),
                        total_24h_revenue=total_revenue,
                        top_earners=top_earners
                    )
                else:
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
                        [p for p in protocols if self._safe_float(p.get("totalNotionalVolume")) > 0],
                        key=lambda x: self._safe_float(x.get("totalNotionalVolume")),
                        reverse=True
                    )[:5]
                    
                    return OptionsData(
                        notional_volume_24h=self._safe_float(data.get("totalNotionalVolume")),
                        premium_volume_24h=self._safe_float(data.get("totalPremiumVolume")),
                        top_protocols=top_protocols
                    )
                else:
                    self.logger.warning(f"DefiLlama Options API error: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching options data: {e}")
            return None

    async def get_defi_fundamentals(self) -> Optional[DeFiFundamentalsData]:
        """Fetch all DeFi fundamentals (Macro + DEX + Fees + Options)."""
        import json
        import os
        from datetime import timezone

        # Check cache freshness
        current_time = datetime.now(timezone.utc)
        if self.last_update and (current_time - self.last_update < self.update_interval):
            if os.path.exists(self.cache_file_path):
                try:
                    with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        if "data" in cached_data:
                            self.logger.debug(f"Using cached DefiLlama data from {self.last_update.isoformat()}")
                            return DeFiFundamentalsData(**cached_data["data"])
                except Exception as e:
                    self.logger.warning(f"Failed to read DefiLlama cache: {e}")
        
        self.logger.debug("Fetching fresh DefiLlama fundamentals...")
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
                
            fundamentals = DeFiFundamentalsData(
                macro=macro,
                dex_volumes=dex,
                fees=fees,
                options=options
            )
            
            # Save to cache
            try:
                cache_payload = {
                    "timestamp": current_time.isoformat(),
                    "data": fundamentals.model_dump()
                }
                # Write to temp file then rename for atomic write
                temp_path = f"{self.cache_file_path}.tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_payload, f, ensure_ascii=False, indent=2)
                
                # Windows atomic replace might need unlink first if exists, but os.replace usually handles it
                if os.path.exists(self.cache_file_path):
                     try:
                         os.remove(self.cache_file_path)
                     except Exception: 
                         pass
                os.rename(temp_path, self.cache_file_path)
                
                self.last_update = current_time
                self.logger.debug("Updated DefiLlama cache")
            except Exception as e:
                self.logger.warning(f"Failed to save DefiLlama cache: {e}")
                
            return fundamentals
            
        except Exception as e:
            self.logger.error(f"Error building DeFi fundamentals: {e}")
            return None
