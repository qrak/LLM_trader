import aiohttp
import asyncio
import json

async def verify_defillama():
    print("\n--- Verifying DefiLlama ---")
    async with aiohttp.ClientSession() as session:
        # 1. Stablecoins
        try:
            async with session.get("https://stablecoins.llama.fi/stablecoins") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pegged_assets = data.get("peggedAssets", [])
                    print(f"[SUCCESS] Fetched {len(pegged_assets)} stablecoins.")
                    if pegged_assets:
                        print(f"Sample: {pegged_assets[0].get('name')} - Circulating: {pegged_assets[0].get('circulating', {}).get('peggedUSD', 'N/A')}")
                else:
                    print(f"[ERROR] Stablecoins endpoint failed: {resp.status}")
        except Exception as e:
            print(f"[ERROR] DefiLlama Stablecoins: {e}")

        # 2. Chains TVL
        try:
            async with session.get("https://api.llama.fi/v2/chains") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"[SUCCESS] Fetched {len(data)} chains TVL.")
                    if data:
                        print(f"Sample: {data[0].get('name')} - TVL: {data[0].get('tvl')}")
                else:
                    print(f"[ERROR] Chains endpoint failed: {resp.status}")
        except Exception as e:
            print(f"[ERROR] DefiLlama Chains: {e}")

async def verify_coingecko():
    print("\n--- Verifying CoinGecko Derivatives ---")
    async with aiohttp.ClientSession() as session:
        # Derivatives / Exchanges
        url = "https://api.coingecko.com/api/v3/derivatives"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"[SUCCESS] Fetched {len(data)} derivative tickers.")
                    if data:
                        print(f"Sample: {data[0].get('market')} - Open Interest: {data[0].get('open_interest_usd')}")
                else:
                    print(f"[ERROR] Derivatives endpoint failed: {resp.status}")
        except Exception as e:
            print(f"[ERROR] CoinGecko Derivatives: {e}")
            
        # DEX Categories (Checking if we can filter pools)
        # Note: CoinGecko doesn't have a direct "DEX Volume" endpoint in free tier easily, usually "exchanges" with types.
        print("\n--- Verifying CoinGecko Exchanges (DEX) ---")
        try:
            async with session.get("https://api.coingecko.com/api/v3/exchanges?per_page=5") as resp:
                 if resp.status == 200:
                    data = await resp.json()
                    print(f"[SUCCESS] Fetched {len(data)} exchanges.")
                    # Check if we can distinguish DEX? usually not explicitly in this list without ID checking
                    for ex in data:
                        print(f" - {ex.get('name')} (Trust: {ex.get('trust_score')})")
                 else:
                    print(f"[ERROR] Exchanges endpoint failed: {resp.status}")
        except Exception as e:
             print(f"[ERROR] CoinGecko Exchanges: {e}")

async def main():
    await verify_defillama()
    await verify_coingecko()

if __name__ == "__main__":
    asyncio.run(main())
