import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dashboard.server import DashboardServer
from httpx import ASGITransport, AsyncClient

async def verify_headers():
    # Setup mock config
    mock_config = MagicMock()
    mock_config.DASHBOARD_ENABLE_CORS = False
    
    server = DashboardServer(
        brain_service=MagicMock(),
        vector_memory=MagicMock(),
        analysis_engine=MagicMock(),
        config=mock_config,
        logger=MagicMock()
    )
    
    app = server.app
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Test API endpoint
        print("Testing /api/performance/history...")
        res = await client.get("/api/performance/history")
        cc = res.headers.get("Cache-Control")
        print(f"  Cache-Control: {cc}")
        assert cc == "public, max-age=30", f"Expected 30s for API, got {cc}"
        
        # Test static .js file
        print("Testing /style.css...")
        res = await client.get("/style.css")
        cc = res.headers.get("Cache-Control")
        print(f"  Cache-Control: {cc}")
        assert cc == "public, max-age=86400", f"Expected 86400s for static, got {cc}"
        
        # Test image file
        print("Testing /logo.png...")
        res = await client.get("/logo.png")
        cc = res.headers.get("Cache-Control")
        print(f"  Cache-Control: {cc}")
        assert cc == "public, max-age=86400", f"Expected 86400s for image, got {cc}"
        
        # Test HTML / Root
        print("Testing /...")
        res = await client.get("/")
        cc = res.headers.get("Cache-Control")
        print(f"  Cache-Control: {cc}")
        assert cc == "public, max-age=300", f"Expected 300s for HTML, got {cc}"

    print("\nALL VERIFICATIONS PASSED!")

if __name__ == "__main__":
    asyncio.run(verify_headers())
