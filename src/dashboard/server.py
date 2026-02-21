import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from collections import defaultdict
import time as time_module
import os
import uvicorn

from .routers import brain, monitor, visuals, performance, ws_router
from .dashboard_state import dashboard_state

class DashboardServer:
    def __init__(self,
                 brain_service,
                 vector_memory,
                 analysis_engine,
                 config,
                 logger,
                 unified_parser=None,
                 persistence=None,
                 exchange_manager=None,
                 host="0.0.0.0",
                 port=8000):
        self.brain_service = brain_service
        self.vector_memory = vector_memory
        self.analysis_engine = analysis_engine
        self.config = config
        self.logger = logger
        self.unified_parser = unified_parser
        self.persistence = persistence
        self.exchange_manager = exchange_manager
        self.host = host
        self.port = port
        self.server_task = None
        self._server = None
        self.dashboard_state = dashboard_state
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            # Startup logic
            print(f"DTO: Dashboard live at http://localhost:{self.port}")
            yield
            # Shutdown logic
            print("DTO: Dashboard shutting down...")

        app = FastAPI(title="LLM Trader Brain", lifespan=lifespan)

        app.add_middleware(GZipMiddleware, minimum_size=500)

        # Security Headers Middleware
        @app.middleware("http")
        async def add_security_headers(request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

            # Content Security Policy (CSP)
            # - script-src: 'self' (dashboard logic), CDNs
            # - style-src: 'self' 'unsafe-inline' (for dashboard styles), CDNs
            # - connect-src: 'self' (for internal API), CDNs if needed
            # - img-src: 'self' data: https: (for content/news images)
            # Cloudflare support: *.cloudflare.com added
            csp = (
                "default-src 'self'; "
                "script-src 'self' https://cdn.jsdelivr.net https://unpkg.com https://*.cloudflare.com https://ajax.cloudflare.com https://static.cloudflareinsights.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https://*.cloudflare.com https://unpkg.com https://cdn.jsdelivr.net;"
            )
            response.headers["Content-Security-Policy"] = csp
            path = request.url.path
            
            # Restrict caching entirely for non-GET/HEAD methods (e.g. POST, PUT, DELETE)
            if request.method not in ("GET", "HEAD"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
                return response

            if path.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.svg', '.webp', '.ico')):
                # Browser: 1 hour | Edge: 1 day
                response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=86400"
            elif path.startswith('/api/'):
                # Most API data changes every ~60s due to internal TTL.
                # Browser: 0 (always ask) | Edge: 30s cache
                response.headers["Cache-Control"] = "public, max-age=0, s-maxage=30"
            elif path.endswith('.html') or path == '/':
                # Browser: 0 | Edge: 60s
                response.headers["Cache-Control"] = "public, max-age=0, s-maxage=60"
            return response

        # Simple Rate Limiting (in-memory, per-IP)
        request_counts = defaultdict(list)
        RATE_LIMIT = 300  # requests per minute
        RATE_WINDOW = 60  # seconds
        MAX_UNIQUE_IPS = 10000  # Prevent memory exhaustion (Defense in Depth behind Cloudflare)

        # Security: State for rate limit cleanup
        state = {"last_cleanup_time": 0.0}
        CLEANUP_INTERVAL = 10.0  # Seconds between full scans

        @app.middleware("http")
        async def rate_limit_middleware(request, call_next):
            # Skip rate limiting for static files
            if request.url.path.startswith("/static") or not request.url.path.startswith("/api"):
                return await call_next(request)

            current_time = time_module.monotonic()

            # Security: Prevent memory exhaustion from too many IPs
            if len(request_counts) > MAX_UNIQUE_IPS:
                # Optimized cleanup: Only scan at most once every CLEANUP_INTERVAL
                if current_time - state["last_cleanup_time"] > CLEANUP_INTERVAL:
                    # Remove inactive IPs
                    keys_to_remove = [
                        ip for ip, timestamps in request_counts.items()
                        if not timestamps or current_time - timestamps[-1] > RATE_WINDOW
                    ]
                    for key in keys_to_remove:
                        del request_counts[key]
                    state["last_cleanup_time"] = current_time

                # If still too large (active attack), drop the oldest entry (FIFO)
                # This degrades gracefully rather than clearing everything (DoS risk)
                while len(request_counts) > MAX_UNIQUE_IPS:
                    try:
                        # defaultdict preserves insertion order in Python 3.7+
                        oldest_ip = next(iter(request_counts))
                        del request_counts[oldest_ip]
                    except StopIteration:
                        break

            client_ip = request.client.host if request.client else "unknown"

            # Clean old requests for current IP
            if client_ip in request_counts:
                request_counts[client_ip] = [
                    t for t in request_counts[client_ip]
                    if current_time - t < RATE_WINDOW
                ]

            if len(request_counts[client_ip]) >= RATE_LIMIT:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded. Try again later."}
                )
            request_counts[client_ip].append(current_time)
            return await call_next(request)

        # CORS Configuration
        # Defaults to False for security. Can be enabled in config.ini.
        enable_cors = getattr(self.config, 'DASHBOARD_ENABLE_CORS', False)

        if enable_cors:
            allowed_origins = getattr(self.config, 'DASHBOARD_CORS_ORIGINS', [])

            # If enabled but empty, log a warning and default to strict (empty list)
            if not allowed_origins:
                print("WARNING: CORS enabled but no origins specified. CORS will effectively be disabled.")

            app.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        app.state.brain_service = self.brain_service
        app.state.vector_memory = self.vector_memory
        app.state.analysis_engine = self.analysis_engine
        app.state.config = self.config
        app.state.logger = self.logger
        app.state.unified_parser = self.unified_parser
        app.state.persistence = self.persistence
        app.state.exchange_manager = self.exchange_manager
        app.state.dashboard_state = self.dashboard_state
        # Expose for testing/monitoring
        app.state.request_counts = request_counts

        app.include_router(brain.router)
        app.include_router(monitor.router)
        app.include_router(visuals.router)
        app.include_router(performance.router)
        app.include_router(ws_router.router)

        # Mount Static Files (Frontend)
        # We assume the static folder is in the same directory as this file
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        if os.path.exists(static_dir):
            app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
        else:
            print(f"WARNING: Static directory not found at {static_dir}")

        return app

    async def start(self):
        """Start the uvicorn server in an asyncio loop."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
            ws="wsproto",
            proxy_headers=True,
            # Cloudflare IPv4 & IPv6 ranges (https://www.cloudflare.com/ips/)
            forwarded_allow_ips="173.245.48.0/20,103.21.244.0/22,103.22.200.0/22,103.31.4.0/22,141.101.64.0/18,108.162.192.0/18,190.93.240.0/20,188.114.96.0/20,197.234.240.0/22,198.41.128.0/17,162.158.0.0/15,104.16.0.0/13,104.24.0.0/14,172.64.0.0/13,131.0.72.0/22,2400:cb00::/32,2606:4700::/32,2803:f800::/32,2405:b500::/32,2405:8100::/32,2a06:98c0::/29,2c0f:f248::/32",
        )
        self._server = uvicorn.Server(config)

        # Disable uvicorn's own signal handling since we handle it ourselves
        self._server.install_signal_handlers = lambda: None

        # Store reference to task
        self.server_task = asyncio.create_task(self._run_server())
        return self.server_task

    async def _run_server(self):
        """Run uvicorn server with exception handling."""
        try:
            await self._server.serve()
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Stop the dashboard server gracefully."""
        if hasattr(self, '_server') and self._server:
            self._server.should_exit = True
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
