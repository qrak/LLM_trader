"""FastAPI server for the Trading Dashboard."""
import asyncio
import hashlib
import os
import time as time_module
from contextlib import asynccontextmanager
from collections import defaultdict

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.gzip import GZipMiddleware

from .routers import brain, monitor, visuals, performance, ws_router
from .dashboard_state import dashboard_state

class DashboardServer:
    """Main application server combining all API routers."""
    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
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
        # pylint: disable=too-many-statements

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            # Startup logic
            print(f"DTO: Dashboard live at http://localhost:{self.port}")
            yield
            # Shutdown logic
            print("DTO: Dashboard shutting down...")

        app = FastAPI(title="LLM Trader Brain", lifespan=lifespan)

        app.add_middleware(GZipMiddleware, minimum_size=500)

        def _set_cache_headers(response, browser_policy, edge_policy):
            response.headers["Cache-Control"] = browser_policy
            response.headers["CDN-Cache-Control"] = edge_policy
            response.headers["Cloudflare-CDN-Cache-Control"] = edge_policy

        def _is_no_store_response(response):
            cache_control = response.headers.get("Cache-Control", "")
            edge_control = response.headers.get("Cloudflare-CDN-Cache-Control", "")
            return "no-store" in cache_control or "no-store" in edge_control

        def _build_etag(request, response, path):
            body = getattr(response, "body", b"")
            if body:
                digest = hashlib.sha256(body).hexdigest()
                return f'W/"{digest}"'

            # GZip/streaming responses may not expose `body` at middleware stage.
            # Use short time-bucketed weak ETags aligned to cache windows.
            if path.startswith('/api/'):
                bucket_seconds = 15
            elif path.endswith('.html') or path == '/':
                bucket_seconds = 30
            else:
                return None

            bucket = int(time_module.time() // bucket_seconds)
            seed = f"{path}?{request.url.query}|{bucket}|{response.headers.get('Content-Type', '')}"
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            return f'W/"{digest}"'

        def _is_not_modified(if_none_match_header, etag):
            if not if_none_match_header or not etag:
                return False
            if if_none_match_header.strip() == "*":
                return True
            candidates = [part.strip() for part in if_none_match_header.split(",")]
            return etag in candidates

        def _is_static_asset(path):
            return path.endswith((
                ".css", ".js", ".mjs", ".png", ".jpg", ".jpeg", ".svg",
                ".webp", ".ico", ".gif", ".woff", ".woff2", ".ttf", ".eot",
            ))

        def _api_cache_policies(path, query_params):
            """Return browser/edge cache policy pair for API routes."""
            # Always bypass CDN cache for highly volatile or user-driven high-cardinality APIs.
            if path.endswith("/refresh-price"):
                return (
                    "no-store, no-cache, must-revalidate, proxy-revalidate",
                    "no-store",
                )

            if path.endswith("/vectors") and query_params.get("query"):
                return (
                    "no-store, no-cache, must-revalidate, proxy-revalidate",
                    "no-store",
                )

            # Keep realtime countdown fresher than the rest of the API surface.
            if path.endswith("/status/countdown"):
                return (
                    "public, max-age=5",
                    "public, max-age=15, stale-while-revalidate=10, stale-if-error=60",
                )

            # Default policy for cache-safe GET APIs (<= 60 seconds staleness budget).
            return (
                "public, max-age=15",
                "public, max-age=60, stale-while-revalidate=30, stale-if-error=300",
            )

        # Security Headers Middleware
        @app.middleware("http")
        async def add_security_headers(request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # Conditionally add HSTS if the request originated over HTTPS.
            # Uvicorn's ProxyHeadersMiddleware processes X-Forwarded-Proto,
            # so request.url.scheme will correctly reflect 'https' if Cloudflare sent it.
            if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

            # Content Security Policy (CSP)
            # - script-src: 'self' (dashboard logic), CDNs
            # - style-src: 'self' 'unsafe-inline' (for dashboard styles), CDNs
            # - connect-src: 'self' (for internal API), CDNs if needed
            # - img-src: 'self' data: https: (for content/news images)
            # Cloudflare support: *.cloudflare.com added
            csp = (
                "default-src 'self'; "
                "frame-ancestors 'none'; "
                "script-src 'self' https://cdn.jsdelivr.net https://unpkg.com "
                "https://*.cloudflare.com https://ajax.cloudflare.com "
                "https://static.cloudflareinsights.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https://*.cloudflare.com https://unpkg.com "
                "https://cdn.jsdelivr.net;"
            )
            response.headers["Content-Security-Policy"] = csp
            path = request.url.path

            # Restrict caching entirely for non-GET/HEAD methods (e.g. POST, PUT, DELETE)
            if request.method not in ("GET", "HEAD"):
                _set_cache_headers(
                    response,
                    "no-store, no-cache, must-revalidate, proxy-revalidate",
                    "no-store",
                )
                return response

            if _is_static_asset(path):
                if request.query_params.get("v"):
                    _set_cache_headers(
                        response,
                        "public, max-age=31536000, immutable",
                        "public, max-age=31536000, stale-while-revalidate=86400, stale-if-error=604800",
                    )
                else:
                    _set_cache_headers(
                        response,
                        "public, max-age=3600",
                        "public, max-age=86400, stale-while-revalidate=3600, stale-if-error=86400",
                    )
            elif path.startswith('/api/'):
                browser_policy, edge_policy = _api_cache_policies(path, request.query_params)
                _set_cache_headers(
                    response,
                    browser_policy,
                    edge_policy,
                )
            elif path.endswith('.html') or path == '/':
                _set_cache_headers(
                    response,
                    "public, max-age=30, must-revalidate",
                    "public, max-age=300, stale-while-revalidate=60, stale-if-error=600",
                )

            # Add conditional ETag handling for cacheable API/HTML responses.
            # Skip static assets because FileResponse already manages validators.
            if (
                request.method in ("GET", "HEAD")
                and response.status_code == 200
                and not _is_static_asset(path)
                and not _is_no_store_response(response)
                and (path.startswith('/api/') or path.endswith('.html') or path == '/')
            ):
                etag = response.headers.get("ETag") or _build_etag(request, response, path)
                if etag:
                    response.headers["ETag"] = etag
                    if _is_not_modified(request.headers.get("if-none-match"), etag):
                        headers = dict(response.headers)
                        headers.pop("content-length", None)
                        return Response(status_code=304, headers=headers)
            return response

        # Simple Rate Limiting (in-memory, per-IP)
        request_counts = defaultdict(list)
        rate_limit = 300  # requests per minute
        rate_window = 60  # seconds
        max_unique_ips = 10000  # Prevent memory exhaustion (Defense in Depth behind Cloudflare)

        # Security: State for rate limit cleanup
        state = {"last_cleanup_time": 0.0}
        cleanup_interval = 10.0  # Seconds between full scans

        @app.middleware("http")
        async def rate_limit_middleware(request, call_next):
            # Skip rate limiting for static files
            if request.url.path.startswith("/static") or not request.url.path.startswith("/api"):
                return await call_next(request)

            current_time = time_module.monotonic()

            # Security: Prevent memory exhaustion from too many IPs
            if len(request_counts) > max_unique_ips:
                # Optimized cleanup: Only scan at most once every CLEANUP_INTERVAL
                if current_time - state["last_cleanup_time"] > cleanup_interval:
                    # Remove inactive IPs
                    keys_to_remove = [
                        ip for ip, timestamps in request_counts.items()
                        if not timestamps or current_time - timestamps[-1] > rate_window
                    ]
                    for key in keys_to_remove:
                        del request_counts[key]
                    state["last_cleanup_time"] = current_time

                # If still too large (active attack), drop the oldest entry (FIFO)
                # This degrades gracefully rather than clearing everything (DoS risk)
                while len(request_counts) > max_unique_ips:
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
                    if current_time - t < rate_window
                ]

            if len(request_counts[client_ip]) >= rate_limit:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded. Try again later."}
                )
            request_counts[client_ip].append(current_time)
            return await call_next(request)

        # CORS Configuration
        # Defaults to False for security. Can be enabled in config.ini.
        enable_cors = self.config.DASHBOARD_ENABLE_CORS

        if enable_cors:
            allowed_origins = self.config.DASHBOARD_CORS_ORIGINS

            # If enabled but empty, log a warning and default to strict (empty list)
            if not allowed_origins:
                print("WARNING: CORS enabled but no origins specified. CORS will effectively be disabled.")

            app.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_credentials=False,  # Wildcard origins are incompatible with credentials
                allow_methods=["GET"],
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

        brain_router = brain.BrainRouter(
            config=self.config,
            logger=self.logger,
            dashboard_state=self.dashboard_state,
            vector_memory=self.vector_memory,
            unified_parser=self.unified_parser,
            persistence=self.persistence,
            exchange_manager=self.exchange_manager
        )
        try:
            rag_engine = self.brain_service.rag_engine if self.brain_service else None
        except AttributeError:
            rag_engine = None
            
        monitor_router = monitor.MonitorRouter(
            config=self.config,
            logger=self.logger,
            dashboard_state=self.dashboard_state,
            analysis_engine=self.analysis_engine,
            rag_engine=rag_engine
        )
        performance_router = performance.PerformanceRouter(
            config=self.config,
            logger=self.logger,
            dashboard_state=self.dashboard_state
        )
        visuals_router = visuals.VisualsRouter(
            analysis_engine=self.analysis_engine
        )
        websocket_router = ws_router.WebSocketRouter(
            manager_instance=ws_router.manager,
            config=self.config,
            dashboard_state=self.dashboard_state
        )

        app.include_router(brain_router.router)
        app.include_router(monitor_router.router)
        app.include_router(visuals_router.router)
        app.include_router(performance_router.router)
        app.include_router(websocket_router.router)

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
        # Guard against double-start: if already running, do nothing
        if self.server_task and not self.server_task.done():
            return self.server_task

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
            ws="wsproto",
            proxy_headers=True,
            # Cloudflare IPv4 & IPv6 ranges — verified 2026-03-02
            # Source: https://www.cloudflare.com/ips-v4/ and /ips-v6/
            # Update periodically: Cloudflare rarely changes these but does occasionally add ranges.
            forwarded_allow_ips=(
                "173.245.48.0/20,103.21.244.0/22,103.22.200.0/22,103.31.4.0/22,"
                "141.101.64.0/18,108.162.192.0/18,190.93.240.0/20,188.114.96.0/20,"
                "197.234.240.0/22,198.41.128.0/17,162.158.0.0/15,104.16.0.0/13,"
                "104.24.0.0/14,172.64.0.0/13,131.0.72.0/22,2400:cb00::/32,"
                "2606:4700::/32,2803:f800::/32,2405:b500::/32,2405:8100::/32,"
                "2a06:98c0::/29,2c0f:f248::/32"
            ),
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
        """Stop the dashboard server gracefully.

        Signals uvicorn to exit, then waits for the server task to finish
        so the socket is fully released before returning.
        """
        if not self._server and not self.server_task:
            return

        # Signal uvicorn's serve() loop to exit cleanly
        if self._server:
            self._server.should_exit = True

        # Wait for the server task to finish naturally (socket release)
        if self.server_task and not self.server_task.done():
            try:
                await asyncio.wait_for(self.server_task, timeout=5.0)
            except asyncio.TimeoutError:
                # Force-cancel if it doesn't stop within 5 seconds
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass

        # Clear references so start() can create fresh instances
        self._server = None
        self.server_task = None
