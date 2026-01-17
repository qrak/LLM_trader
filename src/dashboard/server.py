import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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
        self.dashboard_state = dashboard_state
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup logic
            print(f"DTO: Dashboard live at http://localhost:{self.port}")
            yield
            # Shutdown logic
            print("DTO: Dashboard shutting down...")

        app = FastAPI(title="LLM Trader Brain", lifespan=lifespan)

        # Security Headers Middleware
        @app.middleware("http")
        async def add_security_headers(request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            
            # Content Security Policy (CSP)
            # - script-src: 'self' 'unsafe-inline' (for dashboard logic), CDNs
            # - style-src: 'self' 'unsafe-inline' (for dashboard styles), CDNs
            # - connect-src: 'self' (for internal API), CDNs if needed
            # - img-src: 'self' data: https: (for content/news images)
            # Cloudflare support: *.cloudflare.com added
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com https://*.cloudflare.com https://ajax.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https://*.cloudflare.com;"
            )
            response.headers["Content-Security-Policy"] = csp
            return response

        # Simple Rate Limiting (in-memory, per-IP)
        from collections import defaultdict
        import time as time_module
        request_counts = defaultdict(list)
        RATE_LIMIT = 300  # requests per minute
        RATE_WINDOW = 60  # seconds

        @app.middleware("http")
        async def rate_limit_middleware(request, call_next):
            # Skip rate limiting for static files
            if request.url.path.startswith("/static") or not request.url.path.startswith("/api"):
                return await call_next(request)
            client_ip = request.client.host if request.client else "unknown"
            current_time = time_module.time()
            # Clean old requests
            request_counts[client_ip] = [
                t for t in request_counts[client_ip] 
                if current_time - t < RATE_WINDOW
            ]
            if len(request_counts[client_ip]) >= RATE_LIMIT:
                from fastapi.responses import JSONResponse
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

        app.include_router(brain.router)
        app.include_router(monitor.router)
        app.include_router(visuals.router)
        app.include_router(performance.router)
        app.include_router(ws_router.router)

        # Mount Static Files (Frontend)
        # We assume the static folder is in the same directory as this file
        import os
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
            forwarded_allow_ips="*",
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
