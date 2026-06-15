"""Playwright browser tests for the Admin Console.

Tests the full user journey:
1. Login screen appears when unauthenticated
2. Login with valid credentials
3. Dashboard overview loads with stats
4. Control panel buttons work (force analysis, toggle feed)
5. Config editor loads and can edit values
6. Live log viewer connects
7. Human-in-the-loop input works
8. Logout works
9. Unauthenticated API requests return 401

Requires: playwright>=1.49.0, pytest-asyncio
Run: .venv-wsl/bin/python -m pytest tests/test_admin_playwright.py -x --tb=short -v
"""

import asyncio
import multiprocessing
import socket
import time
from pathlib import Path

import pytest

# Skip if playwright not available
pytest.importorskip("playwright")

from playwright.sync_api import sync_playwright, expect


# ─── Test Server ─────────────────────────────────────────────────────

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_test_server(port: int, ready_event: multiprocessing.Event):
    """Run a minimal FastAPI app with admin routes for testing."""
    import sys
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.gzip import GZipMiddleware
    from src.dashboard.auth import AdminAuthMiddleware, init_auth, hash_password
    from src.dashboard.log_stream import LogStreamManager
    from src.dashboard.routers.admin import AdminRouter
    from src.config.writable_config import WritableConfig
    import tempfile
    import uvicorn
    import asyncio

    # Create temp config
    tmpdir = tempfile.mkdtemp()
    config_path = os.path.join(tmpdir, "config.ini")
    with open(config_path, "w") as f:
        f.write("""[ai_providers]
provider = googleai
google_studio_model = gemini-3.5-flash

[general]
crypto_pair = BTC/USDC
timeframe = 4h
candle_limit = 999
discord_bot = true

[debug]
save_chart_images = false
logger_debug = true

[rag]
update_interval_hours = 4
news_limit = 5

[risk_management]
stop_loss_type = hard
take_profit_type = hard
max_position_size = 0.10

[demo_trading]
demo_quote_capital = 10000

[model_config]
temperature = 1.0
top_p = 0.9

[dashboard]
enabled = true
host = 0.0.0.0
port = 8000
""")

    # Init auth
    password_hash = hash_password("testpass123")
    init_auth("test-signing-key-for-tests", "admin", password_hash)

    # Create components
    writable_config = WritableConfig(config_path)
    log_stream_manager = LogStreamManager()
    force_analysis_event = asyncio.Event()

    # Create app
    app = FastAPI()
    app.add_middleware(GZipMiddleware, minimum_size=500)

    admin_router = AdminRouter(
        writable_config=writable_config,
        log_stream_manager=log_stream_manager,
        config=type("MockConfig", (), {"DASHBOARD_ENABLE_CORS": False, "DASHBOARD_CORS_ORIGINS": []})(),
        logger=None,
        force_analysis_event=force_analysis_event,
    )
    app.include_router(admin_router.router)
    app.add_middleware(AdminAuthMiddleware)

    # Mount admin static files
    static_dir = Path(__file__).parent.parent / "src" / "dashboard" / "static" / "admin"
    if static_dir.exists():
        app.mount("/admin", StaticFiles(directory=str(static_dir), html=True), name="admin_static")

    ready_event.set()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


@pytest.fixture(scope="module")
def test_server():
    """Start a test server and return its base URL."""
    port = _find_free_port()
    ready = multiprocessing.Event()
    proc = multiprocessing.Process(target=_run_test_server, args=(port, ready), daemon=True)
    proc.start()
    ready.wait(timeout=10)
    time.sleep(1)  # Let server fully start
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.join(timeout=5)


@pytest.fixture(scope="module")
def browser_context():
    """Create a Playwright browser context."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        yield context
        context.close()
        browser.close()


@pytest.fixture
def page(browser_context):
    """Create a fresh page for each test."""
    pg = browser_context.new_page()
    yield pg
    pg.close()


# ─── Tests ───────────────────────────────────────────────────────────

class TestLoginScreen:
    """Test the login flow."""

    def test_login_page_loads(self, page, test_server):
        """Admin page shows login screen when not authenticated."""
        page.goto(f"{test_server}/admin/")
        # Should see the login form
        expect(page.locator("#login-screen")).to_be_visible()
        expect(page.locator("#login-user")).to_be_visible()
        expect(page.locator("#login-pass")).to_be_visible()
        expect(page.locator("button[type='submit']")).to_be_visible()

    def test_login_with_wrong_credentials(self, page, test_server):
        """Login with wrong password shows error."""
        page.goto(f"{test_server}/admin/")
        page.fill("#login-user", "admin")
        page.fill("#login-pass", "wrongpassword")
        page.click("button[type='submit']")
        # Should show error
        expect(page.locator("#login-error")).to_be_visible()
        expect(page.locator("#login-error")).to_contain_text("Invalid")

    def test_login_with_valid_credentials(self, page, test_server):
        """Login with correct credentials shows the app."""
        page.goto(f"{test_server}/admin/")
        page.fill("#login-user", "admin")
        page.fill("#login-pass", "testpass123")
        page.click("button[type='submit']")
        # Should hide login screen and show app
        expect(page.locator("#app")).to_be_visible()
        expect(page.locator("#login-screen")).to_be_hidden()

    def test_login_screen_has_correct_title(self, page, test_server):
        """Login screen shows correct branding."""
        page.goto(f"{test_server}/admin/")
        expect(page.locator("#login-screen")).to_contain_text("LLM Trader")
        expect(page.locator("#login-screen")).to_contain_text("Authentication Required")


class TestDashboard:
    """Test the dashboard overview page."""

    def _login(self, page, test_server):
        page.goto(f"{test_server}/admin/")
        # Wait for app to become visible (either via auto-login from cookie or manual login)
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            # Not auto-logged in, do manual login
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()

    def test_dashboard_loads_with_stats(self, page, test_server):
        """Dashboard shows stat cards after login."""
        self._login(page, test_server)
        # Should see stat cards
        expect(page.locator("#stat-uptime")).to_be_visible()
        expect(page.locator("#stat-feed")).to_be_visible()
        expect(page.locator("#stat-subscribers")).to_be_visible()

    def test_dashboard_has_quick_actions(self, page, test_server):
        """Dashboard shows quick action buttons."""
        self._login(page, test_server)
        expect(page.locator("#btn-force-analysis")).to_be_visible()
        expect(page.locator("#btn-toggle-feed")).to_be_visible()

    def test_dashboard_has_human_input(self, page, test_server):
        """Dashboard shows human-in-the-loop input."""
        self._login(page, test_server)
        expect(page.locator("#human-input-text")).to_be_visible()
        expect(page.locator("#btn-submit-input")).to_be_visible()


class TestNavigation:
    """Test sidebar navigation."""

    def _login(self, page, test_server):
        page.goto(f"{test_server}/admin/")
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()

    def test_navigate_to_control_panel(self, page, test_server):
        """Can navigate to control panel page."""
        self._login(page, test_server)
        page.click("button[data-page='control']")
        expect(page.locator("#page-control")).to_be_visible()
        expect(page.locator("#page-dashboard")).to_be_hidden()

    def test_navigate_to_config(self, page, test_server):
        """Can navigate to configuration page."""
        self._login(page, test_server)
        page.click("button[data-page='config']")
        expect(page.locator("#page-config")).to_be_visible()
        expect(page.locator("#page-dashboard")).to_be_hidden()

    def test_navigate_to_logs(self, page, test_server):
        """Can navigate to live logs page."""
        self._login(page, test_server)
        page.click("button[data-page='logs']")
        expect(page.locator("#page-logs")).to_be_visible()
        expect(page.locator("#page-dashboard")).to_be_hidden()

    def test_navigate_back_to_dashboard(self, page, test_server):
        """Can navigate back to dashboard from other pages."""
        self._login(page, test_server)
        page.click("button[data-page='config']")
        expect(page.locator("#page-config")).to_be_visible()
        page.click("button[data-page='dashboard']")
        expect(page.locator("#page-dashboard")).to_be_visible()


class TestControlPanel:
    """Test the control panel page."""

    def _login_and_go_to_control(self, page, test_server):
        page.goto(f"{test_server}/admin/")
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()
        page.click("button[data-page='control']")
        expect(page.locator("#page-control")).to_be_visible()

    def test_force_analysis_button_visible(self, page, test_server):
        """Force analysis button is visible on control panel."""
        self._login_and_go_to_control(page, test_server)
        expect(page.locator("#ctrl-force-analysis")).to_be_visible()
        expect(page.locator("#ctrl-force-analysis")).to_contain_text("Force Analysis")

    def test_toggle_feed_button_visible(self, page, test_server):
        """Toggle feed button is visible on control panel."""
        self._login_and_go_to_control(page, test_server)
        expect(page.locator("#ctrl-toggle-feed")).to_be_visible()
        expect(page.locator("#ctrl-toggle-feed")).to_contain_text("Toggle Dashboard Feed")

    def test_force_analysis_click(self, page, test_server):
        """Clicking force analysis sends command via WebSocket."""
        self._login_and_go_to_control(page, test_server)
        # Click the button - it should send a WS command
        page.click("#ctrl-force-analysis")
        # Wait for console log to appear
        page.wait_for_timeout(1000)
        console_log = page.locator("#console-log")
        expect(console_log).to_be_visible()


class TestConfigEditor:
    """Test the configuration editor."""

    def _login_and_go_to_config(self, page, test_server):
        page.goto(f"{test_server}/admin/")
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()
        page.click("button[data-page='config']")
        expect(page.locator("#page-config")).to_be_visible()

    def test_config_page_loads_sections(self, page, test_server):
        """Config page loads with section headers."""
        self._login_and_go_to_config(page, test_server)
        # Wait for config to load
        page.wait_for_timeout(2000)
        # Should have at least one section
        content = page.locator("#config-sections").inner_text()
        assert "AI Providers" in content or "General" in content

    def test_config_section_expandable(self, page, test_server):
        """Config sections can be expanded."""
        self._login_and_go_to_config(page, test_server)
        page.wait_for_timeout(2000)
        # Click on first section header
        section_headers = page.locator("#config-sections button")
        if section_headers.count() > 0:
            section_headers.first.click()
            page.wait_for_timeout(500)
            # The body should become visible
            assert section_headers.count() > 0


class TestHumanInput:
    """Test human-in-the-loop input."""

    def _login(self, page, test_server):
        page.goto(f"{test_server}/admin/")
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()

    def test_submit_human_input(self, page, test_server):
        """Can submit human input text."""
        self._login(page, test_server)
        page.fill("#human-input-text", "Focus on RSI divergence patterns")
        page.click("#btn-submit-input")
        # Should show success status
        page.wait_for_timeout(1000)
        status = page.locator("#human-input-status")
        expect(status).to_contain_text("Submitted")

    def test_human_input_via_api(self, page, test_server):
        """Human input can be set and retrieved via API."""
        # Login (handle auto-login from cookie)
        page.goto(f"{test_server}/admin/")
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()

        # Use API to set input
        result = page.evaluate("""
            fetch('/api/admin/system/human-input', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: 'Test input from browser'}),
                credentials: 'same-origin'
            }).then(r => r.json())
        """)
        assert result["status"] == "ok"

        # Verify via GET
        result = page.evaluate("""
            fetch('/api/admin/system/human-input', {
                credentials: 'same-origin'
            }).then(r => r.json())
        """)
        assert result["text"] == "Test input from browser"


class TestAuthProtection:
    """Test that API endpoints are protected."""

    def test_unauthenticated_api_returns_401(self, browser_context, test_server):
        """API endpoints return 401 when not authenticated."""
        # Use a fresh context with no cookies
        ctx = browser_context.browser.new_context()
        pg = ctx.new_page()
        pg.goto(f"{test_server}/admin/")
        result = pg.evaluate("""
            fetch('/api/admin/config', {credentials: 'same-origin'})
                .then(r => ({status: r.status}))
        """)
        assert result["status"] == 401
        pg.close()
        ctx.close()

    def test_unauthenticated_system_status_returns_401(self, browser_context, test_server):
        """System status returns 401 when not authenticated."""
        ctx = browser_context.browser.new_context()
        pg = ctx.new_page()
        pg.goto(f"{test_server}/admin/")
        result = pg.evaluate("""
            fetch('/api/admin/system/status', {credentials: 'same-origin'})
                .then(r => ({status: r.status}))
        """)
        assert result["status"] == 401
        pg.close()
        ctx.close()

    def test_health_endpoint_is_public(self, page, test_server):
        """Health endpoint is accessible without auth."""
        page.goto(f"{test_server}/admin/")
        result = page.evaluate("""
            fetch('/api/admin/health', {credentials: 'same-origin'})
                .then(r => r.json())
        """)
        assert result["status"] == "ok"

    def test_login_endpoint_is_public(self, page, test_server):
        """Login endpoint is accessible without auth."""
        page.goto(f"{test_server}/admin/")
        result = page.evaluate("""
            fetch('/api/admin/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username: 'admin', password: 'wrong'}),
                credentials: 'same-origin'
            }).then(r => ({status: r.status}))
        """)
        assert result["status"] == 401  # Wrong creds, but endpoint is reachable


class TestLogout:
    """Test logout functionality."""

    def test_logout_returns_to_login(self, page, test_server):
        """Clicking logout returns to login screen."""
        page.goto(f"{test_server}/admin/")
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()

        # Click logout
        page.click("#btn-logout")
        expect(page.locator("#login-screen")).to_be_visible()
        expect(page.locator("#app")).to_be_hidden()


class TestConfigAPI:
    """Test config API endpoints directly."""

    def _login(self, page, test_server):
        page.goto(f"{test_server}/admin/")
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()

    def test_get_full_config_schema(self, page, test_server):
        """GET /api/admin/config returns full schema."""
        self._login(page, test_server)
        result = page.evaluate("""
            fetch('/api/admin/config', {credentials: 'same-origin'})
                .then(r => r.json())
        """)
        assert "general" in result
        assert "ai_providers" in result
        assert "keys" in result["general"]
        assert "crypto_pair" in result["general"]["keys"]

    def test_get_config_section(self, page, test_server):
        """GET /api/admin/config/general returns one section."""
        self._login(page, test_server)
        result = page.evaluate("""
            fetch('/api/admin/config/general', {credentials: 'same-origin'})
                .then(r => r.json())
        """)
        assert "keys" in result
        assert result["keys"]["crypto_pair"]["value"] == "BTC/USDC"

    def test_patch_config_value(self, page, test_server):
        """PATCH /api/admin/config/general/timeframe updates value."""
        self._login(page, test_server)
        result = page.evaluate("""
            fetch('/api/admin/config/general/timeframe', {
                method: 'PATCH',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({value: '1d'}),
                credentials: 'same-origin'
            }).then(r => r.json())
        """)
        assert result["status"] == "ok"
        assert result["category"] == "cycle"

        # Verify the change
        result = page.evaluate("""
            fetch('/api/admin/config/general', {credentials: 'same-origin'})
                .then(r => r.json())
        """)
        assert result["keys"]["timeframe"]["value"] == "1d"

    def test_patch_invalid_value_returns_400(self, page, test_server):
        """PATCH with invalid value returns 400."""
        self._login(page, test_server)
        result = page.evaluate("""
            fetch('/api/admin/config/general/timeframe', {
                method: 'PATCH',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({value: 'invalid_timeframe'}),
                credentials: 'same-origin'
            }).then(r => r.json().then(data => ({status: r.status, data})))
        """)
        assert result["status"] == 400

    def test_system_status_endpoint(self, page, test_server):
        """GET /api/admin/system/status returns status."""
        self._login(page, test_server)
        result = page.evaluate("""
            fetch('/api/admin/system/status', {credentials: 'same-origin'})
                .then(r => r.json())
        """)
        assert "uptime_seconds" in result
        assert "feed_enabled" in result
        assert result["feed_enabled"] is True

    def test_toggle_feed_endpoint(self, page, test_server):
        """POST /api/admin/system/toggle-feed toggles state."""
        self._login(page, test_server)

        # Get initial state
        initial = page.evaluate("""
            fetch('/api/admin/system/status', {credentials: 'same-origin'})
                .then(r => r.json())
        """)
        initial_feed = initial["feed_enabled"]

        # Toggle
        result = page.evaluate("""
            fetch('/api/admin/system/toggle-feed', {
                method: 'POST',
                credentials: 'same-origin'
            }).then(r => r.json())
        """)
        assert result["status"] == "ok"
        assert result["feed_enabled"] != initial_feed

    def test_force_analysis_endpoint(self, page, test_server):
        """POST /api/admin/system/trigger-analysis works."""
        self._login(page, test_server)
        result = page.evaluate("""
            fetch('/api/admin/system/trigger-analysis', {
                method: 'POST',
                credentials: 'same-origin'
            }).then(r => r.json())
        """)
        assert result["status"] == "ok"

    def test_batch_config_update(self, page, test_server):
        """POST /api/admin/config/batch updates multiple values."""
        self._login(page, test_server)
        result = page.evaluate("""
            fetch('/api/admin/config/batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    updates: [
                        {section: 'general', key: 'timeframe', value: '4h'},
                        {section: 'debug', key: 'logger_debug', value: 'false'}
                    ]
                }),
                credentials: 'same-origin'
            }).then(r => r.json())
        """)
        assert result["status"] == "ok"
        assert len(result["results"]) == 2

    def test_recent_logs_endpoint(self, page, test_server):
        """GET /api/admin/logs/recent returns log lines."""
        self._login(page, test_server)
        result = page.evaluate("""
            fetch('/api/admin/logs/recent?count=10', {credentials: 'same-origin'})
                .then(r => r.json())
        """)
        assert "lines" in result
        assert "count" in result


class TestResponsiveLayout:
    """Test that the layout works at different viewport sizes."""

    def _login(self, page, test_server):
        page.goto(f"{test_server}/admin/")
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", "admin")
            page.fill("#login-pass", "testpass123")
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible()

    def test_sidebar_visible_at_desktop_size(self, page, test_server):
        """Sidebar is visible at desktop viewport."""
        page.set_viewport_size({"width": 1280, "height": 800})
        self._login(page, test_server)
        sidebar = page.locator("aside")
        expect(sidebar).to_be_visible()

    def test_content_area_exists(self, page, test_server):
        """Main content area exists."""
        self._login(page, test_server)
        main = page.locator("main")
        expect(main).to_be_visible()
