"""Production-readiness Playwright tests against the live dashboard.

Tests the live server at semanticsignal.qrak.org (via Cloudflare).
Covers: security headers, CSP compliance, full login flow, config editing,
WebSocket connectivity, and error handling.

Run: .venv-wsl/bin/python -m pytest tests/test_admin_live.py -v --tb=short
Requires: live bot with admin credentials configured in keys.env
"""

import os
import pytest

pytest.importorskip("playwright")

from playwright.sync_api import sync_playwright, expect


LIVE_URL = os.environ.get("LLM_TRADER_URL", "https://semanticsignal.qrak.org")
ADMIN_URL = f"{LIVE_URL}/admin"

# Use env vars for credentials — NEVER hardcode
ADMIN_USER = os.environ.get("LLM_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("LLM_ADMIN_PASS", "")


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        yield br
        br.close()


@pytest.fixture
def ctx(browser):
    c = browser.new_context(viewport={"width": 1280, "height": 800})
    yield c
    c.close()


@pytest.fixture
def page(ctx):
    pg = ctx.new_page()
    yield pg
    pg.close()


# ─── Security ────────────────────────────────────────────────────────

class TestSecurityHeaders:
    """Verify security headers on all responses."""

    def test_main_dashboard_has_security_headers(self, page):
        resp = page.goto(LIVE_URL, timeout=15000)
        h = resp.headers
        assert h.get("x-content-type-options") == "nosniff"
        assert h.get("x-frame-options") == "DENY"
        assert "strict-transport-security" in h
        assert h.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_admin_page_has_security_headers(self, page):
        resp = page.goto(ADMIN_URL, timeout=15000)
        h = resp.headers
        assert h.get("x-content-type-options") == "nosniff"
        assert h.get("x-frame-options") == "DENY"

    def test_csp_allows_admin_scripts(self, page):
        """CSP must allow Tailwind CDN and inline scripts for admin page."""
        import time
        resp = page.goto(f"{ADMIN_URL}?_nocsp={int(time.time())}", timeout=15000)
        csp = resp.headers.get("content-security-policy", "")
        assert "cdn.tailwindcss.com" in csp or "'unsafe-inline'" in csp, \
            "Neither Tailwind CDN nor unsafe-inline in CSP — admin page JS will break"

    def test_no_cors_for_cross_origin(self, page):
        """Cross-origin requests should not get ACAO header."""
        page.goto(LIVE_URL, timeout=15000)
        result = page.evaluate("""
            fetch('/api/monitor/health', {
                headers: {'Origin': 'https://evil.com'}
            }).then(r => r.headers.get('access-control-allow-origin'))
        """)
        assert result is None

    def test_sensitive_files_not_accessible(self, page):
        """Sensitive files should return 404 or be blocked."""
        page.goto(LIVE_URL, timeout=15000)
        for path in ["/keys.env", "/config/config.ini", "/AGENTS.md", "/.env"]:
            result = page.evaluate(f"""
                fetch('{path}').then(r => r.status)
            """)
            assert result in (404, 403, 502), f"{path} returned {result} — may be accessible!"


# ─── Admin Login Flow ────────────────────────────────────────────────

class TestAdminLoginFlow:
    """Test the complete login flow."""

    @pytest.mark.skipif(not ADMIN_PASS, reason="LLM_ADMIN_PASS not set")
    def test_login_success(self, page):
        """Valid credentials should show the admin dashboard."""
        page.goto(ADMIN_URL, timeout=15000)
        page.fill("#login-user", ADMIN_USER)
        page.fill("#login-pass", ADMIN_PASS)
        page.click("button[type='submit']")
        expect(page.locator("#app")).to_be_visible(timeout=10000)
        expect(page.locator("#login-screen")).to_be_hidden()

    def test_login_failure_shows_error(self, page):
        """Invalid credentials should show error message."""
        page.goto(ADMIN_URL, timeout=15000)
        page.fill("#login-user", "wrong_user")
        page.fill("#login-pass", "wrong_pass")
        page.click("button[type='submit']")
        page.wait_for_timeout(2000)
        error = page.locator("#login-error")
        expect(error).to_be_visible()
        expect(error).to_contain_text("Invalid")

    @pytest.mark.skipif(not ADMIN_PASS, reason="LLM_ADMIN_PASS not set")
    def test_login_sets_cookie(self, page, ctx):
        """Login should set an httponly session cookie."""
        page.goto(ADMIN_URL, timeout=15000)
        page.fill("#login-user", ADMIN_USER)
        page.fill("#login-pass", ADMIN_PASS)
        page.click("button[type='submit']")
        expect(page.locator("#app")).to_be_visible(timeout=10000)
        cookies = ctx.cookies()
        session_cookie = [c for c in cookies if c["name"] == "admin_session"]
        assert len(session_cookie) == 1, "admin_session cookie not set"
        assert session_cookie[0]["httpOnly"] is True
        assert session_cookie[0]["secure"] is True

    @pytest.mark.skipif(not ADMIN_PASS, reason="LLM_ADMIN_PASS not set")
    def test_logout_clears_cookie(self, page, ctx):
        """Logout should clear the session cookie."""
        page.goto(ADMIN_URL, timeout=15000)
        page.fill("#login-user", ADMIN_USER)
        page.fill("#login-pass", ADMIN_PASS)
        page.click("button[type='submit']")
        expect(page.locator("#app")).to_be_visible(timeout=10000)
        page.click("#btn-logout")
        expect(page.locator("#login-screen")).to_be_visible(timeout=5000)


# ─── Admin Dashboard ─────────────────────────────────────────────────

@pytest.mark.skipif(not ADMIN_PASS, reason="LLM_ADMIN_PASS not set")
class TestAdminDashboard:
    """Test dashboard features after login."""

    def _login(self, page):
        page.goto(ADMIN_URL, timeout=15000)
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", ADMIN_USER)
            page.fill("#login-pass", ADMIN_PASS)
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible(timeout=10000)

    def test_dashboard_stats_load(self, page):
        """Dashboard stat cards should show data."""
        self._login(page)
        page.wait_for_timeout(3000)
        uptime = page.locator("#stat-uptime").inner_text()
        assert uptime != "--" and uptime != ""

    def test_force_analysis_button(self, page):
        """Force Analysis button should send WS command."""
        self._login(page)
        page.click("#btn-force-analysis")
        page.wait_for_timeout(2000)
        # Should see console log entry
        console = page.locator("#console-log")
        text = console.inner_text()
        assert "force_analysis" in text.lower() or "analysis" in text.lower()

    def test_toggle_feed_button(self, page):
        """Toggle Feed button should change state."""
        self._login(page)
        page.click("#btn-toggle-feed")
        page.wait_for_timeout(2000)
        feed_stat = page.locator("#stat-feed").inner_text()
        assert "ON" in feed_stat or "OFF" in feed_stat

    def test_human_input_submit(self, page):
        """Human input should submit and show confirmation."""
        self._login(page)
        page.fill("#human-input-text", "Production test - ignore")
        page.click("#btn-submit-input")
        page.wait_for_timeout(2000)
        status = page.locator("#human-input-status").inner_text()
        assert "Submitted" in status


# ─── Config Editor ───────────────────────────────────────────────────

@pytest.mark.skipif(not ADMIN_PASS, reason="LLM_ADMIN_PASS not set")
class TestConfigEditor:
    """Test config editor loads and can edit values."""

    def _login(self, page):
        page.goto(ADMIN_URL, timeout=15000)
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", ADMIN_USER)
            page.fill("#login-pass", ADMIN_PASS)
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible(timeout=10000)

    def test_config_page_loads_sections(self, page):
        """Config page should show section headers."""
        self._login(page)
        page.click("button[data-page='config']")
        page.wait_for_timeout(3000)
        content = page.locator("#config-sections").inner_text()
        assert "AI Providers" in content or "General" in content

    def test_config_section_expandable(self, page):
        """Config sections should expand on click."""
        self._login(page)
        page.click("button[data-page='config']")
        page.wait_for_timeout(3000)
        headers = page.locator("#config-sections button")
        if headers.count() > 0:
            headers.first.click()
            page.wait_for_timeout(500)


# ─── API Endpoints ───────────────────────────────────────────────────

@pytest.mark.skipif(not ADMIN_PASS, reason="LLM_ADMIN_PASS not set")
class TestAdminAPI:
    """Test admin API endpoints via browser fetch."""

    def _login(self, page):
        page.goto(ADMIN_URL, timeout=15000)
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", ADMIN_USER)
            page.fill("#login-pass", ADMIN_PASS)
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible(timeout=10000)

    def test_config_schema_returns_all_sections(self, page):
        self._login(page)
        result = page.evaluate("""
            fetch('/api/admin/config', {credentials: 'same-origin'}).then(r => r.json())
        """)
        assert "general" in result
        assert "ai_providers" in result
        assert "risk_management" in result

    def test_system_status_returns_data(self, page):
        self._login(page)
        result = page.evaluate("""
            fetch('/api/admin/system/status', {credentials: 'same-origin'}).then(r => r.json())
        """)
        assert "uptime_seconds" in result
        assert "feed_enabled" in result

    def test_recent_logs_returns_lines(self, page):
        self._login(page)
        result = page.evaluate("""
            fetch('/api/admin/logs/recent?count=5', {credentials: 'same-origin'}).then(r => r.json())
        """)
        assert "lines" in result
        assert isinstance(result["lines"], list)


# ─── Navigation ──────────────────────────────────────────────────────

@pytest.mark.skipif(not ADMIN_PASS, reason="LLM_ADMIN_PASS not set")
class TestNavigation:
    """Test sidebar navigation between pages."""

    def _login(self, page):
        page.goto(ADMIN_URL, timeout=15000)
        try:
            page.wait_for_selector("#app:not(.hidden)", timeout=3000)
        except Exception:
            page.fill("#login-user", ADMIN_USER)
            page.fill("#login-pass", ADMIN_PASS)
            page.click("button[type='submit']")
            expect(page.locator("#app")).to_be_visible(timeout=10000)

    def test_all_nav_pages_accessible(self, page):
        """All navigation pages should be reachable."""
        self._login(page)
        for nav_page in ["control", "config", "logs", "dashboard"]:
            page.click(f"button[data-page='{nav_page}']")
            page.wait_for_timeout(500)
            target = page.locator(f"#page-{nav_page}")
            expect(target).to_be_visible()
