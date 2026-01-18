
from playwright.sync_api import sync_playwright, expect

def test_sidebar_accessibility(page):
    # Load the dashboard
    page.goto("http://localhost:8000/index.html")

    # Wait for JS to initialize (e.g. initUI)
    # The setupSidebarNavigation runs on DOMContentLoaded or initUI
    # We can wait for the sidebar-nav to have role="tablist"

    nav_list = page.locator(".sidebar-nav")
    expect(nav_list).to_have_attribute("role", "tablist", timeout=5000)

    # Get the first tab (Overview)
    tab_overview = page.locator("#nav-tab-overview")
    expect(tab_overview).to_have_attribute("role", "tab")
    expect(tab_overview).to_have_attribute("aria-selected", "true")
    expect(tab_overview).to_have_attribute("tabindex", "0")

    # Get the second tab (Brain)
    tab_brain = page.locator("#nav-tab-brain")
    expect(tab_brain).to_have_attribute("role", "tab")
    expect(tab_brain).to_have_attribute("aria-selected", "false")
    expect(tab_brain).to_have_attribute("tabindex", "-1")

    # Test Arrow Key Navigation
    # Click overview to ensure focus (though it should be focused if we tab to it, but let's click to start)
    # Actually, let's just focus it programmatically or click it
    tab_overview.click()

    # Press ArrowDown
    page.keyboard.press("ArrowDown")

    # Expect Brain tab to be selected and focused
    expect(tab_brain).to_have_attribute("aria-selected", "true")
    expect(tab_brain).to_have_attribute("tabindex", "0")
    expect(tab_overview).to_have_attribute("aria-selected", "false")
    expect(tab_overview).to_have_attribute("tabindex", "-1")

    # Take screenshot of the focus state/active state
    page.screenshot(path="verification/sidebar_accessibility.png")
    print("Verification successful!")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            test_sidebar_accessibility(page)
        except Exception as e:
            print(f"Test failed: {e}")
            page.screenshot(path="verification/sidebar_failure.png")
            raise
        finally:
            browser.close()
