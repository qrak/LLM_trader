from pathlib import Path


DASHBOARD_STATIC = Path("src/dashboard/static")


def test_dashboard_html_exposes_new_lifecycle_and_risk_bindings():
    html = (DASHBOARD_STATIC / "index.html").read_text(encoding="utf-8")

    for element_id in (
        "brain-lifecycle-badge",
        "risk-policy-strip",
        "overview-sl-policy",
        "overview-tp-policy",
        "experience-count",
    ):
        assert f'id="{element_id}"' in html


def test_dashboard_scripts_reference_existing_new_bindings():
    main_js = (DASHBOARD_STATIC / "main.js").read_text(encoding="utf-8")
    websocket_js = (DASHBOARD_STATIC / "modules" / "websocket.js").read_text(encoding="utf-8")
    position_js = (DASHBOARD_STATIC / "modules" / "position_panel.js").read_text(encoding="utf-8")

    assert "overview-sl-policy" in main_js
    assert "overview-tp-policy" in main_js
    assert "brain-lifecycle-badge" in main_js
    assert "brain_rebuild_completed" in websocket_js
    assert "trade-closed-detected" in position_js
