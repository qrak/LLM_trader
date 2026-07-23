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


def test_decision_pathways_panel_bindings():
    html = (DASHBOARD_STATIC / "index.html").read_text(encoding="utf-8")
    main_js = (DASHBOARD_STATIC / "main.js").read_text(encoding="utf-8")

    assert 'id="panel-decision-pathways"' in html
    assert "Decision Pathways" in html
    assert 'id="decision-synopsis"' in html
    assert 'id="decision-graph"' in html
    assert 'id="decision-detail"' in html
    assert 'id="decision-legend"' in html
    assert "decision_pathways_panel.js" in main_js
    assert 'main.js?v=' in (DASHBOARD_STATIC / "index.html").read_text(encoding="utf-8")
    assert "initSynapseNetwork" not in (DASHBOARD_STATIC / "main.js").read_text(encoding="utf-8")
    assert "updateDecisionPathways" in main_js
    assert "synapse_viewer.js" not in main_js
    assert "initSynapseNetwork" not in main_js
    assert "vis-network" in html
    assert "synapse-network" not in html
