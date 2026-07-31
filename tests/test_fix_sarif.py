from pathlib import Path
import json
import tempfile
import pytest

from scripts.fix_sarif import fix_sarif, sanitize_level, fix_uri, main


def test_sanitize_level():
    assert sanitize_level("info") == "note"
    assert sanitize_level("informational") == "note"
    assert sanitize_level("WARNING") == "warning"
    assert sanitize_level("HIGH") == "error"
    assert sanitize_level("critical") == "error"
    assert sanitize_level(None) == "warning"
    assert sanitize_level("unknown_custom_level") == "warning"


def test_fix_sarif_rules_null_and_levels():
    sarif_input = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": None,
                        "rules": None  # Null rules array
                    }
                },
                "results": [
                    {
                        "ruleId": "TEST01",
                        "level": "informational",
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": "/src/src/trading/brain.py"
                                    }
                                }
                            }
                        ]
                    },
                    {
                        "ruleId": "TEST02",
                        "level": "CRITICAL"
                    }
                ]
            }
        ]
    }

    fixed = fix_sarif(sarif_input)

    driver = fixed["runs"][0]["tool"]["driver"]
    assert driver["name"] == "Codacy"
    assert isinstance(driver["rules"], list)  # Must be an array, not null
    assert driver["rules"] == []

    results = fixed["runs"][0]["results"]
    assert results[0]["level"] == "note"
    assert results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "src/trading/brain.py"
    assert results[1]["level"] == "error"


def test_fix_sarif_cli_inplace(monkeypatch, tmp_path):
    sarif_data = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"rules": None}},
                "results": [{"level": "high"}]
            }
        ]
    }
    input_file = tmp_path / "results.sarif"
    input_file.write_text(json.dumps(sarif_data), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["fix_sarif.py", str(input_file)])
    ret = main()
    assert ret == 0

    fixed_data = json.loads(input_file.read_text(encoding="utf-8"))
    assert fixed_data["runs"][0]["tool"]["driver"]["rules"] == []
    assert fixed_data["runs"][0]["results"][0]["level"] == "error"
