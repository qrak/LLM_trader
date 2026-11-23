import json
import os

import pytest

from src.utils import data_utils
from src.utils.data_utils import atomic_write_json


class TestAtomicWriteJsonRetry:
    def test_retries_a_transient_permission_error(self, tmp_path, monkeypatch):
        target = tmp_path / "state.json"
        target.write_text('{"value": 1}', encoding="utf-8")
        real_replace = os.replace
        calls = []

        def flaky_replace(source, destination):
            calls.append(destination)
            if len(calls) < 3:
                raise PermissionError(13, "Odmowa dostępu", str(destination))
            real_replace(source, destination)

        monkeypatch.setattr(data_utils, "_REPLACE_RETRY_DELAYS", (0, 0, 0, 0, 0))
        monkeypatch.setattr(data_utils.os, "replace", flaky_replace)

        atomic_write_json(target, {"value": 2})

        assert len(calls) == 3
        assert json.loads(target.read_text(encoding="utf-8")) == {"value": 2}

    def test_raises_and_removes_the_temp_file_after_a_persistent_permission_error(self, tmp_path, monkeypatch):
        target = tmp_path / "state.json"
        target.write_text('{"value": 1}', encoding="utf-8")

        def always_denied(source, destination):
            raise PermissionError(13, "Odmowa dostępu", str(destination))

        monkeypatch.setattr(data_utils, "_REPLACE_RETRY_DELAYS", (0, 0, 0, 0, 0))
        monkeypatch.setattr(data_utils.os, "replace", always_denied)

        with pytest.raises(PermissionError):
            atomic_write_json(target, {"value": 2})

        assert list(tmp_path.glob("*.tmp")) == []
        assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
