import logging

from src.logger.logger import DailyRotatingFileHandler


def make_handler(tmp_path):
    handler = DailyRotatingFileHandler(
        str(tmp_path / "Bot.log"),
        str(tmp_path),
        "",
        "Bot",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def make_record(message):
    return logging.LogRecord("Bot", logging.INFO, __file__, 1, message, None, None)


def test_the_handler_moves_to_the_new_day_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_19")
    handler = make_handler(tmp_path)
    handler.emit(make_record("first"))
    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_20")
    handler.emit(make_record("second"))
    handler.close()

    first = tmp_path / "Bot" / "2026_11_19" / "Bot.log"
    second = tmp_path / "Bot" / "2026_11_20" / "Bot.log"
    assert first.read_text(encoding="utf-8").strip() == "first"
    assert second.read_text(encoding="utf-8").strip() == "second"


def test_a_closed_stream_does_not_break_the_day_switch(tmp_path, monkeypatch):
    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_19")
    handler = make_handler(tmp_path)
    handler.emit(make_record("first"))
    handler.close()
    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_20")
    handler.emit(make_record("second"))
    handler.close()

    second = tmp_path / "Bot" / "2026_11_20" / "Bot.log"
    assert second.read_text(encoding="utf-8").strip() == "second"


def test_the_midnight_switch_keeps_the_old_lines(tmp_path, monkeypatch):
    old_file = tmp_path / "Bot" / "2026_11_19" / "Bot.log"
    old_file.parent.mkdir(parents=True)
    old_file.write_text("line from the previous day\n", encoding="utf-8")

    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_19")
    handler = make_handler(tmp_path)
    handler.emit(make_record("before midnight"))
    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_20")
    handler.emit(make_record("after midnight"))
    handler.emit(make_record("later still"))
    handler.close()

    assert "line from the previous day" in old_file.read_text(encoding="utf-8")
    assert "before midnight" in old_file.read_text(encoding="utf-8")
    new_file = tmp_path / "Bot" / "2026_11_20" / "Bot.log"
    assert new_file.read_text(encoding="utf-8").split() == ["after", "midnight", "later", "still"]


def test_no_rotated_copy_is_left_behind(tmp_path, monkeypatch):
    today = tmp_path / "Bot" / "2026_11_19" / "Bot.log"
    today.parent.mkdir(parents=True)
    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_19")
    handler = DailyRotatingFileHandler(str(today), str(tmp_path), "", "Bot", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(make_record("today"))
    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_20")
    handler.emit(make_record("tomorrow"))
    handler.close()

    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "Bot/2026_11_19/Bot.log",
        "Bot/2026_11_20/Bot.log",
    ]
    assert today.read_text(encoding="utf-8").strip() == "today"


def test_day_directories_older_than_retention_are_pruned(tmp_path, monkeypatch):
    stale = tmp_path / "Bot" / "2026_10_01" / "Bot.log"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")
    recent = tmp_path / "Bot" / "2026_11_20" / "Bot.log"
    recent.parent.mkdir(parents=True)
    recent.write_text("recent\n", encoding="utf-8")

    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_26")
    handler = make_handler(tmp_path)
    handler.emit(make_record("today"))
    handler.close()

    assert not stale.parent.exists()
    assert recent.exists()


def test_the_error_handler_writes_only_errors_into_its_own_day_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("src.logger.logger.day_stamp", lambda: "2026_11_19")
    handler = DailyRotatingFileHandler(
        str(tmp_path / "errors.log"),
        str(tmp_path),
        "",
        "Bot",
        is_error_handler=True,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.ERROR)
    logger = logging.getLogger("probe_error_handler")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    logger.info("an info line")
    logger.error("a real error")
    handler.close()

    body = (tmp_path / "Bot" / "2026_11_19" / "errors.log").read_text(encoding="utf-8").split()
    assert body == ["a", "real", "error"]
