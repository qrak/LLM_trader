import logging

from src.logger.logger import DailyRotatingFileHandler


def make_handler(tmp_path):
    handler = DailyRotatingFileHandler(
        str(tmp_path / "Bot.log"),
        str(tmp_path),
        "",
        "Bot",
        when="midnight",
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
