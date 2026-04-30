from unittest.mock import MagicMock

from src.rag.news_repository import NewsRepository


def _repo() -> tuple[NewsRepository, MagicMock]:
    logger = MagicMock()
    file_handler = MagicMock()
    repo = NewsRepository(logger=logger, file_handler=file_handler)
    return repo, file_handler


def test_load_recent_articles_uses_default_loader_for_24h() -> None:
    repo, file_handler = _repo()
    file_handler.load_news_articles.return_value = [{"id": "a"}]

    result = repo.load_recent_articles(max_age_seconds=86400)

    assert result == [{"id": "a"}]
    file_handler.load_news_articles.assert_called_once()
    file_handler.filter_articles_by_age.assert_not_called()


def test_load_recent_articles_filters_for_custom_window() -> None:
    repo, file_handler = _repo()
    articles = [{"id": "a"}, {"id": "b"}]
    file_handler.load_news_articles.return_value = articles
    file_handler.filter_articles_by_age.return_value = [{"id": "b"}]

    result = repo.load_recent_articles(max_age_seconds=3600)

    assert result == [{"id": "b"}]
    file_handler.filter_articles_by_age.assert_called_once_with(articles, max_age_seconds=3600)


def test_save_recent_articles_uses_default_persistence_for_24h() -> None:
    repo, file_handler = _repo()
    articles = [{"id": "a"}]

    repo.save_recent_articles(articles, max_age_seconds=86400)

    file_handler.save_news_articles.assert_called_once_with(articles)
    file_handler.filter_articles_by_age.assert_not_called()


def test_save_recent_articles_filters_before_save_for_custom_window() -> None:
    repo, file_handler = _repo()
    articles = [{"id": "a"}, {"id": "b"}]
    filtered = [{"id": "b"}]
    file_handler.filter_articles_by_age.return_value = filtered

    repo.save_recent_articles(articles, max_age_seconds=3600)

    file_handler.filter_articles_by_age.assert_called_once_with(articles, max_age_seconds=3600)
    file_handler.save_news_articles.assert_called_once_with(filtered)


def test_load_fallback_articles_uses_hours_parameter() -> None:
    repo, file_handler = _repo()
    file_handler.load_fallback_articles.return_value = [{"id": "cached"}]

    result = repo.load_fallback_articles(max_age_hours=48)

    assert result == [{"id": "cached"}]
    file_handler.load_fallback_articles.assert_called_once_with(max_age_hours=48)
