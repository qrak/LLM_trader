
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.utils.token_counter import CostStorage

NEWS_FILES = ("crypto_news.json", "news_cache/recent_news.json")

class MonitorRouter:
    """Handles endpoints for system monitoring and news."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, config, logger, dashboard_state, analysis_engine, rag_engine):
        self.router = APIRouter(prefix="/api/monitor", tags=["monitor"])
        self.config = config
        self.logger = logger
        self.dashboard_state = dashboard_state
        self.analysis_engine = analysis_engine
        self.rag_engine = rag_engine

        self.router.add_api_route("/last_prompt", self.get_last_prompt, methods=["GET"])
        self.router.add_api_route("/last_response", self.get_last_response, methods=["GET"])
        self.router.add_api_route("/costs", self.get_api_costs, methods=["GET"])
        self.router.add_api_route("/news", self.get_news, methods=["GET"])
        self.router.add_api_route("/health", self.get_health, methods=["GET"])

    def _load_prev_response_sync(self) -> dict[str, Any]:

        data_dir = self.config.DATA_DIR
        path = Path(data_dir) / "trading" / "previous_response.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as file:
                    return json.load(file)
            except Exception:
                self.logger.error("Error loading previous response", exc_info=True)  # noqa: G201
        return {}

    async def get_last_prompt(self) -> dict[str, Any]:

        if self.analysis_engine and self.analysis_engine.last_generated_prompt:
            return {
                "prompt": self.analysis_engine.last_generated_prompt,
                "source": "memory",
                "timestamp": self.analysis_engine.last_prompt_timestamp,
                "metadata": self.analysis_engine.last_prompt_metadata,
                "lint": self.analysis_engine.last_prompt_lint,
            }
        data = await asyncio.to_thread(self._load_prev_response_sync)
        prompt = data.get("prompt")
        if prompt:
            return {
                "prompt": prompt,
                "source": "disk",
                "timestamp": data.get("timestamp", "unknown")
            }
        return {"prompt": "No prompt generated yet.", "source": None}

    async def get_last_response(self) -> dict[str, Any]:

        if self.analysis_engine and self.analysis_engine.last_llm_response:
            return {
                "response": self.analysis_engine.last_llm_response,
                "source": "memory",
                "timestamp": self.analysis_engine.last_response_timestamp,
                "validation": self.analysis_engine.last_response_validation,
            }
        data = await asyncio.to_thread(self._load_prev_response_sync)
        if data:
            response = data.get("response", {})
            text_analysis = response.get("text_analysis", "No analysis available")
            return {
                "response": text_analysis,
                "source": "disk",
                "timestamp": data.get("timestamp"),
                "indicators": {k: v for k, v in response.items() if k != "text_analysis"}
            }
        return {"response": "No response received yet."}

    async def get_api_costs(self) -> dict[str, Any]:

        cached = self.dashboard_state.get_cached("costs", ttl_seconds=30.0)
        if cached:
            return cached
        storage = await asyncio.to_thread(CostStorage)
        openrouter_cost = storage.get_provider_costs("openrouter").total_cost
        google_cost = storage.get_provider_costs("google").total_cost
        deepseek_cost = storage.get_provider_costs("deepseek").total_cost
        opencode_cost = storage.get_provider_costs("opencode").total_cost
        total = openrouter_cost + google_cost + deepseek_cost + opencode_cost
        result = {
            "costs_by_provider": {
                "openrouter": openrouter_cost,
                "google": google_cost,
                "deepseek": deepseek_cost,
                "opencode": opencode_cost
            },
            "total_session_cost": total,
            "last_request_cost": None,
            "formatted_total": f"${total:.6f}" if total > 0 else "Free"
        }
        self.dashboard_state.set_cached("costs", result)
        return result


    def _read_news_file_sync(self, file_path: Path) -> list:

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("articles", data)

    async def get_news(self, preview: bool = False) -> dict[str, Any]:

        cached = self.dashboard_state.get_cached("news", ttl_seconds=3600.0)
        if cached is not None:
            return self._news_payload(cached, preview)
        articles = []
        if self.rag_engine:
            articles = self.rag_engine.get_news_cache_snapshot()
        if not articles:
            data_dir = self.config.DATA_DIR
            for news_path in NEWS_FILES:
                news_file = Path(data_dir) / news_path
                if news_file.exists():
                    try:
                        parsed_articles = await asyncio.to_thread(self._read_news_file_sync, news_file)
                        articles = parsed_articles
                        if articles:
                            break
                    except Exception:
                        self.logger.error("Failed to load news from %s", news_path, exc_info=True)  # noqa: G201
        self.dashboard_state.set_cached("news", articles)
        return self._news_payload(articles, preview)

    @staticmethod
    def _news_payload(articles: list[dict[str, Any]], preview: bool) -> dict[str, Any]:
        if not preview:
            return {"articles": articles, "count": len(articles)}
        fields = (
            "title", "url", "guid", "published_on", "publishedOn", "source_info",
            "source", "detected_coins", "categories", "tags",
        )
        previews = []
        for article in articles[:30]:
            item = {key: article[key] for key in fields if key in article}
            item["body"] = (article.get("body") or article.get("summary") or "")[:301]
            previews.append(item)
        return {"articles": previews, "count": len(articles)}

    async def get_health(self) -> dict[str, Any]:

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
