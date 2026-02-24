"""Router for dashboard monitoring and news endpoints."""
import json
from pathlib import Path
from typing import Dict, Any

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
        self.router.add_api_route("/system_prompt", self.get_system_prompt, methods=["GET"])
        self.router.add_api_route("/costs", self.get_api_costs, methods=["GET"])
        self.router.add_api_route("/news", self.get_news, methods=["GET"])

    def _load_prev_response(self) -> Dict[str, Any]:
        """Helper to load previous response data."""
        data_dir = self.config.DATA_DIR
        path = Path(data_dir) / "trading" / "previous_response.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception:
                self.logger.error("Error loading previous response", exc_info=True)
        return {}

    async def get_last_prompt(self) -> Dict[str, Any]:
        """Get the last prompt sent to the LLM."""
        if self.analysis_engine and self.analysis_engine.last_generated_prompt:
            return {
                "prompt": self.analysis_engine.last_generated_prompt,
                "source": "memory",
                "timestamp": self.analysis_engine.last_prompt_timestamp
            }
        data = self._load_prev_response()
        prompt = data.get("prompt")
        if prompt:
            return {
                "prompt": prompt,
                "source": "disk",
                "timestamp": data.get("timestamp", "unknown")
            }
        return {"prompt": "No prompt generated yet.", "source": None}

    async def get_last_response(self) -> Dict[str, Any]:
        """Get the last response received from the LLM."""
        if self.analysis_engine and self.analysis_engine.last_llm_response:
            return {
                "response": self.analysis_engine.last_llm_response,
                "source": "memory",
                "timestamp": self.analysis_engine.last_response_timestamp
            }
        data = self._load_prev_response()
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

    async def get_system_prompt(self) -> Dict[str, Any]:
        """Get the last system prompt (contains brain context and trading rules)."""
        system_prompt = self.analysis_engine.last_system_prompt if self.analysis_engine else None
        if system_prompt:
            has_brain = "TRADING BRAIN" in system_prompt
            return {
                "system_prompt": system_prompt,
                "source": "memory",
                "has_brain_context": has_brain
            }
        return {"system_prompt": "No system prompt generated yet.", "source": None, "has_brain_context": False}


    async def get_api_costs(self) -> Dict[str, Any]:
        """Get current API cost tracking data."""
        cached = self.dashboard_state.get_cached("costs", ttl_seconds=30.0)
        if cached:
            return cached
        storage = CostStorage()
        openrouter_cost = storage.get_provider_costs("openrouter").total_cost
        google_cost = storage.get_provider_costs("google").total_cost
        total = openrouter_cost + google_cost
        result = {
            "costs_by_provider": {
                "openrouter": openrouter_cost,
                "google": google_cost
            },
            "total_session_cost": total,
            "last_request_cost": None,
            "formatted_total": f"${total:.6f}" if total > 0 else "Free"
        }
        self.dashboard_state.set_cached("costs", result)
        return result


    async def get_news(self) -> Dict[str, Any]:
        """Get cached news articles from RAG engine or disk."""
        cached = self.dashboard_state.get_cached("news", ttl_seconds=3600.0)
        if cached is not None:
            return {"articles": cached, "count": len(cached)}
        articles = []
        if self.rag_engine:
            news_manager = self.rag_engine.news_manager
            if news_manager:
                articles = news_manager.news_database
        if not articles:
            data_dir = self.config.DATA_DIR
            for news_path in NEWS_FILES:
                news_file = Path(data_dir) / news_path
                if news_file.exists():
                    try:
                        with open(news_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            articles = data.get("articles", data) if isinstance(data, dict) else data
                            if articles:
                                break
                    except Exception:
                        self.logger.error("Failed to load news from %s", news_path, exc_info=True)
        self.dashboard_state.set_cached("news", articles)
        return {"articles": articles, "count": len(articles)}
