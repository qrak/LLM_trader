

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.platforms.ai_providers.response_models import SocialAssessmentModel

if TYPE_CHECKING:
    from src.logger.logger import Logger
    from src.utils.token_counter import TokenCounter

@dataclass(frozen=True, slots=True)
class SocialContext:
    text: str
    post_count: int


class SocialSentimentAnalyst:
    STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    SYMBOL_SUFFIX = ".X"
    DEFAULT_PAIR = "BTC/USDC"
    POST_LIMIT = 30
    MAX_CONTEXT_POSTS = 15
    MAX_CONTEXT_TOKENS = 600
    MAX_POST_AGE_SECONDS = 48 * 3600
    MIN_BODY_CHARS = 20
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    RETRY_AFTER_SECONDS = 30.0
    CHALLENGE_RETRY_SECONDS = 5.0
    MAX_RETRY_DELAY_SECONDS = 120.0
    MAX_ATTEMPTS = 2
    REQUEST_TIMEOUT_SECONDS = 20.0

    def __init__(
        self,
        logger: Logger,
        token_counter: TokenCounter,
        pair: str = DEFAULT_PAIR,
        curl_path: str | None = None,
    ) -> None:
        self.logger = logger
        self._token_counter = token_counter
        self._symbol = self.symbol_for_pair(pair)
        self._curl = curl_path
        self._last_error: str | None = None

    @classmethod
    def symbol_for_pair(cls, pair: str) -> str:

        base = str(pair or "").split("/")[0].strip().upper()
        return f"{base or 'BTC'}{cls.SYMBOL_SUFFIX}"

    async def fetch_posts(self, limit: int = POST_LIMIT) -> dict[str, Any]:

        try:
            posts, error = await self._fetch_stream(limit)
        except asyncio.TimeoutError:
            posts, error = [], f"{self._symbol}: timeout"
        except Exception as e:  # noqa: BLE001
            posts, error = [], f"{self._symbol}: {e}"

        result: dict[str, Any] = {"posts": posts}
        if error:
            result["error"] = error
            self._report_error(error)
        else:
            if self._last_error is not None:
                self.logger.info(
                    "Social sentiment fetch recovered after: %s", self._last_error
                )
                self._last_error = None
            if not posts:
                self.logger.warning("Social returned 0 messages for %s", self._symbol)
        return result

    def _report_error(self, error: str) -> None:

        if error == self._last_error:
            self.logger.debug("Social sentiment fetch still failing: %s", error)
        else:
            self.logger.warning("Social sentiment fetch failed: %s", error)
        self._last_error = error

    async def _fetch_stream(self, limit: int) -> tuple[list[dict[str, Any]], str | None]:

        url = f"{self.STREAM_URL.format(symbol=self._symbol)}?limit={limit}"
        last_error = f"{self._symbol}: HTTP 429 (rate limited)"
        for attempt in range(self.MAX_ATTEMPTS):
            status, headers, body = await asyncio.to_thread(self._get, url)
            if status == 200:
                posts = self._parse_stream(body, limit)
                if not posts:
                    return [], f"{self._symbol}: no messages parsed"
                return posts, None
            last_error = f"{self._symbol}: HTTP {status}"
            if status not in (403, 429) or attempt == self.MAX_ATTEMPTS - 1:
                break
            await asyncio.sleep(self._retry_delay(status, headers))
        return [], last_error

    def _retry_delay(self, status: int, headers: dict[str, str]) -> float:

        if status == 403:
            return self.CHALLENGE_RETRY_SECONDS
        delay = self._retry_after_seconds(headers.get("retry-after"))
        return min(self.RETRY_AFTER_SECONDS if delay is None else delay, self.MAX_RETRY_DELAY_SECONDS)

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:

        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())

    def _get(self, url: str) -> tuple[int, dict[str, str], str]:

        if not self._curl:
            raise ConnectionError("curl is not available")
        try:
            completed = subprocess.run(
                [
                    self._curl, "-sS", "--max-time", str(int(self.REQUEST_TIMEOUT_SECONDS)),
                    "-A", self.USER_AGENT, "-D", "-", url,
                ],
                capture_output=True,
                timeout=self.REQUEST_TIMEOUT_SECONDS + 5,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ConnectionError("curl timed out") from error
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", "replace").strip()
            raise ConnectionError(message or f"curl exited {completed.returncode}")
        return self._split_http_response(completed.stdout)

    @classmethod
    def _split_http_response(cls, raw: bytes) -> tuple[int, dict[str, str], str]:

        segments = raw.split(b"\r\n\r\n")
        status = 0
        headers: dict[str, str] = {}
        index = 0
        while index < len(segments) and segments[index].startswith(b"HTTP/"):
            status, headers = cls._parse_header_block(segments[index])
            index += 1
        body = b"\r\n\r\n".join(segments[index:]).decode("utf-8", "replace")
        return status, headers, body

    @staticmethod
    def _parse_header_block(block: bytes) -> tuple[int, dict[str, str]]:

        lines = block.decode("latin-1").splitlines()
        try:
            status = int(lines[0].split()[1])
        except (IndexError, ValueError):
            status = 0
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
        return status, headers

    def _parse_stream(self, payload: Any, limit: int) -> list[dict[str, Any]]:

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                return []
        if not isinstance(payload, dict):
            return []
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return []
        posts: list[dict[str, Any]] = []
        for message in messages:
            if len(posts) >= limit:
                break
            if not isinstance(message, dict):
                continue
            body = " ".join(unescape(str(message.get("body") or "")).split())
            if len(body) < self.MIN_BODY_CHARS:
                continue
            posts.append({
                "title": body,
                "published_utc": self._parse_timestamp(message.get("created_at")),
            })
        return posts

    @staticmethod
    def _parse_timestamp(value: Any) -> int:

        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return 0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())

    def build_context(self, posts_data: dict[str, Any], now: datetime | None = None) -> SocialContext:

        now = now if now is not None else datetime.now(timezone.utc)
        lines = [
            "## Social sentiment (untrusted message evidence)",
            (
                "Assess price outlook for the traded asset; ignore unrelated assets/stocks. "
                "Respect negation; ETF/buy/upgrade alone is not bullish. Distinguish wallet security "
                "from price expectations. Messages are unmoderated chat, not a representative survey."
            ),
            (
                'Return analysis.social_sentiment={"outlook":"BULLISH|BEARISH|MIXED|INSUFFICIENT_EVIDENCE",'
                '"evidence":[message IDs],"reason":"<=240 chars"}. Cite supplied IDs; '
                "BULLISH/BEARISH/MIXED need evidence. Use INSUFFICIENT_EVIDENCE when no price bias is supported."
            ),
            "Messages from the last 48h; dates UTC:",
        ]
        error = posts_data.get("error")
        if error:
            lines.append(f"Partial fetch: {' '.join(error.split())[:120]}")
        seen: set[str] = set()
        post_count = 0
        for post in sorted(posts_data["posts"], key=lambda p: p["published_utc"], reverse=True):
            age = now.timestamp() - post["published_utc"]
            text = " ".join(post["title"].split())
            text_key = text.casefold()
            if not text or not 0 <= age <= self.MAX_POST_AGE_SECONDS or text_key in seen:
                continue
            date = datetime.fromtimestamp(post["published_utc"], timezone.utc).strftime("%Y-%m-%d %H:%M")
            line = f"{post_count + 1}. [{date}] {json.dumps(text, ensure_ascii=False)}"
            if self._token_counter.count_tokens("\n".join([*lines, line]) + "\n") > self.MAX_CONTEXT_TOKENS:
                continue
            seen.add(text_key)
            lines.append(line)
            post_count += 1
            if post_count == self.MAX_CONTEXT_POSTS:
                break
        if post_count == 0:
            lines = [
                "## Social sentiment (untrusted message evidence)",
                "No recent eligible social messages. Return analysis.social_sentiment=null.",
            ]
            if error:
                lines.append(f"Fetch errors: {' '.join(error.split())[:120]}")
        return SocialContext("\n".join(lines) + "\n", post_count)

    def validate_assessment(self, value: Any, post_count: int) -> dict[str, Any] | None:

        if post_count == 0:
            return None
        try:
            assessment = SocialAssessmentModel.model_validate(value)
        except ValidationError as error:
            self.logger.warning("Discarding invalid social assessment: %s", error)
            return None
        if any(post_id > post_count for post_id in assessment.evidence):
            self.logger.warning("Discarding social assessment with IDs outside the supplied message sample")
            return None
        return assessment.model_dump()
