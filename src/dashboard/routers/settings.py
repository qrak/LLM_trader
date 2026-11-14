import asyncio
import ipaddress
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config.settings_editor import SettingsEditor, SettingsError
from src.dashboard.docs import DocsLibrary


class ConfigUpdate(BaseModel):
    section: str = Field(..., min_length=1, max_length=64)
    key: str = Field(..., min_length=1, max_length=128)
    value: str = Field(..., max_length=4000)


class ConfigUpdateRequest(BaseModel):
    etag: str = Field(..., min_length=64, max_length=64)
    updates: list[ConfigUpdate] = Field(..., min_length=1, max_length=200)


class KeyUpdateRequest(BaseModel):
    etag: str | None = None
    key: str = Field(..., min_length=1, max_length=128)
    value: str = Field(..., max_length=4096)


class SettingsRouter:
    def __init__(
        self,
        logger,
        settings_editor: SettingsEditor | None = None,
        docs_library: DocsLibrary | None = None,
    ):
        self.router = APIRouter(prefix="/api/settings", tags=["settings"])
        self.settings_editor = settings_editor or SettingsEditor(logger=logger)
        self.docs_library = docs_library or DocsLibrary()

        self.router.add_api_route("/status", self.get_status, methods=["GET"])
        self.router.add_api_route(
            "/config", self.get_config, methods=["GET"], response_model=None
        )
        self.router.add_api_route(
            "/config", self.update_config, methods=["POST"], response_model=None
        )
        self.router.add_api_route(
            "/keys", self.get_keys, methods=["GET"], response_model=None
        )
        self.router.add_api_route(
            "/keys", self.update_key, methods=["POST"], response_model=None
        )
        self.router.add_api_route(
            "/keys/init", self.init_keys, methods=["POST"], response_model=None
        )
        self.router.add_api_route(
            "/docs/{slug}", self.get_doc, methods=["GET"], response_model=None
        )

    async def get_status(self, request: Request) -> dict:
        return {
            "local": self._client_is_local(request),
            "docs": self.docs_library.nav(),
        }

    async def get_config(self, request: Request) -> dict | JSONResponse:
        denied = self._denied(request)
        if denied is not None:
            return denied
        try:
            return await asyncio.to_thread(self.settings_editor.read_config)
        except SettingsError as exc:
            return self._error(exc.status, str(exc), exc.section, exc.key)

    async def update_config(
        self,
        body: ConfigUpdateRequest,
        request: Request,
    ) -> dict | JSONResponse:
        denied = self._denied(request)
        if denied is not None:
            return denied
        updates: list[dict[str, str]] = [
            {
                "section": update.section,
                "key": update.key,
                "value": update.value,
            }
            for update in body.updates
        ]
        try:
            return await asyncio.to_thread(
                self.settings_editor.update_config,
                body.etag,
                updates,
            )
        except SettingsError as exc:
            return self._error(exc.status, str(exc), exc.section, exc.key)

    async def get_keys(self, request: Request) -> dict | JSONResponse:
        denied = self._denied(request)
        if denied is not None:
            return denied
        try:
            return await asyncio.to_thread(self.settings_editor.read_keys)
        except SettingsError as exc:
            return self._error(exc.status, str(exc), exc.section, exc.key)

    async def update_key(
        self,
        body: KeyUpdateRequest,
        request: Request,
    ) -> dict | JSONResponse:
        denied = self._denied(request)
        if denied is not None:
            return denied
        try:
            return await asyncio.to_thread(
                self.settings_editor.update_key,
                body.etag,
                body.key,
                body.value,
            )
        except SettingsError as exc:
            return self._error(exc.status, str(exc), exc.section, exc.key)

    async def init_keys(self, request: Request) -> dict | JSONResponse:
        denied = self._denied(request)
        if denied is not None:
            return denied
        try:
            return await asyncio.to_thread(self.settings_editor.create_keys_file)
        except SettingsError as exc:
            return self._error(exc.status, str(exc), exc.section, exc.key)

    async def get_doc(self, slug: str) -> dict | JSONResponse:
        document = self.docs_library.render(slug)
        if document is None:
            return self._error(404, "Document not found.")
        return document

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        if not host:
            return False
        normalized = host.strip().lower()
        if normalized == "localhost":
            return True
        try:
            address = ipaddress.ip_address(normalized)
        except ValueError:
            return False
        if address.is_loopback:
            return True
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            return address.ipv4_mapped.is_loopback
        return False

    @classmethod
    def _host_is_local(cls, host_header: str) -> bool:
        if not host_header:
            return False
        try:
            host = urlsplit(f"//{host_header}").hostname or ""
        except ValueError:
            return False
        return cls._is_loopback_host(host)

    @classmethod
    def _origin_is_local(cls, origin: str) -> bool:
        if not origin:
            return True
        try:
            host = urlsplit(origin).hostname or ""
        except ValueError:
            return False
        return cls._is_loopback_host(host)

    def _client_is_local(self, request: Request) -> bool:
        if request.headers.get("cf-connecting-ip"):
            return False
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded and not self._is_loopback_host(forwarded.split(",")[0].strip()):
            return False
        client = request.client.host if request.client else ""
        if not self._is_loopback_host(client):
            return False
        if not self._host_is_local(request.headers.get("host", "")):
            return False
        origin = request.headers.get("origin", "")
        if origin and not self._origin_is_local(origin):
            return False
        referer = request.headers.get("referer", "")
        return not referer or self._origin_is_local(referer)

    def _denied(self, request: Request) -> JSONResponse | None:
        if self._client_is_local(request):
            return None
        return self._error(
            403,
            "Settings are available only from this machine (localhost).",
        )

    @staticmethod
    def _error(
        status: int,
        message: str,
        section: str | None = None,
        key: str | None = None,
    ) -> JSONResponse:
        payload: dict[str, str] = {"error": message}
        if section is not None:
            payload["section"] = section
        if key is not None:
            payload["key"] = key
        return JSONResponse(status_code=status, content=payload)
