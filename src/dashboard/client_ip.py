

from __future__ import annotations


def resolve_client_ip(direct_ip: str, cf_connecting_ip: str | None) -> str:

    if direct_ip in ("127.0.0.1", "::1"):
        cf_ip = (cf_connecting_ip or "").strip()
        if cf_ip:
            return cf_ip
    return direct_ip
