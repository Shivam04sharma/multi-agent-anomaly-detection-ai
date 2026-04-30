"""Vault secrets fetching utility using secret-proxy.

This module is kept for backward compatibility. The primary integration
now happens via VaultSecretMiddleware which calls secret_proxy.get_secret_async
directly. Use these helpers for ad-hoc secret fetches outside the middleware.
"""

from __future__ import annotations

import logging

from secret_proxy import get_secret_async

logger = logging.getLogger(__name__)


async def get_db_secrets(jwt: str, base_url: str, path: str) -> dict[str, str]:
    """Fetch DB credentials from Vault via secret-proxy."""
    return await get_secret_async(jwt=jwt, path=path, base_url=base_url)
