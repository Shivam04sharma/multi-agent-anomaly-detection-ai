"""Vault-aware middleware — per-tenant DB pool initialization.

Flow:
    1. Request arrives with Authorization: Bearer <jwt>
    2. Decode tenantKey from JWT payload (no signature check — auth dependency does that)
    3. If tenant pool already exists → attach to request.state, pass through
    4. If not → call secret-proxy with JWT → create asyncpg pool → cache it
    5. Routes access request.state.db_pool / request.state.db_schema
"""

from __future__ import annotations

import base64
import json

import structlog
from config import settings
from db.session import get_tenant_pool, init_tenant_pool
from fastapi import Request, Response
from secret_proxy import SecretProxyError, get_secret_async
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger()

# Paths that don't require tenant context
_SKIP_PREFIXES = ("/health", "/actuator/health", "/docs", "/redoc", "/openapi.json")


def _decode_tenant_key(jwt_token: str) -> str | None:
    """Extract tenantKey from JWT payload without verifying signature."""
    try:
        payload_b64 = jwt_token.split(".")[1]
        # Base64url padding
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("tenantKey")
    except Exception:
        return None


def _set_tenant_state(
    request: Request,
    tenant_key: str,
    pool: object | None = None,
    schema: str = "",
) -> None:
    """Attach tenant context to request.state for downstream use."""
    request.state.tenant_key = tenant_key
    request.state.db_pool = pool
    request.state.db_schema = schema


class VaultSecretMiddleware(BaseHTTPMiddleware):
    """Ensures a per-tenant DB pool exists before route handlers execute."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip paths that don't need tenant context
        if any(request.url.path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)

        # Extract JWT
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return await call_next(request)

        jwt_token = auth_header.removeprefix("Bearer ").strip()
        tenant_key = _decode_tenant_key(jwt_token)
        if not tenant_key:
            return await call_next(request)

        # Fast path — pool already cached for this tenant
        tenant_db = get_tenant_pool(tenant_key)
        if tenant_db is not None:
            _set_tenant_state(request, tenant_key, tenant_db.pool, tenant_db.schema)
            return await call_next(request)

        # Slow path — first request from this tenant: fetch secrets → create pool
        try:
            logger.info(
                "vault_fetch_starting",
                tenant=tenant_key,
                vault_proxy_url=settings.vault_proxy_url,
                db_creds_path=settings.db_creds_path,
                db_schema=settings.db_schema,
            )

            secrets = await get_secret_async(
                jwt=jwt_token,
                path=settings.db_creds_path,
                base_url=settings.vault_proxy_url,
            )

            logger.info(
                "vault_fetch_complete",
                tenant=tenant_key,
                secret_keys=list(secrets.keys()) if secrets else [],
                has_db_host="DB_HOST" in secrets,
                has_db_port="DB_PORT" in secrets,
                has_db_name="DB_NAME" in secrets,
                has_db_username="DB_USERNAME" in secrets,
                has_db_password="DB_PASSWORD" in secrets,
            )

            tenant_db = await init_tenant_pool(
                tenant_key=tenant_key, secrets=secrets, schema=settings.db_schema
            )
            _set_tenant_state(request, tenant_key, tenant_db.pool, tenant_db.schema)

            logger.info(
                "vault_tenant_initialized",
                tenant=tenant_key,
                schema=tenant_db.schema,
            )

        except SecretProxyError as exc:
            logger.error(
                "vault_secret_fetch_failed",
                tenant=tenant_key,
                error=str(exc),
                vault_proxy_url=settings.vault_proxy_url,
                db_creds_path=settings.db_creds_path,
                hint=(
                    "Secret-proxy returned an error. Check: "
                    "1) Is VAULT_PROXY_URL correct and reachable from this pod? "
                    "2) Is the JWT valid and not expired? "
                    "3) Does the tenant's Vault namespace exist ({tenant}-kv)? "
                    "4) Does the secret path exist in Vault?"
                ),
            )
            _set_tenant_state(request, tenant_key)

        except Exception as exc:
            logger.error(
                "tenant_pool_creation_failed",
                tenant=tenant_key,
                error=str(exc),
                error_type=type(exc).__name__,
                hint=(
                    "Secrets were fetched but DB pool creation failed. Check: "
                    "1) Are DB_HOST and DB_PORT correct in Vault? "
                    "2) Is the database reachable from this pod (nc -zv <host> <port>)? "
                    "3) Are DB_USERNAME and DB_PASSWORD valid? "
                    "4) Does the database DB_NAME exist? "
                    "See dsn_build_debug log above for the exact values used."
                ),
            )
            _set_tenant_state(request, tenant_key)

        return await call_next(request)
