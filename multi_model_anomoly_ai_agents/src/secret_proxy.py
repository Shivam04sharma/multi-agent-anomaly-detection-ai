"""

The proxy handles caching internally — we just call it every time.
JWT's tenantKey claim routes to the correct Vault namespace ({tenant}-kv).

Handles three response formats:

Format A (flat — legacy):
    {"status": "success", "data": {"DB_HOST": "...", "DB_PORT": "5432", ...}}

Format B (nested value as JSON string):
    {"status": "success", "data": {"value": "{\"data\":{...}}", "tenant": "beta"}}

Format C (Vault KV v2 direct — current infra):
    {"status": "success", "data": {"data": {"DB_HOST": ..., ...}, "metadata": {...}}}
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

DEFAULT_TIMEOUT = 10


class SecretProxyError(Exception):
    """Raised when secret-proxy request fails."""

    pass


def _extract_secrets(body: dict[str, Any]) -> dict[str, Any]:
    """Parse the proxy response and return the flat secret dict.

    Priority order:
      1. Format C (Vault KV v2 direct): body["data"]["data"] + body["data"]["metadata"]
      2. Format B (nested value string): body["data"]["value"] is a JSON string
      3. Format A (flat): body["data"] is already the secrets dict
    """
    data = body.get("data", {})
    logger.debug(
        "extract_secrets_step1_raw_data",
        data_type=type(data).__name__,
        data_keys=list(data.keys()) if isinstance(data, dict) else "NOT_A_DICT",
    )

    if not isinstance(data, dict):
        logger.error(
            "extract_secrets_failed",
            reason="body['data'] is not a dict",
            data_type=type(data).__name__,
        )
        return {}

    # Format C: Vault KV v2 direct — data = {"data": {...secrets...}, "metadata": {...}}
    if "data" in data and "metadata" in data and isinstance(data["data"], dict):
        secrets = data["data"]
        logger.info(
            "extract_secrets_resolved_vault_kv2",
            final_keys=list(secrets.keys()),
            has_db_host="DB_HOST" in secrets,
        )
        return secrets

    # Format B: data has a "value" key containing a JSON string
    raw_value = data.get("value")
    if raw_value is not None:
        parsed = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        logger.debug(
            "extract_secrets_step3_parsed_value",
            parsed_type=type(parsed).__name__,
            parsed_keys=list(parsed.keys()) if isinstance(parsed, dict) else "NOT_A_DICT",
        )
        if isinstance(parsed, dict) and "data" in parsed:
            inner = parsed["data"]
            if isinstance(inner, dict) and "data" in inner and "metadata" in inner:
                secrets = inner["data"]
                logger.info(
                    "extract_secrets_resolved_double_nested",
                    final_keys=list(secrets.keys()) if isinstance(secrets, dict) else "NOT_A_DICT",
                    has_db_host="DB_HOST" in secrets if isinstance(secrets, dict) else False,
                )
                return secrets if isinstance(secrets, dict) else {}
            logger.info(
                "extract_secrets_resolved_single_nested",
                final_keys=list(inner.keys()) if isinstance(inner, dict) else "NOT_A_DICT",
                has_db_host="DB_HOST" in inner if isinstance(inner, dict) else False,
            )
            return inner  # type: ignore[no-any-return]
        return parsed if isinstance(parsed, dict) else {}

    # Format A: data is the flat secret dict directly
    logger.info(
        "extract_secrets_resolved_format_a",
        final_keys=list(data.keys()),
        has_db_host="DB_HOST" in data,
    )
    return data


async def get_secret_async(
    jwt: str,
    path: str,
    base_url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Fetch a secret from Vault via secret-proxy (async).

    Args:
        jwt: Bearer token from the incoming request (contains tenantKey claim).
        path: Vault secret path, e.g. "".
        base_url: Secret-proxy URL, e.g. "".
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict of secret key/value pairs, e.g. {"DB_USERNAME": "admin", ...}.
    """
    url = f"{base_url.rstrip('/')}/v1/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {jwt}"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        body: dict[str, Any] = response.json()

        if body.get("status") != "success":
            logger.warning("secret_proxy_non_success", body=body)
            raise SecretProxyError(f"Proxy returned status: {body.get('status')}")

        secrets = _extract_secrets(body)
        logger.info(
            "secret_proxy_parsed",
            path=path,
            keys=list(secrets.keys()) if secrets else [],
            has_db_host="DB_HOST" in secrets,
            has_db_port="DB_PORT" in secrets,
            has_db_name="DB_NAME" in secrets,
            has_db_username="DB_USERNAME" in secrets,
            has_db_password="DB_PASSWORD" in secrets,
        )
        if not secrets:
            logger.error(
                "secret_proxy_empty_secrets",
                path=path,
                raw_body_keys=list(body.keys()) if isinstance(body, dict) else "NOT_A_DICT",
                hint="Vault returned no usable secrets. Check Vault path and secret structure.",
            )
        missing = [
            k for k in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USERNAME", "DB_PASSWORD")
            if k not in secrets
        ]
        if missing:
            logger.error(
                "secret_proxy_missing_db_keys",
                path=path,
                missing_keys=missing,
                actual_keys=list(secrets.keys()),
                hint=(
                    f"Expected DB credential keys not found in parsed secrets. "
                    f"Got keys: {list(secrets.keys())}. "
                    f"This usually means the Vault response has a different nesting "
                    f"structure than expected. Ask DevOps to verify the secret at '{path}'."
                ),
            )
        return secrets

    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        logger.error("secret_proxy_http_error", status=code, path=path)
        if code == 401:
            raise SecretProxyError("Unauthorized: Invalid or expired JWT (401)") from e
        if code == 404:
            raise SecretProxyError(f"Secret not found at '{path}' (404)") from e
        if code == 502:
            raise SecretProxyError("Vault unreachable and no cache (502)") from e
        raise SecretProxyError(f"HTTP {code} fetching {path}") from e

    except httpx.ConnectError as e:
        logger.error("secret_proxy_connect_error", url=base_url)
        raise SecretProxyError(f"Cannot reach secret-proxy at {base_url}") from e

    except httpx.TimeoutException as e:
        logger.error("secret_proxy_timeout", timeout=timeout)
        raise SecretProxyError(f"Proxy timeout after {timeout}s") from e

    except json.JSONDecodeError as e:
        logger.error("secret_proxy_json_error", error=str(e))
        raise SecretProxyError(f"Failed to parse proxy response: {e}") from e
