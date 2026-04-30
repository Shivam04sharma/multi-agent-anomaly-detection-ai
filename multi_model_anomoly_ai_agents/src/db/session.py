"""Per-tenant async PostgreSQL connection pools.

Each tenant gets its own asyncpg pool, created lazily on first request.
Credentials are fetched from Vault via the secret-proxy using the JWT.

Pool lifecycle:
    1. First request from tenant X arrives with JWT
    2. Middleware calls secret-proxy → gets DB creds for tenant X
    3. init_tenant_pool() creates a pool and stores it in _pools[tenant_key]
    4. Subsequent requests from tenant X reuse the pool
    5. On shutdown, close_all_pools() drains every pool
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import asyncpg
import structlog

logger = structlog.get_logger()


@dataclass
class TenantDB:
    """Holds the pool and schema for a single tenant."""

    pool: asyncpg.Pool
    schema: str


# tenant_key → TenantDB
_tenants: dict[str, TenantDB] = {}

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _build_dsn(secrets: dict[str, str]) -> str:
    """Build asyncpg DSN from Vault secrets dict."""
    user = secrets.get("DB_USERNAME", "")
    pwd = quote(secrets.get("DB_PASSWORD", ""), safe="")
    host = secrets.get("DB_HOST", "localhost").removeprefix("postgresql://")
    port = secrets.get("DB_PORT", "5432")
    name = secrets.get("DB_NAME", "")

    logger.info(
        "dsn_build_debug",
        db_host=host,
        db_port=port,
        db_name=name,
        db_username=user,
        has_password=bool(pwd),
        host_was_default=("DB_HOST" not in secrets),
        port_was_default=("DB_PORT" not in secrets),
        name_was_default=("DB_NAME" not in secrets),
        user_was_default=("DB_USERNAME" not in secrets),
        hint=(
            "If host=localhost or any field shows was_default=True, "
            "the Vault secrets dict is missing that key. "
            "Check secret_proxy logs above for parsing details."
        ),
    )

    dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
    # Log sanitized DSN (mask password)
    safe_dsn = f"postgresql://{user}:***@{host}:{port}/{name}"
    logger.info("dsn_built", dsn=safe_dsn)
    return dsn


async def init_tenant_pool(
    tenant_key: str,
    secrets: dict[str, str],
    schema: str = "",
    min_size: int = 2,
    max_size: int = 10,
) -> TenantDB:
    """Create (or return existing) connection pool for a tenant.

    Args:
        tenant_key: Tenant identifier from JWT.
        secrets: DB credentials from Vault (DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD).
        schema: PostgreSQL schema name — from env var DB_SCHEMA, not from Vault.
        min_size: Minimum pool connections.
        max_size: Maximum pool connections.
    """
    if tenant_key in _tenants:
        return _tenants[tenant_key]

    dsn = _build_dsn(secrets)
    pool = await asyncpg.create_pool(dsn=dsn, min_size=min_size, max_size=max_size)

    if schema:
        await _run_migrations(pool, schema)

    tenant_db = TenantDB(pool=pool, schema=schema)
    _tenants[tenant_key] = tenant_db
    logger.info(
        "tenant_pool_created",
        tenant=tenant_key,
        schema=schema,
        host=secrets.get("DB_HOST"),
    )
    return tenant_db


def get_tenant_pool(tenant_key: str) -> TenantDB | None:
    """Get the TenantDB for a tenant, or None if not yet initialized."""
    return _tenants.get(tenant_key)


async def close_all_pools() -> None:
    """Shutdown hook — close every tenant pool."""
    for tenant, tdb in _tenants.items():
        await tdb.pool.close()
        logger.info("tenant_pool_closed", tenant=tenant)
    _tenants.clear()


async def _run_migrations(pool: asyncpg.Pool, schema: str) -> None:
    """Run pending SQL migrations for a tenant's schema.

    Tracks applied migrations in {schema}.schema_migrations table.
    SQL files: db/migrations/{env}/V{n}__{description}.sql
    {schema} placeholder is replaced at runtime.
    """
    from config import settings

    env = settings.env.lower()
    migration_dir = _MIGRATIONS_DIR / env

    if not migration_dir.exists():
        logger.warning("migrations_dir_not_found", path=str(migration_dir))
        return

    sql_files = sorted(
        migration_dir.glob("V*.sql"),
        key=lambda f: int(f.stem.split("__")[0].lstrip("V")),
    )

    if not sql_files:
        logger.info("no_migrations_found", env=env)
        return

    async with pool.acquire() as conn:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{schema}".schema_migrations (
                version     VARCHAR(100) PRIMARY KEY,
                applied_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            );
        """)

        for sql_file in sql_files:
            version = sql_file.stem
            already = await conn.fetchval(
                f'SELECT 1 FROM "{schema}".schema_migrations WHERE version = $1',
                version,
            )
            if already:
                continue

            sql = sql_file.read_text(encoding="utf-8").replace("{schema}", schema)
            await conn.execute(sql)
            await conn.execute(
                f'INSERT INTO "{schema}".schema_migrations (version) VALUES ($1)',
                version,
            )
            logger.info("migration_applied", version=version, schema=schema)
