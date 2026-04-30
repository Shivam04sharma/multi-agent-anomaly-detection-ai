"""Local config — runs on developer machine."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env.local sits one level above src/ — resolve path so it works
# whether you run from project root or from src/
_ENV_FILE = Path(__file__).resolve().parents[2] / "your .env.local file"


class LocalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")

    env: str = "your local environment name (e.g. 'local' or your initials)"
    service_name: str = "your-service-name"
    port: int = your_port_number
    log_level: str = "debug"
    debug: bool = True

    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "")
    redis_port: int = int(os.environ.get("REDIS_PORT", ""))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    redis_db: int = 0

    # Storage
    minio_url: str = os.getenv("MINIO_URL", "your local MinIO URL")
    aws_access_key: str = os.getenv("AWS_ACCESS_KEY", "")
    aws_secret_key: str = os.getenv("AWS_SECRET_KEY", "")
    aws_s3_bucket: str = os.getenv("AWS_S3_BUCKET", "")
    aws_region: str = os.getenv("AWS_S3_REGION", "")
    minio_secure: bool = False

 
    openai_model: str = "gpt-4o"
    embedding_model: str = "models/all-MiniLM-L6-v2"

    

    # Auth
    auth_enabled: bool = True
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = "your JWT algorithm"

    # Database schema — same for all tenants, injected via env var
    db_schema: str = os.getenv("DB_SCHEMA", "your_db_schema_name")

    # Vault Sidecar
    vault_proxy_url: str = "your local vault proxy URL "
    db_creds_path: str = "your vault path for DB credentials"


    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"


settings = LocalSettings()
