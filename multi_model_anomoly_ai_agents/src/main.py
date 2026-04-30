"""Multi Model Anomaly AI Agents Service."""

from __future__ import annotations

import socket
from contextlib import asynccontextmanager

import structlog
import uvicorn
from config import settings
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes.anomaly_routes import router as anomaly_router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load embedding model (avoids cold-start on first request)
    from services.embedding_layer import _get_model

    _get_model()

    logger.info(
        "startup",
        service=settings.service_name,
        env=settings.env,
        port=settings.port,
        auth_enabled=settings.auth_enabled,
    )
    yield

    from db.session import close_all_pools

    await close_all_pools()
    logger.info("shutdown", service=settings.service_name)


app = FastAPI(
    title="multi_model_anomoly_ai_agents",
    version="1.0.0",
    description=(
        "Universal plug-and-play anomaly detection pipeline. "
        "Accepts any tabular dataset — auto-validates, normalizes, "
        "builds features, detects anomalies with Isolation Forest / LOF / Z-score, "
        "fuses statistical + vector + rule signals, and generates LLM explanations."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
    )


app.include_router(anomaly_router, prefix="/api/v1", tags=["anomaly"])


@app.get("/health", tags=["health"])
@app.get("/actuator/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": app.version,
        "env": settings.env,
        "app_instance": socket.gethostname(),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=settings.debug)
