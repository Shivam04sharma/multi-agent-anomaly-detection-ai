"""Anomaly Detection API Routes.

Endpoints:
  POST /anomaly/analyze       — run full pipeline on JSON dataset
  POST /anomaly/upload-csv    — upload a CSV file and run full pipeline
  POST /anomaly/predict       — single-record real-time prediction
  GET  /anomaly/sessions/{id} — fetch saved alerts for a session
  GET  /anomaly/sessions      — list all sessions with stats
"""

from __future__ import annotations

import asyncio
import json
import uuid

import structlog
from config.auth import CurrentUser, get_current_user
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from schemas.anomaly_schemas import (
    AlertPreview,
    AnomalyAnalyzeRequest,
    AnomalyAnalyzeResponse,
    AnomalyDetected,
    AnomalyPredictRequest,
    AnomalyPredictResponse,
    SessionsListResponse,
    SessionSummary,
)
from services import (
    detection_engine,
    embedding_layer,
    embedding_store,
    explanation_engine,
    feature_builder,
    ingestion,
    normalization,
    prompt_builder,
    result_store,
    scoring_engine,
    validation_agent,
    vector_store,
)
from services.llm_client import get_llm_client

logger = structlog.get_logger()

router = APIRouter()


def _get_tenant_ctx(request: Request) -> tuple:
    """Extract tenant DB pool and schema from request.state."""
    pool = getattr(request.state, "db_pool", None)
    schema = getattr(request.state, "db_schema", "")
    return pool, schema


@router.post(
    "/anomaly/analyze",
    response_model=AnomalyAnalyzeResponse,
    summary="Run full anomaly detection pipeline on a dataset",
    description=(
        "Accepts any tabular dataset (any columns). "
        "Runs Steps 1–10: ingest → validate → normalize → features → "
        "intent → embed → vector search → detect → score → explain."
    ),
)
async def analyze(
    req: AnomalyAnalyzeRequest,
    request: Request,
) -> AnomalyAnalyzeResponse:
    pool, schema = _get_tenant_ctx(request)
    processing_notes: list[str] = []

    # ── Step 1: Ingest ──────────────────────────────────────────────────────
    try:
        df = ingestion.ingest_payload(req.data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    original_df = df.copy()

    # ── Step 2: Validate ────────────────────────────────────────────────────
    val_result = validation_agent.validate(df)
    if val_result.dataset_status == "INVALID":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset validation failed: {val_result.reason}",
        )

    processing_notes.extend(val_result.warnings)

    if val_result.duplicate_count > 0:
        processing_notes.append(f"Found {val_result.duplicate_count} duplicate rows (kept).")

    # ── Step 3: Normalize ───────────────────────────────────────────────────
    norm_df, norm_result = normalization.normalize(df, val_result)
    if norm_result.transform_log:
        processing_notes.append(f"Log-transformed {len(norm_result.transform_log)} skewed columns.")

    # ── Step 4: Build Features ──────────────────────────────────────────────
    feature_df, feat_result = feature_builder.build_features(norm_df, val_result)
    row_texts = feature_builder.build_row_texts(original_df)

    if feat_result.dropped_columns:
        processing_notes.append(f"Dropped {len(feat_result.dropped_columns)} low-signal columns.")

    # ── Step 5: Parse Intent ────────────────────────────────────────────────
    llm = get_llm_client()
    intent = await prompt_builder.parse_intent(
        user_request=req.user_request,
        available_columns=feat_result.feature_names,
        llm_client=llm,
        sensitivity_override=req.sensitivity,
    )
    processing_notes.append(f"Intent: {intent.anomaly_type} (source: {intent.intent_source}).")

    # ── Step 6: Embed ───────────────────────────────────────────────────────
    embeddings = embedding_layer.embed(row_texts)

    # ── Step 6b: Save embeddings to PostgreSQL ──────────────────────────────
    session_id = uuid.uuid4().hex
    try:
        await embedding_store.save_embeddings(
            pool=pool,
            schema=schema,
            session_id=session_id,
            row_texts=row_texts,
            embeddings=embeddings,
        )
    except Exception as exc:
        logger.warning("embedding_save_failed", error=str(exc))

    # ── Step 7: Vector Outlier Scores ───────────────────────────────────────
    try:
        vector_scores, vector_outliers, _ = vector_store.compute_vector_outlier_scores(embeddings)
    except Exception as exc:
        logger.warning("vector_store_failed_fallback", error=str(exc))
        vector_scores = [0.0] * len(row_texts)
        vector_outliers = []
        processing_notes.append("Vector similarity unavailable — Qdrant not reachable.")

    # ── Step 8: Detect ──────────────────────────────────────────────────────
    stat_scores, algorithm, selection_reason, _ = detection_engine.select_and_detect(
        feature_df=feature_df,
        intent=intent,
        latency_mode="batch",
    )
    processing_notes.append(selection_reason)

    # ── Step 9: Score (fuse signals) ────────────────────────────────────────
    scored = scoring_engine.fuse_scores(
        statistical_scores=stat_scores,
        vector_scores=vector_scores,
        vector_outlier_indices=vector_outliers,
    )

    # ── Step 10: Explain ────────────────────────────────────────────────────
    explained = await explanation_engine.explain_async(
        scored_results=scored,
        original_df=original_df,
        feature_df=feature_df,
        llm_client=llm,
    )

    # ── Build anomaly list ──────────────────────────────────────────────────
    anomalies = [AnomalyDetected(**r) for r in explained if r["anomaly_flag"]]
    anomalies.sort(key=lambda a: a.anomaly_score, reverse=True)

    # ── Optional narrative ──────────────────────────────────────────────────
    narrative: str | None = None
    if req.narrate:
        narrative = await explanation_engine.generate_narrative(
            anomalies=[a.model_dump() for a in anomalies],
            total_rows=len(df),
            llm_client=llm,
        )

    total_rows = len(df)
    anomaly_rate = round(len(anomalies) / total_rows, 4) if total_rows > 0 else 0.0

    # ── Save results to DB ──────────────────────────────────────────────────
    try:
        await result_store.save_session(
            pool=pool,
            schema=schema,
            session_id=session_id,
            source_name="json_payload",
            total_rows=total_rows,
            anomalies_found=len(anomalies),
            anomaly_rate=anomaly_rate,
            algorithm_used=algorithm,
            intent_type=intent.anomaly_type,
            narrative=narrative,
        )
        await result_store.save_alerts(
            pool=pool,
            schema=schema,
            session_id=session_id,
            anomalies=[a.model_dump() for a in anomalies],
        )
    except Exception as exc:
        logger.warning("db_save_failed", error=str(exc))

    logger.info(
        "analyze_complete",
        session_id=session_id,
        total_rows=total_rows,
        anomalies=len(anomalies),
        anomaly_rate=anomaly_rate,
        algorithm=algorithm,
    )

    return AnomalyAnalyzeResponse(
        status="success",
        total_rows=total_rows,
        anomalies_found=len(anomalies),
        anomaly_rate=anomaly_rate,
        anomalies=anomalies,
        narrative=narrative,
        algorithm_used=algorithm,
        model_used=llm.model_name,
        validation=val_result,
        intent=intent,
        processing_notes=processing_notes,
    )


@router.post(
    "/anomaly/predict",
    response_model=AnomalyPredictResponse,
    summary="Real-time single-record anomaly prediction",
    description=(
        "Fast statistical check on a single record. "
        "Uses Z-score (< 50ms latency). "
        "Provide reference_data for baseline context, or use raw column values."
    ),
)
async def predict(
    req: AnomalyPredictRequest,
) -> AnomalyPredictResponse:
    import pandas as pd

    llm = get_llm_client()

    row_df = pd.DataFrame([req.record])

    if req.reference_data:
        try:
            ref_df = ingestion.ingest_payload(req.reference_data)
            combined = pd.concat([ref_df, row_df], ignore_index=True)
        except Exception:
            combined = row_df
    else:
        combined = row_df

    val = validation_agent.validate(combined if len(combined) >= 10 else combined)
    if val.dataset_status == "INVALID" or not val.numeric_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot predict: no numeric columns found in record.",
        )

    norm_combined, _ = normalization.normalize(combined, val)
    feature_combined, _ = feature_builder.build_features(norm_combined, val)

    from services.detection_engine import _zscore_scores

    scores = _zscore_scores(feature_combined)

    target_score = float(scores[-1])
    anomaly_flag = target_score >= 0.65
    severity = "high" if target_score >= 0.80 else ("medium" if target_score >= 0.55 else "low")

    explained = explanation_engine.explain(
        scored_results=[
            {
                "row_index": len(feature_combined) - 1,
                "anomaly_score": target_score,
                "anomaly_flag": anomaly_flag,
                "severity": severity,
                "signal_breakdown": {"stat": target_score, "vector": 0.0, "rule": 0.0},
                "override_applied": target_score >= 0.90,
                "signal_sources": ["statistical"] if target_score >= 0.60 else [],
            }
        ],
        original_df=combined,
        feature_df=feature_combined,
    )
    record_result = explained[0]

    return AnomalyPredictResponse(
        anomaly=anomaly_flag,
        anomaly_score=round(target_score, 4),
        severity=severity,
        explanation=record_result.get("explanation_text", ""),
        signal_breakdown=record_result.get("signal_breakdown", {}),
        model_used=llm.model_name,
    )


@router.post(
    "/anomaly/upload-csv",
    response_model=AnomalyAnalyzeResponse,
    summary="Upload a CSV file and run full anomaly detection pipeline",
    description=(
        "Upload any CSV file — columns are auto-detected. "
        "Runs the same full pipeline as /analyze. Results saved to DB."
    ),
)
async def upload_csv(
    request: Request,
    file: UploadFile = File(..., description="CSV file to analyze"),
    narrate: bool = Form(True, description="Generate LLM narrative summary (true/false)."),
    user_request: str = Form(
        "",
        description=(
            "What to look for, e.g. 'detect fraud in payments' or 'find unusual visitor patterns'. "
            "Leave blank for automatic detection."
        ),
    ),
    sensitivity: float | None = Form(
        None,
        description=(
            "[Advanced / Optional] Override anomaly rate: 0.05 = ~5% flagged, 0.25 = ~25% of rows. "
            "Leave blank — algorithm auto-detects the right threshold from your data."
        ),
    ),
) -> AnomalyAnalyzeResponse:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only .csv files are supported.",
        )

    content = await file.read()
    try:
        df = ingestion.ingest_csv(content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV: {exc}",
        )

    from schemas.anomaly_schemas import DataPayload

    payload = DataPayload(
        columns=list(df.columns),
        rows=df.values.tolist(),
    )
    req = AnomalyAnalyzeRequest(
        data=payload,
        narrate=narrate,
        user_request=user_request.strip() or None,
        sensitivity=sensitivity if sensitivity > 0 else None,  # type: ignore[operator]
    )

    return await analyze(req, request)


@router.get(
    "/anomaly/sessions/{session_id}",
    summary="Fetch saved anomaly alerts for a session",
    description="Retrieve all anomalous records saved to DB for a specific session_id.",
)
async def get_session(
    session_id: str,
    request: Request,
) -> dict:
    pool, schema = _get_tenant_ctx(request)
    try:
        alerts = await result_store.get_session_alerts(
            pool=pool, schema=schema, session_id=session_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")
    return {"session_id": session_id, "alerts": alerts, "count": len(alerts)}


@router.get(
    "/anomaly/sessions",
    response_model=SessionsListResponse,
    summary="List all sessions with summarized stats and top alert previews",
    description=(
        "Returns a paginated list of all anomaly detection sessions. "
        "Each session includes: metadata, anomaly rate, severity breakdown "
        "(high/medium/low counts), narrative summary, and a preview of the "
        "top 3 highest-scoring anomalies. "
        "Filter by intent_type (fraud/security/operational/pattern/custom) "
        "or min_anomaly_rate to narrow results."
    ),
)
async def list_sessions(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    intent_type: str | None = None,
    min_anomaly_rate: float | None = None,
) -> SessionsListResponse:
    pool, schema = _get_tenant_ctx(request)
    try:
        sessions_raw, total = await asyncio.gather(
            result_store.get_all_sessions(
                pool=pool,
                schema=schema,
                limit=limit,
                offset=offset,
                intent_type=intent_type,
                min_anomaly_rate=min_anomaly_rate,
            ),
            result_store.get_sessions_count(
                pool=pool,
                schema=schema,
                intent_type=intent_type,
                min_anomaly_rate=min_anomaly_rate,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")

    sessions = [
        SessionSummary(
            **{k: v for k, v in s.items() if k != "top_alerts_preview"},
            top_alerts_preview=[
                AlertPreview(
                    row_index=a["row_index"],
                    anomaly_score=a["anomaly_score"],
                    severity=a["severity"],
                    explanation_text=a.get("explanation_text") or "",
                    top_features=(
                        json.loads(a["top_features"])
                        if isinstance(a.get("top_features"), str)
                        else (a.get("top_features") or [])
                    ),
                )
                for a in s.get("top_alerts_preview", [])
            ],
        )
        for s in sessions_raw
    ]

    logger.info("sessions_listed", total=total, returned=len(sessions), offset=offset)

    return SessionsListResponse(
        total=total,
        limit=limit,
        offset=offset,
        sessions=sessions,
    )
