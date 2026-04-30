"""Step 6 — Embedding Layer.

Converts row_to_text descriptions into dense 384-dim vectors using
sentence-transformers/all-MiniLM-L6-v2 — fully OFFLINE, no API calls.

First run: model downloads automatically (~90MB).
Subsequent runs: served from local cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
from config import settings

# Resolve model path relative to the app root (/app in Docker, or src/../ locally)
_APP_ROOT = Path(__file__).resolve().parent.parent.parent

logger = structlog.get_logger()

EMBEDDING_DIM = 384

_model = None  # lazy-loaded singleton


def _get_model():
    global _model
    model_name = settings.embedding_model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("embedding_model_loading", model=model_name)
        model_path = _APP_ROOT / model_name
        if model_path.exists():
            model_name = str(model_path)
            _model = SentenceTransformer(model_name, local_files_only=True)
        else:
            # First run: download from HuggingFace, then served from cache
            _model = SentenceTransformer(model_name, local_files_only=False)
        logger.info("embedding_model_ready", model=model_name, dim=EMBEDDING_DIM)
    return _model


def embed(row_text_list: list[str]) -> np.ndarray:
    """Convert row text descriptions into dense embedding vectors.

    Args:
        row_text_list: One natural language sentence per dataset row.

    Returns:
        np.ndarray of shape (n_rows, 384) — one vector per row.
        Fully offline. No API call made.
    """
    if not row_text_list:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(
        row_text_list,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=True,  # unit vectors → cosine similarity = dot product
    )
    logger.info(
        "embeddings_generated",
        rows=len(row_text_list),
        shape=embeddings.shape,
        model=settings.embedding_model,
    )
    return embeddings.astype(np.float32)
