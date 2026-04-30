"""Step 7 — Vector Storage & Similarity Search using Qdrant.

Stores embeddings and identifies semantically unusual records using
nearest-neighbour cosine similarity search.

Rows whose vector is far from all others → semantic outliers.
"""

from __future__ import annotations

import uuid

import numpy as np
import structlog
from config import settings

logger = structlog.get_logger()

_COLLECTION_PREFIX = "anomaly"
_K_NEIGHBOURS = 5
_OUTLIER_PERCENTILE = 95  # top 5% most isolated = outlier candidates


def _get_client():
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def compute_vector_outlier_scores(
    embeddings: np.ndarray,
    collection_suffix: str | None = None,
) -> tuple[list[float], list[int], float]:
    """Store embeddings in Qdrant, compute cosine distance to K nearest neighbours.

    High average distance to neighbours → semantically unusual → outlier.

    Returns:
        similarity_scores: per-row average distance to K nearest neighbours (0=similar, 1=isolated)
        outlier_candidates: row indices above 95th percentile
        threshold: the 95th percentile distance value used
    """
    from qdrant_client.models import Distance, PointStruct, VectorParams

    n_rows, dim = embeddings.shape
    suffix = collection_suffix or uuid.uuid4().hex[:8]
    collection_name = f"{_COLLECTION_PREFIX}_{suffix}"

    client = _get_client()

    # Create fresh temporary collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Upload all row vectors
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={"row_index": i},
        )
        for i in range(n_rows)
    ]
    client.upsert(collection_name=collection_name, points=points)

    # For each row: find K nearest neighbours, compute average cosine distance
    similarity_scores: list[float] = []
    for i in range(n_rows):
        results = client.search(
            collection_name=collection_name,
            query_vector=embeddings[i].tolist(),
            limit=_K_NEIGHBOURS + 1,  # +1 because self is always returned first
        )
        # Exclude self (score ≈ 1.0), take up to K remaining
        neighbour_scores = [r.score for r in results if r.score < 0.9999][:_K_NEIGHBOURS]
        if neighbour_scores:
            # Distance = 1 - similarity (higher = more isolated)
            avg_distance = 1.0 - (sum(neighbour_scores) / len(neighbour_scores))
        else:
            avg_distance = 1.0
        similarity_scores.append(round(avg_distance, 6))

    # Threshold at 95th percentile
    threshold = float(np.percentile(similarity_scores, _OUTLIER_PERCENTILE))
    outlier_candidates = [i for i, s in enumerate(similarity_scores) if s >= threshold]

    # Cleanup temporary collection
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    logger.info(
        "vector_outliers_computed",
        total_rows=n_rows,
        outliers_found=len(outlier_candidates),
        threshold=round(threshold, 4),
        k=_K_NEIGHBOURS,
    )
    return similarity_scores, outlier_candidates, threshold
