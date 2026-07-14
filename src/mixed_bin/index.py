"""In-process vector index over the warehouse catalog.

The key modelling decision: a SKU is not one vector, it is a *set* of vectors.

Apparel is deformable. A single product-shot embedding is brittle: the same
t-shirt crumpled in a tote, half-occluded, or stuffed in a reflective polybag
lands far from its pristine catalog photo. So we store each SKU as several
reference views (front, back, folded, crumpled, on-hanger) and let Qdrant's
multivector MAX_SIM late-interaction score decide the match.

MAX_SIM, for each query vector, takes the maximum similarity across all of a
SKU's stored views, then sums. A crumpled crop simply lights up whichever
stored view is closest. That single choice is what turns "matches the catalog
photo" into "matches the item, however it happens to be lying in the bin."

`LocalQdrantIndex` uses the standard client's in-process local mode
(`QdrantClient(path=...)`): no server, no network, runs today. On a robot you
would swap in Qdrant Edge's `EdgeShard` runtime (see edge_shard.py) which is the
same idea with a smaller footprint and no background threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient, models


@dataclass
class SkuRecord:
    sku: str
    title: str
    # One row per reference view. Shape: (num_views, dim).
    view_vectors: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class SkuHit:
    sku: str
    title: str
    score: float
    metadata: dict


class BinIndex:
    """Interface implemented by every backend (local mode, Edge, ...)."""

    def build(self, records: Sequence[SkuRecord]) -> None: ...

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SkuHit]: ...


class LocalQdrantIndex(BinIndex):
    def __init__(self, path: str, collection: str, dim: int):
        # `path` -> on-disk, embedded, single-process Qdrant. ":memory:" also works.
        self.client = QdrantClient(path=path)
        self.collection = collection
        self.dim = dim

    def build(self, records: Sequence[SkuRecord]) -> None:
        # MAX_SIM comparator is what makes this a late-interaction / multivector
        # collection: each point carries a matrix of vectors, not one vector.
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        points = []
        for i, rec in enumerate(records):
            matrix = np.asarray(rec.view_vectors, dtype="float32")
            if matrix.ndim != 2 or matrix.shape[1] != self.dim:
                raise ValueError(
                    f"SKU {rec.sku}: expected (num_views, {self.dim}) matrix, "
                    f"got {matrix.shape}"
                )
            points.append(
                models.PointStruct(
                    id=i,
                    vector=matrix.tolist(),  # list-of-vectors == multivector
                    payload={"sku": rec.sku, "title": rec.title, **rec.metadata},
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SkuHit]:
        query = np.asarray(query_vector, dtype="float32")
        # A single crop is a one-row query matrix; MAX_SIM scores it against
        # every stored view of every SKU and keeps the best per SKU.
        result = self.client.query_points(
            collection_name=self.collection,
            query=[query.tolist()],
            limit=top_k,
            with_payload=True,
        )
        hits: list[SkuHit] = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                SkuHit(
                    sku=payload.get("sku", "?"),
                    title=payload.get("title", ""),
                    score=float(point.score),
                    metadata={k: v for k, v in payload.items() if k not in ("sku", "title")},
                )
            )
        return hits
