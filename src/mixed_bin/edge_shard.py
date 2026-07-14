"""Qdrant Edge (`EdgeShard`) adapter: the production edge deployment path.

Qdrant Edge is an embedded vector search engine compiled as a library that runs
*inside* your application process, with no background optimizer or update
threads and a footprint measured in single-digit megabytes. On a robot's
compute module it is what you'd ship instead of the standard client's local
mode.

STATUS: as of mid-2026 Qdrant Edge is in private beta and access is gated, so
`pip install qdrant-edge-py` is not open to everyone yet. This adapter mirrors
the API shown in Qdrant's on-device write-up so your code is ready the moment
you have beta access. It imports lazily and never runs in CI. Treat method
signatures as beta-stable, not frozen, and re-check against the docs:
https://qdrant.tech/documentation/edge/

The single-vector `Query.Nearest` form below is the one Qdrant documents for
Edge today. Multivector MAX_SIM on Edge specifically is not yet confirmed in the
public docs, so for a mixed bin you would either (a) collapse each SKU's views
to a mean vector for the Edge shard, or (b) keep the multivector index in local
mode until Edge exposes it. We do (a) here and note the trade-off.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mixed_bin.index import BinIndex, SkuHit, SkuRecord


class EdgeShardIndex(BinIndex):
    def __init__(self, shard_dir: str, dim: int, vector_name: str = "vision"):
        try:
            from qdrant_edge import EdgeShard  # noqa: F401
        except ImportError as exc:  # pragma: no cover - beta package, gated
            raise ImportError(
                "Qdrant Edge (qdrant-edge-py) is a gated beta package and is not "
                "installed. Request access at https://qdrant.tech/edge/ . Until "
                "then, use LocalQdrantIndex, which runs the same in-process idea "
                "with the standard client."
            ) from exc
        self._shard_dir = shard_dir
        self.dim = dim
        self.vector_name = vector_name
        self._shard = None  # created in build() / load()

    def build(self, records: Sequence[SkuRecord]) -> None:
        from qdrant_edge import EdgeShard, UpdateOperation
        from qdrant_client import models

        # One dense named vector per point. We mean-pool each SKU's views into a
        # single robust prototype vector (see module docstring for why).
        self._shard = EdgeShard.create(
            self._shard_dir,
            vectors={
                self.vector_name: models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                )
            },
        )
        points = []
        for i, rec in enumerate(records):
            proto = _mean_unit(np.asarray(rec.view_vectors, dtype="float32"))
            points.append(
                models.PointStruct(
                    id=i,
                    vector={self.vector_name: proto.tolist()},
                    payload={"sku": rec.sku, "title": rec.title, **rec.metadata},
                )
            )
        self._shard.update(UpdateOperation.upsert_points(points))
        self._shard.flush()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SkuHit]:
        from qdrant_edge import Query, QueryRequest

        if self._shard is None:
            from qdrant_edge import EdgeShard

            self._shard = EdgeShard.load(self._shard_dir)
        query = np.asarray(query_vector, dtype="float32")
        response = self._shard.query(
            QueryRequest(
                query=Query.Nearest(query.tolist(), using=self.vector_name),
                limit=top_k,
            )
        )
        hits: list[SkuHit] = []
        for point in response:
            payload = getattr(point, "payload", {}) or {}
            hits.append(
                SkuHit(
                    sku=payload.get("sku", "?"),
                    title=payload.get("title", ""),
                    score=float(getattr(point, "score", 0.0)),
                    metadata={k: v for k, v in payload.items() if k not in ("sku", "title")},
                )
            )
        return hits


def _mean_unit(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype("float32") if norm else mean.astype("float32")
