"""On-robot vector index over the warehouse catalog, backed by Qdrant Edge.

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

The backend is Qdrant Edge: an embedded search engine compiled into this
process, with no server, no network and no background optimizer threads. The
whole index is a directory on the robot's local disk. This is the same engine
that ships on the robot, not a stand-in for it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from qdrant_edge import (
    CountRequest,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    MultiVectorComparator,
    MultiVectorConfig,
    Point,
    Query,
    QueryRequest,
    UpdateOperation,
)


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
    """Interface the pick loop depends on, so tests can substitute a stub."""

    def build(self, records: Sequence[SkuRecord]) -> None: ...

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SkuHit]: ...


class EdgeShardIndex(BinIndex):
    """A Qdrant Edge shard holding one multivector point per SKU.

    Usage is two-phase, which mirrors how a fleet actually runs. A back-office
    machine calls `build()` to embed the catalog and write the shard, the shard
    is synced down to each robot, and the robot itself only ever calls
    `search()`, which reads the existing files off local disk.
    """

    def __init__(self, shard_path: str | Path, dim: int, vector_name: str = "vision"):
        self.shard_path = Path(shard_path)
        self.dim = dim
        self.vector_name = vector_name
        self._shard: EdgeShard | None = None

    def _config(self) -> EdgeConfig:
        # multivector_config is what makes this a late-interaction index: each
        # point carries a matrix of vectors rather than a single vector.
        return EdgeConfig(
            vectors={
                self.vector_name: EdgeVectorParams(
                    size=self.dim,
                    distance=Distance.Cosine,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MaxSim
                    ),
                )
            }
        )

    def build(self, records: Sequence[SkuRecord]) -> None:
        """Embed a catalog into a fresh shard, replacing anything already there."""
        self.close()
        # Edge refuses to create over existing segment data, and it expects the
        # target directory to exist already.
        if self.shard_path.exists():
            shutil.rmtree(self.shard_path)
        self.shard_path.mkdir(parents=True, exist_ok=True)

        shard = EdgeShard.create(str(self.shard_path), self._config())
        points = []
        for i, rec in enumerate(records):
            matrix = np.asarray(rec.view_vectors, dtype="float32")
            if matrix.ndim != 2 or matrix.shape[1] != self.dim:
                raise ValueError(
                    f"SKU {rec.sku}: expected (num_views, {self.dim}) matrix, "
                    f"got {matrix.shape}"
                )
            points.append(
                Point(
                    id=i,
                    # A list of lists is one multivector point.
                    vector={self.vector_name: matrix.tolist()},
                    payload={"sku": rec.sku, "title": rec.title, **rec.metadata},
                )
            )
        shard.update(UpdateOperation.upsert_points(points))
        shard.flush()
        self._shard = shard

    def load(self) -> None:
        """Open an existing shard from disk. This is all a robot does at boot."""
        if not self.shard_path.exists():
            raise FileNotFoundError(
                f"No Edge shard at {self.shard_path}. Build one first with "
                f"'mixed-bin build --catalog data/catalog'."
            )
        self._shard = EdgeShard.load(str(self.shard_path))

    @property
    def shard(self) -> EdgeShard:
        if self._shard is None:
            self.load()
        assert self._shard is not None
        return self._shard

    def count(self) -> int:
        return self.shard.count(CountRequest())

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SkuHit]:
        # A single crop is a one-row query matrix. MAX_SIM scores it against
        # every stored view of every SKU and keeps the best view per SKU.
        # Several rows also work, for instance multiple crops of one garment,
        # and Edge sums the per-row maxima into the late-interaction score.
        query = np.atleast_2d(np.asarray(query_vector, dtype="float32"))
        response = self.shard.query(
            QueryRequest(
                query=Query.Nearest(query.tolist(), using=self.vector_name),
                limit=top_k,
                with_payload=True,
            )
        )
        hits: list[SkuHit] = []
        for point in response:
            payload = point.payload or {}
            hits.append(
                SkuHit(
                    sku=payload.get("sku", "?"),
                    title=payload.get("title", ""),
                    score=float(point.score),
                    metadata={
                        k: v for k, v in payload.items() if k not in ("sku", "title")
                    },
                )
            )
        return hits

    def close(self) -> None:
        if self._shard is not None:
            self._shard.close()
            self._shard = None

    def __enter__(self) -> "EdgeShardIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
