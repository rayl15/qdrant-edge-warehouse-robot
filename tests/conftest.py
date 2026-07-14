import numpy as np
import pytest

from mixed_bin.embeddings import FakeEmbedder
from mixed_bin.index import SkuRecord


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)


@pytest.fixture
def catalog(embedder) -> list[SkuRecord]:
    """A tiny catalog: each SKU stored as three reference views."""
    skus = [
        ("SKU-1001", "navy crew tshirt"),
        ("SKU-1002", "black hoodie"),
        ("SKU-1003", "red polo"),
    ]
    records = []
    for sku, title in skus:
        views = np.stack([embedder.embed_label(sku, view=v) for v in (1, 2, 3)])
        records.append(SkuRecord(sku=sku, title=title, view_vectors=views))
    return records
