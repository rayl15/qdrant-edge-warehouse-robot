"""Turn a folder of catalog images into SKU records.

Layout on disk:

    catalog/
      SKU-1001_navy-crew-tshirt/
        front.jpg
        back.jpg
        crumpled.jpg
      SKU-1002_black-hoodie/
        ...

Each subfolder is one SKU; every image in it is a reference view. We embed each
view and stack them into the (num_views, dim) matrix that the multivector index
expects. Adding a new season's line is just dropping in a new folder and
re-running: that is the zero-shot property in practice, no retraining.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mixed_bin.embeddings import Embedder
from mixed_bin.index import SkuRecord

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _title_from_dirname(name: str) -> tuple[str, str]:
    # "SKU-1001_navy-crew-tshirt" -> ("SKU-1001", "navy crew tshirt")
    sku, _, slug = name.partition("_")
    title = slug.replace("-", " ").strip() or sku
    return sku, title


def build_records(catalog_dir: str | Path, embedder: Embedder) -> list[SkuRecord]:
    root = Path(catalog_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Catalog dir not found: {root}")

    records: list[SkuRecord] = []
    for sku_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        view_paths = sorted(p for p in sku_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        if not view_paths:
            continue
        vectors = []
        for view_path in view_paths:
            with Image.open(view_path) as img:
                # Tag with the folder label so FakeEmbedder stays deterministic.
                img.info.setdefault("label", sku_dir.name)
                vectors.append(embedder.embed_image(img))
        sku, title = _title_from_dirname(sku_dir.name)
        records.append(
            SkuRecord(
                sku=sku,
                title=title,
                view_vectors=np.stack(vectors),
                metadata={"num_views": len(vectors)},
            )
        )
    if not records:
        raise ValueError(f"No SKU folders with images under {root}")
    return records
