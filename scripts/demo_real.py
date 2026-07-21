"""Real-image demo: MobileCLIP2 + Qdrant multivector on the sample ABO catalog.

For each product we index all views but one, then query with the held-out view
(often a very different shot, e.g. on-model instead of studio) and check that
multivector MAX_SIM still retrieves the right product.

    pip install -e '.[clip]'
    python scripts/make_synthetic_data.py   # not needed; real images ship in data/catalog
    python scripts/demo_real.py

Needs the 'clip' extra (torch + open_clip). Weights download once, then offline.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from mixed_bin.config import Settings
from mixed_bin.embeddings import MobileClipEmbedder
from mixed_bin.index import EdgeShardIndex, SkuRecord

CATALOG_DIR = Path(__file__).resolve().parent.parent / "data" / "catalog"
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_views(sku_dir: Path) -> list[Path]:
    return sorted(p for p in sku_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)


def main() -> None:
    settings = Settings()
    print(f"Loading {settings.model_name} ...")
    embedder = MobileClipEmbedder(settings.model_name)
    print(f"Encoder ready: {embedder.dim}-d on {embedder.device}\n")

    sku_dirs = sorted(p for p in CATALOG_DIR.iterdir() if p.is_dir())
    records, queries = [], []
    for sku_dir in sku_dirs:
        views = load_views(sku_dir)
        if len(views) < 3:
            continue
        catalog_views, held_out = views[:-1], views[-1]
        mat = np.stack([embedder.embed_image(Image.open(p)) for p in catalog_views])
        sku, _, slug = sku_dir.name.partition("_")
        records.append(SkuRecord(sku=sku, title=slug.replace("-", " "), view_vectors=mat))
        queries.append((sku, held_out, embedder.embed_image(Image.open(held_out))))

    index = EdgeShardIndex(settings.shard_path, embedder.dim, settings.vector_name)
    index.build(records)
    print(f"Indexed {len(records)} products as multivector points "
          f"({sum(len(load_views(d)) - 1 for d in sku_dirs if len(load_views(d)) >= 3)} catalog views).\n")

    correct, total_ms = 0, 0.0
    print("Held-out-view retrieval (query view was NOT in the index):")
    for true_sku, held_out, qvec in queries:
        start = time.perf_counter()
        hits = index.search(qvec, top_k=3)
        total_ms += (time.perf_counter() - start) * 1000
        top = hits[0]
        ok = top.sku == true_sku
        correct += ok
        mark = "ok " if ok else "MISS"
        print(f"  [{mark}] {held_out.parent.name:42} -> {top.sku} "
              f"'{top.title}' ({top.score:.3f})")

    n = len(queries)
    print(f"\nTop-1 accuracy on held-out views: {correct}/{n} = {correct / n:.0%}")
    print(f"Mean search time: {total_ms / n:.2f} ms/query (in-process, exact).")


if __name__ == "__main__":
    main()
