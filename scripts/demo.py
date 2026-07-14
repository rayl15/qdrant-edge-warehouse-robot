"""End-to-end demo with zero downloads.

Builds a small multivector catalog with the FakeEmbedder, then plans picks for a
simulated mixed bin. Proves the whole wiring (embed -> multivector index ->
sub-ms search -> grip point) without torch or model weights.

    python scripts/demo.py

Swap FakeEmbedder for MobileClipEmbedder and LabelledDetector for
YoloBinDetector to run the exact same code on real images at the edge.
"""

from __future__ import annotations

import tempfile
import time

import numpy as np
from PIL import Image

from mixed_bin.config import Settings
from mixed_bin.detector import Crop
from mixed_bin.embeddings import FakeEmbedder
from mixed_bin.index import LocalQdrantIndex, SkuRecord
from mixed_bin.search import MixedBinPicker

CATALOG = [
    ("SKU-1001", "navy crew tshirt"),
    ("SKU-1002", "black hoodie"),
    ("SKU-1003", "red polo"),
    ("SKU-1004", "white oxford shirt"),
]


class LabelledDetector:
    """Stand-in for YOLO: yields crops tagged with a known garment label."""

    def __init__(self, items):
        self.items = items  # (label, view, box)

    def detect(self, _bin_image):
        crops = []
        for label, view, box in self.items:
            img = Image.new("RGB", (64, 64))
            img.info.update(label=label, view=view)
            crops.append(Crop(image=img, box=box))
        return crops


def main() -> None:
    emb = FakeEmbedder(dim=64)

    # Each SKU is stored as several reference views (front / folded / crumpled).
    records = [
        SkuRecord(sku, title, np.stack([emb.embed_label(sku, view=v) for v in (1, 2, 3)]))
        for sku, title in CATALOG
    ]

    with tempfile.TemporaryDirectory() as tmp:
        index = LocalQdrantIndex(f"{tmp}/shard", "demo_catalog", dim=64)
        index.build(records)
        print(f"Indexed {len(records)} SKUs as multivector points.\n")

        # A messy tote: three garments, each seen from an unfamiliar angle (view 7/8/9).
        bin_items = [
            ("SKU-1003", 7, (20, 15, 180, 175)),
            ("SKU-1001", 8, (200, 40, 360, 200)),
            ("SKU-1004", 9, (60, 210, 220, 370)),
        ]
        picker = MixedBinPicker(
            LabelledDetector(bin_items), emb, index, Settings(min_confidence=0.3)
        )

        start = time.perf_counter()
        targets = picker.plan_picks(Image.new("RGB", (400, 400)))
        elapsed_ms = (time.perf_counter() - start) * 1000

        print("Pick plan (most confident first):")
        for t in targets:
            flag = "" if t.confident else "  [LOW CONF -> human]"
            print(f"  {t.sku:10} {t.title:20} score={t.confidence:.3f} grip={t.grip_point}{flag}")
        print(f"\nParsed + searched {len(targets)} garments in {elapsed_ms:.1f} ms (in-process).")


if __name__ == "__main__":
    main()
