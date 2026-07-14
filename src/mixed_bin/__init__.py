"""Offline visual search for apparel-picking robots.

Pairs an on-device Vision-Language embedder (MobileCLIP2) with an in-process
Qdrant multivector index so a robot arm can recognise a crumpled garment in a
mixed bin and pick it, entirely offline.
"""

from mixed_bin.config import Settings
from mixed_bin.embeddings import Embedder, FakeEmbedder, MobileClipEmbedder
from mixed_bin.index import BinIndex, LocalQdrantIndex, SkuRecord, SkuHit

__all__ = [
    "Settings",
    "Embedder",
    "FakeEmbedder",
    "MobileClipEmbedder",
    "BinIndex",
    "LocalQdrantIndex",
    "SkuRecord",
    "SkuHit",
]
