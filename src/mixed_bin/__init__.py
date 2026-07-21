"""Offline visual search for apparel-picking robots.

Pairs an on-device Vision-Language embedder (MobileCLIP2) with a Qdrant Edge
multivector shard running in-process, so a robot arm can recognise a crumpled
garment in a mixed bin and pick it, entirely offline.
"""

from mixed_bin.config import Settings
from mixed_bin.embeddings import Embedder, FakeEmbedder, MobileClipEmbedder
from mixed_bin.index import BinIndex, EdgeShardIndex, SkuRecord, SkuHit

__all__ = [
    "Settings",
    "Embedder",
    "FakeEmbedder",
    "MobileClipEmbedder",
    "BinIndex",
    "EdgeShardIndex",
    "SkuRecord",
    "SkuHit",
]
