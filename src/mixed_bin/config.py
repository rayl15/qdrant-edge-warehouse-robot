"""Central configuration.

Everything a robot's compute module needs to know is here: which encoder to
load, where the on-disk shard lives, and how many candidates to return per
garment. Override via environment variables so the same image runs on a laptop
and on a Jetson without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # OpenCLIP model tag. MobileCLIP2-S0 (512-d) is the edge sweet spot:
    # ~11M image-encoder params, single-digit-ms encode, 512-d vectors that
    # are cheap to index. Step up to MobileCLIP2-L-14 (768-d) only with
    # Orin-class headroom.
    model_name: str = "hf-hub:timm/MobileCLIP2-S0-OpenCLIP"

    # Where the in-process Qdrant shard is persisted on the robot's local disk.
    shard_path: str = "storage/mixed_bin"
    collection: str = "warehouse_catalog"

    # How many SKU candidates to retrieve for each detected garment crop.
    top_k: int = 5

    # Cosine similarity below this is treated as "unknown item, ask a human"
    # rather than a confident pick. Tune against your own catalog.
    min_confidence: float = 0.25

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            model_name=os.environ.get("MIXED_BIN_MODEL", Settings.model_name),
            shard_path=os.environ.get("MIXED_BIN_SHARD", Settings.shard_path),
            collection=os.environ.get("MIXED_BIN_COLLECTION", Settings.collection),
            top_k=int(os.environ.get("MIXED_BIN_TOP_K", Settings.top_k)),
            min_confidence=float(
                os.environ.get("MIXED_BIN_MIN_CONFIDENCE", Settings.min_confidence)
            ),
        )
