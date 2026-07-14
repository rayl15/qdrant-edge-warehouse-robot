"""The pick loop: bin image -> per-garment SKU + grip point.

This is the whole point of the system, and it is deliberately small. The robot
detects garments, embeds each crop with the local Vision model, and searches the
in-process index. Because retrieval never leaves the process, the search is the
cheapest step in the loop, not a 200-500 ms cloud round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from mixed_bin.config import Settings
from mixed_bin.detector import Crop
from mixed_bin.embeddings import Embedder
from mixed_bin.index import BinIndex, SkuHit


@dataclass
class PickTarget:
    sku: str
    title: str
    confidence: float
    grip_point: tuple[int, int]
    box: tuple[int, int, int, int]
    confident: bool  # False -> below threshold, escalate to a human


class Detector:  # structural type; SimpleBinDetector / YoloBinDetector both fit
    def detect(self, bin_image: Image.Image) -> list[Crop]: ...


class MixedBinPicker:
    def __init__(
        self,
        detector: Detector,
        embedder: Embedder,
        index: BinIndex,
        settings: Settings | None = None,
    ):
        self.detector = detector
        self.embedder = embedder
        self.index = index
        self.settings = settings or Settings()

    def plan_picks(self, bin_image: Image.Image) -> list[PickTarget]:
        """Parse the bin once and return an ordered pick list."""
        targets: list[PickTarget] = []
        for crop in self.detector.detect(bin_image):
            vector = self.embedder.embed_image(crop.image)
            hits = self.index.search(vector, top_k=self.settings.top_k)
            if not hits:
                continue
            best: SkuHit = hits[0]
            targets.append(
                PickTarget(
                    sku=best.sku,
                    title=best.title,
                    confidence=best.score,
                    grip_point=crop.grip_point,
                    box=crop.box,
                    confident=best.score >= self.settings.min_confidence,
                )
            )
        # Pick the most confident matches first.
        targets.sort(key=lambda t: t.confidence, reverse=True)
        return targets
