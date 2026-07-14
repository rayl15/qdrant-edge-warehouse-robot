"""Scene parsing: a bin image in, a list of garment crops out.

A mixed bin holds several deformable items. Before we can search, we have to
localise each candidate garment and hand the picker a physical grip point. The
detector returns `Crop`s: the sub-image plus the pixel-space centre of its
bounding box, which is the point the arm's inverse-kinematics targets.

`SimpleBinDetector` is dependency-free (it tiles the image) so the repo runs
anywhere. `YoloBinDetector` is the drop-in you'd actually deploy.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Crop:
    image: Image.Image
    # Bounding box in pixels: (left, top, right, bottom).
    box: tuple[int, int, int, int]

    @property
    def grip_point(self) -> tuple[int, int]:
        """Pixel-space centre of the box: where the arm goes."""
        left, top, right, bottom = self.box
        return (left + right) // 2, (top + bottom) // 2


class SimpleBinDetector:
    """No-ML fallback. Splits the bin into a grid of candidate regions.

    Useful for demos, tests, and sanity checks. It obviously over-generates
    crops (empty tiles included), but downstream confidence thresholding drops
    the ones that match nothing in the catalog.
    """

    def __init__(self, rows: int = 2, cols: int = 2):
        self.rows = rows
        self.cols = cols

    def detect(self, bin_image: Image.Image) -> list[Crop]:
        w, h = bin_image.size
        tile_w, tile_h = w // self.cols, h // self.rows
        crops: list[Crop] = []
        for r in range(self.rows):
            for c in range(self.cols):
                box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
                crop_img = bin_image.crop(box)
                # Carry any label metadata through so FakeEmbedder still works.
                crop_img.info.update(bin_image.info)
                crops.append(Crop(image=crop_img, box=box))
        return crops


class YoloBinDetector:
    """Real garment detection via Ultralytics YOLO.

    Install the optional extra:  pip install '.[detect]'
    Point `weights` at a model fine-tuned on your warehouse's garment classes,
    or start from a general checkpoint and let CLIP do the fine-grained SKU work.
    """

    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "YoloBinDetector needs the 'detect' extra: pip install '.[detect]'"
            ) from exc
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, bin_image: Image.Image) -> list[Crop]:
        results = self.model.predict(bin_image, conf=self.conf, verbose=False)
        crops: list[Crop] = []
        for result in results:
            for xyxy in result.boxes.xyxy.tolist():
                box = tuple(int(v) for v in xyxy)  # type: ignore[assignment]
                crop_img = bin_image.crop(box)
                crop_img.info.update(bin_image.info)
                crops.append(Crop(image=crop_img, box=box))  # type: ignore[arg-type]
        return crops
