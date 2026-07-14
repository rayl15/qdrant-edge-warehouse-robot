from PIL import Image

from mixed_bin.config import Settings
from mixed_bin.detector import Crop, SimpleBinDetector
from mixed_bin.index import LocalQdrantIndex
from mixed_bin.search import MixedBinPicker


class LabelledDetector:
    """Test detector that returns crops tagged with a known label/view."""

    def __init__(self, items):
        self.items = items  # list of (label, view, box)

    def detect(self, bin_image):
        crops = []
        for label, view, box in self.items:
            img = Image.new("RGB", (32, 32))
            img.info["label"] = label
            img.info["view"] = view
            crops.append(Crop(image=img, box=box))
        return crops


def test_picker_matches_each_garment(tmp_path, embedder, catalog):
    index = LocalQdrantIndex(str(tmp_path / "shard"), "cat", dim=64)
    index.build(catalog)

    detector = LabelledDetector([
        ("SKU-1001", 5, (0, 0, 100, 100)),
        ("SKU-1003", 6, (100, 0, 200, 100)),
    ])
    picker = MixedBinPicker(detector, embedder, index, Settings(min_confidence=0.3))
    targets = picker.plan_picks(Image.new("RGB", (200, 100)))

    found = {t.sku for t in targets}
    assert found == {"SKU-1001", "SKU-1003"}
    assert all(t.confident for t in targets)
    # Grip point is the centre of the bounding box.
    first = next(t for t in targets if t.sku == "SKU-1001")
    assert first.grip_point == (50, 50)


def test_unknown_item_is_low_confidence(tmp_path, embedder, catalog):
    index = LocalQdrantIndex(str(tmp_path / "shard"), "cat", dim=64)
    index.build(catalog)

    detector = LabelledDetector([("NOT-IN-CATALOG", 1, (0, 0, 10, 10))])
    picker = MixedBinPicker(detector, embedder, index, Settings(min_confidence=0.5))
    targets = picker.plan_picks(Image.new("RGB", (10, 10)))

    assert len(targets) == 1
    assert not targets[0].confident  # escalate to a human


def test_simple_detector_tiles_the_bin():
    crops = SimpleBinDetector(rows=2, cols=2).detect(Image.new("RGB", (200, 100)))
    assert len(crops) == 4
    assert crops[0].box == (0, 0, 100, 50)
