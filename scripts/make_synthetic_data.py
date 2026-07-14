"""Generate a synthetic on-disk catalog so the `mixed-bin` CLI runs out of the box.

Creates folders named by SKU (the FakeEmbedder keys off the folder name), each
with a few solid-colour 'view' images, plus a couple of dummy bin images.

    python scripts/make_synthetic_data.py
    mixed-bin build --catalog data/catalog
    mixed-bin pick  --image  data/bins/tote_01.png    # with --real for MobileCLIP

These are placeholders. For a real run, drop actual product photos into
data/catalog/<SKU>_<slug>/ and real tote photos into data/bins/.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

CATALOG = {
    "SKU-1001_navy-crew-tshirt": (18, 32, 74),
    "SKU-1002_black-hoodie": (24, 24, 24),
    "SKU-1003_red-polo": (150, 30, 40),
    "SKU-1004_white-oxford-shirt": (235, 235, 230),
}
VIEWS = ["front", "folded", "crumpled"]
ROOT = Path("data")


def main() -> None:
    for name, colour in CATALOG.items():
        d = ROOT / "catalog" / name
        d.mkdir(parents=True, exist_ok=True)
        for i, view in enumerate(VIEWS):
            # Slightly vary brightness per view so they are not identical.
            shade = tuple(min(255, c + i * 12) for c in colour)
            Image.new("RGB", (256, 256), shade).save(d / f"{view}.png")

    bins = ROOT / "bins"
    bins.mkdir(parents=True, exist_ok=True)
    for n in (1, 2):
        Image.new("RGB", (512, 512), (60, 60, 60)).save(bins / f"tote_0{n}.png")

    print(f"Wrote {len(CATALOG)} SKUs and 2 bins under {ROOT}/")


if __name__ == "__main__":
    main()
