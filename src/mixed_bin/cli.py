"""Command-line entry point.

    mixed-bin build  --catalog data/catalog
    mixed-bin pick   --image  data/bins/tote_01.jpg

By default it uses the FakeEmbedder so it runs with zero downloads. Pass
--real to load MobileCLIP2 (needs the 'clip' extra).
"""

from __future__ import annotations

import argparse
import sys

from PIL import Image

from mixed_bin.catalog import build_records
from mixed_bin.config import Settings
from mixed_bin.detector import SimpleBinDetector
from mixed_bin.embeddings import Embedder, FakeEmbedder, MobileClipEmbedder
from mixed_bin.index import LocalQdrantIndex
from mixed_bin.search import MixedBinPicker


def _make_embedder(real: bool, settings: Settings) -> Embedder:
    if real:
        return MobileClipEmbedder(settings.model_name)
    return FakeEmbedder()


def _cmd_build(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    embedder = _make_embedder(args.real, settings)
    records = build_records(args.catalog, embedder)
    index = LocalQdrantIndex(settings.shard_path, settings.collection, embedder.dim)
    index.build(records)
    print(f"Indexed {len(records)} SKUs ({embedder.dim}-d) into {settings.shard_path}")
    return 0


def _cmd_pick(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    embedder = _make_embedder(args.real, settings)
    index = LocalQdrantIndex(settings.shard_path, settings.collection, embedder.dim)
    picker = MixedBinPicker(SimpleBinDetector(), embedder, index, settings)
    with Image.open(args.image) as bin_image:
        targets = picker.plan_picks(bin_image.copy())
    if not targets:
        print("No pickable items found.")
        return 0
    for t in targets:
        flag = "" if t.confident else "  [LOW CONF -> escalate]"
        print(f"{t.sku:12} {t.title:24} score={t.confidence:.3f} grip={t.grip_point}{flag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mixed-bin", description=__doc__)
    parser.add_argument("--real", action="store_true", help="Use MobileCLIP2 instead of FakeEmbedder")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Embed a catalog folder into the local shard")
    p_build.add_argument("--catalog", required=True)
    p_build.set_defaults(func=_cmd_build)

    p_pick = sub.add_parser("pick", help="Plan picks for one bin image")
    p_pick.add_argument("--image", required=True)
    p_pick.set_defaults(func=_cmd_pick)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
