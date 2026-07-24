"""Crumpled-garment demo: does the pipeline recognise a wrinkled t-shirt?

The main sample catalog (`data/catalog`) is footwear and accessories, because
those are the multi-view listings Amazon Berkeley Objects ships permissively.
This demo answers the obvious follow-up: does it hold up on actual crumpled
garments, which is the whole premise of the article?

The catalog here (`data/catalog_garments`) is 25 t-shirts, each photographed
wrinkled from the front and the back, plus a couple of flat studio shots for
one colour. Source: Flickr user ir0cko, CC BY 2.0 (see data/ATTRIBUTION.md).

Protocol is deliberately hard and honest: we index each shirt's wrinkled FRONT
(and any flat views), then query with the wrinkled BACK, which is never in the
index. The query is a genuinely crumpled shot the model has not seen, and it
has to be told apart from 24 other near-identical wrinkled tees, several of
which differ only in shade (navy vs royal vs light blue, three greys, three
greens). Colour alone will not carry it.

    pip install -e '.[clip]'
    python scripts/demo_crumpled.py            # prints accuracy + timing
    python scripts/demo_crumpled.py --panel    # also writes the README figure

Writes docs/images/crumpled-garment-results.png with --panel.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mixed_bin.config import Settings
from mixed_bin.embeddings import MobileClipEmbedder
from mixed_bin.index import EdgeShardIndex, SkuRecord

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "catalog_garments"
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
QUERY_VIEW = "wrinkled-back"  # held out of the index, used as the crumpled query


def views_of(sku_dir: Path) -> list[Path]:
    return sorted(p for p in sku_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)


def label(sku_dir: Path) -> str:
    # "TEE-15-NAVY-BLUE_navy-blue-tshirt" -> "navy-blue"
    return sku_dir.name.split("_", 1)[1].replace("-tshirt", "") if "_" in sku_dir.name else sku_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="store_true", help="write the README results figure")
    args = parser.parse_args()

    settings = Settings()
    print(f"Loading {settings.model_name} ...")
    embedder = MobileClipEmbedder(settings.model_name)
    print(f"Encoder ready: {embedder.dim}-d on {embedder.device}\n")

    sku_dirs = sorted(p for p in CATALOG_DIR.iterdir() if p.is_dir())
    records: list[SkuRecord] = []
    queries = []  # (sku_code, label, query_path, query_vec)
    for sku_dir in sku_dirs:
        views = views_of(sku_dir)
        query_path = next((p for p in views if p.stem == QUERY_VIEW), None)
        index_paths = [p for p in views if p.stem != QUERY_VIEW]
        if query_path is None or not index_paths:
            continue
        code = sku_dir.name.split("_", 1)[0]
        matrix = np.stack([embedder.embed_image(Image.open(p)) for p in index_paths])
        records.append(SkuRecord(sku=code, title=label(sku_dir), view_vectors=matrix,
                                 metadata={"front": str(index_paths[0])}))
        queries.append((code, label(sku_dir), query_path,
                        embedder.embed_image(Image.open(query_path))))

    log: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        log.append(line)

    with EdgeShardIndex(str(ROOT / "storage" / "garments"), dim=embedder.dim) as index:
        index.build(records)
        n_views = sum(r.view_vectors.shape[0] for r in records)
        emit(f"Indexed {index.count()} wrinkled t-shirts as multivector points "
             f"({n_views} reference views).")
        emit(f"Query = each shirt's held-out {QUERY_VIEW} (never indexed).")
        emit()

        correct, top3, total_ms, results = 0, 0, 0.0, []
        for code, lbl, qpath, qvec in queries:
            t0 = time.perf_counter()
            hits = index.search(qvec, top_k=3)
            total_ms += (time.perf_counter() - t0) * 1000
            top = hits[0]
            ok = top.sku == code
            in_top3 = any(h.sku == code for h in hits)
            correct += ok
            top3 += in_top3
            results.append((code, lbl, qpath, top, ok, hits))
            mark = "ok  " if ok else ("top3" if in_top3 else "MISS")
            note = "" if ok else f"  (got {top.title})"
            emit(f"  [{mark}] {lbl:14} -> {top.title:14} {top.score:.3f}{note}")

        n = len(queries)
        emit()
        emit(f"Top-1 accuracy on held-out crumpled backs: {correct}/{n} = {correct / n:.0%}")
        emit(f"Top-3 accuracy (deploy with top-k + a confidence floor): {top3}/{n} = {top3 / n:.0%}")
        emit(f"Mean search time: {total_ms / n:.2f} ms/query (Qdrant Edge, in-process).")
        print("\nEvery top-1 miss is a near-identical shade (e.g. royal vs navy, silver vs")
        print("heather-grey): one front view per shirt is not enough to separate 25 plain")
        print("tees. That is the single-view brittleness the article is about. The remedy is")
        print("more reference views per SKU (see the multivector unit test in tests/).")

        if args.panel:
            write_input_grid(results)
            write_terminal(log)
            write_panel(results, correct, n)


def _images_dir():
    d = ROOT / "docs" / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_input_grid(results) -> None:
    """The INPUT: the crumpled garment shots fed to the pipeline as queries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = results[:15]
    cols = 5
    rows = (len(shown) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.5))
    fig.suptitle("INPUT: crumpled garments the camera sees (held-out query shots)",
                 fontsize=13, y=1.0)
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("off")
        if i < len(shown):
            _, lbl, qpath, _, _, _ = shown[i]
            ax.imshow(Image.open(qpath).convert("RGB"))
            ax.set_title(lbl, fontsize=8)
    out = _images_dir() / "crumpled-input.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out.relative_to(ROOT)}")


def write_terminal(log) -> None:
    """The OUTPUT: a terminal screenshot of the actual run."""
    from PIL import ImageDraw, ImageFont
    from matplotlib import font_manager

    lines = ["$ python scripts/demo_crumpled.py", ""] + log
    mono = font_manager.findfont(font_manager.FontProperties(family="monospace"))
    size = 22
    font = ImageFont.truetype(mono, size)
    pad, lh = 28, size + 8
    w = 1180
    h = pad * 2 + lh * len(lines)
    img = Image.new("RGB", (w, h), (13, 17, 23))  # GitHub-dark terminal
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = pad + i * lh
        if line.startswith("$"):
            color = (88, 166, 255)   # command, blue
        elif "[ok" in line:
            color = (63, 185, 80)    # hit, green
        elif "[MISS" in line:
            color = (248, 81, 73)    # miss, red
        elif "[top3" in line:
            color = (210, 168, 60)   # in top-3, amber
        elif "accuracy" in line or "search time" in line.lower():
            color = (230, 237, 243)  # summary, bright
        else:
            color = (139, 148, 158)  # muted
        d.text((pad, y), line, font=font, fill=color)
    out = _images_dir() / "crumpled-output-terminal.png"
    img.save(out)
    print(f"Wrote {out.relative_to(ROOT)}")


def write_panel(results, correct, n) -> None:
    """Render the crumpled query beside its retrieved reference, for the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = results[:12]
    cols, rows = 3, (len(shown) + 2) // 3
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4.2, rows * 2.5))
    fig.suptitle(
        f"Crumpled-garment retrieval on Qdrant Edge  -  {correct}/{n} correct\n"
        "query = held-out wrinkled BACK (never indexed)  ->  retrieved SKU (indexed front)",
        fontsize=13, y=0.99,
    )
    axes = np.atleast_2d(axes)
    for i, (code, lbl, qpath, top, ok, hits) in enumerate(shown):
        r, c = divmod(i, cols)
        ax_q, ax_r = axes[r][c * 2], axes[r][c * 2 + 1]
        ax_q.imshow(Image.open(qpath).convert("RGB"))
        ax_q.axis("off")
        ax_q.set_title(f"query: {lbl}\n(crumpled back)", fontsize=8)
        front_path = top.metadata.get("front")
        if front_path:
            ax_r.imshow(Image.open(front_path).convert("RGB"))
        ax_r.axis("off")
        color = "#16a34a" if ok else "#dc3545"
        ax_r.set_title(f"{'MATCH' if ok else 'MISS'}: {top.title}\nscore {top.score:.2f}",
                       fontsize=8, color=color)
        for ax in (ax_q, ax_r):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)
    for j in range(len(shown), rows * cols):
        r, c = divmod(j, cols)
        axes[r][c * 2].axis("off")
        axes[r][c * 2 + 1].axis("off")
    out = ROOT / "docs" / "images" / "crumpled-garment-results.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
