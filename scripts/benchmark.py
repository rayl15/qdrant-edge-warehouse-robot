"""Measure Qdrant Edge multivector search latency on this machine.

The claim the whole architecture rests on is that retrieval stops being the
bottleneck once it runs in-process. This script is how you check that on your
own hardware rather than taking anyone's word for it, including ours.

It builds a synthetic catalog of the size you ask for, then times warm queries
against the Edge shard. No model weights, no network, so it runs anywhere.

    python scripts/benchmark.py                    # 500 SKUs x 4 views, 512-d
    python scripts/benchmark.py --skus 5000 --dim 512

On a Jetson-class board expect the numbers to be a few times slower than a
laptop, and still far inside a control-loop budget.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time

import numpy as np

from mixed_bin.embeddings import FakeEmbedder
from mixed_bin.index import EdgeShardIndex, SkuRecord


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skus", type=int, default=500, help="catalog size")
    parser.add_argument("--views", type=int, default=4, help="reference views per SKU")
    parser.add_argument("--dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--queries", type=int, default=500, help="timed queries")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    emb = FakeEmbedder(dim=args.dim)
    records = [
        SkuRecord(
            sku=f"SKU-{i:05d}",
            title="synthetic garment",
            view_vectors=np.stack(
                [emb.embed_label(f"SKU-{i:05d}", view=v) for v in range(1, args.views + 1)]
            ),
        )
        for i in range(args.skus)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        with EdgeShardIndex(f"{tmp}/shard", dim=args.dim) as index:
            start = time.perf_counter()
            index.build(records)
            build_ms = (time.perf_counter() - start) * 1000

            total_vectors = args.skus * args.views
            print(
                f"Catalog: {index.count()} SKUs x {args.views} views "
                f"= {total_vectors} vectors, {args.dim}-d"
            )
            print(f"Shard build: {build_ms:.0f} ms")

            # Warm the first query out of the measurement: it pays one-off
            # setup costs a long-running robot process pays only at boot.
            index.search(emb.embed_label("SKU-00000", view=99), top_k=args.top_k)

            timings = []
            for i in range(args.queries):
                query = emb.embed_label(f"SKU-{i % args.skus:05d}", view=99)
                start = time.perf_counter()
                index.search(query, top_k=args.top_k)
                timings.append((time.perf_counter() - start) * 1000)

            timings.sort()
            p95 = timings[int(0.95 * len(timings))]
            print(
                f"Search over {args.queries} warm queries: "
                f"mean {statistics.mean(timings):.3f} ms | "
                f"median {statistics.median(timings):.3f} ms | "
                f"p95 {p95:.3f} ms | max {timings[-1]:.3f} ms"
            )
            print("\nFor comparison, a cloud round-trip from a warehouse floor is 200-500 ms,")
            print("and that is when the Wi-Fi is working.")


if __name__ == "__main__":
    main()
