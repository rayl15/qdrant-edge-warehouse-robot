# Mixed Bin: Offline Visual Search for Apparel Robots

Recognise a crumpled garment in a mixed bin and pick it, entirely offline.

A robotic picking arm in a fashion fulfillment center cannot wait 200-500 ms for
a cloud round-trip, and the warehouse is a Faraday cage of steel shelving with
dead Wi-Fi anyway. So both the vision model and the vector search have to live
on the robot. This repo pairs an on-device Vision-Language encoder
([Apple MobileCLIP2](https://github.com/apple/ml-mobileclip)) with an in-process
[Qdrant](https://qdrant.tech/) multivector index. No server, no network, no GPU
required to try it.

It accompanies the article *"Solving the Mixed Bin: Offline Visual Search for
Apparel Robotics."*

---

## The idea in one paragraph

Apparel is deformable, so a single product-shot embedding is brittle: the same
t-shirt crumpled in a tote lands far from its pristine catalog photo. We store
each SKU as **several reference views** (front, back, folded, crumpled) and let
Qdrant's multivector **MAX_SIM** late-interaction score decide the match. For
each query crop, MAX_SIM takes the best-matching stored view per SKU, so a
crumpled crop simply lights up whichever view is closest. That is the difference
between "matches the catalog photo" and "matches the item, however it is lying."

## Quickstart (no downloads, no GPU)

```bash
git clone https://github.com/rayl15/qdrant-edge-warehouse-robot.git
cd qdrant-edge-warehouse-robot
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

python scripts/demo.py     # build a catalog, plan picks for a simulated bin
pytest                     # includes the multivector-vs-single-view test
```

The demo runs on a deterministic `FakeEmbedder` so nothing is downloaded. The
pipeline code is identical to the real path; only the embedder and detector swap.

## Run it for real (MobileCLIP2 on real images)

The repo ships a small sample catalog of real product photos under
`data/catalog/` (9 items, several views each, from Amazon Berkeley Objects, CC
BY 4.0, see `data/ATTRIBUTION.md`). `scripts/demo_real.py` indexes all views but
one per item, then queries with the held-out view to show multivector retrieval
identifies the product from a shot it never indexed:

```bash
pip install -e '.[clip]'          # torch + open_clip + transformers; weights cache once
python scripts/demo_real.py
```

Real output on the sample catalog (CPU laptop):

```
Encoder ready: 512-d on cpu
Indexed 9 products as multivector points (25 catalog views).
Held-out-view retrieval (query view was NOT in the index):
  [ok ] ABO-001_amazon-merk-vinden-dames  -> ABO-001 (0.829)
  ...
Top-1 accuracy on held-out views: 9/9 = 100%
Mean search time: 0.33 ms/query (in-process, exact).
```

Note on the sample data: ABO's permissively-licensed multi-view listings are
footwear and accessories, so the sample is shoes, bags, hats, watches and
sunglasses rather than garments. The pipeline is identical for apparel, only the
reference photos differ. Drop your own product folders into `data/catalog/` to
use it on garments.

Default encoder is `MobileCLIP2-S0` (512-d, single-digit-ms image encode, the
edge sweet spot). Override with `MIXED_BIN_MODEL`; step up to
`MobileCLIP2-L-14` (768-d) with Orin-class headroom.

## How it fits together

```
bin image ─▶ Detector ─▶ crop ─▶ MobileCLIP2 ─▶ vector ─▶ Qdrant MAX_SIM ─▶ SKU + grip point
            (YOLO)                (VLM, on-device)         (in-process, multivector)
```

| Module | Responsibility |
| --- | --- |
| `embeddings.py` | `MobileClipEmbedder` (real) and `FakeEmbedder` (deps-free) behind one `Embedder` protocol |
| `detector.py` | Bin image to garment crops + pixel-space grip points (`SimpleBinDetector`, `YoloBinDetector`) |
| `index.py` | `LocalQdrantIndex`: in-process multivector collection, each SKU a matrix of views |
| `edge_shard.py` | `EdgeShardIndex`: the Qdrant Edge deployment path (beta, gated) |
| `catalog.py` | A folder of SKU images to multivector records |
| `search.py` | The pick loop: detect to embed to search to ranked pick plan |

## Deploying on the robot: Qdrant Edge

`LocalQdrantIndex` already runs Qdrant in-process. On a Jetson-class board you
would swap in [**Qdrant Edge**](https://qdrant.tech/documentation/edge/), an
embedded engine compiled into your process with no background threads and a
single-digit-MB footprint. `edge_shard.py` mirrors its `EdgeShard` API so the
code is ready. Qdrant Edge is in **private beta** as of mid-2026, so the package
is gated: request access at <https://qdrant.tech/edge/>. Everything else here
runs today without it.

## What is deliberately simple

- `SimpleBinDetector` tiles the image instead of detecting garments. Install the
  `detect` extra and use `YoloBinDetector` for real bounding boxes.
- `FakeEmbedder` is for tests and demos. It is not a model; it just gives stable,
  separable vectors so the retrieval wiring is provable in CI.
- Latency numbers in the article for Qdrant Edge itself (~0.1 ms on-device) are
  Qdrant's own published figures, not measured here. `scripts/demo.py` prints the
  real local-mode timing on your machine.

## License

MIT. MobileCLIP2 weights are under Apple's ML Research Model license; review it
before commercial deployment.
