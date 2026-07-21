import numpy as np

from mixed_bin.index import EdgeShardIndex, SkuRecord


def _unit(v):
    return (v / np.linalg.norm(v)).astype("float32")


def test_multivector_search_returns_correct_sku(tmp_path, embedder, catalog):
    index = EdgeShardIndex((tmp_path / "shard"), dim=64)
    index.build(catalog)

    # An unseen "view" of SKU-1002 should still retrieve SKU-1002 first.
    query = embedder.embed_label("SKU-1002", view=9)
    hits = index.search(query, top_k=3)

    assert hits[0].sku == "SKU-1002"
    assert hits[0].score > hits[1].score


def test_multivector_beats_single_view_for_deformed_item(tmp_path):
    """The article's core claim, as a test.

    A garment photographed 'front' looks different crumpled. If the catalog
    stores only the front view, a crumpled query can lose to an unrelated SKU.
    Storing multiple views and scoring with MAX_SIM recovers the right match.
    """
    dim = 64
    rng = np.random.default_rng(0)
    base_a = _unit(rng.standard_normal(dim))       # SKU-A identity
    deform = _unit(rng.standard_normal(dim))        # "crumpled" direction

    front_a = base_a
    crumpled_a = _unit(0.5 * base_a + 0.9 * deform)
    query = _unit(0.35 * base_a + 1.0 * deform)     # a crumpled crop of A

    # Distractor B: a different SKU whose front view happens to sit at cosine
    # ~0.6 to the query. That beats A's front view (~0.33) but loses to A's
    # crumpled view (~0.98). Built by mixing the query with an orthogonal axis.
    e = rng.standard_normal(dim)
    e = _unit(e - (e @ query) * query)              # orthogonal to the query
    front_b = _unit(0.6 * query + np.sqrt(1 - 0.6**2) * e)

    single = EdgeShardIndex((tmp_path / "single"), dim=dim)
    single.build([
        SkuRecord("SKU-A", "item a", np.stack([front_a])),
        SkuRecord("SKU-B", "item b", np.stack([front_b])),
    ])
    multi = EdgeShardIndex((tmp_path / "multi"), dim=dim)
    multi.build([
        SkuRecord("SKU-A", "item a", np.stack([front_a, crumpled_a])),
        SkuRecord("SKU-B", "item b", np.stack([front_b])),
    ])

    single_hits = single.search(query, top_k=2)
    multi_hits = multi.search(query, top_k=2)

    # Front-only index is fooled by the distractor; multivector picks A.
    assert single_hits[0].sku == "SKU-B"
    assert multi_hits[0].sku == "SKU-A"

    # And A scores strictly higher once its crumpled view is in the index.
    a_single = next(h.score for h in single_hits if h.sku == "SKU-A")
    a_multi = next(h.score for h in multi_hits if h.sku == "SKU-A")
    assert a_multi > a_single
