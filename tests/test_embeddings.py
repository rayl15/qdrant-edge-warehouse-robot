import numpy as np

from mixed_bin.embeddings import FakeEmbedder


def _cos(a, b):
    return float(np.dot(a, b))  # inputs are already unit vectors


def test_vectors_are_unit_and_right_shape():
    emb = FakeEmbedder(dim=64)
    v = emb.embed_label("SKU-1001", view=1)
    assert v.shape == (64,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_same_label_views_are_close():
    emb = FakeEmbedder(dim=64)
    v1 = emb.embed_label("SKU-1001", view=1)
    v2 = emb.embed_label("SKU-1001", view=2)
    assert _cos(v1, v2) > 0.9


def test_different_labels_are_far():
    emb = FakeEmbedder(dim=64)
    a = emb.embed_label("SKU-1001", view=1)
    b = emb.embed_label("SKU-1002", view=1)
    assert _cos(a, b) < 0.4


def test_deterministic_across_instances():
    a = FakeEmbedder(dim=64).embed_label("SKU-1003", view=2)
    b = FakeEmbedder(dim=64).embed_label("SKU-1003", view=2)
    assert np.allclose(a, b)
