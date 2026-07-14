"""Vision-Language embedders.

The pipeline only needs one thing from a model: turn an image (or a text prompt)
into a single L2-normalised vector in a shared space. That contract is the
`Embedder` protocol below.

Two implementations ship here:

* `MobileClipEmbedder` - the real thing. Apple's MobileCLIP2 via OpenCLIP.
  Runs offline on a Jetson-class board once the weights are cached locally.
* `FakeEmbedder` - a deterministic, dependency-free stand-in. It produces stable
  vectors keyed on a label so the whole pipeline (and CI) runs without torch or
  a single byte of downloaded weights. Same-label crops land near each other;
  different labels stay apart. Good enough to prove the retrieval wiring.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class Embedder(Protocol):
    """Anything that maps images/text to unit vectors of a fixed dimension."""

    dim: int

    def embed_image(self, image: Image.Image) -> np.ndarray: ...

    def embed_text(self, prompts: Sequence[str]) -> np.ndarray: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype("float32")


class MobileClipEmbedder:
    """Apple MobileCLIP2 through OpenCLIP.

    Install the optional extra first:  pip install '.[clip]'
    Weights download once to the HuggingFace cache, then the encoder is fully
    offline, which is the entire point for a warehouse Faraday cage.
    """

    def __init__(self, model_name: str = "hf-hub:timm/MobileCLIP2-S0-OpenCLIP", device: str | None = None):
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ImportError(
                "MobileClipEmbedder needs the 'clip' extra. Install with:\n"
                "    pip install '.[clip]'\n"
                "or use FakeEmbedder for a dependency-free run."
            ) from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # MobileCLIP has BatchNorm layers, so eval() is mandatory for stable output.
        self.model.eval().to(self.device)
        # Probe the output dimension once rather than hardcoding per variant.
        self.dim = int(self.embed_text(["_"]).shape[1])

    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            vec = self.model.encode_image(tensor)
        return _l2_normalise(vec.cpu().numpy())[0]

    def embed_text(self, prompts: Sequence[str]) -> np.ndarray:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        with self._torch.no_grad():
            vecs = self.model.encode_text(tokens)
        return _l2_normalise(vecs.cpu().numpy())


class FakeEmbedder:
    """Deterministic embedder for tests, demos, and CI. No model, no downloads.

    Vectors are derived from a label attached to each image (or from the prompt
    text). A small amount of seeded noise simulates view-to-view variation so
    multivector matching has something non-trivial to do.
    """

    def __init__(self, dim: int = 512, jitter: float = 0.015):
        # jitter is tuned so different views of the same label stay at ~0.95
        # cosine while different labels sit near 0 (near-orthogonal in high-d).
        self.dim = dim
        self.jitter = jitter

    def _seed(self, key: str) -> int:
        # hashlib, not builtin hash(): str hashing is salted per process.
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

    def _base_vector(self, label: str) -> np.ndarray:
        rng = np.random.default_rng(self._seed(label))
        return _l2_normalise(rng.standard_normal(self.dim).reshape(1, -1))[0]

    def embed_label(self, label: str, view: int = 0) -> np.ndarray:
        """Embed by label directly. Views of the same label are close but not identical."""
        base = self._base_vector(label)
        if view:
            rng = np.random.default_rng(self._seed(f"{label}#view{view}"))
            base = base + self.jitter * rng.standard_normal(self.dim)
        return _l2_normalise(base.reshape(1, -1))[0]

    def embed_image(self, image: Image.Image) -> np.ndarray:
        # A real crop has no label, so fall back to the label stashed in the
        # PIL image's `info` dict by the test/demo fixtures.
        label = image.info.get("label", "unknown")
        view = int(image.info.get("view", 0))
        return self.embed_label(label, view)

    def embed_text(self, prompts: Sequence[str]) -> np.ndarray:
        return np.stack([self._base_vector(p) for p in prompts])
