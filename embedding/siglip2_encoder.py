"""
SigLIP2 encoder.

WHAT WAS WRONG
--------------
The previous version claimed to project 1152-dim SigLIP2 features into a
"unified" space with a linear layer:

    self.proj = nn.Linear(1152, 768, bias=False)
    self.proj.weight.data.zero_()
    for i in range(min(input_dim, output_dim)):     # range(768)
        self.proj.weight.data[i, i] = 1.0

For nn.Linear the weight is (out_features, in_features) = (768, 1152).
Setting weight[i, i] = 1 for i < 768 and zero everywhere else makes
output[j] = input[j]. That is not a projection -- it keeps the first 768
coordinates and discards the last 384. The layer had no gradient and was
never trained. SigLIP 2 is not Matryoshka-trained, so its coordinates carry
no ordering by importance: those 384 discarded dimensions were a third of
the model's actual signal, chosen arbitrarily.

The docstring said 1024, the comment said 1024, and UNIFIED_DIM was 768 --
a good sign nobody had revisited this since it was written.

NOW
---
Default is `native`: full 1152 dimensions, L2-normalised, no projection.
`pca` is available for a genuine dimensionality reduction fitted on real
embeddings. `truncate` is kept only so the old behaviour can be reproduced
for comparison; it is not a sane default.

The encoder also exposes `logit_scale` and `logit_bias` so downstream code
can work in the model's own logit space instead of raw cosine. This is what
makes reported confidence meaningful: a 0.28 cosine is a strong SigLIP
match, and only the model's temperature says so.
"""

import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from loguru import logger

# Import the standard open_clip, not the vendored TULIP fork, which can
# shadow the real package on sys.path.
_tulip_paths = [p for p in sys.path if "tulip" in p and "open_clip" in p]
for _p in _tulip_paths:
    sys.path.remove(_p)
import open_clip  # noqa: E402
sys.path.extend(_tulip_paths)

from config import settings  # noqa: E402

_MODE_NATIVE = "native"
_MODE_PCA = "pca"
_MODE_TRUNCATE = "truncate"


class _PCAProjection(nn.Module):
    """Real dimensionality reduction: (x - mean) @ components.T.

    Unlike truncation this keeps the directions of maximum variance, so
    cosine similarity in the reduced space tracks the original closely.
    Fit it with scripts/build_siglip2_pca.py and point SIGLIP2_PCA_PATH at
    the resulting .npz.
    """

    def __init__(self, components: np.ndarray, mean: np.ndarray):
        super().__init__()
        self.register_buffer("components", torch.from_numpy(components).float())
        self.register_buffer("mean", torch.from_numpy(mean).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) @ self.components.to(x.dtype).t()


class SigLIP2Encoder:
    """SigLIP2 image/text encoder with honest dimensionality handling."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Initializing SigLIP2 encoder on {self.device}")

        pretrained = settings.OPENCLIP_MODEL_DIR or settings.SIGLIP2_PRETRAINED
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=settings.SIGLIP2_MODEL_NAME,
            pretrained=pretrained,
            device=self.device,
        )
        if settings.USE_FP16 and "cuda" in self.device:
            self.model = self.model.half()
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(settings.SIGLIP2_MODEL_NAME)

        self.native_dim = int(settings.EMBEDDING_DIM_SIGLIP2)
        self.mode = str(getattr(settings, "SIGLIP2_PROJECTION_MODE", _MODE_NATIVE)).lower()
        self.projection: Optional[nn.Module] = None
        self._out_dim = self.native_dim

        if self.mode == _MODE_PCA:
            self._init_pca()
        elif self.mode == _MODE_TRUNCATE:
            self._out_dim = int(settings.EMBEDDING_DIM_SIGLIP2_STORED)
            logger.warning(
                f"SigLIP2 running in TRUNCATE mode: keeping the first {self._out_dim} of "
                f"{self.native_dim} coordinates and discarding the rest. This is lossy and "
                f"exists only for reproducing the old behaviour. Prefer 'native' or 'pca'."
            )
        elif self.mode != _MODE_NATIVE:
            logger.warning(f"Unknown SIGLIP2_PROJECTION_MODE={self.mode!r}; using native.")
            self.mode = _MODE_NATIVE

        # Model-native calibration constants. SigLIP is sigmoid-trained and
        # ships a bias; CLIP-family models have scale only.
        self.logit_scale, self.logit_bias = self._read_logit_params()

        logger.info(
            f"SigLIP2 ready: {settings.SIGLIP2_MODEL_NAME} mode={self.mode} "
            f"native_dim={self.native_dim} output_dim={self._out_dim} "
            f"logit_scale={self.logit_scale:.2f} logit_bias={self.logit_bias:.3f}"
        )

    # ------------------------------------------------------------------ #

    @property
    def output_dim(self) -> int:
        """Dimensionality actually written to the vector store."""
        return self._out_dim

    def _read_logit_params(self):
        from retrieval.scoring import get_logit_params
        return get_logit_params(self.model)

    def _init_pca(self):
        path = str(getattr(settings, "SIGLIP2_PCA_PATH", "") or "")
        try:
            data = np.load(path)
            components = np.asarray(data["components"], dtype=np.float32)
            mean = np.asarray(data["mean"], dtype=np.float32)
        except Exception as e:
            logger.error(
                f"SIGLIP2_PROJECTION_MODE=pca but the PCA file could not be loaded "
                f"({path!r}: {e}). Falling back to native {self.native_dim}-dim output."
            )
            self.mode = _MODE_NATIVE
            self._out_dim = self.native_dim
            return

        if components.shape[1] != self.native_dim:
            logger.error(
                f"PCA components have input dim {components.shape[1]}, expected "
                f"{self.native_dim}. Falling back to native."
            )
            self.mode = _MODE_NATIVE
            self._out_dim = self.native_dim
            return

        self.projection = _PCAProjection(components, mean).to(self.device)
        if settings.USE_FP16 and "cuda" in self.device:
            self.projection = self.projection.half()
        self._out_dim = int(components.shape[0])
        logger.info(f"SigLIP2 PCA projection loaded: {self.native_dim} -> {self._out_dim}")

    def _post(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the configured projection, then L2-normalise.

        Normalisation happens last in every mode so stored vectors are unit
        length and cosine similarity is a plain dot product.
        """
        if self.mode == _MODE_TRUNCATE:
            features = features[..., : self._out_dim]
        elif self.projection is not None:
            features = self.projection(features)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # ------------------------------------------------------------------ #

    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            t = self.preprocess(image).unsqueeze(0).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                t = t.half()
            f = self._post(self.model.encode_image(t))
        return f.float().cpu().numpy().astype(np.float32).flatten()

    def encode_images_batch(self, images: list) -> np.ndarray:
        with torch.no_grad():
            t = torch.stack([self.preprocess(i) for i in images]).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                t = t.half()
            f = self._post(self.model.encode_image(t))
        return f.float().cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            f = self._post(self.model.encode_text(tokens))
        return f.float().cpu().numpy().astype(np.float32).flatten()

    def encode_texts_batch(self, texts: list) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            f = self._post(self.model.encode_text(tokens))
        return f.float().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #

    def get_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Raw cosine. Kept for compatibility -- prefer `get_logit`."""
        return float(np.dot(image_features, text_features))

    def get_logit(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Cosine mapped into the model's trained logit space."""
        return float(np.dot(image_features, text_features)) * self.logit_scale + self.logit_bias

    def get_probability(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """SigLIP is sigmoid-trained, so sigmoid of the logit is the model's
        own estimate that this image and text are a matching pair."""
        z = self.get_logit(image_features, text_features)
        return float(1.0 / (1.0 + np.exp(-z))) if z >= 0 else float(np.exp(z) / (1.0 + np.exp(z)))
