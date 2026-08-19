"""
SigLIP2 Encoder for image and text embeddings.
Uses OpenCLIP with ViT-SO400M-14-SigLIP2 backbone producing 1152-dimensional embeddings,
with a linear projection layer to map to unified 1024-dimensional space as described in the paper.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from loguru import logger

# Import standard open_clip_torch, NOT the TULIP fork.
# The TULIP fork at tulip/open_clip/src/ can shadow the real package on sys.path.
_tulip_paths = [p for p in sys.path if 'tulip' in p and 'open_clip' in p]
for _p in _tulip_paths:
    sys.path.remove(_p)
import open_clip
sys.path.extend(_tulip_paths)

from config import settings

class LinearProjection(nn.Module):
    """Linear projection layer to map SigLIP2 1152-dim to unified 1024-dim space."""

    def __init__(self, input_dim: int = 1152, output_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        # Initialize with identity-like weights for the first 1024 dimensions
        with torch.no_grad():
            self.proj.weight.data.zero_()
            for i in range(min(input_dim, output_dim)):
                self.proj.weight.data[i, i] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class SigLIP2Encoder:
    """
    SigLIP2-based encoder producing 1152-dimensional embeddings,
    projected to unified 1024-dim space via a linear projection layer.
    """

    def __init__(self, device: str = None):
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Initializing SigLIP2 encoder on {self.device}")

        # SigLIP2 uses ViT-SO400M-14 backbone with siglip pretraining
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
        self.native_dim = settings.EMBEDDING_DIM_SIGLIP2  # 1152
        self.unified_dim = settings.UNIFIED_DIM  # 1024

        # Linear projection to unified space
        self.projection = LinearProjection(
            input_dim=self.native_dim,
            output_dim=self.unified_dim,
        ).to(self.device)
        if settings.USE_FP16 and "cuda" in self.device:
            self.projection = self.projection.half()

        logger.info(
            f"SigLIP2 encoder loaded: {settings.SIGLIP2_MODEL_NAME} "
            f"(native_dim={self.native_dim}, unified_dim={self.unified_dim})"
        )

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        """Project SigLIP2 features to unified dimension space."""
        projected = self.projection(features)
        return projected / projected.norm(dim=-1, keepdim=True)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL Image to a projected SigLIP2 embedding."""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                image_tensor = image_tensor.half()
            features = self.model.encode_image(image_tensor)
            features = self._project(features)
        return features.cpu().numpy().astype(np.float32).flatten()

    def encode_images_batch(self, images: list) -> np.ndarray:
        """Encode a batch of PIL Images to projected SigLIP2 embeddings."""
        with torch.no_grad():
            image_tensors = torch.stack(
                [self.preprocess(img) for img in images]
            ).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                image_tensors = image_tensors.half()
            features = self.model.encode_image(image_tensors)
            features = self._project(features)
        return features.cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query to a projected SigLIP2 embedding."""
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(tokens)
            features = self._project(features)
        return features.cpu().numpy().astype(np.float32).flatten()

    def encode_texts_batch(self, texts: list) -> np.ndarray:
        """Encode multiple text queries to projected SigLIP2 embeddings."""
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            features = self.model.encode_text(tokens)
            features = self._project(features)
        return features.cpu().numpy().astype(np.float32)

    def get_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Compute cosine similarity between image and text features."""
        return float(np.dot(image_features, text_features))
