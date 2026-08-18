"""
CLIP Encoder for image and text embeddings.
Uses OpenCLIP with ViT-L-14 backbone and dfn2b pretraining as described in the paper.
"""

import sys
import torch
import numpy as np
from PIL import Image
from loguru import logger

# Import standard open_clip_torch, NOT the TULIP fork.
_tulip_paths = [p for p in sys.path if 'tulip' in p and 'open_clip' in p]
for _p in _tulip_paths:
    sys.path.remove(_p)
import open_clip
sys.path.extend(_tulip_paths)

from config import settings

class CLIPEncoder:
    """
    CLIP-based encoder producing 1024-dimensional embeddings.
    Uses ViT-L-14-quickgelu with dfn2b pretraining.
    """

    def __init__(self, device: str = None):
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Initializing CLIP encoder on {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=settings.CLIP_MODEL_NAME,
            pretrained=settings.CLIP_PRETRAINED,
            device=self.device,
        )

        if settings.USE_FP16 and "cuda" in self.device:
            self.model = self.model.half()

        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(settings.CLIP_MODEL_NAME)
        self.embedding_dim = settings.EMBEDDING_DIM_CLIP

        logger.info(
            f"CLIP encoder loaded: {settings.CLIP_MODEL_NAME} "
            f"(dim={self.embedding_dim}, fp16={settings.USE_FP16})"
        )

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL Image to a normalized CLIP embedding."""
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                image_tensor = image_tensor.half()
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).flatten()

    def encode_images_batch(self, images: list) -> np.ndarray:
        """Encode a batch of PIL Images to normalized CLIP embeddings."""
        with torch.no_grad():
            image_tensors = torch.stack(
                [self.preprocess(img) for img in images]
            ).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                image_tensors = image_tensors.half()
            features = self.model.encode_image(image_tensors)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query to a normalized CLIP embedding."""
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).flatten()

    def encode_texts_batch(self, texts: list) -> np.ndarray:
        """Encode multiple text queries to normalized CLIP embeddings."""
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def get_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Compute cosine similarity between image and text features."""
        return float(np.dot(image_features, text_features))
