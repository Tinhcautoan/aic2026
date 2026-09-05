"""
C2LIP encoder -- replaces the SigLIP2 branch (the `siglip2_embedding` field /
`siglip2_encoder` slot in HybridEmbeddingEngine) when settings.USE_C2LIP is
True. Same swap pattern as pecore_encoder.py replacing the CLIP slot.

Checkpoint identity (verified against c2lipweight.pt on this machine):
  state_dict = {"epoch": 5, "name": "siglip_cc30m_3m_lla-...", "state_dict":
  {...}, "optimizer": {...}}. The 314 `state_dict` tensors load into
  open_clip's `ViT-B-16-SigLIP` (after stripping the "module." DDP prefix)
  with zero missing/unexpected keys -- it is a fine-tuned SigLIP ViT-B/16
  checkpoint, not a new architecture. Output is 768-dim (text_projection is
  768x768), matching what's already stored in
  data/persistent_store/embeddings.pkl's 'siglip2' field on this machine, so
  no projection/truncation logic is needed here (unlike siglip2_encoder.py's
  native/pca/truncate modes for the 1152-dim ViT-SO400M-14-SigLIP2).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger

# Same isolation as siglip2_encoder.py / pecore_encoder.py: keep the
# vendored TULIP open_clip fork from shadowing the real package on sys.path.
_tulip_paths = [p for p in sys.path if "tulip" in p and "open_clip" in p]
for _p in _tulip_paths:
    sys.path.remove(_p)
import open_clip  # noqa: E402
sys.path.extend(_tulip_paths)

from config import settings  # noqa: E402


class C2LIPEncoder:
    """C2LIP (fine-tuned SigLIP ViT-B/16) image/text encoder."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or (
            f"cuda:{settings.GPU_DEVICE_ID}"
            if settings.GPU_ENABLED and torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Initializing C2LIP encoder on {self.device}")

        checkpoint_path = str(settings.C2LIP_CHECKPOINT)
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"C2LIP_CHECKPOINT={checkpoint_path!r} not found. Set it in .env "
                f"to the path of c2lipweight.pt on this machine."
            )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.C2LIP_MODEL_NAME,
            pretrained=None,
            device="cpu",
        )
        self._load_checkpoint(checkpoint_path)

        if settings.USE_FP16 and "cuda" in self.device:
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(settings.C2LIP_MODEL_NAME)

        self.embedding_dim = int(settings.EMBEDDING_DIM_C2LIP)
        self.logit_scale, self.logit_bias = self._read_logit_params()

        logger.info(
            f"C2LIP ready: {settings.C2LIP_MODEL_NAME} ckpt={checkpoint_path} "
            f"dim={self.embedding_dim} logit_scale={self.logit_scale:.2f} "
            f"logit_bias={self.logit_bias:.3f}"
        )

    # ------------------------------------------------------------------ #

    def _load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        # Trained under DataParallel/DDP -- strip the "module." prefix.
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                f"C2LIP checkpoint load: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys. missing={missing[:5]} "
                f"unexpected={unexpected[:5]}"
            )

    @property
    def output_dim(self) -> int:
        """Dimensionality actually written to the vector store."""
        return self.embedding_dim

    def _read_logit_params(self):
        from retrieval.scoring import get_logit_params
        return get_logit_params(self.model)

    def _normalize(self, f: torch.Tensor) -> torch.Tensor:
        return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # ------------------------------------------------------------------ #

    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            t = self.preprocess(image).unsqueeze(0).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                t = t.half()
            f = self._normalize(self.model.encode_image(t))
        return f.float().cpu().numpy().astype(np.float32).flatten()

    def encode_images_batch(self, images: list) -> np.ndarray:
        with torch.no_grad():
            t = torch.stack([self.preprocess(i) for i in images]).to(self.device)
            if settings.USE_FP16 and "cuda" in self.device:
                t = t.half()
            f = self._normalize(self.model.encode_image(t))
        return f.float().cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            f = self._normalize(self.model.encode_text(tokens))
        return f.float().cpu().numpy().astype(np.float32).flatten()

    def encode_texts_batch(self, texts: list) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            f = self._normalize(self.model.encode_text(tokens))
        return f.float().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #

    def get_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Raw cosine. Kept for compatibility -- prefer `get_logit`."""
        return float(np.dot(image_features, text_features))

    def get_logit(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Cosine mapped into the model's trained logit space."""
        return float(np.dot(image_features, text_features)) * self.logit_scale + self.logit_bias

    def get_probability(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """SigLIP-style sigmoid-trained model: sigmoid(logit) is the model's
        own estimate that this image and text are a matching pair."""
        z = self.get_logit(image_features, text_features)
        return float(1.0 / (1.0 + np.exp(-z))) if z >= 0 else float(np.exp(z) / (1.0 + np.exp(z)))
