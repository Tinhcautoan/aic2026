import os
import sys
import re
import pickle
import numpy as np
from PIL import Image
from loguru import logger

# Add Colab paths to python path
aicv5_dir = "/content/v8_c2lip_extracted/v8_c2lip/aicv5"
if os.path.isdir(aicv5_dir):
    sys.path.insert(0, aicv5_dir)
    sys.path.insert(0, os.path.dirname(aicv5_dir))

from config import settings
from embedding.hybrid_engine import HybridEmbeddingEngine

# Configure paths for Google Colab
EMB_PKL_PATH = "/content/v8_c2lip_extracted/v8_c2lip/aicv5/data/persistent_store/embeddings.pkl"
META_PKL_PATH = "/content/v8_c2lip_extracted/v8_c2lip/aicv5/data/persistent_store/metadata.pkl"
KEYFRAME_DIR = "/content/v8_c2lip_extracted/v8_c2lip/aicv5/data/keyframes"

def get_image_path(frame_id: str) -> str:
    """Find the keyframe image file path for a given frame_id (video_id:frame_idx)"""
    video_id, frame_idx = frame_id.split(":")
    
    # Try different filename formats
    filename_formats = [
        f"{video_id}_frame_{int(frame_idx):06d}.jpg",
        f"{video_id}_frame_{frame_idx}.jpg",
        f"{int(frame_idx):06d}.jpg",
        f"{frame_idx}.jpg"
    ]
    
    for fmt in filename_formats:
        # Since files are directly inside KEYFRAME_DIR (flat structure)
        path = os.path.join(KEYFRAME_DIR, fmt)
        if os.path.exists(path):
            return path
                
    # Default fallback path
    return os.path.join(KEYFRAME_DIR, f"{video_id}_frame_{int(frame_idx):06d}.jpg")

def run_export_from_images():
    logger.info("Initializing PE-Core-G-only embedding extraction on Colab...")
    
    import open_clip
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading hf-hub:timm/PE-Core-bigG-14-448 on {device}...")
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:timm/PE-Core-bigG-14-448',
            device=device
        )
        if torch.cuda.is_available():
            model = model.half()
        model.eval()
        
        # Determine embedding dimension dynamically by encoding a dummy tensor (TimmModel does not have visual.output_dim)
        dummy_tensor = torch.zeros((1, 3, 448, 448), device=device)
        if torch.cuda.is_available():
            dummy_tensor = dummy_tensor.half()
        with torch.no_grad():
            dummy_out = model.encode_image(dummy_tensor)
        embedding_dim = dummy_out.shape[-1]
        
        logger.info(f"PE-Core-G model loaded successfully. Embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to load PE-Core-G model: {e}")
        return
    
    if not os.path.exists(META_PKL_PATH):
        logger.error(f"Metadata file not found at: {META_PKL_PATH}")
        return
        
    if not os.path.exists(EMB_PKL_PATH):
        logger.error(f"Embeddings file not found at: {EMB_PKL_PATH}")
        return

    logger.info(f"Loading {META_PKL_PATH}...")
    with open(META_PKL_PATH, "rb") as f:
        meta_data = pickle.load(f)
    ids = meta_data["ids"]
    
    logger.info(f"Loading {EMB_PKL_PATH}...")
    with open(EMB_PKL_PATH, "rb") as f:
        embeddings = pickle.load(f)

    # Initialize pe_core_g list if not present
    if "pe_core_g" not in embeddings:
        embeddings["pe_core_g"] = [None] * len(ids)
    
    pe_core_g_list = embeddings["pe_core_g"]
    
    # Resize list if metadata has grown
    if len(pe_core_g_list) < len(ids):
        pe_core_g_list.extend([None] * (len(ids) - len(pe_core_g_list)))

    # Find which indices need to be computed
    indices_to_compute = [i for i, x in enumerate(pe_core_g_list) if x is None]
    total_frames = len(ids)
    computed_count = total_frames - len(indices_to_compute)
    
    logger.info(f"Total keyframes: {total_frames}")
    logger.info(f"Already computed PE-Core-G: {computed_count}")
    logger.info(f"Remaining to compute: {len(indices_to_compute)}")
    
    if not indices_to_compute:
        logger.info("All PE-Core-G embeddings are already computed!")
        return

    batch_size = 32
    logger.info(f"Starting batch encoding (batch_size={batch_size})...")
    
    # Process keyframes in batches to optimize GPU speed
    for b_start in range(0, len(indices_to_compute), batch_size):
        chunk_indices = indices_to_compute[b_start : b_start + batch_size]
        chunk_images = []
        valid_indices = []
        
        for idx in chunk_indices:
            frame_id = ids[idx]
            filepath = get_image_path(frame_id)
            
            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath).convert("RGB")
                    chunk_images.append(img)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.error(f"Failed to load image {filepath}: {e}")
            else:
                logger.debug(f"Keyframe image not found: {filepath}")
                
        if not chunk_images:
            continue
            
        try:
            # Generate Embeddings (PE-Core-G only)
            with torch.no_grad():
                tensors = torch.stack([preprocess(img) for img in chunk_images]).to(device)
                if torch.cuda.is_available():
                    tensors = tensors.half()
                features = model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
            pe_core_g_embs = features.cpu().numpy().astype(np.float32)
            
            for i, idx in enumerate(valid_indices):
                pe_core_g_list[idx] = pe_core_g_embs[i]
                
        except Exception as e:
            logger.error(f"Error encoding batch starting at index {chunk_indices[0]}: {e}")
            
        # Save progress checkpoint periodically (every 16 batches / 512 images)
        if (b_start // batch_size) % 16 == 0:
            embeddings["pe_core_g"] = pe_core_g_list
            with open(EMB_PKL_PATH, "wb") as f:
                pickle.dump(embeddings, f)
            logger.info(f"Progress: {computed_count + b_start + len(chunk_indices)}/{total_frames} frames completed")

    # Fill any remaining None values (e.g. from missing files) with zero vectors to avoid downstream errors
    zero_vector = np.zeros(embedding_dim, dtype=np.float32)
    filled_count = 0
    for idx in range(len(pe_core_g_list)):
        if pe_core_g_list[idx] is None:
            pe_core_g_list[idx] = zero_vector
            filled_count += 1
            
    if filled_count > 0:
        logger.warning(f"Filled {filled_count} missing/failed keyframes with zero vectors.")

    # Final save back to pkl
    embeddings["pe_core_g"] = pe_core_g_list
    with open(EMB_PKL_PATH, "wb") as f:
        pickle.dump(embeddings, f)
        
    logger.info("Successfully updated PE-Core-G embeddings directly into embeddings.pkl!")

if __name__ == "__main__":
    run_export_from_images()

