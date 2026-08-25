"""
Video Retrieval System — Launcher (Milvus Lite).

Runs the FastAPI app directly. Milvus Lite is file-based (milvus.db), so no
Docker daemon, containers, or port 19530 are required.
"""

import os
from pathlib import Path

# Must be set before torch/open_clip/transformers/huggingface_hub are
# imported anywhere downstream -- they read HF_HOME once to compute cache
# paths. Model weights (PE-Core, SigLIP2, translator, cross-encoder) add up
# to 15-20GB; defaulting into the project directory avoids silently filling
# a small system drive. Set HF_HOME yourself (env var or .env) to override.
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "hf-cache"))


def print_instructions():
    print()
    print("=" * 60)
    print("  HOW TO USE")
    print("=" * 60)
    print()
    print("  Step 1: Upload a video")
    print("    Open http://localhost:8000/process")
    print("    Paste the path to a video file or folder")
    print('    Choose "Single File" or "Entire Folder"')
    print("    Click Process and wait for it to finish")
    print()
    print("  Step 2: Search your video")
    print("    Open http://localhost:8000/ui")
    print('    Type what you\'re looking for (e.g. "a dog running")')
    print("    Press Enter or click Search")
    print()
    print("  Step 3: View results")
    print("    Matching frames appear with similarity scores")
    print("    Best matches are ranked first")
    print()
    print("  Tip: Check system status at http://localhost:8000/api/status")
    print("=" * 60)
    print()


def main():
    print()
    print("=" * 60)
    print("  Video Retrieval System (Milvus Lite)")
    print("=" * 60)
    print("  Vector DB: file-based Milvus Lite (milvus.db) — no Docker required.")
    print()
    print_instructions()

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
