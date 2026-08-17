#!/usr/bin/env python
"""
Video Retrieval System - Main Entry Point
Team 

Usage:
    python main.py serve          # Start the FastAPI server
    python main.py process VIDEO  # Process and index a video
    python main.py search QUERY   # Search with a text query
    python main.py evaluate       # Run evaluation on test queries
    python main.py status         # Check system status
"""

import sys
import argparse
from loguru import logger

from config import settings
from embedding.hybrid_engine import HybridEmbeddingEngine
from retrieval.reranker import HybridReranker
from retrieval.milvus_store import MilvusVectorStore
from retrieval.redis_cache import RedisCacheManager
from preprocessing.pipeline import PreprocessingPipeline
from evaluation.metrics import evaluate_retrieval, run_evaluation_from_file

def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    logger.info(f"Starting server at http://{settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=args.reload,
        log_level="info",
    )

def cmd_process(args):
    """Process and index a video."""
    engine = HybridEmbeddingEngine()
    milvus = MilvusVectorStore()
    pipeline = PreprocessingPipeline(
        embedding_engine=engine,
        milvus_store=milvus,
    )

    result = pipeline.process_video(
        video_path=args.video_path,
        video_id=args.video_id,
        save_keyframes=args.save_keyframes,
    )

    logger.info(f"Result: {result}")
    milvus.close()

def cmd_search(args):
    """Search with a text query."""
    engine = HybridEmbeddingEngine()
    milvus = MilvusVectorStore()
    reranker = HybridReranker()

    # Encode query
    hybrid_embs = engine.encode_text(args.query)

    # Search
    candidates = milvus.hybrid_search(
        clip_query=hybrid_embs["clip"],
        siglip2_query=hybrid_embs["siglip2"],
        top_k=args.top_k * 3,
    )

    # Rerank
    results = reranker.rerank(candidates, top_k=args.top_k)

    print(f"\n{'='*70}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print(f"{'='*70}")

    for i, r in enumerate(results):
        print(f"\n[{i+1}] Video: {r['video_id']} | Frame: {r['frame_idx']} | "
              f"Time: {r['timestamp']:.1f}s")
        print(f"    CLIP: {r['clip_score']:.4f} | SigLIP2: {r['siglip2_score']:.4f} | "
              f"Final: {r['final_score']:.4f}")

    milvus.close()

def cmd_evaluate(args):
    """Run evaluation."""
    if args.queries_file:
        results = run_evaluation_from_file(
            queries_file=args.queries_file,
            output_file=args.output,
        )
    else:
        logger.error("Please provide --queries-file path to query JSON")
        return

    print(f"\nFinal Results:")
    metrics = results["overall_metrics"]
    for key, val in metrics.items():
        if not key.endswith("_std"):
            print(f"  {key}: {val:.4f}")

def cmd_status(args):
    """Check system status."""
    print(f"\n{'='*50}")
    print(f"Video Retrieval System")
    print(f"{'='*50}")
    print(f"GPU Available: {_check_gpu()}")
    print(f"Config: {settings.PROJECT_ROOT}")

    try:
        milvus = MilvusVectorStore()
        stats = milvus.get_collection_stats()
        print(f"Milvus: {stats}")
        milvus.close()
    except Exception as e:
        print(f"Milvus: Not available ({e})")

    try:
        import redis.asyncio as aioredis
        print(f"Redis: Checking... (requires running Redis server)")
    except ImportError:
        print(f"Redis: Not available")

def _check_gpu() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"✅ Yes ({torch.cuda.get_device_name(0)})"
        return "❌ No (CPU only)"
    except ImportError:
        return "❓ torch not installed"

def main():
    parser = argparse.ArgumentParser(
        description="Video Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py serve
  python main.py process /path/to/video.mp4 --video-id my_video
  python main.py search "a person presenting slides" --top-k 10
  python main.py evaluate --queries-file queries.json
  python main.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--reload", action="store_true", default=False,
                              help="Enable auto-reload")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process and index a video")
    process_parser.add_argument("video_path", help="Path to video file")
    process_parser.add_argument("--video-id", help="Custom video identifier")
    process_parser.add_argument("--save-keyframes", action="store_true",
                                help="Save keyframe images")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search with text query")
    search_parser.add_argument("query", help="Text query")
    search_parser.add_argument("--top-k", type=int, default=10,
                               help="Number of results (default: 10)")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--queries-file", required=True,
                             help="Path to queries JSON file")
    eval_parser.add_argument("--output", help="Path to save results JSON")

    # Status command
    subparsers.add_parser("status", help="Check system status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "serve": cmd_serve,
        "process": cmd_process,
        "search": cmd_search,
        "evaluate": cmd_evaluate,
        "status": cmd_status,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)

if __name__ == "__main__":
    main()
