"""
Evaluation Metrics for  Video Retrieval System.
Implements Recall@k and Mean Reciprocal Rank (MRR) as described in Section 4 of the paper.

The evaluation protocol follows:
1. Fetch queries from a central evaluation server
2. Execute retrieval on GPU
3. Submit results back for validation
4. Repeat 5 times per query, take mean
5. Compute final Recall@k and MRR across all queries
"""

import time
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from loguru import logger

from config import settings
from embedding.hybrid_engine import HybridEmbeddingEngine
from retrieval.reranker import HybridReranker
from retrieval.milvus_store import MilvusVectorStore

def compute_recall_at_k(
    ground_truth: List[str],
    retrieved_ids: List[str],
    k: int = 10,
) -> float:
    """
    Compute Recall@k.
    
    Recall@k = |{relevant items} ∩ {retrieved top-k}| / |{relevant items}|
    
    Args:
        ground_truth: List of ground truth item IDs
        retrieved_ids: List of retrieved item IDs in ranked order
        k: Cutoff rank (default 10)
    
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not ground_truth:
        return 0.0

    # Take top-k retrieved items
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(ground_truth)

    # Intersection
    hits = len(top_k_retrieved.intersection(relevant_set))

    return hits / len(relevant_set)

def compute_mrr(
    ground_truth: List[str],
    retrieved_ids: List[str],
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    MRR = 1/rank of first relevant item
    If no relevant item found, MRR = 0.
    
    Args:
        ground_truth: List of ground truth item IDs
        retrieved_ids: List of retrieved item IDs in ranked order
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not ground_truth:
        return 0.0

    relevant_set = set(ground_truth)

    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_set:
            return 1.0 / rank

    return 0.0

def evaluate_retrieval(
    engine: HybridEmbeddingEngine,
    milvus_store: MilvusVectorStore,
    reranker: HybridReranker,
    queries: List[dict],
    k_values: List[int] = None,
    num_repeats: int = 5,
) -> dict:
    """
    Run full evaluation on a set of queries following the  protocol.
    
    Each query dict should have:
        - 'query': the text query
        - 'ground_truth': list of ground truth item IDs
        - 'task_type': one of 'KIS', 'VQA', 'TRAKE'
    
    Args:
        engine: HybridEmbeddingEngine instance
        milvus_store: MilvusVectorStore instance
        reranker: HybridReranker instance
        queries: List of query dicts with ground truth
        k_values: List of k values for Recall@k (default: [1, 5, 10])
        num_repeats: Number of times to repeat each query (default 5 per paper)
    
    Returns:
        Dict with evaluation metrics
    """
    if k_values is None:
        k_values = [1, 5, 10]

    all_results = []
    per_k_recall = {k: [] for k in k_values}
    mrr_scores = []

    total_queries = len(queries)
    logger.info(f"Evaluating {total_queries} queries ({num_repeats} repeats each)...")

    for query_idx, query_data in enumerate(queries):
        query_text = query_data["query"]
        ground_truth = query_data["ground_truth"]

        query_recall = {k: [] for k in k_values}
        query_mrr = []

        for repeat in range(num_repeats):
            # Encode query
            hybrid_embs = engine.encode_text(query_text)

            # Hybrid search
            candidates = milvus_store.hybrid_search(
                clip_query=hybrid_embs["clip"],
                siglip2_query=hybrid_embs["siglip2"],
                top_k=100,
            )

            # Rerank
            reranked = reranker.rerank(candidates, top_k=100)

            # Extract retrieved IDs (video_id_frame_idx as unique key)
            retrieved_ids = [
                f"{r['video_id']}_{r['frame_idx']}" for r in reranked
            ]

            # Compute metrics
            for k in k_values:
                recall = compute_recall_at_k(ground_truth, retrieved_ids, k)
                query_recall[k].append(recall)

            mrr = compute_mrr(ground_truth, retrieved_ids)
            query_mrr.append(mrr)

        # Average over repeats for this query
        query_result = {
            "query": query_text,
            "task_type": query_data.get("task_type", "KIS"),
            "metrics": {},
        }

        for k in k_values:
            avg_recall = np.mean(query_recall[k])
            per_k_recall[k].append(avg_recall)
            query_result["metrics"][f"Recall@{k}"] = float(avg_recall)

        avg_mrr = np.mean(query_mrr)
        mrr_scores.append(avg_mrr)
        query_result["metrics"]["MRR"] = float(avg_mrr)

        all_results.append(query_result)

        logger.info(
            f"Query {query_idx + 1}/{total_queries}: "
            f"Recall@10={avg_mrr:.3f}, MRR={avg_mrr:.3f} "
            f"'{query_text[:60]}...'"
        )

    # Compute overall metrics
    overall_metrics = {}
    for k in k_values:
        overall_metrics[f"Recall@{k}"] = float(np.mean(per_k_recall[k]))
        overall_metrics[f"Recall@{k}_std"] = float(np.std(per_k_recall[k]))

    overall_metrics["MRR"] = float(np.mean(mrr_scores))
    overall_metrics["MRR_std"] = float(np.std(mrr_scores))

    summary = {
        "total_queries": total_queries,
        "num_repeats_per_query": num_repeats,
        "k_values": k_values,
        "overall_metrics": overall_metrics,
        "per_query_results": all_results,
    }

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total queries: {total_queries}")
    for k in k_values:
        logger.info(f"Recall@{k}: {overall_metrics[f'Recall@{k}']:.4f} "
                     f"(±{overall_metrics[f'Recall@{k}_std']:.4f})")
    logger.info(f"MRR: {overall_metrics['MRR']:.4f} "
                f"(±{overall_metrics['MRR_std']:.4f})")
    logger.info("=" * 60)

    return summary

def run_evaluation_from_file(
    queries_file: str,
    engine: HybridEmbeddingEngine = None,
    milvus_store: MilvusVectorStore = None,
    reranker: HybridReranker = None,
    output_file: str = None,
) -> dict:
    """
    Run evaluation loading queries from a JSON file.
    
    Expected JSON format:
    [
        {
            "query": "text query here",
            "ground_truth": ["video_id_frame_idx", ...],
            "task_type": "KIS"
        },
        ...
    ]
    """
    # Load queries
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Initialize components if not provided
    _own_engine = engine is None
    _own_milvus = milvus_store is None

    if engine is None:
        engine = HybridEmbeddingEngine()
    if milvus_store is None:
        milvus_store = MilvusVectorStore()
    if reranker is None:
        reranker = HybridReranker()

    try:
        results = evaluate_retrieval(
            engine=engine,
            milvus_store=milvus_store,
            reranker=reranker,
            queries=queries,
        )

        # Save results if output file specified
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")

        return results

    finally:
        if _own_milvus:
            milvus_store.close()
        # engine cleanup handled by torch

if __name__ == "__main__":
    # Example: Run evaluation with sample queries
    logger.info(" Evaluation Module")
    logger.info("Usage: python -m evaluation.metrics --queries queries.json")
