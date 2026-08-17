"""
System Test Suite for Video Retrieval System.
Tests all components: config, reranker, API endpoints, and end-to-end flow.
"""

import sys
import os
import json
import time
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestConfig(unittest.TestCase):
    """Test configuration loading."""

    def test_settings_load(self):
        from config import settings
        self.assertEqual(settings.API_PORT, 8000)
        self.assertEqual(settings.CLIP_MODEL_NAME, "ViT-L-14")
        self.assertEqual(settings.RRF_K, 60)
        self.assertEqual(settings.FUSION_ALPHA, 0.5)
        self.assertTrue(settings.UNIFIED_DIM == 1024)
        print("  ✅ Config loaded correctly")

    def test_settings_types(self):
        from config import settings
        self.assertIsInstance(settings.MILVUS_PORT, int)
        self.assertIsInstance(settings.FUSION_ALPHA, float)
        self.assertIsInstance(settings.OCR_ENABLED, bool)
        self.assertIsInstance(settings.ALLOWED_ORIGINS, list)
        print("  ✅ Config types correct")

class TestReranker(unittest.TestCase):
    """Test the RRF + CombMNZ reranking algorithm."""

    def setUp(self):
        from retrieval.reranker import HybridReranker
        self.reranker = HybridReranker(rrf_k=60, fusion_alpha=0.5)

    def test_rerank_basic(self):
        """Basic reranking with two models."""
        candidates = [
            {"id": 1, "video_id": "v1", "frame_idx": 0, "timestamp": 0.0,
             "clip_score": 0.9, "siglip2_score": 0.85, "metadata": {}},
            {"id": 2, "video_id": "v1", "frame_idx": 10, "timestamp": 1.0,
             "clip_score": 0.7, "siglip2_score": 0.95, "metadata": {}},
            {"id": 3, "video_id": "v2", "frame_idx": 5, "timestamp": 0.5,
             "clip_score": 0.5, "siglip2_score": 0.5, "metadata": {}},
        ]
        results = self.reranker.rerank(candidates, top_k=3)
        self.assertEqual(len(results), 3)
        self.assertIn("final_score", results[0])
        # Best candidate should have highest final_score
        self.assertGreaterEqual(results[0]["final_score"], results[1]["final_score"])
        print("  ✅ Reranker basic test passed")
        print(f"     Top result: final_score={results[0]['final_score']:.4f}")

    def test_rerank_empty(self):
        results = self.reranker.rerank([], top_k=10)
        self.assertEqual(len(results), 0)
        print("  ✅ Reranker empty list handled")

    def test_rerank_single(self):
        candidates = [
            {"id": 1, "video_id": "v1", "frame_idx": 0, "timestamp": 0.0,
             "clip_score": 0.8, "siglip2_score": 0.8, "metadata": {}},
        ]
        results = self.reranker.rerank(candidates, top_k=5)
        self.assertEqual(len(results), 1)
        self.assertIn("final_score", results[0])
        print("  ✅ Reranker single candidate handled")

    def test_rrf_computation(self):
        """Test RRF score computation."""
        clip_ranks = np.array([0, 1, 2], dtype=np.float32)
        siglip2_ranks = np.array([2, 0, 1], dtype=np.float32)
        rrf_scores = self.reranker._reciprocal_rank_fusion(clip_ranks, siglip2_ranks)
        self.assertEqual(len(rrf_scores), 3)
        # Rank 0 in both gets highest RRF
        self.assertGreater(rrf_scores[0], rrf_scores[2])
        print("  ✅ RRF computation correct")

    def test_comb_mnz_computation(self):
        """Test CombMNZ score computation."""
        clip_scores = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        siglip2_scores = np.array([0.85, 0.6, 0.05], dtype=np.float32)
        comb_scores = self.reranker._comb_mnz(clip_scores, siglip2_scores)
        self.assertEqual(len(comb_scores), 3)
        # Best scores get higher CombMNZ
        self.assertGreater(comb_scores[0], comb_scores[2])
        print("  ✅ CombMNZ computation correct")

class TestAPIModels(unittest.TestCase):
    """Test API request/response models."""

    def test_query_request(self):
        from api.models import QueryRequest
        req = QueryRequest(query="test query", top_k=10, model="fusion")
        self.assertEqual(req.query, "test query")
        self.assertEqual(req.model, "fusion")
        self.assertEqual(req.top_k, 10)
        print("  ✅ QueryRequest model valid")

    def test_query_request_validation(self):
        from api.models import QueryRequest
        with self.assertRaises(Exception):
            QueryRequest(query="", top_k=-1)  # Should fail validation
        print("  ✅ QueryRequest validation works")

class TestCLI(unittest.TestCase):
    """Test CLI commands (main.py)."""

    def test_main_import(self):
        """Test that main.py imports cleanly."""
        import main
        self.assertTrue(hasattr(main, 'main'))
        print("  ✅ main.py imports successfully")

def test_api_endpoints():
    """Test the running API server endpoints."""
    import urllib.request
    import urllib.error

    base_url = "http://localhost:8000"
    endpoints = [
        ("GET", "/api/health", 200),
        ("GET", "/api/status", 200),
        ("GET", "/", 200),
        ("GET", "/ui", 200),
    ]

    print("\n" + "=" * 50)
    print("Testing API Endpoints (server must be running)")
    print("=" * 50)

    all_passed = True
    for method, path, expected_status in endpoints:
        url = base_url + path
        try:
            req = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(req, timeout=5) as resp:
                status = resp.getcode()
                body = resp.read().decode()
                if status == expected_status:
                    print(f"  ✅ {method} {path} → {status}")
                else:
                    print(f"  ❌ {method} {path} → {status} (expected {expected_status})")
                    all_passed = False
        except urllib.error.HTTPError as e:
            if e.code == expected_status:
                print(f"  ✅ {method} {path} → {e.code}")
            else:
                print(f"  ❌ {method} {path} → {e.code} (expected {expected_status})")
                all_passed = False
        except Exception as e:
            print(f"  ⚠️  {method} {path} → Error: {e}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print(" System Test Suite")
    print("=" * 60)

    # Run unittest for config, reranker, models
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestReranker))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIModels))
    suite.addTests(loader.loadTestsFromTestCase(TestCLI))

    runner = unittest.TextTestRunner(verbosity=2)
    print("\n📋 Unit Tests:")
    result = runner.run(suite)

    # Test live API if server is running
    api_ok = test_api_endpoints()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Unit tests: {'✅ ALL PASSED' if result.wasSuccessful() else '❌ FAILURES'}")
    print(f"  API tests:  {'✅ ALL PASSED' if api_ok else '❌ FAILURES'}")
    print(f"  Total: {result.testsRun} unit tests run, {len(result.failures)} failures")
    print("=" * 60)
    print("\n💡 To test the full search pipeline, you need:")
    print("   1. torch installed: pip install torch torchvision")
    print("   2. Milvus running: docker run -d --name milvus milvusdb/milvus:v2.4.0")
    print("   3. Redis running: docker run -d --name redis -p 6379:6379 redis:7-alpine")
    print("   4. Process a video: POST /api/process with video_path")
    print("   5. Run a search: POST /api/search with query text")
    print("=" * 60)
