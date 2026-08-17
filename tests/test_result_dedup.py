"""Unit tests for result deduplication and expansion rank preservation."""

from retrieval.result_dedup import deduplicate_results
from retrieval.hybrid_fusion import _merge_expansion_preserving_rank, _frame_key


def _frame(video_id: str, frame_idx: int, score: float, timestamp: float = 0.0) -> dict:
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "final_rerank_score": score,
        "clip_score": score,
    }


def _neighbor(source: dict, frame_idx: int, score: float) -> dict:
    return {
        **source,
        "frame_idx": frame_idx,
        "final_rerank_score": score,
        "is_expanded_neighbor": True,
        "expanded_from_frame": source["frame_idx"],
    }


class TestTemporalNMS:
    def test_keeps_one_of_nearby_frames(self):
        results = [
            _frame("V1", 100, 1.0, 10.0),
            _frame("V1", 105, 0.9, 10.2),
            _frame("V1", 120, 0.8, 12.0),
        ]
        out = deduplicate_results(results, top_k=2, min_frame_gap=15, scene_window_sec=0)
        assert len(out) == 2
        assert out[0]["frame_idx"] == 100
        assert out[1]["frame_idx"] == 120

    def test_different_videos_not_suppressed(self):
        results = [
            _frame("V1", 100, 1.0, 10.0),
            _frame("V2", 100, 0.9, 10.0),
        ]
        out = deduplicate_results(results, top_k=2, min_frame_gap=15, scene_window_sec=0)
        assert len(out) == 2


class TestSceneBucket:
    def test_keeps_one_per_scene_window(self):
        results = [
            _frame("V1", 100, 1.0, 116.8),
            _frame("V1", 200, 0.9, 119.0),
            _frame("V1", 300, 0.8, 130.0),
        ]
        out = deduplicate_results(
            results,
            top_k=2,
            min_frame_gap=0,
            scene_window_sec=8.0,
        )
        assert len(out) == 2
        assert out[0]["frame_idx"] == 100
        assert out[1]["frame_idx"] == 300


class TestBackfill:
    def test_fills_to_top_k_when_dedup_is_strict(self):
        results = [_frame("V1", i * 5, 1.0 - i * 0.01, float(i)) for i in range(20)]
        out = deduplicate_results(results, top_k=10, min_frame_gap=15, scene_window_sec=0)
        assert len(out) == 10


class TestMergeExpansionPreservingRank:
    def test_neighbor_stays_after_source_not_above_earlier_hits(self):
        base = [
            _frame("V1", 100, 1.0),
            _frame("V1", 200, 0.8),
        ]
        # Neighbor of frame 100 with artificially high score should not jump above frame 200's rank
        neighbors = [_neighbor(base[0], 105, 0.99)]
        merged = _merge_expansion_preserving_rank(base, neighbors)
        keys = [_frame_key(r) for r in merged]
        assert keys.index("V1_100") < keys.index("V1_105")
        assert keys.index("V1_105") < keys.index("V1_200")

    def test_skips_neighbor_when_real_frame_exists(self):
        base = [_frame("V1", 100, 1.0), _frame("V1", 105, 0.95)]
        neighbors = [_neighbor(base[0], 105, 0.99)]
        merged = _merge_expansion_preserving_rank(base, neighbors)
        frame105 = [r for r in merged if r["frame_idx"] == 105]
        assert len(frame105) == 1
        assert not frame105[0].get("is_expanded_neighbor")
