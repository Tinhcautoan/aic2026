"""Tests for retrieval.trake_aligner (event alignment by embedding search)."""

import sys
import os
import unittest
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.trake_aligner import align_video_events, rank_videos


def _fake_search(
    event_results: dict[str, list[tuple[int, float]]],
):
    """Build a fake search_fn from {event_text: [(frame_idx, score), ...]}."""
    def search_fn(event_text: str, candidates: List[dict]) -> List[Tuple[int, float]]:
        return event_results.get(event_text, [])
    return search_fn


class TestAlignVideoEvents(unittest.TestCase):
    """Align events within a single video's candidates."""

    def test_returns_one_frame_per_event(self):
        search = _fake_search({
            "giậm nhảy": [(200, 0.9)],
            "bay qua xà": [(400, 0.85)],
            "tiếp đất": [(600, 0.8)],
        })
        alignment = align_video_events(
            video_id="V1",
            candidate_frames=[{"frame_idx": i, "score": 0.5} for i in range(0, 1000, 5)],
            event_texts=["giậm nhảy", "bay qua xà", "tiếp đất"],
            query_score=0.7,
            event_search_fn=search,
        )
        self.assertEqual(len(alignment.event_frames), 3)

    def test_total_score_combines_query_and_events(self):
        search = _fake_search({
            "giậm nhảy": [(200, 0.9)],
            "bay qua xà": [(400, 0.9)],
        })
        alignment = align_video_events(
            video_id="V1",
            candidate_frames=[],
            event_texts=["giậm nhảy", "bay qua xà"],
            query_score=1.0,
            event_search_fn=search,
        )
        self.assertAlmostEqual(alignment.total_score, 0.9)

    def test_empty_event_results_gives_zero_score(self):
        search = _fake_search({"giậm nhảy": []})
        alignment = align_video_events(
            video_id="V1",
            candidate_frames=[],
            event_texts=["giậm nhảy"],
            query_score=0.8,
            event_search_fn=search,
        )
        self.assertEqual(alignment.event_frames[0].frame_idx, -1)
        self.assertAlmostEqual(alignment.total_score, 0.0)

    def test_frame_order_matches_event_order(self):
        search = _fake_search({
            "first": [(100, 0.9)],
            "second": [(500, 0.8)],
            "third": [(900, 0.7)],
        })
        alignment = align_video_events(
            video_id="V1",
            candidate_frames=[],
            event_texts=["first", "second", "third"],
            query_score=0.6,
            event_search_fn=search,
        )
        indices = [ef.frame_idx for ef in alignment.event_frames]
        self.assertEqual(indices, [100, 500, 900])


class TestRankVideos(unittest.TestCase):
    """Rank multiple video candidates by combined score."""

    def test_best_video_first(self):
        alignments = [
            align_video_events(
                "V2", [], ["e1"], 0.8, _fake_search({"e1": [(10, 0.5)]}),
            ),
            align_video_events(
                "V1", [], ["e1"], 1.0, _fake_search({"e1": [(10, 0.9)]}),
            ),
        ]
        ranked = rank_videos(alignments)
        self.assertEqual(ranked[0].video_id, "V1")
        self.assertEqual(ranked[1].video_id, "V2")

    def test_max_rows_limit(self):
        alignments = [
            align_video_events(
                f"V{i}", [], ["e1"], 1.0, _fake_search({"e1": [(10, 0.9)]}),
            )
            for i in range(20)
        ]
        ranked = rank_videos(alignments, max_rows=5)
        self.assertEqual(len(ranked), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
