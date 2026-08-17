"""Tests for retrieval.frame_sampler (window delta around keyframes)."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.frame_sampler import sample_window, sample_video_windows


class TestSampleWindow(unittest.TestCase):
    """Test a single keyframe window."""

    def test_returns_frames_in_step_sequence(self):
        frames = sample_window(
            video_id="V1", keyframe_idx=500, keyframe_score=0.9,
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        indices = [f.frame_idx for f in frames]
        self.assertIn(500, indices)
        self.assertEqual(indices, sorted(indices))

    def test_step_size(self):
        frames = sample_window(
            video_id="V1", keyframe_idx=500, keyframe_score=0.9,
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        indices = [f.frame_idx for f in frames]
        diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        self.assertTrue(all(d == 5 for d in diffs))

    def test_clamps_at_start(self):
        frames = sample_window(
            video_id="V1", keyframe_idx=10, keyframe_score=0.9,
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        self.assertEqual(frames[0].frame_idx, 0)

    def test_clamps_at_end(self):
        frames = sample_window(
            video_id="V1", keyframe_idx=990, keyframe_score=0.9,
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        self.assertTrue(frames[-1].frame_idx <= 999)
        self.assertTrue(frames[-1].frame_idx > 990)

    def test_keyframe_always_present(self):
        frames = sample_window(
            video_id="V1", keyframe_idx=500, keyframe_score=0.9,
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        kf = [f for f in frames if f.is_keyframe]
        self.assertEqual(len(kf), 1)
        self.assertEqual(kf[0].frame_idx, 500)

    def test_keyframe_score_applied_to_all_frames(self):
        frames = sample_window(
            video_id="V1", keyframe_idx=500, keyframe_score=0.9,
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        self.assertTrue(all(f.score == 0.9 for f in frames))

    def test_all_frames_have_video_id(self):
        frames = sample_window(
            video_id="V1", keyframe_idx=500, keyframe_score=0.9,
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        self.assertTrue(all(f.video_id == "V1" for f in frames))

    def test_small_delta_one_frame(self):
        """When window is tiny, at least the keyframe itself is returned."""
        frames = sample_window(
            video_id="V1", keyframe_idx=500, keyframe_score=0.9,
            frame_count=1000, delta_ratio=0.0, frame_step=5,
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_idx, 500)


class TestSampleVideoWindows(unittest.TestCase):
    """Test sampling across multiple keyframes for one video."""

    def test_two_keyframes_union(self):
        sampled = sample_video_windows(
            video_id="V1",
            keyframes=[
                {"frame_idx": 200, "score": 0.9},
                {"frame_idx": 800, "score": 0.8},
            ],
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        indices = {f.frame_idx for f in sampled}
        self.assertIn(200, indices)
        self.assertIn(800, indices)

    def test_deduplicates_overlapping_windows(self):
        sampled = sample_video_windows(
            video_id="V1",
            keyframes=[
                {"frame_idx": 250, "score": 0.9},
                {"frame_idx": 260, "score": 0.85},
            ],
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        all_indices = [f.frame_idx for f in sampled]
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_multiple_videos(self):
        sampled = sample_video_windows(
            video_id="V1",
            keyframes=[{"frame_idx": 500, "score": 0.9}],
            frame_count=1000, delta_ratio=0.05, frame_step=5,
        )
        self.assertTrue(all(f.video_id == "V1" for f in sampled))


if __name__ == "__main__":
    unittest.main(verbosity=2)
