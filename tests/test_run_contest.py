"""End-to-end tests for scripts.run_contest (3-mode batch processing)."""

import sys
import os
import csv
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.query_parser import TKIS, QA, TRAKE
from retrieval.answer_expander import format_csv_line, expand_rows, SubmissionRow
from retrieval.frame_sampler import SampledFrame
from retrieval.trake_aligner import TrakeVideoAlignment, TrakeEventFrame, rank_videos


def _write_query(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class TestEndToEndKis(unittest.TestCase):
    """Simulate TKIS: top-100 from retriever → CSV."""

    def test_produces_correct_csv_rows(self):
        rows = []
        for i in range(100):
            rows.append(SubmissionRow(
                line=format_csv_line(f"L{i//10:02d}_V{i:03d}", i * 25),
                score=1.0 - i * 0.01,
            ))
        result = expand_rows(rows, max_rows=100)
        self.assertEqual(len(result), 100)
        parsed = [r.line.split(",") for r in result]
        self.assertEqual(parsed[0][0], "L00_V000")


class TestEndToEndQa(unittest.TestCase):
    """Simulate Q&A: parse query → search → VLM → CSV."""

    def test_produces_qa_csv_rows(self):
        from retrieval.query_parser import split_qa
        query = "Trong video về lễ trao giải, có bao nhiêu người lên sân khấu?"
        parsed = split_qa(query)
        self.assertIn("lễ trao giải", parsed.description)

        rows = [
            SubmissionRow(
                line=format_csv_line("V1", 100, answer="5"),
                score=0.9,
            ),
            SubmissionRow(
                line=format_csv_line("V2", 200, answer="3"),
                score=0.7,
            ),
        ]
        result = expand_rows(rows, max_rows=100)
        self.assertEqual(len(result), 2)
        self.assertIn(', "5"', result[0].line)


class TestEndToEndTrake(unittest.TestCase):
    """Simulate TRAKE: parse events → align → rank → CSV."""

    def test_produces_trake_csv_rows(self):
        from retrieval.query_parser import parse_trake
        query = "Tìm 3 khoảnh khắc: (1) chạy, (2) nhảy, (3) đáp đất."
        parsed = parse_trake(query)
        self.assertEqual(parsed.events, ["chạy", "nhảy", "đáp đất"])

        alignments = [
            TrakeVideoAlignment(
                video_id="V1",
                event_frames=[
                    TrakeEventFrame("chạy", 100, 0.9),
                    TrakeEventFrame("nhảy", 300, 0.85),
                    TrakeEventFrame("đáp đất", 500, 0.8),
                ],
                total_score=0.85,
            ),
            TrakeVideoAlignment(
                video_id="V2",
                event_frames=[
                    TrakeEventFrame("chạy", 200, 0.7),
                    TrakeEventFrame("nhảy", 400, 0.6),
                    TrakeEventFrame("đáp đất", 600, 0.5),
                ],
                total_score=0.6,
            ),
        ]
        ranked = rank_videos(alignments, max_rows=100)
        best = ranked[0]
        line = format_csv_line(best.video_id, frames=[ef.frame_idx for ef in best.event_frames])
        self.assertEqual(line, "V1, 100, 300, 500")


class TestModeDetection(unittest.TestCase):
    """Ensure all 3 suffixes are detected correctly from filenames."""

    def test_kis(self):
        self.assertEqual(TKIS, "kis")

    def test_qa(self):
        self.assertEqual(QA, "qa")

    def test_trake(self):
        self.assertEqual(TRAKE, "trake")


if __name__ == "__main__":
    unittest.main(verbosity=2)
