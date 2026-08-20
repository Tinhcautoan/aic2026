"""Tests for retrieval.answer_expander (R@k-aware answer expansion)."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.answer_expander import SubmissionRow, expand_rows, format_csv_line


class TestFormatCsvLine(unittest.TestCase):
    """Mode-specific CSV line formatting."""

    def test_kis_line(self):
        line = format_csv_line(video_id="L01_V028", frame_idx=3450)
        self.assertEqual(line, "L01_V028, 3450")

    def test_qa_line_simple_answer(self):
        line = format_csv_line(video_id="L01_V028", frame_idx=3450, answer="5")
        self.assertEqual(line, 'L01_V028, 3450, "5"')

    def test_qa_line_answer_with_comma(self):
        line = format_csv_line(video_id="L01_V028", frame_idx=3450, answer="Năm người, gồm cả")
        self.assertEqual(line, 'L01_V028, 3450, "Năm người, gồm cả"')

    def test_trake_line(self):
        line = format_csv_line(video_id="L10_V001", frames=[1200, 1850, 2100, 2450])
        self.assertEqual(line, "L10_V001, 1200, 1850, 2100, 2450")


class TestExpandRows(unittest.TestCase):
    """Sort by score, cap at max_rows."""

    def test_sorts_descending(self):
        rows = [
            SubmissionRow(line="V1,10", score=0.5),
            SubmissionRow(line="V1,20", score=0.9),
            SubmissionRow(line="V2,10", score=0.7),
        ]
        result = expand_rows(rows, max_rows=100)
        self.assertEqual(result[0].line, "V1,20")
        self.assertEqual(result[1].line, "V2,10")
        self.assertEqual(result[2].line, "V1,10")

    def test_caps_at_max_rows(self):
        rows = [SubmissionRow(line=f"V1,{i}", score=1.0 - i * 0.01) for i in range(200)]
        result = expand_rows(rows, max_rows=100)
        self.assertEqual(len(result), 100)
        self.assertEqual(result[0].score, 1.0)

    def test_empty_input(self):
        self.assertEqual(expand_rows([], max_rows=100), [])

    def test_deduplicates_same_frame(self):
        rows = [
            SubmissionRow(line="V1,10", score=0.9),
            SubmissionRow(line="V1,10", score=0.8),
        ]
        result = expand_rows(rows, max_rows=100)
        self.assertEqual(len(result), 2)

    def test_returns_as_many_as_available_under_limit(self):
        rows = [SubmissionRow(line="V1,10", score=0.9)]
        result = expand_rows(rows, max_rows=100)
        self.assertEqual(len(result), 1)


class TestExpandRowsIntegration(unittest.TestCase):
    """End-to-end: expand_rows returns only formatted lines."""

    def test_returns_only_string_lines(self):
        rows = [
            SubmissionRow(line='V1,100,"answer"', score=0.9),
            SubmissionRow(line="V2,200", score=0.7),
        ]
        result = expand_rows(rows, max_rows=2)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], SubmissionRow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
