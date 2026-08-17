"""Tests for retrieval.query_parser (query text -> main / events / question)."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.query_parser import parse_trake, split_qa, detect_mode, TKIS, QA, TRAKE


class TestDetectMode(unittest.TestCase):
    def test_suffix_kis(self):
        self.assertEqual(detect_mode("query-1-kis.txt"), TKIS)

    def test_suffix_qa(self):
        self.assertEqual(detect_mode("query-2-qa.txt"), QA)

    def test_suffix_trake(self):
        self.assertEqual(detect_mode("query-3-trake.txt"), TRAKE)

    def test_uppercase_suffix(self):
        self.assertEqual(detect_mode("QUERY-1-TRAKE.TXT"), TRAKE)

    def test_unknown_suffix_raises(self):
        with self.assertRaises(ValueError):
            detect_mode("query-1.txt")


class TestParseTrake(unittest.TestCase):
    def test_parses_numbered_events(self):
        text = (
            "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: "
            "(1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
        )
        parsed = parse_trake(text)
        self.assertEqual(parsed.main, "vận động viên thực hiện cú nhảy")
        self.assertEqual(parsed.events, ["giậm nhảy", "bay qua xà", "tiếp đất", "đứng dậy"])
        self.assertEqual(parsed.event_count, 4)

    def test_events_in_order(self):
        text = "Quy trình nấu phở: (1) ninh xương, (2) chần bánh phở, (3) chan nước dùng."
        parsed = parse_trake(text)
        self.assertEqual(parsed.events, ["ninh xương", "chần bánh phở", "chan nước dùng"])
        self.assertEqual(parsed.main, "Quy trình nấu phở")

    def test_parenthesized_number_without_word_nhap(self):
        text = "Cú nhảy cao (1) chạy đà (2) giậm nhảy (3) bay qua xà"
        parsed = parse_trake(text)
        self.assertEqual(parsed.events, ["chạy đà", "giậm nhảy", "bay qua xà"])

    def test_raises_when_no_events(self):
        with self.assertRaises(ValueError):
            parse_trake("Chỉ là một mô tả thông thường, không có event nào.")


class TestSplitQa(unittest.TestCase):
    def test_splits_description_and_question(self):
        text = (
            "Trong video về lễ trao giải thưởng âm nhạc, "
            "có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?"
        )
        parsed = split_qa(text)
        self.assertIn("lễ trao giải thưởng âm nhạc", parsed.description)
        self.assertEqual(parsed.question, "có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?")

    def test_no_question_mark_uses_whole_text(self):
        text = "Diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời"
        parsed = split_qa(text)
        self.assertEqual(parsed.description, text)
        self.assertEqual(parsed.question, text)

    def test_multiple_questions_keeps_last(self):
        text = "Ai nói chuyện với ai? Có bao nhiêu người trong phòng?"
        parsed = split_qa(text)
        self.assertEqual(parsed.question, "Có bao nhiêu người trong phòng?")


if __name__ == "__main__":
    unittest.main(verbosity=2)
