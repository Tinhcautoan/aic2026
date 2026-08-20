"""Tests for retrieval.media_meta (contest media-info → per-frame text)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.media_meta import (
    clean_description,
    date_variants_text,
    frame_metadata_fields,
    load_media_info,
    normalize_video_id,
    video_frame_count,
)
from retrieval.bm25_search import (
    BM25Index,
    build_bm25_index_from_store,
    build_frame_document_text,
    expand_query_dates,
    simple_tokenize,
)


class TestMediaInfoLoading(unittest.TestCase):
    def test_loads_real_contest_json(self):
        info = load_media_info("L21_V001")
        self.assertIsNotNone(info)
        self.assertIn("60 Giây", info.get("title", ""))
        self.assertEqual(info.get("publish_date"), "01/08/2024")

    def test_missing_video_returns_none(self):
        self.assertIsNone(load_media_info("ZZZ_999"))

    def test_normalize_video_id(self):
        self.assertEqual(normalize_video_id("L21_V001.mp4"), "L21_V001")
        self.assertEqual(normalize_video_id("data/L21_V001"), "L21_V001")


class TestDateVariants(unittest.TestCase):
    def test_dmy_publish_date(self):
        text = date_variants_text("01/08/2024")
        for expected in ("01/08/2024", "1/8/2024", "01082024", "2024-08-01", "ngày 1 tháng 8 năm 2024"):
            self.assertIn(expected, text)

    def test_compact_title_date(self):
        text = date_variants_text("Ngày 01082024 - HTV Tin Tức")
        self.assertIn("01/08/2024", text)
        self.assertIn("1/8/2024", text)

    def test_unknown_date_passthrough(self):
        self.assertEqual(date_variants_text("không có ngày"), "không có ngày")


class TestCleanDescription(unittest.TestCase):
    def test_strips_urls_and_boilerplate(self):
        desc = (
            "60 Giây Sáng - Ngày 01/08/2024 - HTV Tin Tức Mới Nhất 2024\n"
            "► Đăng ký KÊNH: https://bit.ly/2HoUna4\n"
            "©️ Bản quyền thuộc HTV\n"
            "#Tintuc #HTV\n"
            "Tổng hợp tin tức nổi bật trong ngày.\n"
        )
        cleaned = clean_description(desc)
        self.assertIn("60 Giây Sáng", cleaned)
        self.assertIn("Tổng hợp tin tức", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("Bản quyền", cleaned)

    def test_empty(self):
        self.assertEqual(clean_description(""), "")


class TestFrameMetadataFields(unittest.TestCase):
    def test_real_video_fields(self):
        fields = frame_metadata_fields("L21_V001")
        self.assertEqual(fields["video_publish_date"], "01/08/2024")
        self.assertIn("video_title", fields)
        self.assertIsInstance(fields["video_keywords"], list)
        self.assertTrue(fields["video_title"])

    def test_missing_video(self):
        self.assertEqual(frame_metadata_fields("ZZZ_999"), {})

    def test_video_frame_count(self):
        self.assertEqual(video_frame_count("L21_V001"), 1262 * 25)


class TestFrameDocumentText(unittest.TestCase):
    def test_video_fields_in_document(self):
        meta = frame_metadata_fields("L21_V001")
        text = build_frame_document_text(meta)
        self.assertIn("60 Giây", text)
        self.assertIn("1/8/2024", text)  # date variants expanded

    def test_empty_metadata(self):
        self.assertEqual(build_frame_document_text({}), "")


class TestBM25WithMediaText(unittest.TestCase):
    def _index(self):
        idx = BM25Index()
        # A 5-video corpus so a term appearing in a single doc gets positive
        # BM25 IDF (with only 2 docs, df=1 terms get idf=log(1)=0 and scores tie).
        vids = ["L21_V001", "L21_V002", "L21_V003", "L21_V005", "L21_V006"]
        for i, vid in enumerate(vids):
            idx.add_documents([
                {
                    "video_id": vid,
                    "frame_idx": 100 + i,
                    "metadata": frame_metadata_fields(vid),
                },
            ])
        idx.build()
        return idx

    def test_date_query_matches(self):
        idx = self._index()
        hits = idx.search("bản tin 60 giây ngày 1/8/2024", top_k=5)
        self.assertTrue(hits)
        # L21_V001 is dated 01/08/2024, L21_V002 is 02/08/2024 — the date
        # tokens must push the matching video to the top.  (Note: rank_bm25
        # returns negative scores for terms that appear in every doc, so only
        # compare relative ranking, not the sign of the score.)
        self.assertEqual(hits[0]["video_id"], "L21_V001")
        self.assertNotEqual(hits[0]["bm25_score"], 0.0)
        self.assertGreater(hits[0]["bm25_score"], hits[1]["bm25_score"])

    def test_tokenizer_keeps_vietnamese(self):
        toks = simple_tokenize("60 Giây Sáng ngày 01/08/2024")
        self.assertIn("giây", toks)
        self.assertIn("2024", toks)

    def test_build_from_store(self):
        class FakeStore:
            def get_all_frame_records(self, batch_size=10000):
                return [
                    {"video_id": "L21_V001", "frame_idx": 1, "metadata": frame_metadata_fields("L21_V001")},
                ]

        idx = BM25Index()
        n = build_bm25_index_from_store(FakeStore(), idx)
        self.assertEqual(n, 1)
        self.assertEqual(idx.size, 1)

    def test_versioned_persistence_rebuild(self):
        import os
        import tempfile

        idx = BM25Index()
        idx.add_documents([
            {"video_id": "L21_V001", "frame_idx": 1, "metadata": frame_metadata_fields("L21_V001")},
        ])
        idx.build()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bm25.pkl")
            idx.save(path)
            # Corrupt the version to simulate a stale index.
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            data["version"] = data["version"] - 1
            with open(path, "wb") as f:
                pickle.dump(data, f)

            fresh = BM25Index()
            self.assertFalse(fresh.load(path))  # stale -> rebuild required


class TestExpandQueryDates(unittest.TestCase):
    def test_dmy_query_expansion(self):
        out = expand_query_dates("bản tin ngày 1/8/2024")
        self.assertIn("01082024", out)
        self.assertIn("01/08/2024", out)
        self.assertIn("ngày 1 tháng 8 năm 2024", out)

    def test_no_date_passthrough(self):
        self.assertEqual(expand_query_dates("thời sự trong ngày"), "thời sự trong ngày")


if __name__ == "__main__":
    unittest.main()
