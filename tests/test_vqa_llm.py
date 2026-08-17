"""
Tests for the Gemini-powered VQA module (retrieval.vqa_llm).
Uses a mocked httpx client — no real API calls.
"""

import sys
import os
import unittest
import base64
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vqa_llm import (
    answer_from_llm,
    build_llm_payload,
    _extract_answer,
    _empty,
)


def _make_jpeg() -> bytes:
    """Produce a tiny but valid JPEG using Pillow."""
    from PIL import Image
    import io

    img = Image.new("RGB", (16, 16), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


TINY_JPEG = _make_jpeg()


class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class TestBuildPayload(unittest.TestCase):
    def test_payload_contains_question_and_images(self):
        payload = build_llm_payload("What color is the car?", [TINY_JPEG, TINY_JPEG])
        contents = payload["contents"][0]
        self.assertIn("Question: What color is the car?", contents["parts"][0]["text"])
        images = [p for p in contents["parts"] if "inline_data" in p]
        self.assertEqual(len(images), 2)
        for img in images:
            self.assertEqual(img["inline_data"]["mime_type"], "image/jpeg")
            decoded = base64.b64decode(img["inline_data"]["data"])
            self.assertEqual(decoded, TINY_JPEG)


class TestExtractAnswer(unittest.TestCase):
    def test_extracts_text(self):
        resp = {"candidates": [{"content": {"parts": [{"text": "Red car."}]}}]}
        self.assertEqual(_extract_answer(resp), "Red car.")

    def test_skips_non_text_parts(self):
        resp = {"candidates": [{"content": {"parts": [{"thoughtSignature": "x"}, {"text": "Blue"}]}}]}
        self.assertEqual(_extract_answer(resp), "Blue")

    def test_empty_on_no_candidates(self):
        self.assertEqual(_extract_answer({"candidates": []}), "")


class TestAnswerFromLLM(unittest.IsolatedAsyncioTestCase):
    def _results(self):
        return [
            {"video_id": "v1", "frame_idx": 10, "timestamp": 0.4, "final_score": 0.9},
            {"video_id": "v1", "frame_idx": 11, "timestamp": 0.44, "final_score": 0.8},
            {"video_id": "v1", "frame_idx": 12, "timestamp": 0.48, "final_score": 0.7},
        ]

    async def _loader(self, video_id, frame_idx):
        return TINY_JPEG

    async def test_returns_llm_answer_and_evidence(self):
        resp = FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "A red car."}]}}]})
        client = AsyncMock()
        client.post.return_value = resp

        out = await answer_from_llm(
            question="What color is the car?",
            results=self._results(),
            api_key="test-key",
            top_k=3,
            frame_loader=self._loader,
            client=client,
        )

        self.assertEqual(out["answer"], "A red car.")
        self.assertEqual(out["source"], "llm")
        self.assertEqual(out["evidence_video_id"], "v1")
        self.assertEqual(out["evidence_frame_idx"], 10)
        self.assertGreater(out["confidence"], 0)

        call_kwargs = client.post.call_args[1]
        self.assertEqual(call_kwargs["headers"]["x-goog-api-key"], "test-key")
        payload = call_kwargs["json"]
        parts = payload["contents"][0]["parts"]
        image_parts = [p for p in parts if "inline_data" in p]
        self.assertEqual(len(image_parts), 3)

    async def test_falls_back_to_empty_on_http_error(self):
        resp = FakeResponse(500, {"error": "boom"})
        client = AsyncMock()
        client.post.return_value = resp

        out = await answer_from_llm(
            question="q",
            results=self._results(),
            api_key="k",
            frame_loader=self._loader,
            client=client,
        )
        self.assertEqual(out["answer"], "")
        self.assertEqual(out["source"], "none")

    async def test_empty_result_without_key(self):
        out = await answer_from_llm(
            question="q", results=self._results(), api_key="", client=AsyncMock()
        )
        self.assertEqual(out, _empty("q"))

    async def test_dedupes_frames(self):
        resp = FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]})
        client = AsyncMock()
        client.post.return_value = resp

        await answer_from_llm(
            question="q",
            results=self._results() + [{"video_id": "v1", "frame_idx": 10, "timestamp": 0.4}],
            api_key="k",
            top_k=3,
            frame_loader=self._loader,
            client=client,
        )
        payload = client.post.call_args[1]["json"]
        parts = payload["contents"][0]["parts"]
        image_parts = [p for p in parts if "inline_data" in p]
        self.assertEqual(len(image_parts), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
