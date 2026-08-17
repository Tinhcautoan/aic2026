"""Tests for retrieval.vqa_llm VLM backend (answer_from_vlm, build_vlm_payload)."""

import sys
import os
import unittest
import base64
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vqa_llm import build_vlm_payload, _parse_vlm_answer, answer_from_vlm


def _make_jpeg() -> bytes:
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


class TestBuildVlmPayload(unittest.TestCase):
    def test_payload_has_images_and_text(self):
        frames = [
            {"video_id": "V1", "frame_idx": 100, "jpeg": TINY_JPEG, "score": 0.9},
            {"video_id": "V1", "frame_idx": 105, "jpeg": TINY_JPEG, "score": 0.8},
        ]
        payload = build_vlm_payload(
            question="How many people on stage?",
            frames=frames,
            instruction="Answer ONLY: <video_id>, <frame_id>, \"<answer>\"",
        )
        messages = payload["messages"]
        self.assertEqual(len(messages), 1)
        content = messages[0]["content"]
        images = [p for p in content if p["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        text_part = [p for p in content if p["type"] == "text"][0]
        self.assertIn("How many people on stage?", text_part["text"])

    def test_frame_labels_in_prompt(self):
        frames = [{"video_id": "V1", "frame_idx": 100, "jpeg": TINY_JPEG, "score": 0.9}]
        payload = build_vlm_payload("q", frames, instruction="Answer.")
        text_part = payload["messages"][0]["content"][-1]["text"]
        self.assertIn("V1", text_part)
        self.assertIn("100", text_part)


class TestParseVlmAnswer(unittest.TestCase):
    def test_parses_vqa_format(self):
        result = _parse_vlm_answer('L01_V028,3450,"5"', mode="qa")
        self.assertEqual(result["video_id"], "L01_V028")
        self.assertEqual(result["frame_idx"], 3450)
        self.assertEqual(result["answer"], "5")

    def test_parses_vqa_format_unquoted(self):
        result = _parse_vlm_answer("L01_V028,3450,5", mode="qa")
        self.assertEqual(result["video_id"], "L01_V028")
        self.assertEqual(result["frame_idx"], 3450)
        self.assertEqual(result["answer"], "5")

    def test_parses_trake_format(self):
        result = _parse_vlm_answer("L10_V001,1200,1850,2100,2450", mode="trake")
        self.assertEqual(result["video_id"], "L10_V001")
        self.assertEqual(result["frames"], [1200, 1850, 2100, 2450])

    def test_returns_none_on_unparseable(self):
        result = _parse_vlm_answer("Some random text with no commas")
        self.assertIsNone(result)

    def test_auto_mode_qa_preferred(self):
        """Without mode, VQA format is tried before TRAKE."""
        result = _parse_vlm_answer('L01_V028,3450,"red"')
        self.assertIsNotNone(result)
        self.assertEqual(result["frame_idx"], 3450)


class TestAnswerFromVlm(unittest.IsolatedAsyncioTestCase):
    def _frames(self):
        return [
            {"video_id": "V1", "frame_idx": 100, "jpeg": TINY_JPEG, "score": 0.9},
            {"video_id": "V1", "frame_idx": 105, "jpeg": TINY_JPEG, "score": 0.85},
        ]

    async def test_returns_parsed_answer(self):
        resp = FakeResponse(200, {
            "choices": [{"message": {"content": 'V1,100,"Red car."'}}]
        })
        client = AsyncMock()
        client.post.return_value = resp

        out = await answer_from_vlm(
            question="What color is the car?",
            frames=self._frames(),
            model="test-model",
            client=client,
        )
        self.assertEqual(out["answer"], "Red car.")
        self.assertEqual(out["video_id"], "V1")
        self.assertEqual(out["frame_idx"], 100)
        self.assertEqual(out["source"], "vlm")

    async def test_empty_on_http_error(self):
        resp = FakeResponse(500, {"error": "boom"})
        client = AsyncMock()
        client.post.return_value = resp

        out = await answer_from_vlm(
            question="q",
            frames=self._frames(),
            model="test-model",
            client=client,
        )
        self.assertEqual(out["answer"], "")
        self.assertEqual(out["source"], "none")

    async def test_sends_correct_endpoint(self):
        resp = FakeResponse(200, {
            "choices": [{"message": {"content": 'V1,100,"OK"'}}]
        })
        client = AsyncMock()
        client.post.return_value = resp

        await answer_from_vlm(
            question="q",
            frames=self._frames(),
            model="qwen2.5-vl-32b",
            vlm_url="http://localhost:8080",
            client=client,
        )
        call_args = client.post.call_args
        self.assertEqual(call_args[0][0], "http://localhost:8080/v1/chat/completions")
        self.assertEqual(call_args[1]["json"]["model"], "qwen2.5-vl-32b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
