"""
Gemini-powered VideoQA answering.

Sends the question together with the top retrieved evidence frames (decoded
on-demand from the source video) to a Gemini vision model, then returns a
short text answer plus the evidence frame that backed it.

If the LLM call fails for any reason, callers should fall back to the
metadata-based answer from retrieval.vqa.answer_from_retrieval_results.
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

import httpx
from loguru import logger

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

MAX_FRAME_DIM = 1024  # downscale very large frames before sending


def _encode_jpeg(frame_bytes: bytes, max_dim: int = MAX_FRAME_DIM) -> Optional[bytes]:
    """Re-encode a raw decoded frame into JPEG, downscaling if needed."""
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max_dim and max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes() if ok else None
    except Exception as e:  # pragma: no cover - depends on cv2 availability
        logger.warning(f"Failed to encode frame: {e}")
        return None


def build_llm_payload(
    question: str,
    jpegs: List[bytes],
    instruction: str | None = None,
) -> Dict[str, Any]:
    """Build the Gemini generateContent request body."""
    if instruction is None:
        instruction = (
            "You are a video question-answering assistant. Below are the most "
            "relevant frames captured from the video. Answer the user's question "
            "based ONLY on what is visible in these frames. If the frames do not "
            "contain enough information, say so briefly. Keep the answer short "
            "(1-3 sentences) and in the same language as the question."
        )
    parts: List[Dict[str, Any]] = [{"text": f"{instruction}\n\nQuestion: {question}"}]
    for jpeg in jpegs:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(jpeg).decode("ascii"),
            }
        })
    return {"contents": [{"role": "user", "parts": parts}]}


def _extract_answer(resp: Dict[str, Any]) -> str:
    """Pull the final text out of a Gemini generateContent response."""
    try:
        candidates = resp.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        for part in parts:
            text = part.get("text")
            if text:
                return text.strip()
        return ""
    except Exception:
        return ""


async def answer_from_llm(
    question: str,
    results: List[dict],
    api_key: str,
    model: str = "gemini-3.5-flash",
    top_k: int = 3,
    timeout: float = 60.0,
    frame_loader: Optional[Callable[[str, int], Optional[bytes]]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """
    Answer a VQA question with Gemini using the top retrieved frames.

    Args:
        question: the VQA question text.
        results: ranked retrieval hits (each has video_id, frame_idx, timestamp).
        api_key: Google AI Studio API key.
        model: Gemini model id (default gemini-3.5-flash).
        top_k: how many evidence frames to send.
        timeout: HTTP timeout in seconds.
        frame_loader: callable(video_id, frame_idx) -> raw JPEG bytes; used to
            decode frames on demand. Defaults to a loader that decodes from the
            video files on disk via cv2.
        client: optional shared httpx.AsyncClient (used for testing).

    Returns:
        dict with keys: answer, evidence_video_id, evidence_frame_idx,
        evidence_timestamp, confidence, source ("llm").
    """
    if not api_key:
        return _empty(question)
    if not results:
        return _empty(question)

    loader = frame_loader or _default_frame_loader

    seen = set()
    frames: List[dict] = []
    for r in results:
        vid = r.get("video_id", "")
        fidx = r.get("frame_idx")
        if fidx is None:
            continue
        key = (vid, fidx)
        if key in seen:
            continue
        seen.add(key)
        frames.append(r)
        if len(frames) >= top_k:
            break

    if not frames:
        return _empty(question)

    jpegs: List[bytes] = []
    for r in frames:
        try:
            raw = await loader(r["video_id"], r["frame_idx"])
        except Exception as e:  # pragma: no cover
            logger.warning(f"Frame load failed {r.get('video_id')}:{r.get('frame_idx')} - {e}")
            raw = None
        jpeg = _encode_jpeg(raw) if raw else None
        if jpeg:
            jpegs.append(jpeg)

    if not jpegs:
        return _empty(question)

    payload = build_llm_payload(question, jpegs)

    start = time.time()
    try:
        url = GEMINI_GENERATE_URL.format(model=model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        if client is not None:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
        else:
            async with httpx.AsyncClient() as c:
                resp = await c.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Gemini VQA request failed: {e}")
        return _empty(question)

    answer = _extract_answer(data)
    best = frames[0]
    latency = time.time() - start

    if not answer:
        return _empty(question)

    return {
        "answer": answer,
        "evidence_video_id": best.get("video_id", ""),
        "evidence_frame_idx": best.get("frame_idx"),
        "evidence_timestamp": best.get("timestamp"),
        "confidence": float(best.get("final_score", best.get("clip_score", 0.0)) or 0.0),
        "source": "llm",
        "latency_ms": latency * 1000.0,
        "model": model,
    }


def _empty(question: str) -> Dict[str, Any]:
    return {
        "answer": "",
        "evidence_video_id": "",
        "evidence_frame_idx": None,
        "evidence_timestamp": None,
        "confidence": 0.0,
        "source": "none",
    }


async def _default_frame_loader(video_id: str, frame_idx: int) -> Optional[bytes]:
    """Decode a single frame from the source video on disk."""
    try:
        import cv2
        from api.app import resolve_video_path

        path = resolve_video_path(video_id)
        if not path:
            return None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        ok, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes() if ok else None
    except Exception as e:  # pragma: no cover
        logger.warning(f"Default frame loader failed for {video_id}:{frame_idx} - {e}")
        return None


# ------------------------------------------------------------------ #
# Self-hosted VLM backend (vLLM OpenAI-compatible endpoint)
# ------------------------------------------------------------------ #

VLM_DEFAULT_INSTRUCTION = (
    "Below are candidate frames from videos. Each is labeled [video_id, frame_idx].\n"
    "Answer the question based ONLY on what is visible in these frames.\n"
    "Respond in the SAME language as the question.\n"
    "For Q&A: answer in the exact format: <video_id>, <frame_id>, \"<answer>\"\n"
    "For TRAKE: answer in the exact format: <video_id>, <frame1>, <frame2>, ..., <frameN>\n"
    "Do NOT add any explanation."
)


def build_vlm_payload(
    question: str,
    frames: List[Dict[str, Any]],
    instruction: str | None = None,
) -> Dict[str, Any]:
    """
    Build an OpenAI-compatible chat/completions request body for a VLM.

    Each frame dict must contain ``video_id``, ``frame_idx``, ``jpeg``
    (raw JPEG bytes), and ``score``.
    """
    if instruction is None:
        instruction = VLM_DEFAULT_INSTRUCTION

    content: List[Dict[str, Any]] = []

    # Add images with labels in the text
    for i, frame in enumerate(frames):
        jpeg = frame.get("jpeg")
        if jpeg is None:
            continue
        b64 = base64.b64encode(jpeg).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    # Build labeled prompt
    labels = ", ".join(
        f"[{f['video_id']}, {f['frame_idx']}]" for f in frames
    )
    prompt_text = (
        f"{instruction}\n\n"
        f"Frames: {labels}\n\n"
        f"Question: {question}"
    )
    content.append({"type": "text", "text": prompt_text})

    return {
        "model": "",  # filled by caller
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 150,
    }


_VQA_RE = re.compile(
    r"^([^,]+),\s*(\d+),\s*(?:\"(.+?)\"|(.+))$",
    re.DOTALL,
)
_TRAKE_RE = re.compile(
    r"^([^,]+),\s*(\d+(?:\s*,\s*\d+)+)$",
)


def _parse_vlm_answer(raw: str, mode: str | None = None) -> Optional[Dict[str, Any]]:
    """
    Parse the VLM output into a structured answer dict.

    Args:
        raw: the raw text from the VLM.
        mode: "qa" to parse as Q&A only, "trake" to parse as TRAKE only,
            or None to try both (Q&A first, then TRAKE).

    Returns ``{"video_id", "frame_idx", "answer"}`` for Q&A,
    ``{"video_id", "frames": [...]}`` for TRAKE, or ``None`` on failure.
    """
    raw = raw.strip()
    if not raw:
        return None

    try_qa = mode in (None, "qa")
    try_trake = mode in (None, "trake")

    # Try TRAKE first when explicitly requested
    if try_trake:
        m = _TRAKE_RE.match(raw)
        if m:
            vid = m.group(1)
            frames = [int(x) for x in m.group(2).split(",")]
            if len(frames) >= 2:
                return {"video_id": vid, "frames": frames, "source": "vlm"}

    # Try Q&A
    if try_qa:
        m = _VQA_RE.match(raw)
        if m:
            vid = m.group(1)
            frame_idx = int(m.group(2))
            answer = (m.group(3) or m.group(4) or "").strip()
            return {"video_id": vid, "frame_idx": frame_idx, "answer": answer, "source": "vlm"}

    # Fallback: TRAKE with 1 frame (degenerate case)
    if try_trake:
        m = _TRAKE_RE.match(raw)
        if m:
            vid = m.group(1)
            frames = [int(x) for x in m.group(2).split(",")]
            return {"video_id": vid, "frames": frames, "source": "vlm"}

    return None


async def answer_from_vlm(
    question: str,
    frames: List[Dict[str, Any]],
    model: str = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    vlm_url: str = "http://localhost:8080",
    timeout: float = 120.0,
    instruction: str | None = None,
    client: Optional[Any] = None,
    mode: str | None = None,
) -> Dict[str, Any]:
    """
    Answer a question using a self-hosted VLM via OpenAI-compatible API.

    Args:
        question: VQA or TRAKE question text.
        frames: list of dicts, each with ``video_id``, ``frame_idx``,
            ``jpeg`` (bytes), and ``score``.
        model: model name to pass to vLLM.
        vlm_url: base URL of the vLLM server (no trailing slash).
        timeout: HTTP timeout in seconds.
        instruction: system prompt override.
        client: optional shared ``httpx.AsyncClient`` (for testing).

    Returns:
        dict with keys: answer, video_id, frame_idx (Q&A) or frames (TRAKE),
        source ("vlm"), confidence, latency_ms, model.
    """
    if not frames:
        return _vlm_empty()

    payload = build_vlm_payload(question, frames, instruction)
    payload["model"] = model

    url = f"{vlm_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    start = time.time()
    try:
        if client is not None:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
        else:
            async with httpx.AsyncClient() as c:
                resp = await c.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"VLM request failed: {e}")
        return _vlm_empty()

    raw_text = ""
    try:
        choices = data.get("choices") or []
        if choices:
            raw_text = choices[0].get("message", {}).get("content", "")
    except Exception:
        pass

    parsed = _parse_vlm_answer(raw_text, mode=mode)
    latency = time.time() - start

    if not parsed:
        return _vlm_empty()

    best_frame = frames[0] if frames else {}
    result = {
        "answer": parsed.get("answer", ""),
        "video_id": parsed.get("video_id", best_frame.get("video_id", "")),
        "frame_idx": parsed.get("frame_idx"),
        "frames": parsed.get("frames"),
        "confidence": float(best_frame.get("score", 0.0)),
        "source": "vlm",
        "latency_ms": latency * 1000.0,
        "model": model,
    }
    return result


def _vlm_empty() -> Dict[str, Any]:
    return {
        "answer": "",
        "video_id": "",
        "frame_idx": None,
        "frames": None,
        "confidence": 0.0,
        "source": "none",
    }
