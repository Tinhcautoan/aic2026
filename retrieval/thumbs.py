"""
Thumbnails and video lookup for the UI.

Order of preference for a requested (video, frame):
  1. the extracted keyframe JPEG  <dir>/<video>_frame_<idx:06d>.jpg   (fast, ~2 ms with JPEG draft mode)
  2. the nearest INDEXED keyframe's JPEG  (the operator picked a time between keyframes)
  3. decode that frame from the source video

Finished thumbnails live in an in-memory LRU, and `prefetch` lets the search endpoint
warm the cache for a result page in background threads, so by the time the browser asks
for the images they are already encoded.
"""

from __future__ import annotations

import io
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from loguru import logger
from PIL import Image

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


class ThumbService:
    def __init__(self, keyframe_dirs: Sequence[Path], video_dir: Path, width: int = 384,
                 cache_items: int = 6000, index=None, workers: int = 6):
        self.dirs = [Path(d) for d in keyframe_dirs if str(d).strip()]
        self.video_dir = Path(video_dir)
        self.width = int(width)
        self.index = index                                   # FrameIndex (optional) for nearest-keyframe lookup
        self._cache: "OrderedDict[Tuple[str, int], bytes]" = OrderedDict()
        self._cache_max = int(cache_items)
        self._lock = threading.Lock()
        self._videos: Optional[dict] = None
        self._vlock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="thumb")

    # ---------------------------------------------------------------- videos

    def video_path(self, video_id: str) -> Optional[Path]:
        if self._videos is None:
            with self._vlock:
                if self._videos is None:
                    found = {}
                    if self.video_dir.is_dir():
                        for root, _, files in os.walk(self.video_dir):
                            for f in files:
                                if f.lower().endswith(VIDEO_EXT):
                                    found.setdefault(Path(f).stem, Path(root) / f)
                    self._videos = found
                    logger.info(f"ThumbService: {len(found)} video files under {self.video_dir}")
        return self._videos.get(Path(video_id).stem)

    # ----------------------------------------------------------------- cache

    def _get_cached(self, key):
        with self._lock:
            v = self._cache.get(key)
            if v is not None:
                self._cache.move_to_end(key)
            return v

    def _put(self, key, data: bytes) -> None:
        with self._lock:
            self._cache[key] = data
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

    # ----------------------------------------------------------------- files

    def _jpeg_path(self, video_id: str, frame_idx: int) -> Optional[Path]:
        name = f"{video_id}_frame_{frame_idx:06d}.jpg"
        for d in self.dirs:
            p = d / name
            try:
                if p.is_file():
                    return p
            except OSError:                 # a drive that dropped out (e.g. WinError 433) must not become a 500
                continue
        return None

    def _encode(self, im: Image.Image, width: Optional[int] = None) -> bytes:
        w = width or self.width
        im = im.convert("RGB")
        if im.width > w:
            im.thumbnail((w, w * 3), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return buf.getvalue()

    def _from_jpeg(self, path: Path, width: Optional[int] = None) -> Optional[bytes]:
        try:
            with Image.open(path) as im:
                im.draft("RGB", (width or self.width, width or self.width))      # decode at 1/2, 1/4, 1/8 scale: several x faster
                return self._encode(im, width)
        except Exception as e:
            logger.warning(f"thumbnail read failed for {path}: {e}")
            return None

    def _from_video(self, video_id: str, frame_idx: int, width: Optional[int] = None) -> Optional[bytes]:
        vp = self.video_path(video_id)
        if vp is None:
            return None
        try:
            import cv2
            cap = cv2.VideoCapture(str(vp))
            try:
                if not cap.isOpened():
                    return None
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, bgr = cap.read()
            finally:
                cap.release()
            if not ok:
                return None
            im = Image.fromarray(bgr[:, :, ::-1])
            return self._encode(im, width)
        except Exception as e:
            logger.warning(f"video frame decode failed for {video_id}@{frame_idx}: {e}")
            return None

    # ------------------------------------------------------------------- api

    def get(self, video_id: str, frame_idx: int, width: Optional[int] = None) -> Optional[bytes]:
        key = (video_id, int(frame_idx), int(width or 0))
        data = self._get_cached(key)
        if data is not None:
            return data
        data = None
        p = self._jpeg_path(video_id, frame_idx)
        if p is None and self.index is not None and self.index.has_video(video_id):
            try:
                row = self.index.find_row(video_id, frame_idx)
                p = self._jpeg_path(video_id, int(self.index.frames[row]))
            except KeyError:
                p = None
        if p is not None:
            data = self._from_jpeg(p, width)
        if data is None:
            data = self._from_video(video_id, frame_idx, width)
        if data is not None:
            self._put(key, data)
        return data

    def prefetch(self, items: Iterable[Tuple[str, int]]) -> None:
        for v, f in items:
            if self._get_cached((v, int(f), 0)) is None:
                self._pool.submit(self.get, v, int(f))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
