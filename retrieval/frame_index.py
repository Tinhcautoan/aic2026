"""
Compact frame index for the PE-Core-only retrieval engine.

Everything lives in one fp16 matrix on the GPU. A query is a single matrix-vector
product over ALL frames (232k x 1280 in ~1 ms on an RTX 4060), so there is no
approximate search, no vector database and no candidate pool that could miss a
frame. Rows are sorted by (video, frame_idx), so every video is one contiguous
slice -- which is what the TRAKE per-video aggregation and DP alignment need.

On-disk layout (written by scripts/build_index.py):
    emb.npy        float16 [N, D]   L2-normalised PE-Core image embeddings
    frames.npy     int32   [N]      original frame index inside the video
    vid.npy        int32   [N]      row -> index into videos.json (non-decreasing)
    ms.npy         int32   [N]      frame time in milliseconds
    videos.json    {"dim", "count", "videos":[{"id","fps","start","end","source"}, ...]}
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["FrameIndex", "save_index"]


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_index(out_dir: Path, emb: np.ndarray, frames: np.ndarray, vid: np.ndarray, ms: np.ndarray,
               videos: List[dict], extra: Optional[dict] = None) -> None:
    """Write an index. Validates the invariants the search code relies on."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    if emb.shape[0] != n or len(vid) != n or len(ms) != n:
        raise ValueError("emb/frames/vid/ms length mismatch")
    if emb.dtype != np.float16:
        raise ValueError("emb must be float16")
    if n and (np.diff(vid) < 0).any():
        raise ValueError("vid must be non-decreasing (rows sorted by video)")
    # strictly increasing frame_idx inside each video
    same = np.diff(vid) == 0
    if n and (np.diff(frames)[same] <= 0).any():
        raise ValueError("frame_idx must be strictly increasing inside each video")
    _atomic_save_npy(out_dir / "emb.npy", np.ascontiguousarray(emb))
    _atomic_save_npy(out_dir / "frames.npy", frames.astype(np.int32))
    _atomic_save_npy(out_dir / "vid.npy", vid.astype(np.int32))
    _atomic_save_npy(out_dir / "ms.npy", ms.astype(np.int32))
    meta = {"dim": int(emb.shape[1]), "count": int(n), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos": videos, **(extra or {})}
    tmp = out_dir / "videos.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_dir / "videos.json")   # written last: its presence means the index is complete


@dataclass
class Hit:
    pos: int
    video_id: str
    frame_idx: int
    ms: int
    fps: float
    score: float


class FrameIndex:
    def __init__(self, emb: torch.Tensor, frames: np.ndarray, vid: np.ndarray, ms: np.ndarray, videos: List[dict]):
        self.emb = emb                                   # [N, D] on the search device
        self.frames = frames.astype(np.int64)
        self.vid = vid.astype(np.int64)
        self.ms = ms.astype(np.int64)
        self.videos = videos
        self.device = emb.device
        self.n, self.dim = int(emb.shape[0]), int(emb.shape[1])
        self.video_ids: List[str] = [v["id"] for v in videos]
        self.fps = np.array([float(v.get("fps") or 25.0) for v in videos], dtype=np.float64)
        self.start = np.array([int(v["start"]) for v in videos], dtype=np.int64)
        self.end = np.array([int(v["end"]) for v in videos], dtype=np.int64)
        self._by_id: Dict[str, int] = {vid_: i for i, vid_ in enumerate(self.video_ids)}
        self.vid_t = torch.from_numpy(self.vid).to(self.device)          # row -> video, for segment reductions
        self.frame_counts = (self.end - self.start)

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, index_dir: Path, device: Optional[str] = None) -> "FrameIndex":
        index_dir = Path(index_dir)
        meta_path = index_dir / "videos.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Chưa có index tại {index_dir} (thiếu videos.json). Chạy: python scripts/build_index.py")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        emb = np.load(index_dir / "emb.npy", mmap_mode="r")
        frames = np.load(index_dir / "frames.npy")
        vid = np.load(index_dir / "vid.npy")
        ms = np.load(index_dir / "ms.npy")
        n, d = int(meta["count"]), int(meta["dim"])
        if emb.shape != (n, d) or len(frames) != n or len(vid) != n or len(ms) != n:
            raise ValueError(f"Index hỏng/không đồng bộ: emb{emb.shape} frames{len(frames)} vid{len(vid)} ms{len(ms)} vs videos.json ({n},{d})")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.from_numpy(np.ascontiguousarray(emb))
        t = t.to(device) if device.startswith("cuda") else t.float()
        return cls(t, frames, vid, ms, meta["videos"])

    # ---------------------------------------------------------------- lookup

    def video_index(self, video_id: str) -> int:
        try:
            return self._by_id[video_id]
        except KeyError:
            raise KeyError(f"video không có trong index: {video_id}")

    def has_video(self, video_id: str) -> bool:
        return video_id in self._by_id

    def slice_of(self, v: int) -> Tuple[int, int]:
        return int(self.start[v]), int(self.end[v])

    def find_row(self, video_id: str, frame_idx: int) -> int:
        """Row of the indexed keyframe nearest to `frame_idx` in that video."""
        v = self.video_index(video_id)
        s, e = self.slice_of(v)
        if e <= s:
            raise KeyError(f"video không có khung nào: {video_id}")
        fr = self.frames[s:e]
        j = int(np.searchsorted(fr, frame_idx))
        if j <= 0:
            return s
        if j >= len(fr):
            return e - 1
        return s + (j - 1 if frame_idx - fr[j - 1] <= fr[j] - frame_idx else j)

    def neighbors(self, video_id: str, frame_idx: int, n: int) -> List[dict]:
        row = self.find_row(video_id, frame_idx)
        v = int(self.vid[row])
        s, e = self.slice_of(v)
        lo, hi = max(s, row - n), min(e, row + n + 1)
        return [{"frame_idx": int(self.frames[r]), "ms": int(self.ms[r])} for r in range(lo, hi)]

    def hit(self, pos: int, score: float) -> Hit:
        v = int(self.vid[pos])
        return Hit(pos, self.video_ids[v], int(self.frames[pos]), int(self.ms[pos]), float(self.fps[v]), float(score))

    # ---------------------------------------------------------------- search

    def _q(self, q: torch.Tensor) -> torch.Tensor:
        return q.to(self.device, dtype=self.emb.dtype).reshape(-1)

    @torch.inference_mode()
    def top_frames(self, q: torch.Tensor, k: int, nms_sec: float = 6.0,
                   video_ids: Optional[Sequence[str]] = None, pool: Optional[int] = None,
                   per_video: int = 0) -> List[Hit]:
        """Best `k` frames for one query vector, at most one per `nms_sec` window of a video and, when
        `per_video` > 0, at most that many frames per video (densely sampled videos would otherwise fill the list).

        The ranking pool comes from an fp16 matrix-vector product over every frame; the
        pool is then re-scored in fp32 so ties that fp16 rounding would create are broken
        by the true cosine. If the caps leave the list short, the pool is widened.
        """
        q16 = self._q(q)
        scores = torch.mv(self.emb, q16)
        if video_ids:
            mask = torch.full((len(self.video_ids),), float("-inf"), device=self.device, dtype=scores.dtype)
            for vid_ in video_ids:
                if vid_ in self._by_id:
                    mask[self._by_id[vid_]] = 0
            scores = scores + mask[self.vid_t]
        pool = min(self.n, pool or max(600, 12 * k))
        while True:
            kept = self._pick(scores, q16, pool, k, nms_sec, per_video)
            if len(kept) >= k or pool >= self.n:
                return kept
            pool = min(self.n, pool * 4)

    def _pick(self, scores: torch.Tensor, q16: torch.Tensor, pool: int, k: int, nms_sec: float,
              per_video: int) -> List[Hit]:
        vals, idx = torch.topk(scores, pool)
        rows = self.emb[idx].float()
        exact = torch.mv(rows, q16.float())
        exact = torch.where(torch.isfinite(vals), exact, torch.full_like(exact, float("-inf")))   # keep filtered-out rows out
        order = torch.argsort(exact, descending=True)
        idx_np = idx[order].cpu().numpy()
        exact_np = exact[order].cpu().numpy()

        win = int(nms_sec * 1000)
        kept: List[Hit] = []
        taken: Dict[int, List[int]] = {}
        for pos, sc in zip(idx_np, exact_np):
            if not np.isfinite(sc):
                continue
            v = int(self.vid[pos])
            t = int(self.ms[pos])
            ts = taken.setdefault(v, [])
            if per_video > 0 and len(ts) >= per_video:
                continue
            if win > 0 and any(abs(t - x) < win for x in ts):
                continue
            ts.append(t)
            kept.append(self.hit(int(pos), float(sc)))
            if len(kept) >= k:
                break
        return kept
