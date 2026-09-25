"""scripts/build_index.py: batch2 shards + extra old-corpus shards (fps travels inside the .npz)."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from retrieval.frame_index import FrameIndex

ROOT = Path(__file__).resolve().parent.parent


def unit_rows(n, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, 1280)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def shard(path, frames, seed, fps=None, scale=1.0):
    kw = {} if fps is None else {"fps": np.float32(fps)}
    np.savez(path, frame_idx=np.array(frames, np.int32), pe_core=unit_rows(len(frames), seed) * scale, **kw)


def make_video(path, fps):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for _ in range(10):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()


def run(tmp_path, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_index.py"),
         "--npz-dir", str(tmp_path / "b2"), "--videos", str(tmp_path / "vid"), "--keyframes", str(tmp_path / "none"),
         "--out", str(tmp_path / "index"), *extra],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT)


def setup(tmp_path):
    (tmp_path / "b2").mkdir()
    (tmp_path / "vid").mkdir()
    (tmp_path / "extra").mkdir()
    shard(tmp_path / "b2" / "N001-V001.npz", [0, 30, 60], 1)
    make_video(tmp_path / "vid" / "N001-V001.avi", 30)


def test_extra_shards_are_indexed_with_their_own_fps(tmp_path):
    setup(tmp_path)
    shard(tmp_path / "extra" / "L26_V074.npz", [0, 25, 50, 75], 2, fps=25.0)
    shard(tmp_path / "extra" / "L21_V001.npz", [0, 5, 10], 3, fps=29.97)
    r = run(tmp_path, "--extra-npz-dir", str(tmp_path / "extra"))
    assert r.returncode == 0, r.stdout + r.stderr
    idx = FrameIndex.load(tmp_path / "index", device="cpu")
    assert idx.video_ids == ["N001-V001", "L21_V001", "L26_V074"]
    assert idx.n == 3 + 3 + 4
    v = idx.video_index("L26_V074")
    s, e = idx.slice_of(v)
    assert list(idx.frames[s:e]) == [0, 25, 50, 75]
    assert list(idx.ms[s:e]) == [0, 1000, 2000, 3000]              # frame * 1000 / 25 fps
    assert idx.videos[v]["source"] == "batch1-old" and idx.videos[v]["fps"] == 25.0
    assert idx.videos[idx.video_index("N001-V001")]["source"] == "batch2"
    v2 = idx.video_index("L21_V001")
    s2, e2 = idx.slice_of(v2)
    assert list(idx.ms[s2:e2]) == [0, 167, 334]                    # 5 / 29.97 s, rounded


def test_bad_extra_shards_are_reported_and_left_out(tmp_path):
    setup(tmp_path)
    shard(tmp_path / "extra" / "L26_V074.npz", [0, 25], 2, fps=25.0)             # good
    shard(tmp_path / "extra" / "L26_V079.npz", [0, 25], 4, fps=25.0, scale=1.5)  # not unit norm
    shard(tmp_path / "extra" / "L26_V328.npz", [0, 25], 5)                       # no fps stored
    shard(tmp_path / "extra" / "L26_V340.npz", [0, 25], 6, fps=0.0)              # fps 0
    shard(tmp_path / "extra" / "N001-V001.npz", [0, 25], 7, fps=25.0)            # id already indexed
    r = run(tmp_path, "--extra-npz-dir", str(tmp_path / "extra"))
    assert r.returncode == 2, r.stdout + r.stderr
    out = r.stdout
    assert "L26_V079: not unit norm" in out
    assert "L26_V328: no usable fps" in out and "L26_V340: no usable fps" in out
    assert "N001-V001 already indexed" in out
    idx = FrameIndex.load(tmp_path / "index", device="cpu")      # the good videos are still indexed
    assert idx.video_ids == ["N001-V001", "L26_V074"]
    assert idx.n == 3 + 2


def test_extra_keyframe_folder_is_cross_checked(tmp_path):
    setup(tmp_path)
    shard(tmp_path / "extra" / "L26_V074.npz", [0, 25, 50], 2, fps=25.0)
    shard(tmp_path / "extra" / "L26_V079.npz", [0, 25, 50], 3, fps=25.0)
    kf = tmp_path / "kf"
    kf.mkdir()
    for f in (0, 25, 50):
        (kf / f"L26_V074_frame_{f:06d}.jpg").write_bytes(b"x")
    for f in (0, 25):                                                            # one keyframe missing on disk
        (kf / f"L26_V079_frame_{f:06d}.jpg").write_bytes(b"x")
    r = run(tmp_path, "--extra-npz-dir", str(tmp_path / "extra"), "--extra-keyframes", str(kf))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "L26_V079: frame set differs from extracted keyframes (3 vs 2)" in r.stdout
    idx = FrameIndex.load(tmp_path / "index", device="cpu")
    assert "L26_V074" in idx.video_ids and "L26_V079" not in idx.video_ids


def test_without_extra_dir_nothing_changes(tmp_path):
    setup(tmp_path)
    r = run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    idx = FrameIndex.load(tmp_path / "index", device="cpu")
    assert idx.video_ids == ["N001-V001"] and idx.n == 3


def make_zip(path, video, frames, jpeg=b"\xff\xd8fake\xff\xd9"):
    import zipfile
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for f in frames:
            z.writestr(f"{video}_frame_{f:06d}.jpg", jpeg)


def test_ingest_script_indexes_only_validated_videos(tmp_path):
    setup(tmp_path)
    work = tmp_path / "work"
    (work / "embeddings_pecore_old_missing").mkdir(parents=True)
    (work / "keyframes_old_missing").mkdir()
    emb, zips = work / "embeddings_pecore_old_missing", work / "keyframes_old_missing"
    shard(emb / "L26_V074.npz", [0, 25, 50], 2, fps=25.0)              # good
    make_zip(zips / "L26_V074.zip", "L26_V074", [0, 25, 50])
    shard(emb / "L26_V079.npz", [0, 25, 50], 3, fps=25.0)              # zip is missing one image
    make_zip(zips / "L26_V079.zip", "L26_V079", [0, 25])
    shard(emb / "L26_V328.npz", [0, 25], 4, fps=25.0)                  # no zip at all
    kf = tmp_path / "kf"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ingest_old_missing.py"), "--skip-download", "--work", str(work),
         "--keyframes", str(kf), "--out", str(tmp_path / "index"), "--old-dir", "", "--npz-dir", str(tmp_path / "b2"),
         "--videos", str(tmp_path / "vid"), "--batch2-keyframes", str(tmp_path / "none")],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "L26_V079: zip lỗi" in r.stdout and "L26_V328: có vector nhưng thiếu zip" in r.stdout
    assert sorted(p.name for p in kf.iterdir()) == [f"L26_V074_frame_{f:06d}.jpg" for f in (0, 25, 50)]
    idx = FrameIndex.load(tmp_path / "index", device="cpu")
    assert idx.video_ids == ["N001-V001", "L26_V074"] and idx.n == 6
    assert idx.videos[1]["source"] == "batch1-old"
    # a second run changes nothing and still succeeds
    r2 = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ingest_old_missing.py"), "--skip-download", "--work", str(work),
         "--keyframes", str(kf), "--out", str(tmp_path / "index"), "--old-dir", "", "--npz-dir", str(tmp_path / "b2"),
         "--videos", str(tmp_path / "vid"), "--batch2-keyframes", str(tmp_path / "none")],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert not list(kf.glob("*.part"))
