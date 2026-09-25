"""Build the compact PE-Core frame index from the Colab embedding shards (one .npz per video).

    python scripts/build_index.py

Reads   D:\\aic_pecore_embeddings\\embeddings_pecore\\<video>.npz   (frame_idx int32, pe_core float32 [n,1280])
        the video files (only their headers, for fps)                 -> ms = frame_idx * 1000 / fps
Checks  every shard (finite, unit norm, strictly increasing frames) and, where the keyframe JPEGs are on
        disk, that the frame set is EXACTLY the extracted keyframes.
Writes  INDEX_DIR (default D:\\aic_index): emb.npy fp16, frames/vid/ms .npy, videos.json

Old-corpus videos that were embedded later on Colab (--extra-npz-dir) carry their own fps in the shard
(frame_idx int32, pe_core float32 [n,1280], fps float32) and are indexed as source "batch1-old".
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.frame_index import save_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--npz-dir", default=r"D:\aic_pecore_embeddings\embeddings_pecore")
ap.add_argument("--videos", default=r"E:\AIC2026\aicv5\data\videos")
ap.add_argument("--keyframes", default=r"D:\aic_batch2_keyframes")
ap.add_argument("--out", default=os.environ.get("INDEX_DIR", r"D:\aic_index"))
ap.add_argument("--old-dir", default="", help="folder with metadata/embeddings_OLDCORPUS_SAFE.pkl -> also index the old corpus")
ap.add_argument("--extra-npz-dir", default="", help="Colab shards for old-corpus videos missing from the old store (fps stored in each .npz)")
ap.add_argument("--extra-keyframes", default="", help="folder with the extracted keyframe JPEGs of those videos (frame sets are checked when given)")
a = ap.parse_args()

t0 = time.time()
npz_dir, vid_dir = Path(a.npz_dir), Path(a.videos)
shards = sorted(p for p in npz_dir.glob("*.npz") if not p.name.endswith(".tmp.npz"))
print(f"{len(shards)} shards in {npz_dir}")

# ---- fps per video (header read only)
exts = {}
for f in os.listdir(vid_dir):
    exts[Path(f).stem] = f


def fps_of(v):
    import cv2
    p = vid_dir / exts[v] if v in exts else None
    if p is None:
        return None
    for _ in range(3):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 1:
            return float(fps)
    return None


names = [p.stem for p in shards]
with ThreadPoolExecutor(16) as ex:
    fps = dict(zip(names, ex.map(fps_of, names)))
missing_fps = [v for v, f in fps.items() if f is None]
print(f"fps read for {len(names) - len(missing_fps)}/{len(names)} videos in {time.time()-t0:.0f}s; distribution:",
      {round(k, 2): sum(1 for f in fps.values() if f and abs(f - k) < 0.01) for k in sorted({round(f, 2) for f in fps.values() if f})})
if missing_fps:
    print("!! no fps for", missing_fps, "-> assuming 25")
    for v in missing_fps:
        fps[v] = 25.0

# ---- keyframe files on disk (to prove the shards line up with what was extracted)
pat = re.compile(r"^(.+?)_frame_(\d+)\.jpg$")
on_disk = {}
kdir = Path(a.keyframes)
if kdir.is_dir():
    for f in os.listdir(kdir):
        m = pat.match(f)
        if m:
            on_disk.setdefault(m.group(1), set()).add(int(m.group(2)))

def check_shard(fr, e):
    """-> (frames sorted, embeddings, None) or (fr, e, reason)."""
    if e.ndim != 2 or e.shape[1] != 1280 or e.shape[0] != len(fr) or len(fr) == 0:
        return fr, e, f"bad shape {e.shape} / {len(fr)} frames"
    if not np.isfinite(e).all():
        return fr, e, "NaN/inf"
    if abs(np.linalg.norm(e, axis=1) - 1).max() > 5e-3:
        return fr, e, "not unit norm"
    if (np.diff(fr) <= 0).any():
        order = np.argsort(fr)
        fr, e = fr[order], e[order]
        if (np.diff(fr) <= 0).any():
            return fr, e, "duplicate frame_idx"
    return fr, e, None


embs, frames, vids, mss, meta, problems = [], [], [], [], [], []
pos = 0
for vi, p in enumerate(shards):
    v = p.stem
    d = np.load(p)
    fr, e, why = check_shard(d["frame_idx"].astype(np.int64), d["pe_core"])
    if why is None and v in on_disk and set(fr.tolist()) != on_disk[v]:
        why = f"frame set differs from extracted keyframes ({len(fr)} vs {len(on_disk[v])})"
    if why:
        problems.append((v, why)); print(f"  !! {v}: {why}")
        continue
    embs.append(e.astype(np.float16)); frames.append(fr.astype(np.int32)); vids.append(np.full(len(fr), len(meta), np.int32))
    mss.append(np.rint(fr * 1000.0 / fps[v]).astype(np.int32))
    meta.append({"id": v, "fps": fps[v], "start": pos, "end": pos + len(fr), "source": "batch2"})
    pos += len(fr)

if a.old_dir and Path(a.old_dir).is_dir():
    # ---- old corpus (L21..L30): PE-Core 1280-d vectors already in the old persistent store backup
    import collections, pickle
    md = pickle.load(open(Path(a.old_dir) / "metadata_OLDCORPUS_SAFE.pkl", "rb"))
    em = pickle.load(open(Path(a.old_dir) / "embeddings_OLDCORPUS_SAFE.pkl", "rb"))
    ids, clip = md["ids"], em["clip"]
    assert len(ids) == len(clip), (len(ids), len(clip))
    per = collections.defaultdict(list)
    for i, key in enumerate(ids):
        v, f = key.rsplit(":", 1)
        per[v].append((int(f), float(md["metadata"][key].get("timestamp", 0.0)), i))
    taken = {m["id"] for m in meta}
    n_old = 0
    for v in sorted(per):
        if v in taken:
            print(f"  !! old video {v} collides with a batch2 id, skipped"); continue
        rows = sorted(per[v])
        fr = np.array([r[0] for r in rows], np.int64)
        if (np.diff(fr) <= 0).any():
            print(f"  !! old video {v}: duplicate frame ids, skipped"); continue
        e = np.stack([clip[r[2]] for r in rows]).astype(np.float32)
        if e.shape[1] != 1280 or not np.isfinite(e).all() or abs(np.linalg.norm(e, axis=1) - 1).max() > 5e-3:
            print(f"  !! old video {v}: bad vectors, skipped"); continue
        ts = np.array([r[1] for r in rows])
        pos_t = (ts > 0) & (fr > 0)
        fps_v = float(np.median(fr[pos_t] / ts[pos_t])) if pos_t.any() else 25.0
        embs.append(e.astype(np.float16)); frames.append(fr.astype(np.int32)); vids.append(np.full(len(fr), len(meta), np.int32))
        mss.append(np.rint(ts * 1000.0).astype(np.int32))
        meta.append({"id": v, "fps": round(fps_v, 3), "start": pos, "end": pos + len(fr), "source": "batch1-old"})
        pos += len(fr); n_old += len(fr)
    print(f"old corpus added: {len(per)} videos, {n_old:,} frames")

if a.extra_npz_dir and Path(a.extra_npz_dir).is_dir():
    # ---- old-corpus videos embedded afterwards on Colab (fps travels inside the shard)
    extra_disk = {}
    if a.extra_keyframes and Path(a.extra_keyframes).is_dir():
        for f in os.listdir(a.extra_keyframes):
            m = pat.match(f)
            if m:
                extra_disk.setdefault(m.group(1), set()).add(int(m.group(2)))
    taken = {m["id"] for m in meta}
    n_extra = 0
    for p in sorted(x for x in Path(a.extra_npz_dir).glob("*.npz") if not x.name.endswith(".tmp.npz")):
        v = p.stem
        if v in taken:
            print(f"  !! extra video {v} already indexed, skipped"); continue
        d = np.load(p)
        fps_v = float(d["fps"]) if "fps" in d.files else 0.0
        fr, e, why = check_shard(d["frame_idx"].astype(np.int64), d["pe_core"])
        if why is None and not fps_v > 1:
            why = f"no usable fps in the shard ({fps_v})"
        if why is None and v in extra_disk and set(fr.tolist()) != extra_disk[v]:
            why = f"frame set differs from extracted keyframes ({len(fr)} vs {len(extra_disk[v])})"
        if why:
            problems.append((v, why)); print(f"  !! {v}: {why}")
            continue
        embs.append(e.astype(np.float16)); frames.append(fr.astype(np.int32)); vids.append(np.full(len(fr), len(meta), np.int32))
        mss.append(np.rint(fr * 1000.0 / fps_v).astype(np.int32))
        meta.append({"id": v, "fps": round(fps_v, 3), "start": pos, "end": pos + len(fr), "source": "batch1-old"})
        pos += len(fr); n_extra += len(fr); taken.add(v)
    print(f"extra old videos added: {n_extra:,} frames")

if not embs:
    sys.exit("nothing to index")
emb = np.concatenate(embs); del embs
frames, vids, mss = np.concatenate(frames), np.concatenate(vids), np.concatenate(mss)
print(f"{len(meta)} videos, {len(frames):,} frames, emb {emb.shape} fp16 = {emb.nbytes/1e6:.0f} MB; verified against JPEGs for {sum(1 for m in meta if m['id'] in on_disk)} videos")
save_index(Path(a.out), emb, frames, vids, mss, meta, extra={"model": "PE-Core-bigG-14-448", "note": "batch2 (Colab shards)"})
print(f"index written to {a.out} in {time.time()-t0:.0f}s")
if problems:
    print(f"!! {len(problems)} shard(s) skipped:", problems)
    sys.exit(2)
