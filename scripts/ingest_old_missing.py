"""Pull the Colab results for the old-corpus videos that had no vectors, unpack their keyframes and rebuild the index.

    python scripts/ingest_old_missing.py                # download from HF -> unzip -> build D:\\aic_index_all_new
    python scripts/ingest_old_missing.py --skip-download --work D:\\some\\folder   # reuse files already on disk

Colab notebook: colab_embed_old_missing11.ipynb; results on 5clnl/idk in
    embeddings_pecore_old_missing/<video>.npz   (frame_idx, pe_core, fps)
    keyframes_old_missing/<video>.zip           (<video>_frame_NNNNNN.jpg)

The live index is NOT touched: the new one is written next to it (--out) and swapped in by hand after the
server has been stopped:   move aic_index_all aic_index_all_prev  &  move aic_index_all_new aic_index_all
"""
import argparse, os, re, subprocess, sys, zipfile
from pathlib import Path

import numpy as np

REPO = "5clnl/idk"
EMB_DIR, KF_DIR = "embeddings_pecore_old_missing", "keyframes_old_missing"
ROOT = Path(__file__).resolve().parent.parent
FRAME_RE = re.compile(r"^(.+?)_frame_(\d+)\.jpg$")

ap = argparse.ArgumentParser()
ap.add_argument("--work", default=r"D:\aic_old_missing", help="download folder")
ap.add_argument("--keyframes", default=r"D:\aic_old_keyframes", help="where the keyframe JPEGs are unpacked (already on the server's KEYFRAME_DIRS)")
ap.add_argument("--out", default=r"D:\aic_index_all_new")
ap.add_argument("--old-dir", default=r"D:\aic_oldcorpus_backup")
ap.add_argument("--npz-dir", default=r"D:\aic_pecore_embeddings\embeddings_pecore")
ap.add_argument("--videos", default=r"E:\AIC2026\aicv5\data\videos")
ap.add_argument("--batch2-keyframes", default=r"D:\aic_batch2_keyframes")
ap.add_argument("--skip-download", action="store_true")
ap.add_argument("--no-build", action="store_true")
a = ap.parse_args()

work = Path(a.work)
emb_dir, zip_dir, kf_dir = work / EMB_DIR, work / KF_DIR, Path(a.keyframes)
for d in (emb_dir, zip_dir, kf_dir):
    d.mkdir(parents=True, exist_ok=True)


def download():
    os.environ.pop("HF_HUB_OFFLINE", None)                 # the launchers force offline; this script must reach the hub
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith((EMB_DIR + "/", KF_DIR + "/"))]
    if not files:
        sys.exit("Chưa có kết quả nào trên Hugging Face — chạy notebook Colab trước.")
    sizes = {i.path: i.size for i in api.get_paths_info(REPO, files, repo_type="dataset")}
    for f in sorted(files):
        dst = work / f
        if dst.exists() and dst.stat().st_size == sizes[f]:
            continue
        print(f"tải {f} ({sizes[f] / 1e6:.1f} MB)", flush=True)
        for attempt in range(1, 6):
            try:
                hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(work))
                if (work / f).stat().st_size == sizes[f]:
                    break
                print(f"   kích thước sai (lần {attempt})")
            except Exception as e:
                print(f"   lỗi lần {attempt}: {str(e)[:120]}")
        else:
            sys.exit(f"không tải được {f}")


def unpack():
    """Unpack every keyframe zip (CRC-checked); a video counts only if the JPEGs match its .npz frame list."""
    ok = []
    for npz in sorted(emb_dir.glob("*.npz")):
        v = npz.stem
        z = zip_dir / f"{v}.zip"
        if not z.exists():
            print(f"!! {v}: có vector nhưng thiếu zip keyframe -> bỏ qua")
            continue
        want = set(int(x) for x in np.load(npz)["frame_idx"])
        with zipfile.ZipFile(z) as zf:
            bad = zf.testzip()
            got = {int(m.group(2)) for n in zf.namelist() if (m := FRAME_RE.match(n)) and m.group(1) == v}
            if bad or got != want:
                print(f"!! {v}: zip lỗi ({bad}) hoặc khung không khớp ({len(got)} ảnh vs {len(want)} vector) -> bỏ qua")
                continue
            for n in zf.namelist():
                dst = kf_dir / n
                if dst.exists() and dst.stat().st_size == zf.getinfo(n).file_size:
                    continue
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes(zf.read(n))
                tmp.replace(dst)
        ok.append(v)
        print(f"  {v}: {len(want)} keyframe đã giải nén vào {kf_dir}")
    return ok


if not a.skip_download:
    download()
done = unpack()
print(f"\n{len(done)} video sẵn sàng: {', '.join(done)}")
if not done:
    sys.exit("không có video nào hợp lệ")
if a.no_build:
    sys.exit(0)

# only shards whose keyframes were unpacked and matched go into the index (a vector without a thumbnail is useless)
ready = work / "ready"
if ready.exists():
    for old in ready.glob("*.npz"):
        old.unlink()
ready.mkdir(exist_ok=True)
for v in done:
    (ready / f"{v}.npz").write_bytes((emb_dir / f"{v}.npz").read_bytes())

cmd = [sys.executable, str(ROOT / "scripts" / "build_index.py"), "--npz-dir", a.npz_dir,
       "--videos", a.videos, "--keyframes", a.batch2_keyframes, "--old-dir", a.old_dir, "--out", a.out,
       "--extra-npz-dir", str(ready), "--extra-keyframes", str(kf_dir)]
print("\n" + " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd, cwd=ROOT))
