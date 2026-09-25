"""Pull the Colab dense-sampling shards (N-series) and rebuild the index with them merged in.

    python scripts/ingest_dense.py                     # download from HF -> check against the local thumbnails -> build D:\\aic_index_all_new
    python scripts/ingest_dense.py --skip-download     # reuse the shards already in --work

Colab notebook: colab_embed_N_dense.ipynb -> 5clnl/idk :: embeddings_pecore_N_dense/<video>.npz (frame_idx, pe_core, fps).
Thumbnails come from scripts/make_dense_thumbs.py (D:\\aic_N_dense_thumbs, must be in KEYFRAME_DIRS). A video whose
thumbnail list differs from the shard is still indexed (the server cuts a missing thumbnail from the video on demand),
but it is reported.

The index is written next to the live one (--out); it also carries the 11 old-corpus videos embedded earlier
(D:\\aic_old_missing\\ready) when that folder exists. Swap it in by hand after stopping the server.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

REPO = "5clnl/idk"
EMB_DIR = "embeddings_pecore_N_dense"
ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--work", default=r"D:\aic_N_dense", help="download folder")
ap.add_argument("--thumbs", default=r"D:\aic_N_dense_thumbs")
ap.add_argument("--out", default=r"D:\aic_index_all_new")
ap.add_argument("--old-dir", default=r"D:\aic_oldcorpus_backup")
ap.add_argument("--npz-dir", default=r"D:\aic_pecore_embeddings\embeddings_pecore")
ap.add_argument("--videos", default=r"E:\AIC2026\aicv5\data\videos")
ap.add_argument("--batch2-keyframes", default=r"D:\aic_keyframes_all")
ap.add_argument("--extra-npz-dir", default=r"D:\aic_old_missing\ready")
ap.add_argument("--extra-keyframes", default=r"D:\aic_keyframes_all")
ap.add_argument("--skip-download", action="store_true")
a = ap.parse_args()

work = Path(a.work)
emb_dir = work / EMB_DIR
emb_dir.mkdir(parents=True, exist_ok=True)


def download():
    os.environ.pop("HF_HUB_OFFLINE", None)                 # the launchers force offline; this script must reach the hub
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith(EMB_DIR + "/") and f.endswith(".npz")]
    if not files:
        sys.exit("Chưa có kết quả nào trên Hugging Face — chạy notebook Colab (colab_embed_N_dense) trước.")
    sizes = {}
    for i in range(0, len(files), 100):
        sizes.update({x.path: x.size for x in api.get_paths_info(REPO, files[i:i + 100], repo_type="dataset")})
    todo = [f for f in sorted(files) if not ((work / f).exists() and (work / f).stat().st_size == sizes[f])]
    print(f"{len(files)} shard trên HF, cần tải {len(todo)}", flush=True)
    for n, f in enumerate(todo, 1):
        for attempt in range(1, 6):
            try:
                hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(work))
                if (work / f).stat().st_size == sizes[f]:
                    break
                print(f"   {f}: kích thước sai (lần {attempt})")
            except Exception as e:
                print(f"   {f}: lỗi lần {attempt}: {str(e)[:120]}")
        else:
            sys.exit(f"không tải được {f}")
        if n % 25 == 0 or n == len(todo):
            print(f"  đã tải {n}/{len(todo)}", flush=True)


if not a.skip_download:
    download()

shards = sorted(emb_dir.glob("*.npz"))
if not shards:
    sys.exit("không có shard nào")

# ---- compare each shard with the thumbnails made from the local videos (same frame list expected)
thumbs = Path(a.thumbs)
n_match = n_diff = n_nomark = 0
for p in shards:
    d = np.load(p)
    fr = [int(x) for x in d["frame_idx"]]
    mk = thumbs / "_done" / f"{p.stem}.json"
    if not mk.exists():
        n_nomark += 1
        print(f"  {p.stem}: chưa có thumbnail cục bộ (server sẽ cắt từ video khi cần)")
        continue
    got = json.loads(mk.read_text(encoding="utf-8"))["idx"]
    if got == fr:
        n_match += 1
    else:
        n_diff += 1
        miss = sorted(set(fr) - set(got))
        print(f"  !! {p.stem}: thumbnail lệch shard ({len(got)} ảnh vs {len(fr)} vector; thiếu {len(miss)} ảnh, ví dụ {miss[:3]}) — vẫn nạp, ảnh thiếu sẽ cắt từ video")
print(f"{len(shards)} shard: thumbnail khớp {n_match}, lệch {n_diff}, chưa có {n_nomark}")

cmd = [sys.executable, str(ROOT / "scripts" / "build_index.py"), "--npz-dir", a.npz_dir, "--videos", a.videos,
       "--keyframes", a.batch2_keyframes, "--old-dir", a.old_dir, "--out", a.out, "--dense-npz-dir", str(emb_dir)]
if a.extra_npz_dir and Path(a.extra_npz_dir).is_dir():
    cmd += ["--extra-npz-dir", a.extra_npz_dir, "--extra-keyframes", a.extra_keyframes]
print("\n" + " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd, cwd=ROOT))
