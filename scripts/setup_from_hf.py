"""Tải dữ liệu cho server từ Hugging Face (5clnl/idk) về MỘT thư mục, giải nén keyframe và kiểm tra đủ chưa.
Chạy được trên Windows và Linux.

    python scripts/setup_from_hf.py D:\\aic_data                 # tất cả: index_all + keyframes_all + videos (~490 GB)
    python scripts/setup_from_hf.py /data/aic --no-videos        # chỉ chỉ mục + keyframe (~135 GB), không xem được video
    python scripts/setup_from_hf.py /data/aic --delete-zips      # xóa zip keyframe sau khi giải nén (bớt ~67 GB)

Kết quả (đúng cấu trúc mà start_server.sh / start_server_hf.bat dùng):
    <DATA>/index_all/     emb.npy, frames.npy, vid.npy, ms.npy, videos.json
    <DATA>/keyframes/     234.083 ảnh <video>_frame_<frame 6 chữ số>.jpg (phẳng, không thư mục con)
    <DATA>/videos/        1.487 video (mới + cũ)
Chạy lại lúc nào cũng được: file đã tải đủ được bỏ qua, file đang dở được tải tiếp.
"""
import argparse, json, os, shutil, sys, zipfile
from pathlib import Path


def fetch(folder, data, repo, local_source=None):
    if local_source:                                   # dùng cho kiểm thử: "repo" là một thư mục trên máy
        src = Path(local_source) / folder
        dst = Path(data) / folder
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, dst / p.name)
        return
    from huggingface_hub import snapshot_download
    for attempt in range(1, 6):
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=[f"{folder}/*"], local_dir=str(data), max_workers=8)
            return
        except Exception as e:
            print(f"   lỗi mạng lần {attempt}: {type(e).__name__}: {str(e)[:150]} -> thử lại (phần đã tải được giữ lại)", flush=True)
    sys.exit(f"không tải được {folder}/ — chạy lại lệnh này")


def extract(data, delete_zips):
    kdir = Path(data) / "keyframes"
    kdir.mkdir(parents=True, exist_ok=True)
    zips = sorted((Path(data) / "keyframes_all").glob("*.zip"))
    for z in zips:
        print(f"   giải nén {z.name} ...", flush=True)
        with zipfile.ZipFile(z) as zf:
            for zi in zf.infolist():
                if zi.is_dir():
                    continue
                dst = kdir / Path(zi.filename).name          # luôn phẳng
                if dst.exists() and dst.stat().st_size == zi.file_size:
                    continue
                tmp = dst.with_name(dst.name + ".part")
                with zf.open(zi) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                os.replace(tmp, dst)
        if delete_zips:
            z.unlink()
    return len(zips)


def verify(data):
    import numpy as np
    idx = Path(data) / "index_all"
    meta = json.load(open(idx / "videos.json", encoding="utf-8"))
    frames = np.load(idx / "frames.npy")
    kf = set(os.listdir(Path(data) / "keyframes"))
    missing, dense = 0, 0
    for v in meta["videos"]:
        for f in frames[v["start"]:v["end"]]:
            if f"{v['id']}_frame_{int(f):06d}.jpg" not in kf:
                if v.get("dense"):
                    dense += 1                  # khung dày của video N không có ảnh: server cắt từ video khi cần
                else:
                    missing += 1
    vdir = Path(data) / "videos"
    have = {p.stem for p in vdir.iterdir()} if vdir.is_dir() else set()
    no_video = [v["id"] for v in meta["videos"] if v["id"] not in have]
    print(f"chỉ mục: {len(meta['videos'])} video, {meta['count']:,} khung")
    print(f"keyframe: {len(kf):,} ảnh; khung thiếu ảnh: {missing} (khung dày của N, cắt từ video khi cần: {dense:,})")
    print(f"video: có {len(meta['videos']) - len(no_video)}/{len(meta['videos'])}" + (f", thiếu ví dụ {no_video[:3]}" if no_video else ""))
    return missing == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=str(Path.home() / "aic_data"))
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--delete-zips", action="store_true")
    ap.add_argument("--repo", default="5clnl/idk")
    ap.add_argument("--local-source", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    print(f"== 1/3 chỉ mục (index_all, ~0,8 GB) -> {data}", flush=True)
    fetch("index_all", data, a.repo, a.local_source)
    print("== 2/3 keyframe (keyframes_all, ~67 GB, 7 file zip)", flush=True)
    fetch("keyframes_all", data, a.repo, a.local_source)
    n = extract(data, a.delete_zips)
    print(f"   đã giải nén {n} file zip vào {data / 'keyframes'}")
    if a.no_videos:
        print("== 3/3 bỏ qua video (--no-videos)")
    else:
        print("== 3/3 video (videos, ~350 GB, 1.487 file)", flush=True)
        fetch("videos", data, a.repo, a.local_source)
    print("== kiểm tra")
    ok = verify(data)
    print("OK" if ok else "!! THIẾU ẢNH KEYFRAME — chạy lại lệnh này")
    print(f"Chạy server:  Linux: bash start_server.sh \"{data}\"   |   Windows: start_server_hf.bat \"{data}\"")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
