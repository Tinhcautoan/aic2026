"""Tải dữ liệu cho server từ Hugging Face (5clnl/idk) về MỘT thư mục, giải nén và kiểm tra đủ chưa.
Chạy được trên Windows và Linux.

    python scripts/setup_from_hf.py D:\\aic_data --delete-zips              # chỉ mục + keyframe + video (từng file)
    python scripts/setup_from_hf.py D:\\aic_data --videos-zip --delete-zips # video tải dạng 9 file zip thay vì 1.487 file
    python scripts/setup_from_hf.py /data/aic --no-videos --delete-zips      # không video (~70 GB), không xem được video

    --delete-zips : xóa từng zip ngay sau khi giải nén xong (không cần chỗ trống gấp đôi)

Kết quả (đúng cấu trúc mà start_server.sh / start_server_hf.bat dùng):
    <DATA>/index_all/     emb.npy, frames.npy, vid.npy, ms.npy, videos.json
    <DATA>/keyframes/     234.083 ảnh <video>_frame_<frame 6 chữ số>.jpg (phẳng, không thư mục con)
    <DATA>/videos/        1.487 video (mới + cũ)
Chạy lại lúc nào cũng được: file đã tải đủ được bỏ qua, file đang dở được tải tiếp, zip đã giải nén xong không tải lại.
"""
import argparse, json, os, shutil, sys, zipfile
from pathlib import Path, PurePosixPath


class Source:
    """Hugging Face dataset, or a local folder with the same layout (--local-source, for tests)."""

    def __init__(self, repo, local=""):
        self.repo, self.local = repo, Path(local) if local else None

    def list(self, folder):
        if self.local:
            d = self.local / folder
            return sorted(f"{folder}/{p.name}" for p in d.iterdir() if p.is_file()) if d.is_dir() else []
        from huggingface_hub import HfApi
        return sorted(f for f in HfApi().list_repo_files(self.repo, repo_type="dataset") if f.startswith(folder + "/"))

    def get_folder(self, folder, data):
        if self.local:
            for f in self.list(folder):
                self._copy(f, data)
            return
        from huggingface_hub import snapshot_download
        self._retry(lambda: snapshot_download(repo_id=self.repo, repo_type="dataset", allow_patterns=[f"{folder}/*"],
                                              local_dir=str(data), max_workers=8), f"{folder}/")

    def get_file(self, path, data):
        if self.local:
            return self._copy(path, data)
        from huggingface_hub import hf_hub_download
        return Path(self._retry(lambda: hf_hub_download(self.repo, path, repo_type="dataset", local_dir=str(data)), path))

    def _copy(self, rel, data):
        dst = Path(data) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.local / rel, dst)
        return dst

    @staticmethod
    def _retry(fn, what):
        for attempt in range(1, 6):
            try:
                return fn()
            except Exception as e:
                print(f"   lỗi mạng lần {attempt}: {type(e).__name__}: {str(e)[:150]} -> thử lại (phần đã tải được giữ lại)", flush=True)
        sys.exit(f"không tải được {what} — chạy lại lệnh này")


def unzip_one(zpath, data, flat_into=None, keep_prefix=None):
    """flat_into: every entry goes into that folder by its file name (keyframes).
    keep_prefix: entries must live under that prefix and keep their path below <DATA> (videos/<file>)."""
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            if flat_into is not None:
                dst = Path(flat_into) / PurePosixPath(zi.filename).name
            else:
                parts = PurePosixPath(zi.filename).parts
                if not parts or parts[0] != keep_prefix or ".." in parts:
                    raise SystemExit(f"{zpath.name}: mục lạ trong zip: {zi.filename}")
                dst = Path(data).joinpath(*parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            n += 1
            if dst.exists() and dst.stat().st_size == zi.file_size:
                continue
            tmp = dst.with_name(dst.name + ".part")
            with zf.open(zi) as src, open(tmp, "wb") as out:           # zipfile kiểm CRC khi đọc hết mục
                shutil.copyfileobj(src, out, 4 << 20)
            os.replace(tmp, dst)
    return n


def zip_folder(src, folder, data, delete_zips, **unzip_kw):
    """Tải và giải nén TỪNG zip một (zip xong mới sang zip sau). Zip đã giải nén xong được đánh dấu để không tải lại."""
    done_dir = Path(data) / ".done"
    done_dir.mkdir(parents=True, exist_ok=True)
    zips = [f for f in src.list(folder) if f.endswith(".zip")]
    if not zips:
        sys.exit(f"không thấy file zip nào trong {folder}/ trên Hugging Face")
    for i, rel in enumerate(zips, 1):
        name = PurePosixPath(rel).name
        mark = done_dir / f"{folder}__{name}.done"
        if mark.exists():
            print(f"   [{i}/{len(zips)}] {name}: đã giải nén từ trước", flush=True)
            continue
        print(f"   [{i}/{len(zips)}] tải {name} ...", flush=True)
        zpath = src.get_file(rel, data)
        n = unzip_one(zpath, data, **unzip_kw)
        mark.write_text(str(n), encoding="utf-8")
        if delete_zips:
            zpath.unlink()
        print(f"   [{i}/{len(zips)}] {name}: giải nén {n:,} file" + (" (đã xóa zip)" if delete_zips else ""), flush=True)


def verify(data):
    import numpy as np
    idx = Path(data) / "index_all"
    meta = json.load(open(idx / "videos.json", encoding="utf-8"))
    frames = np.load(idx / "frames.npy")
    kdir = Path(data) / "keyframes"
    kf = set(os.listdir(kdir)) if kdir.is_dir() else set()
    missing, dense = 0, 0
    for v in meta["videos"]:
        for f in frames[v["start"]:v["end"]]:
            if f"{v['id']}_frame_{int(f):06d}.jpg" not in kf:
                if v.get("dense"):
                    dense += 1                  # khung dày của video N không có ảnh: server cắt từ video khi cần
                else:
                    missing += 1
    vdir = Path(data) / "videos"
    have = {p.stem for p in vdir.iterdir() if not p.name.endswith(".part")} if vdir.is_dir() else set()
    no_video = [v["id"] for v in meta["videos"] if v["id"] not in have]
    print(f"chỉ mục: {len(meta['videos'])} video, {meta['count']:,} khung")
    print(f"keyframe: {len(kf):,} ảnh; khung thiếu ảnh: {missing} (khung dày của N, cắt từ video khi cần: {dense:,})")
    print(f"video: có {len(meta['videos']) - len(no_video)}/{len(meta['videos'])}" + (f", thiếu ví dụ {no_video[:3]}" if no_video else ""))
    return missing == 0, len(no_video)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=str(Path.home() / "aic_data"))
    ap.add_argument("--no-videos", action="store_true", help="không tải video")
    ap.add_argument("--videos-zip", action="store_true", help="tải video từ videos_all/ (9 zip) thay vì videos/ (1.487 file)")
    ap.add_argument("--delete-zips", action="store_true", help="xóa từng zip sau khi giải nén")
    ap.add_argument("--repo", default="5clnl/idk")
    ap.add_argument("--local-source", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    src = Source(a.repo, a.local_source)

    print(f"== 1/3 chỉ mục (index_all, ~0,8 GB) -> {data}", flush=True)
    src.get_folder("index_all", data)
    print("== 2/3 keyframe (keyframes_all: 7 zip, ~66 GB)", flush=True)
    zip_folder(src, "keyframes_all", data, a.delete_zips, flat_into=data / "keyframes")
    if a.no_videos:
        print("== 3/3 bỏ qua video (--no-videos)")
    elif a.videos_zip:
        print("== 3/3 video (videos_all: 9 zip, ~350 GB)", flush=True)
        zip_folder(src, "videos_all", data, a.delete_zips, keep_prefix="videos")
    else:
        print("== 3/3 video (videos/: 1.487 file, ~350 GB)", flush=True)
        src.get_folder("videos", data)
    print("== kiểm tra")
    ok, no_video = verify(data)
    if not ok:
        print(f"!! THIẾU ẢNH KEYFRAME — xóa thư mục {data / '.done'} rồi chạy lại lệnh này")
    elif no_video and not a.no_videos:
        print("!! THIẾU VIDEO — chạy lại lệnh này" + (f" (nếu vẫn thiếu: xóa {data / '.done'} rồi chạy lại)" if a.videos_zip else ""))
    else:
        print("OK")
    print(f"Chạy server:  Linux: bash start_server.sh \"{data}\"   |   Windows: start_server_hf.bat \"{data}\"")
    sys.exit(0 if ok and (a.no_videos or not no_video) else 1)


if __name__ == "__main__":
    main()
