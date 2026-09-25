"""Pack every video into a few STORED zip files and upload them to Hugging Face (5clnl/idk :: videos_all/).

    python scripts/zip_videos_to_hf.py                    # all parts
    python scripts/zip_videos_to_hf.py --only videos_M_1of3.zip
    python scripts/zip_videos_to_hf.py --plan             # print the parts, do nothing

Inside each zip the entries are `videos/<file>` (not compressed: mp4/mov are already compressed), so unzipping every
part into one folder gives <folder>/videos/ with all 1,487 videos. The videos are already on HF under videos/ and HF
stores files in content-defined chunks (Xet), so uploading a stored zip of the same bytes only sends what is new —
the script prints how many bytes really went over the network. Parts already on HF are skipped (restartable).
"""
import argparse, json, os, sys, time, zipfile
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
REPO, OUT_DIR = "5clnl/idk", "videos_all"
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_of(name):
    if name.startswith("L"):
        return "L"
    return "M" if name.startswith("M") else "N_S"


def plan_parts(files, cap):
    """files: [(name, size)] -> [(part name, [(name, size)])], grouped by series, each part <= cap bytes."""
    out = []
    for g in ("M", "N_S", "L"):
        items = sorted(f for f in files if group_of(f[0]) == g)
        parts, cur, cur_b = [], [], 0
        for it in items:
            if cur and cur_b + it[1] > cap:
                parts.append(cur)
                cur, cur_b = [], 0
            cur.append(it)
            cur_b += it[1]
        if cur:
            parts.append(cur)
        out += [(f"videos_{g}_{i}of{len(parts)}.zip", p) for i, p in enumerate(parts, 1)]
    return out


def net_sent():
    try:
        import psutil
        return psutil.net_io_counters().bytes_sent
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=r"E:\AIC2026\aicv5\data\videos")
    ap.add_argument("--stage", default=r"D:\aic_work\zip_stage")
    ap.add_argument("--part-gb", type=float, default=45.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    vdir, stage = Path(a.videos), Path(a.stage)
    local = sorted((p.name, p.stat().st_size) for p in vdir.iterdir() if p.suffix.lower() in VIDEO_EXT and p.is_file())
    remote_files = api.list_repo_files(REPO, repo_type="dataset")
    remote_v = [f for f in remote_files if f.startswith("videos/")]
    rsize = {}
    for i in range(0, len(remote_v), 200):
        rsize.update({x.path.split("/", 1)[1]: x.size for x in api.get_paths_info(REPO, remote_v[i:i + 200], repo_type="dataset")})
    lsize = dict(local)
    only_local = sorted(set(lsize) - set(rsize))
    only_remote = sorted(set(rsize) - set(lsize))
    differ = sorted(n for n in set(lsize) & set(rsize) if lsize[n] != rsize[n])
    log(f"máy: {len(local)} video ({sum(lsize.values()) / 1e9:.1f} GB) | HF videos/: {len(rsize)} | chỉ có trên máy {len(only_local)} "
        f"| chỉ có trên HF {len(only_remote)} | khác kích thước {len(differ)}")
    if only_remote or differ:
        sys.exit(f"!! máy thiếu hoặc khác HF: {only_remote[:5]} {differ[:5]} — dừng để không gói bản sai")

    parts = plan_parts(local, int(a.part_gb * 1e9))
    for name, items in parts:
        log(f"  {name}: {len(items)} video, {sum(s for _, s in items) / 1e9:.1f} GB")
    if a.plan:
        return

    stage.mkdir(parents=True, exist_ok=True)
    done_remote = {f.split("/", 1)[1] for f in remote_files if f.startswith(OUT_DIR + "/")}
    manifest, t_all = {}, time.time()
    for name, items in parts:
        manifest[name] = {n: s for n, s in items}
        if a.only and name != a.only:
            continue
        if name in done_remote:
            log(f"[bỏ qua] {name}: đã có trên HF")
            continue
        path, tmp = stage / name, stage / (name + ".tmp")
        t0 = time.time()
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for n, s in items:
                z.write(vdir / n, arcname=f"videos/{n}")
        os.replace(tmp, path)
        t1 = time.time()
        with zipfile.ZipFile(path) as z:                       # read back: every entry, full CRC
            infos = z.infolist()
            bad = z.testzip()
        want = {f"videos/{n}": s for n, s in items}
        if bad is not None or {i.filename: i.file_size for i in infos} != want:
            sys.exit(f"!! {name}: zip lỗi ({bad}) hoặc danh sách khác kế hoạch — dừng")
        size = path.stat().st_size
        log(f"[{name}] đã gói {len(items)} video, {size / 1e9:.1f} GB, kiểm CRC xong ({(t1 - t0) / 60:.1f} + {(time.time() - t1) / 60:.1f} phút)")
        for attempt in range(1, 6):
            s0, t2 = net_sent(), time.time()
            try:
                api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{OUT_DIR}/{name}", repo_id=REPO, repo_type="dataset",
                                commit_message=f"videos_all: {name} ({len(items)} video)")
                got = api.get_paths_info(REPO, [f"{OUT_DIR}/{name}"], repo_type="dataset")
                sent = (net_sent() - s0) / 1e9 if s0 is not None else float("nan")
                if got and got[0].size == size:
                    log(f"[{name}] upload xong {size / 1e9:.1f} GB trong {(time.time() - t2) / 60:.1f} phút; mạng thực gửi ≈ {sent:.2f} GB")
                    break
                log(f"[{name}] kích thước trên HF khác ({got[0].size if got else None} vs {size}) -> thử lại")
            except Exception as e:
                log(f"[{name}] upload lỗi lần {attempt}: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(30 * attempt)
        else:
            sys.exit(f"!! {name}: upload không thành công — file vẫn ở {path}, chạy lại lệnh để thử tiếp")
        path.unlink()

    if not a.only:
        ops = [CommitOperationAdd(path_in_repo=f"{OUT_DIR}/manifest.json", path_or_fileobj=json.dumps(
            {"parts": manifest, "videos": sum(len(v) for v in manifest.values()),
             "bytes": sum(sum(v.values()) for v in manifest.values())}, ensure_ascii=False, indent=1).encode("utf-8"))]
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message="videos_all: manifest")
        log(f"XONG: {len(parts)} file trong {OUT_DIR}/ ({(time.time() - t_all) / 60:.1f} phút)")


if __name__ == "__main__":
    main()
