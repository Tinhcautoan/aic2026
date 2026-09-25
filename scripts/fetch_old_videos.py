"""Put every old-corpus video (L21-L30, 873 files of the 83 GB videos.zip on Hugging Face) into the SAME folder as the
new videos, without downloading the zip as a whole.

    python scripts/fetch_old_videos.py                       # -> E:\\AIC2026\\aicv5\\data\\videos, 4 parallel range reads
    python scripts/fetch_old_videos.py --parallel 2 --dest D:\\some\\folder

For each video:
  * already in --dest with the right size ............ kept (checked against the zip's CRC32 at the end with --verify)
  * already extracted in one of --reuse folders ...... copied locally, CRC32 checked while copying
  * otherwise ........................................ read out of videos.zip with HTTP Range (retrieval/video_fetch.py:
                                                         resumes after a dropped connection, CRC32 checked on the stream)
Every file written to --dest is read back and its CRC32 compared with the zip's, because the external drive has flipped
bits before. Restartable at any time: finished videos are skipped.
"""
import argparse, json, os, shutil, sys, threading, time, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retrieval.video_fetch import VideoFetcher  # noqa: E402


def crc_of(path, bufsize=8 << 20):
    c = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return c & 0xFFFFFFFF
            c = zlib.crc32(b, c)


def copy_with_crc(src, dst, bufsize=8 << 20):
    tmp = dst.with_name(dst.name + ".part")
    c = 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            b = fi.read(bufsize)
            if not b:
                break
            fo.write(b)
            c = zlib.crc32(b, c)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    return c & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=r"E:\AIC2026\aicv5\data\videos")
    ap.add_argument("--reuse", default=r"D:\aic_old_videos", help="';'-separated folders with videos already extracted")
    ap.add_argument("--index", default=str(ROOT / "config" / "old_videos_zip_index.json"))
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--verify", action="store_true", help="also CRC-check videos that were already in --dest")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    entries = {}
    for e in json.loads(Path(a.index).read_text(encoding="utf-8")):
        entries[Path(e["name"]).stem] = e
    reuse = [Path(d) for d in a.reuse.split(";") if d.strip()]
    log_lock = threading.Lock()

    def log(msg):
        with log_lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def good_size(p, e):
        try:
            return p.is_file() and p.stat().st_size == int(e["size"])
        except OSError:
            return False

    vids = sorted(entries)
    if a.limit:
        vids = vids[:a.limit]
    to_fetch, bad = [], []
    n_have = n_copied = 0
    for v in vids:
        e, target = entries[v], dest / f"{v}.mp4"
        if good_size(target, e):
            if a.verify and crc_of(target) != int(e["crc"]):
                log(f"{v}: CRC sai trong {dest} -> lấy lại")
                target.unlink()
            else:
                n_have += 1
                continue
        src = next((d / f"{v}.mp4" for d in reuse if good_size(d / f"{v}.mp4", e)), None)
        if src is not None:
            try:
                if copy_with_crc(src, target) == int(e["crc"]) and crc_of(target) == int(e["crc"]):
                    n_copied += 1
                    continue
                log(f"{v}: bản ở {src.parent} sai CRC -> tải lại từ Hugging Face")
                target.unlink(missing_ok=True)
            except OSError as ex:
                log(f"{v}: chép từ {src.parent} lỗi ({ex}) -> tải lại từ Hugging Face")
        to_fetch.append(v)
    total_gb = sum(int(entries[v]["csize"]) for v in to_fetch) / 1e9
    log(f"{len(vids)} video cũ: đã có {n_have}, chép từ ổ khác {n_copied}, cần tải {len(to_fetch)} ({total_gb:.1f} GB) -> {dest}")

    f = VideoFetcher(Path(a.index), dest, max_parallel=a.parallel,
                     has_local=lambda v: good_size(dest / f"{v}.mp4", entries[v]))
    pending, active, tries, done = list(to_fetch), set(), {}, []
    done_bytes, t0, last_report = 0, time.time(), 0.0
    while pending or active:
        while pending and len(active) < a.parallel:
            v = pending.pop(0)
            tries[v] = tries.get(v, 0) + 1
            f.start(v)
            active.add(v)
        time.sleep(1)
        for v in list(active):
            st = f.status(v)
            if st["state"] not in ("done", "error"):
                continue
            active.discard(v)
            e, target = entries[v], dest / f"{v}.mp4"
            ok = st["state"] == "done"
            if ok:
                try:
                    ok = crc_of(target) == int(e["crc"])              # read back: the drive may corrupt what was written
                    if not ok:
                        log(f"{v}: đọc lại thấy CRC sai sau khi ghi")
                except OSError as ex:
                    ok = False
                    log(f"{v}: không đọc lại được ({ex})")
            if ok:
                done.append(v)
                done_bytes += int(e["csize"])
                continue
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            if tries[v] < 3:
                log(f"{v}: lỗi ({st.get('error') or 'CRC'}) -> thử lại lần {tries[v] + 1}")
                with f._lock:
                    f._state.pop(v, None)
                pending.append(v)
            else:
                bad.append(v)
                log(f"{v}: BỎ sau 3 lần: {st.get('error')}")
        now = time.time()
        if now - last_report >= 60 or (not pending and not active):
            last_report = now
            run_bytes = sum(f.status(v).get("done", 0) for v in active)
            got = (done_bytes + run_bytes) / 1e9
            rate = got / max(now - t0, 1e-6)
            left = (total_gb - got) / rate / 60 if rate > 0 else float("nan")
            log(f"xong {len(done)}/{len(to_fetch)} video | {got:.1f}/{total_gb:.1f} GB | {rate * 1000:.1f} MB/s | còn ≈ {left:.0f} phút | lỗi {len(bad)}")
    log(f"KẾT THÚC: tải {len(done)}, lỗi {len(bad)} {bad[:20]} | tổng video cũ trong {dest}: "
        f"{sum(1 for v in vids if good_size(dest / f'{v}.mp4', entries[v]))}/{len(vids)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
