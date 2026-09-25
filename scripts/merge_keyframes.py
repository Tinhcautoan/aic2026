"""Gather every keyframe JPEG (new batch2 + old L21-L30 + the 11 old videos embedded later) into ONE flat folder.

    python scripts/merge_keyframes.py            # copy E:\\AIC2026\\keyframes_backup and D:\\aic_old_keyframes into D:\\aic_batch2_keyframes

The destination is the folder the running server already reads, so thumbnails keep working while this runs; rename it
(e.g. to D:\\aic_keyframes_all) and point KEYFRAME_DIRS at it afterwards. Files are copied (never moved), written to a
.part name first, size-checked, and skipped when an identical-size copy already exists, so the script can be re-run.
"""
import argparse, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAT = re.compile(r"^(.+?)_frame_(\d+)\.jpg$")


def copy_one(src: Path, dst: Path, tries: int = 3) -> str:
    size = src.stat().st_size
    if dst.exists() and dst.stat().st_size == size:
        return "skip"
    tmp = dst.with_name(dst.name + ".part")
    last = None
    for _ in range(tries):
        try:
            data = src.read_bytes()
            if len(data) != size:
                raise OSError(f"short read {len(data)}/{size}")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            if dst.stat().st_size != size:
                raise OSError("size differs after write")
            return "copied"
        except OSError as e:                       # the external drive sometimes drops a read: try again
            last = e
            time.sleep(1)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    raise OSError(f"{src}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=r"D:\aic_keyframes_all")
    ap.add_argument("--src", default=r"E:\AIC2026\keyframes_backup;D:\aic_old_keyframes")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in [Path(x) for x in a.src.split(";") if x.strip()]:
        names = [n for n in os.listdir(d) if PAT.match(n)]
        print(f"{d}: {len(names):,} keyframe", flush=True)
        jobs += [(d / n, dest / n) for n in names]
    t0, counts, bad = time.time(), {"skip": 0, "copied": 0}, []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, (fut, (s, _)) in enumerate(zip(ex.map(lambda j: _safe(copy_one, *j), jobs), jobs), 1):
            if isinstance(fut, Exception):
                bad.append(str(fut))
            else:
                counts[fut] += 1
            if i % 10000 == 0:
                print(f"  {i:,}/{len(jobs):,} ({(time.time() - t0) / 60:.1f} phút)", flush=True)
    n_dest = sum(1 for n in os.listdir(dest) if PAT.match(n))
    print(f"xong: chép {counts['copied']:,}, đã có {counts['skip']:,}, lỗi {len(bad)} | {dest} hiện có {n_dest:,} keyframe "
          f"({(time.time() - t0) / 60:.1f} phút)", flush=True)
    for b in bad[:20]:
        print("  lỗi:", b)
    sys.exit(1 if bad else 0)


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return e


if __name__ == "__main__":
    main()
