"""Small thumbnails for the N-series dense sampling, cut from the local videos (no network).

    python scripts/make_dense_thumbs.py                    # every 2 s, all N* videos, low CPU priority
    python scripts/make_dense_thumbs.py --videos E:\\AIC2026\\aicv5\\data\\videos --out D:\\aic_N_dense_thumbs

Picks exactly the frames the Colab notebook embeds (scripts/dense_sample.py) and writes them as
<video>_frame_<idx>.jpg (max width 448, quality 80) into --out, which must be listed in KEYFRAME_DIRS.
Restartable: a finished video leaves _done/<video>.json (frame list) and is skipped next time.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dense_sample  # noqa: E402


def _low_priority():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL
    except Exception:
        try:
            os.nice(10)
        except Exception:
            pass


def work(path, out_dir, interval, width, quality):
    r = dense_sample.sample_video(path, out_dir, interval, max_width=width, quality=quality)
    done = Path(out_dir) / "_done"
    done.mkdir(exist_ok=True)
    tmp = done / (r["video"] + ".json.part")
    tmp.write_text(json.dumps({k: r[k] for k in ("video", "fps", "declared", "idx", "planned", "decoder")}), encoding="utf-8")
    os.replace(tmp, done / (r["video"] + ".json"))
    return r["video"], len(r["idx"]), r["planned"], r["decoder"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=r"E:\AIC2026\aicv5\data\videos")
    ap.add_argument("--out", default=r"D:\aic_N_dense_thumbs")
    ap.add_argument("--prefix", default="N")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (for a test)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(f for f in os.listdir(a.videos) if f.startswith(a.prefix) and f.lower().endswith((".mov", ".mp4")))
    todo = [f for f in vids if not (out / "_done" / (Path(f).stem + ".json")).exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(vids)} videos, {len(vids) - len([f for f in vids if f in todo])} already done, {len(todo)} to do (every {a.interval:g}s, width {a.width})", flush=True)
    t0, n_frames, bad = time.time(), 0, []
    with ProcessPoolExecutor(a.procs, initializer=_low_priority) as ex:
        futs = {ex.submit(work, str(Path(a.videos) / f), str(out), a.interval, a.width, a.quality): f for f in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            f = futs[fu]
            try:
                v, n, planned, dec = fu.result()
                n_frames += n
                flag = "" if n >= 0.98 * planned else f"  (only {n}/{planned}: the end of the video cannot be decoded)"
                if n < 0.85 * planned:
                    flag = f"  !! only {n}/{planned}"
                    bad.append(v)
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {v}: {n} thumbnails ({dec}){flag}   {n_frames:,} total, {el / 60:.1f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)
            except Exception as e:
                bad.append(f)
                print(f"[{i}/{len(todo)}] {f}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done: {n_frames:,} thumbnails in {(time.time() - t0) / 60:.1f} min; problems: {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
