"""Uniform (time-based) frame sampling of a video: one frame every `interval` seconds.

Used by BOTH sides of the N-series dense re-sampling so they pick exactly the same frames:
  * the Colab notebook (full-resolution JPEGs that are embedded with PE-Core), and
  * scripts/make_dense_thumbs.py (small thumbnails for the web UI).

Frame k is the decoded frame with index round(k * interval * fps); indices count decoded frames from 0, exactly like
the `frame_idx` of the keyframes already in the index (ms = frame_idx * 1000 / fps).  cv2 / av are imported lazily,
so a notebook can inline this file verbatim.
"""
import os


def plan(fps, n_frames, interval):
    """Frame indices to keep (strictly increasing) for a video of n_frames frames."""
    if n_frames <= 0:
        raise ValueError("unknown frame count")
    step = float(interval) * float(fps)
    out, k = [], 0
    while True:
        idx = int(round(k * step))
        if idx >= n_frames:
            break
        if not out or idx > out[-1]:
            out.append(idx)
        k += 1
    return out


def _probe_cv2(path):
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0, 0.0
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()


def _probe_av(path):
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate or 0)
        n = int(st.frames or 0)
        if n <= 0 and st.duration and fps:
            n = int(float(st.duration * st.time_base) * fps)
        return n, fps


def _decode_cv2(path, wanted):
    """Yield (idx, bgr) for the wanted indices (sorted), decoding sequentially; stops at the first unreadable frame."""
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        i, w = 0, 0
        while w < len(wanted):
            if i == wanted[w]:
                ok, frame = cap.read()
                if not ok:
                    return
                yield i, frame
                w += 1
            elif not cap.grab():
                return
            i += 1
    finally:
        cap.release()


def _decode_av(path, wanted):
    import av
    want = set(wanted)
    last = wanted[-1] if wanted else -1
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            if i in want:
                yield i, fr.reformat(format="bgr24", src_colorspace="ITU601").to_ndarray()
            if i >= last:
                return


def sample_video(path, out_dir, interval, name=None, max_width=None, quality=95, decoder="auto"):
    """Write <name>_frame_<idx:06d>.jpg for every planned frame into out_dir.

    max_width: shrink (INTER_AREA) so the width is at most this; None keeps the full resolution.
    Returns {"video", "fps", "declared", "idx": [indices actually written], "planned", "decoder"}.
    """
    import cv2
    name = name or os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    declared, fps = _probe_cv2(path)
    if declared <= 0 or fps < 1:                      # OpenCV cannot tell (or cannot open it): ask PyAV; it raises on garbage
        n_av, fps_av = _probe_av(path)
        declared, fps = declared if declared > 0 else n_av, fps if fps >= 1 else fps_av
    if declared <= 0:
        raise RuntimeError(f"{name}: cannot determine the number of frames")
    if fps < 1:
        fps = 25.0
    wanted = plan(fps, declared, interval)
    if not wanted:
        raise RuntimeError(f"{name}: nothing to sample (declared {declared} frames, fps {fps})")
    written, used = [], None
    for dec in (("cv2", "av") if decoder == "auto" else (decoder,)):
        gen = _decode_cv2(path, wanted) if dec == "cv2" else _decode_av(path, wanted)
        try:
            for idx, frame in gen:
                if max_width and frame.shape[1] > max_width:
                    h = max(1, round(frame.shape[0] * max_width / frame.shape[1]))
                    frame = cv2.resize(frame, (max_width, h), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if not ok:
                    raise RuntimeError(f"{name}@{idx}: JPEG encode failed")
                dst = os.path.join(out_dir, f"{name}_frame_{idx:06d}.jpg")
                tmp = dst + ".part"
                with open(tmp, "wb") as f:
                    f.write(buf.tobytes())
                os.replace(tmp, dst)
                written.append(idx)
        except Exception as e:                        # e.g. cv2 cannot decode this codec: try the next decoder
            if dec == "av" or decoder != "auto":
                raise
            written.clear()
            continue
        # accept when the decoder reached (almost) the end of the plan; otherwise let the next decoder try
        if len(written) >= 0.98 * len(wanted) or dec == "av" or decoder != "auto":
            used = dec
            break
        written.clear()
    if used is None or not written:
        raise RuntimeError(f"{name}: could not decode any planned frame")
    return {"video": name, "fps": fps, "declared": declared, "idx": written, "planned": len(wanted), "decoder": used}
