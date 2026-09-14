"""
Batch ASR extraction over data/videos (video/*.mp4) using faster-whisper.

Resumable: writes one JSON per video to data/asr/<VIDEO_ID>.json and skips
any video whose output file already exists and is non-empty, so an
interruption only loses the one video that was mid-transcription.

Output schema per video (data/asr/<VIDEO_ID>.json):
    [{"start": float_seconds, "end": float_seconds, "text": str}, ...]

This is a preprocessing step to populate the `asr_transcript` field that
retrieval/bm25_search.py's build_frame_document_text() already expects but
which has never been populated (see SESSION_HANDOFF.md) -- BM25 has always
had 0 documents on this corpus. Run scripts/merge_asr_into_metadata.py
after this finishes (or partially finishes) to wire the transcripts into
data/persistent_store/metadata.pkl.

Usage:
    python scripts/extract_asr.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/hf-cache")

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR = PROJECT_ROOT / "video"
ASR_DIR = PROJECT_ROOT / "data" / "asr"
LOG_FILE = PROJECT_ROOT / "data" / "asr" / "_progress.log"
MODEL_NAME = "large-v3-turbo"
LANGUAGE = "vi"

ASR_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    from faster_whisper import WhisperModel

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        log(f"No videos found in {VIDEO_DIR}")
        return

    log(f"Found {len(videos)} videos. Loading {MODEL_NAME}...")
    t_load = time.time()
    model = WhisperModel(
        MODEL_NAME, device="cuda", compute_type="int8_float16",
        download_root="D:/hf-cache/faster-whisper",
    )
    log(f"Model loaded in {time.time() - t_load:.1f}s")

    done = 0
    skipped = 0
    failed = 0
    total_video_sec = 0.0
    t_start = time.time()

    for i, video_path in enumerate(videos, 1):
        video_id = video_path.stem
        out_path = ASR_DIR / f"{video_id}.json"

        if out_path.exists() and out_path.stat().st_size > 2:
            skipped += 1
            continue

        try:
            t0 = time.time()
            segments, info = model.transcribe(
                str(video_path), language=LANGUAGE, vad_filter=True, beam_size=1,
            )
            segs = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segments
            ]
            # Write to a temp file then rename -- atomic, so a crash mid-write
            # never leaves a corrupt/partial JSON that looks "done" on resume.
            tmp_path = out_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(segs, f, ensure_ascii=False)
            tmp_path.replace(out_path)

            elapsed = time.time() - t0
            total_video_sec += info.duration
            done += 1
            rtf = info.duration / elapsed if elapsed > 0 else 0
            log(
                f"[{i}/{len(videos)}] {video_id}: {len(segs)} segs, "
                f"{info.duration:.0f}s video in {elapsed:.1f}s ({rtf:.1f}x). "
                f"done={done} skipped={skipped} failed={failed}"
            )
        except Exception as e:
            failed += 1
            log(f"[{i}/{len(videos)}] {video_id}: FAILED -- {e}")

    wall = time.time() - t_start
    log(
        f"COMPLETE. done={done} skipped={skipped} failed={failed} "
        f"wall_time={wall/3600:.2f}h total_video_processed={total_video_sec/3600:.2f}h"
    )


if __name__ == "__main__":
    main()
