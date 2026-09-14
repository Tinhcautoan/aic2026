"""
Merge per-video ASR transcripts (data/asr/<VIDEO_ID>.json, produced by
scripts/extract_asr.py) into data/persistent_store/metadata.pkl's per-frame
metadata as an `asr_transcript` field.

WHY: retrieval/bm25_search.py's build_frame_document_text() has always
expected an `asr_transcript` field per frame, but nothing ever populated it
-- BM25 has had 0 documents on this corpus (confirmed by exhaustive scan,
see SESSION_HANDOFF.md). This script closes that gap using real ASR output.

Safe to re-run as more of data/asr/*.json finishes (extract_asr.py runs for
hours): each run recomputes asr_transcript for every frame from whatever
ASR files currently exist, and always backs up metadata.pkl first (never
overwritten, never auto-deleted).

Matching rule: a frame gets the text of the ASR segment whose [start, end]
window contains its timestamp; if none contains it exactly (keyframes don't
line up with spoken-sentence boundaries), the nearest segment within
ASR_MATCH_TOLERANCE_SEC is used instead. A frame with no ASR file yet for
its video, or no segment within tolerance (silent/music-only stretch), gets
no asr_transcript.

Usage:
    python scripts/merge_asr_into_metadata.py
"""

import json
import pickle
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STORE_DIR = PROJECT_ROOT / "data" / "persistent_store"
METADATA_PATH = STORE_DIR / "metadata.pkl"
ASR_DIR = PROJECT_ROOT / "data" / "asr"
ASR_MATCH_TOLERANCE_SEC = 5.0


def load_asr_segments(video_id: str):
    path = ASR_DIR / f"{video_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_segment_text(segments, timestamp: float, tolerance: float) -> str:
    """`segments` need not be pre-sorted. Returns the containing segment's
    text, or the nearest one within `tolerance` seconds, else ''."""
    best_text = ""
    best_dist = tolerance
    for seg in segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return seg["text"]
        dist = (seg["start"] - timestamp) if timestamp < seg["start"] else (timestamp - seg["end"])
        if dist < best_dist:
            best_dist = dist
            best_text = seg["text"]
    return best_text


def main():
    if not METADATA_PATH.exists():
        print(f"No metadata at {METADATA_PATH}")
        return

    with open(METADATA_PATH, "rb") as f:
        data = pickle.load(f)
    ids = data.get("ids", [])
    metadata = data.get("metadata", {})

    backup_path = STORE_DIR / f"metadata.pkl.bak_before_asr_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(METADATA_PATH, backup_path)
    print(f"Backed up original metadata to {backup_path}")

    by_video = {}
    for fid in ids:
        vid = metadata[fid].get("video_id", "")
        by_video.setdefault(vid, []).append(fid)

    videos_with_asr = 0
    frames_updated = 0
    asr_cache = {}
    for vid, fids in by_video.items():
        if vid not in asr_cache:
            segs = load_asr_segments(vid)
            asr_cache[vid] = sorted(segs, key=lambda s: s["start"]) if segs else None
        segments = asr_cache[vid]
        if segments is None:
            continue
        videos_with_asr += 1
        for fid in fids:
            ts = float(metadata[fid].get("timestamp", 0.0))
            text = find_segment_text(segments, ts, ASR_MATCH_TOLERANCE_SEC)
            if text:
                metadata[fid]["asr_transcript"] = text
                frames_updated += 1

    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"ids": ids, "metadata": metadata}, f)

    print(f"Videos with ASR available: {videos_with_asr}/{len(by_video)}")
    print(f"Frames updated with asr_transcript: {frames_updated}/{len(ids)}")


if __name__ == "__main__":
    main()
