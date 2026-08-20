"""
Video-level metadata (contest media-info) -> per-frame text enrichment.

The AIC25 contest ships ``data/media-info-aic25-b1/media-info/<VIDEO_ID>.json``
for every contest video, e.g. L21_V001.json:

    title, description, keywords[], publish_date, author, channel_id,
    channel_url, length, thumbnail_url, watch_url

We merge these fields into every frame's ``metadata`` dict at ingestion time,
which unlocks the previously-dead Stage 2 (BM25) and Stage 4 (cross-encoder)
of the retrieval funnel:

  - BM25 now matches date / keyword / title-based queries exactly
    (e.g. "bản tin 60 giây ngày 1/8/2024" -> L21_V001's title + keywords).
  - The cross-encoder gets a real (query, video-text) document to score
    instead of skipping every candidate because the metadata was empty.

Design notes:
  - The description is mostly YouTube boilerplate (copyright/social links), so
    only a short cleaned headline is kept in the text signal.
  - publish_date is expanded into common date spellings so BM25 (pure token
    overlap) can match the many date formats queries use.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from config import settings

_CACHE: Dict[str, Optional[dict]] = {}
_LOCK = threading.Lock()

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_HASHTAG_RE = re.compile(r"#[\w\u00C0-\u1EF9]+")
# Lines that are pure YouTube/promotion boilerplate — drop them entirely.
_BOILERPLATE_RE = re.compile(
    r"(bản quyền|copyright|đăng ký kênh|đăng ký kÊnh|theo dõi|subcribe|subscribe|"
    r"xem tv|xem television|htvc|hplus|facebook|instagram|tiktok|lotus|youtube|"
    r"mobile|smart tv|android|ios|play store|app store|wap|web)",
    re.IGNORECASE,
)

_TITLE_VIDEO_DATE_RE = re.compile(
    r"ng\s?[àa]y\s*[:\-]?\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})",
    re.IGNORECASE,
)
_TITLE_COMPACT_DATE_RE = re.compile(
    r"(?:ng\s?[àa]y\s*)?\b(\d{8})\b", re.IGNORECASE,
)
_DMY_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")
_YMD_RE = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$")

# Common Vietnamese month names used when spelling a date out in queries.
_MONTHS_VI = {
    1: "tháng 1", 2: "tháng 2", 3: "tháng 3", 4: "tháng 4", 5: "tháng 5",
    6: "tháng 6", 7: "tháng 7", 8: "tháng 8", 9: "tháng 9", 10: "tháng 10",
    11: "tháng 11", 12: "tháng 12",
}


def normalize_video_id(video_id: Optional[str]) -> str:
    """Strip extension / path so 'foo.mp4', 'dir/foo' -> 'foo'."""
    if not video_id:
        return ""
    return Path(str(video_id).strip()).stem


def media_info_dir() -> Path:
    return settings.MEDIA_INFO_DIR


def load_media_info(video_id: Optional[str]) -> Optional[dict]:
    """Read (and cache) the media-info JSON for a contest video id.

    Returns None when the file is missing or unparseable.  Thread-safe and
    cheap after the first lookup per video.
    """
    vid = normalize_video_id(video_id)
    if not vid:
        return None

    with _LOCK:
        if vid in _CACHE:
            return _CACHE[vid]

    path = media_info_dir() / f"{vid}.json"
    info: Optional[dict] = None
    if path.exists():
        try:
            import json

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                info = raw
        except Exception as e:
            logger.warning(f"Failed to read media-info {path}: {e}")

    with _LOCK:
        _CACHE[vid] = info
    return info


def clean_description(description: Optional[str]) -> str:
    """Strip URLs/emails/hashtags/boilerplate; keep a short headline.

    The contest descriptions are mostly YouTube channel boilerplate
    (copyright, social links, app promotion).  For BM25 / cross-encoder we
    only want the actual news headline, i.e. the first meaningful line(s).
    """
    if not description:
        return ""
    lines = []
    for line in str(description).splitlines():
        line = _URL_RE.sub(" ", line)
        line = _EMAIL_RE.sub(" ", line)
        line = _HASHTAG_RE.sub(" ", line)
        line = re.sub(r"[►✅►©️\u00A0]+", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if len(line) < 8:
            continue
        if _BOILERPLATE_RE.search(line):
            continue
        lines.append(line)
        if len(lines) >= 2:
            break
    if not lines:
        return ""
    # Keep at most 2 short lines (headline + subtitle), 200 chars each.
    kept = []
    for line in lines:
        kept.append(line[:200])
        if len(kept) >= 2:
            break
    return " ".join(kept)


def _date_to_variants(d: int, m: int, y: int) -> List[str]:
    """All common spellings for a (day, month, year) triple."""
    yy = y % 100
    variants = {
        f"{d:02d}/{m:02d}/{y:04d}",
        f"{d}/{m}/{y:04d}",
        f"{d:02d}-{m:02d}-{y:04d}",
        f"{d}-{m}-{y:04d}",
        f"{d:02d}{m:02d}{y:04d}",
        f"{d}{m}{yy:02d}",
        f"{y:04d}-{m:02d}-{d:02d}",
        f"{y:04d}{m:02d}{d:02d}",
        f"{d:02d}/{m:02d}/{yy:02d}",
        f"{d}/{m}/{yy:02d}",
        f"ngày {d} tháng {m} năm {y:04d}",
        f"ngày {d}/{m}/{y:04d}",
        f"{_MONTHS_VI.get(m, '')} năm {y:04d}",
    }
    return sorted(v for v in variants if v)


def date_variants_text(date_str: Optional[str]) -> str:
    """Expand a publish-date / title date into every common spelling.

    ``"01/08/2024"`` -> ``"01/08/2024 | 1/8/2024 | 01-08-2024 | 01082024 | ..."``
    BM25 is pure token overlap, so spelling variants are what make date
    queries ("1/8/2024", "ngày 1 tháng 8", "01082024") actually match.
    """
    if not date_str:
        return ""
    s = str(date_str).strip()
    found: List[tuple] = []  # (d, m, y)

    m = _DMY_RE.match(s)
    if m:
        d, mm, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= d <= 31:
            found.append((d, mm, y))
    else:
        m = _YMD_RE.match(s)
        if m:
            y, mm, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mm <= 12 and 1 <= d <= 31:
                found.append((d, mm, y))
        else:
            m = _TITLE_COMPACT_DATE_RE.search(s)
            if m:
                digits = m.group(1)
                if len(digits) == 8:
                    # "Ngày 01082024" is DDMMYYYY in the contest titles.
                    try:
                        cands = [
                            (int(digits[:2]), int(digits[2:4]), int(digits[4:8])),  # DDMMYYYY
                            (int(digits[6:8]), int(digits[4:6]), int(digits[:4])),  # YYYYMMDD
                        ]
                    except ValueError:
                        cands = []
                    for dd, mm, y in cands:
                        if 1 <= mm <= 12 and 1 <= dd <= 31 and len(f"{y:04d}") == 4:
                            found.append((dd, mm, y))
                            break
            # Also try "Ngày 01/08/2024" in a title.
            m = _TITLE_VIDEO_DATE_RE.search(s)
            if m and not found:
                d, mm, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                y = y + 2000 if y < 100 else y
                if 1 <= mm <= 12 and 1 <= d <= 31:
                    found.append((d, mm, y))

    if not found:
        return s
    return " | ".join(
        v for triple in found for v in _date_to_variants(*triple)
    )


def frame_metadata_fields(video_id: Optional[str]) -> dict:
    """Video-level metadata to merge into every frame's ``metadata`` dict."""
    info = load_media_info(video_id)
    if not info:
        return {}
    return {
        "video_title": info.get("title", "") or "",
        "video_keywords": list(info.get("keywords") or []),
        "video_publish_date": info.get("publish_date", "") or "",
        "video_description": clean_description(info.get("description", "")),
        "video_author": info.get("author", "") or "",
        "video_channel": info.get("channel_id", "") or "",
        "video_length": info.get("length") or 0,
    }


def video_frame_count(video_id: Optional[str]) -> Optional[int]:
    """Estimated total frame count of a video from media-info ``length`` (s).

    Used by the contest Q&A / TRAKE pipelines to size their delta windows.
    Falls back to None when no media-info is available.
    """
    info = load_media_info(video_id)
    if not info:
        return None
    length = info.get("length")
    if not length:
        return None
    fps = settings.MEDIA_FRAME_COUNT_FPS
    return int(float(length) * fps)

    vid = normalize_video_id(video_id)
    if not vid:
        return None

    with _LOCK:
        if vid in _CACHE:
            return _CACHE[vid]

    path = media_info_dir() / f"{vid}.json"
    info: Optional[dict] = None
    if path.exists():
        try:
            import json

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                info = raw
        except Exception as e:
            logger.warning(f"Failed to read media-info {path}: {e}")

    with _LOCK:
        _CACHE[vid] = info
    return info
