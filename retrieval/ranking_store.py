"""
Manual ranking store — persist UI-edited result orderings to plain-text
submission files under ``<PROJECT_ROOT>/submission/``.

This is a DISPLAY-LAYER artifact only. Editing a ranking here rewrites the
on-disk text file that holds the *shown / exported* result order for a
query. It never touches embeddings, the vector store, BM25, or any model
output — think of it as "save the human's re-ordered answer sheet".

Line format matches ``retrieval.answer_expander.format_csv_line`` / the BTC
spec (no header row, row 1 == rank 1 == best):

    KIS    ->  video_id, frame_idx
    Q&A    ->  video_id, frame_idx, "answer"
    TRAKE  ->  video_id, f1, f2, ..., fN
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from retrieval.answer_expander import format_csv_line

RANKING_DIRNAME = "submission"
MAX_ROWS = 100

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_NUM_RE = re.compile(r"^-?\d+$")


def ranking_dir() -> Path:
    """The submission/ folder, created on first use."""
    d = settings.PROJECT_ROOT / RANKING_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(name: str, default: str = "manual_ranking") -> str:
    """Sanitize a user-supplied name to a bare ``<stem>.<txt|csv>`` basename.

    Directory components are stripped, so a caller can never escape
    ``submission/``.
    """
    raw = (name or "").strip().replace("\\", "/").split("/")[-1]
    stem, dot, ext = raw.partition(".")
    stem = _SAFE_RE.sub("_", stem).strip("._-") or default
    ext = _SAFE_RE.sub("", ext).lower() if dot else ""
    if ext not in ("txt", "csv"):
        ext = "txt"
    return f"{stem}.{ext}"


def _row_to_line(row: Dict[str, Any], mode: str) -> Optional[str]:
    """Serialize one ranking row to a submission line, or None to skip it."""
    vid = str(row.get("video_id") or "").strip()
    if not vid:
        return None

    mode = (mode or "tkis").lower()
    frames = row.get("frames")
    answer = row.get("answer")
    fidx = row.get("frame_idx")

    if mode == "trake" and frames:
        fr = [int(f) for f in frames if str(f).strip() != ""]
        return format_csv_line(vid, frames=fr) if fr else None

    if mode in ("qa", "q&a", "qna") and answer not in (None, ""):
        try:
            fi = int(fidx)
        except (TypeError, ValueError):
            fi = 0
        return format_csv_line(vid, frame_idx=fi, answer=str(answer))

    # default: KIS  ->  video_id, frame_idx
    try:
        fi = int(fidx)
    except (TypeError, ValueError):
        return None
    return format_csv_line(vid, frame_idx=fi)


def save_ranking(
    name: str,
    rows: List[Dict[str, Any]],
    mode: str = "tkis",
    max_rows: int = MAX_ROWS,
) -> Dict[str, Any]:
    """Write ``rows`` (index 0 == Top 1) to ``submission/<name>``.

    Overwrites any existing file with that name. Returns a small summary.
    """
    fname = safe_name(name)
    lines: List[str] = []
    skipped = 0
    for row in (rows or [])[:max_rows]:
        line = _row_to_line(row, mode)
        if line is None:
            skipped += 1
            continue
        lines.append(line)

    path = ranking_dir() / fname
    with open(path, "w", newline="", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    return {
        "name": fname,
        "path": str(path),
        "rows_written": len(lines),
        "rows_skipped": skipped,
        "mode": (mode or "tkis").lower(),
    }


def _parse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one submission line back into a row dict."""
    line = line.strip()
    if not line:
        return None
    try:
        parts = next(csv.reader([line], skipinitialspace=True))
    except Exception:
        parts = [p.strip() for p in line.split(",")]
    parts = [p.strip() for p in parts]
    if not parts or not parts[0]:
        return None

    vid = parts[0]
    nums: List[int] = []
    answer: Optional[str] = None
    for tok in parts[1:]:
        if _NUM_RE.match(tok):
            nums.append(int(tok))
        elif tok:
            answer = tok  # first non-numeric extra token == answer text

    row: Dict[str, Any] = {"video_id": vid}
    if answer is not None:
        row["frame_idx"] = nums[0] if nums else 0
        row["answer"] = answer
    elif len(nums) >= 2:
        row["frames"] = nums
        row["frame_idx"] = nums[0]
    else:
        row["frame_idx"] = nums[0] if nums else 0
    return row


def load_ranking(name: str) -> Dict[str, Any]:
    """Read ``submission/<name>`` back into ordered rows."""
    fname = safe_name(name)
    path = ranking_dir() / fname
    if not path.is_file():
        raise FileNotFoundError(f"Ranking file not found: {fname}")

    raw = path.read_text(encoding="utf-8")
    rows: List[Dict[str, Any]] = []
    for ln in raw.splitlines():
        parsed = _parse_line(ln)
        if parsed:
            parsed["rank"] = len(rows) + 1
            rows.append(parsed)

    mode_guess = "tkis"
    if any("answer" in r for r in rows):
        mode_guess = "qa"
    elif any(len(r.get("frames") or []) >= 2 for r in rows):
        mode_guess = "trake"

    return {"name": fname, "mode": mode_guess, "rows": rows, "raw": raw}


def list_rankings() -> List[Dict[str, Any]]:
    """Every saved ranking file, newest first."""
    d = ranking_dir()
    seen = set()
    out: List[Dict[str, Any]] = []
    for p in list(d.glob("*.txt")) + list(d.glob("*.csv")):
        if p.name in seen:
            continue
        seen.add(p.name)
        try:
            st = p.stat()
            n = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
        except Exception:
            continue
        out.append({
            "name": p.name,
            "rows": n,
            "modified": st.st_mtime,
            "size": st.st_size,
        })
    out.sort(key=lambda item: item["modified"], reverse=True)
    return out


def delete_ranking(name: str) -> bool:
    fname = safe_name(name)
    path = ranking_dir() / fname
    if path.is_file():
        path.unlink()
        return True
    return False
