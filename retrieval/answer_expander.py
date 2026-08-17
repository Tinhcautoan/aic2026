"""
Answer Expander: format and rank candidate rows for R@k scoring.

R@k = max R-Score among the first k rows, Final = mean(R@1,R@5,R@20,R@50,R@100).
To maximize this, the highest-confidence answer must sit at position 1
(affects all 5 R@k values), while positions 2–100 serve as a hedge
against uncertainty.  This module sorts candidate rows by descending
confidence and caps at ``max_rows`` (default 100).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SubmissionRow:
    """A single candidate answer row with its confidence score."""
    line: str
    score: float


def format_csv_line(
    video_id: str,
    frame_idx: Optional[int] = None,
    frames: Optional[List[int]] = None,
    answer: Optional[str] = None,
) -> str:
    """
    Format a single answer line according to the BTC CSV spec.

    - TKIS:  ``video_id,frame_idx``
    - Q&A:   ``video_id,frame_idx,"answer"``
    - TRAKE: ``video_id,f1,f2,...,fN``

    No header row.  Answers containing commas or quotes are properly
    quoted using Python's csv module for RFC-4180 compliance.
    """
    parts: List[str] = [video_id]
    if frames is not None:
        parts.extend(str(f) for f in frames)
        return ",".join(parts)

    if frame_idx is not None:
        parts.append(str(frame_idx))

    if answer is not None:
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([answer])
        parts.append(buf.getvalue().rstrip("\r\n"))

    return ",".join(parts)


def expand_rows(
    rows: List[SubmissionRow],
    max_rows: int = 100,
) -> List[SubmissionRow]:
    """
    Sort candidate rows by descending score and cap at ``max_rows``.

    Duplicates (same line) are kept — they don't hurt and may occur
    when multiple videos produce identical frames.
    """
    ranked = sorted(rows, key=lambda r: r.score, reverse=True)
    return ranked[:max_rows]
