"""
Parse contest query text into structured parts.

Three contest query modes (selected by file suffix):
  - TKIS:  a natural-language description of an event
  - Q&A:   a description plus a question about that event
  - TRAKE: a description plus a numbered list of N event moments

This module only parses *text*; it does not know about retrieval or videos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

TKIS = "kis"
QA = "qa"
TRAKE = "trake"


@dataclass
class ParsedTrake:
    main: str
    events: List[str]
    event_count: int


@dataclass
class ParsedQA:
    description: str
    question: str


def detect_mode(filename: str) -> str:
    """Detect the query mode from the file's name suffix (-kis/-qa/-trake)."""
    stem = filename.rsplit(".", 1)[0].lower()
    suffix = stem.rsplit("-", 1)[-1].strip()
    if suffix == TKIS:
        return TKIS
    if suffix == QA:
        return QA
    if suffix == TRAKE:
        return TRAKE
    raise ValueError(f"Cannot detect query mode from filename: {filename!r}")


_EVENT_SPLIT_RE = re.compile(r"\(\s*\d+\s*\)")
_TRAKE_PREFIX_RE = re.compile(
    r"^\s*(?:tìm\s+\d+\s+)?khoảnh\s+khắc\s+[^:]*?\s+khi\s*",
    re.IGNORECASE,
)


def parse_trake(text: str) -> ParsedTrake:
    """
    Split a TRAKE query into its main description and its ordered events.

    Events are the parenthesized-number items, e.g. "(1) giậm nhảy".
    The main query is the text before the first numbered event, with any
    leading "Tìm N khoảnh khắc ..." instruction removed.
    """
    text = text.strip()
    parts = _EVENT_SPLIT_RE.split(text)
    if len(parts) < 2:
        raise ValueError("TRAKE query has no numbered events: " + text[:80])

    events: List[str] = []
    for part in parts[1:]:
        event = part.strip().rstrip(".,;")
        if event:
            events.append(event)
    if not events:
        raise ValueError("TRAKE query has empty events: " + text[:80])

    main = parts[0].strip().rstrip(":.,;")
    main = _TRAKE_PREFIX_RE.sub("", main).strip()
    if not main:
        main = parts[0].strip().rstrip(":.,;")

    return ParsedTrake(main=main, events=events, event_count=len(events))


def split_qa(text: str) -> ParsedQA:
    """
    Split a Q&A query into its description (used for retrieval) and its
    question (asked of the VLM).

    Heuristics:
      - The last sentence ending in '?' is the question.
      - If the whole text is one sentence, everything after the last comma
        is treated as the question when the text ends in '?'.
    """
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    last = sentences[-1] if sentences else ""
    if last.endswith("?"):
        if len(sentences) > 1:
            return ParsedQA(description=" ".join(sentences[:-1]), question=last)
        idx = text.rfind(",")
        if idx != -1 and text[idx + 1:].strip().endswith("?"):
            return ParsedQA(
                description=text[:idx].strip(),
                question=text[idx + 1:].strip(),
            )
        return ParsedQA(description=text, question=text)

    return ParsedQA(description=text, question=text)
