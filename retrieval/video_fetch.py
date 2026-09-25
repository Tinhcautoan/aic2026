"""
On-demand download of ONE video out of the big old-corpus zip on Hugging Face (5clnl/idk :: videos.zip, 83 GB).

The zip cannot be downloaded as a whole on a weak connection, so the operator asks for the single video they want to
look at: the entry is read with HTTP Range requests (offsets come from the zip's central directory, saved once in
config/old_videos_zip_index.json), inflated on the fly, checked against the zip's own size + CRC32 and only then
renamed to <dest>/<video>.mp4. A dropped connection resumes where it stopped (the inflater keeps its state).
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Callable, Dict, Optional

import httpx
from loguru import logger

DEFAULT_URL = "https://huggingface.co/datasets/5clnl/idk/resolve/main/videos.zip"


class FetchError(Exception):
    """A failure worth retrying (network hiccup, early end of stream)."""


class FatalFetchError(FetchError):
    """A failure that retrying cannot fix (bad header, missing file, corrupt data, wrong CRC)."""


class VideoFetcher:
    def __init__(self, zip_index: Path, dest_dir: Path, url: str = DEFAULT_URL, max_parallel: int = 2,
                 has_local: Optional[Callable[[str], bool]] = None, on_done: Optional[Callable[[str, Path], None]] = None,
                 max_attempts: int = 25, transport: Optional[httpx.BaseTransport] = None, retry_sleep: float = 3.0):
        self.entries: Dict[str, dict] = {}
        try:
            for e in json.loads(Path(zip_index).read_text(encoding="utf-8")):
                name = e["name"]
                stem = os.path.splitext(os.path.basename(name))[0]
                self.entries[stem] = e
        except (OSError, ValueError, KeyError) as ex:
            logger.warning(f"VideoFetcher: cannot read {zip_index}: {ex}")
        self.dest = Path(dest_dir)
        self.url = url
        self.has_local = has_local or (lambda v: (self.dest / f"{v}.mp4").is_file())
        self.on_done = on_done
        self.max_attempts = max_attempts
        self.retry_sleep = retry_sleep
        self._transport = transport
        self._sem = threading.Semaphore(max_parallel)
        self._lock = threading.RLock()
        self._state: Dict[str, dict] = {}

    # ------------------------------------------------------------------ info

    def can_fetch(self, video_id: str) -> bool:
        return video_id in self.entries

    def size_bytes(self, video_id: str) -> Optional[int]:
        e = self.entries.get(video_id)
        return int(e["size"]) if e else None

    def status(self, video_id: str) -> dict:
        with self._lock:
            st = dict(self._state.get(video_id) or {})
        if not st:
            st = {"state": "done" if self.has_local(video_id) else "idle"}
        st.setdefault("done", 0)
        st.setdefault("total", int(self.entries[video_id]["csize"]) if video_id in self.entries else 0)
        st["fetchable"] = self.can_fetch(video_id)
        return st

    # ----------------------------------------------------------------- start

    def start(self, video_id: str) -> dict:
        if video_id not in self.entries:
            raise FetchError(f"{video_id} không có trong videos.zip trên Hugging Face")
        if self.has_local(video_id):
            with self._lock:
                self._state[video_id] = {"state": "done", "done": int(self.entries[video_id]["csize"]),
                                         "total": int(self.entries[video_id]["csize"])}
            return self.status(video_id)
        with self._lock:
            cur = self._state.get(video_id)
            if cur and cur.get("state") in ("queued", "running"):
                return self.status(video_id)
            self._state[video_id] = {"state": "queued", "done": 0, "total": int(self.entries[video_id]["csize"]),
                                     "started": time.time(), "speed": 0.0, "attempt": 0, "error": ""}
        threading.Thread(target=self._run, args=(video_id,), name=f"fetch-{video_id}", daemon=True).start()
        return self.status(video_id)

    # ------------------------------------------------------------------- work

    def _set(self, video_id: str, **kw) -> None:
        with self._lock:
            self._state.setdefault(video_id, {}).update(kw)

    def _client(self) -> httpx.Client:
        return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, connect=20.0), transport=self._transport,
                            headers={"User-Agent": "aic-video-fetch/1.0"})

    def _data_start(self, client: httpx.Client, e: dict) -> int:
        """Offset of the entry's data = local header offset + 30 + name length + extra length (read from the local header)."""
        off = int(e["off"])
        r = client.get(self.url, headers={"Range": f"bytes={off}-{off + 30 + 1024}"})
        self._check_status(r.status_code, "khi đọc header")
        h = r.content
        if len(h) < 30 or h[:4] != b"PK\x03\x04":
            raise FatalFetchError("header cục bộ của zip không hợp lệ")
        method = struct.unpack("<H", h[8:10])[0]
        nlen, elen = struct.unpack("<HH", h[26:30])
        if method != int(e["method"]):
            raise FatalFetchError(f"phương thức nén khác kỳ vọng ({method} vs {e['method']})")
        return off + 30 + nlen + elen

    @staticmethod
    def _check_status(code: int, what: str = "") -> None:
        if code == 206:
            return
        msg = f"HTTP {code} {what}".strip()
        if code in (200, 400, 401, 403, 404, 416):        # 200 = the server ignored Range: retrying will not help
            raise FatalFetchError(msg)
        raise FetchError(msg)

    def _run(self, video_id: str) -> None:
        e = self.entries[video_id]
        csize, size, crc_want, method = int(e["csize"]), int(e["size"]), int(e["crc"]), int(e["method"])
        self.dest.mkdir(parents=True, exist_ok=True)
        final, part = self.dest / f"{video_id}.mp4", self.dest / f"{video_id}.mp4.part"
        with self._sem:
            self._set(video_id, state="running")
            pos, crc, written = 0, 0, 0
            inflater = zlib.decompressobj(-15) if method == 8 else None
            data_start, last_err = None, ""
            t_mark, pos_mark, speed = time.time(), 0, 0.0
            try:
                with open(part, "wb") as out:
                    for attempt in range(1, self.max_attempts + 1):
                        self._set(video_id, attempt=attempt)
                        try:
                            with self._client() as client:
                                if data_start is None:
                                    data_start = self._data_start(client, e)
                                if pos >= csize:
                                    break
                                with client.stream("GET", self.url, headers={"Range": f"bytes={data_start + pos}-{data_start + csize - 1}"}) as r:
                                    self._check_status(r.status_code)
                                    for chunk in r.iter_bytes():                   # pieces as they arrive, so a drop loses almost nothing
                                        pos += len(chunk)
                                        raw = inflater.decompress(chunk) if inflater else chunk
                                        if raw:
                                            out.write(raw)
                                            crc = zlib.crc32(raw, crc)
                                            written += len(raw)
                                        now = time.time()
                                        if now - t_mark >= 1.0:
                                            inst = (pos - pos_mark) / (now - t_mark)
                                            speed = inst if speed == 0 else 0.7 * speed + 0.3 * inst
                                            t_mark, pos_mark = now, pos
                                            self._set(video_id, done=pos, speed=round(speed))
                                        else:
                                            self._set(video_id, done=pos)
                            if pos >= csize:
                                break
                            raise FetchError(f"kết thúc sớm ở {pos}/{csize} byte")
                        except FatalFetchError:
                            raise
                        except FetchError as ex:
                            last_err = str(ex)
                        except (httpx.HTTPError, OSError) as ex:
                            last_err = f"{type(ex).__name__}: {str(ex)[:120]}"
                        except zlib.error as ex:
                            raise FatalFetchError(f"dữ liệu nén hỏng: {ex}")
                        logger.warning(f"fetch {video_id}: lần {attempt}/{self.max_attempts} lỗi ({last_err}); tiếp tục từ byte {pos}")
                        self._set(video_id, error=last_err)
                        time.sleep(self.retry_sleep * min(attempt, 5))
                    else:
                        raise FetchError(f"quá {self.max_attempts} lần thử: {last_err}")
                    if inflater:
                        tail = inflater.flush()
                        if tail:
                            out.write(tail)
                            crc = zlib.crc32(tail, crc)
                            written += len(tail)
                    out.flush()
                    os.fsync(out.fileno())
                if written != size:
                    raise FatalFetchError(f"kích thước sau giải nén {written} khác {size}")
                if (crc & 0xFFFFFFFF) != crc_want:
                    raise FatalFetchError("CRC32 không khớp — file tải về bị hỏng")
                os.replace(part, final)
                self._set(video_id, state="done", done=csize, error="", speed=0)
                logger.info(f"fetched {video_id} ({size / 1e6:.0f} MB) -> {final}")
                if self.on_done:
                    try:
                        self.on_done(video_id, final)
                    except Exception as ex:
                        logger.warning(f"on_done failed: {ex}")
            except Exception as ex:
                logger.warning(f"fetch {video_id} failed: {ex}")
                self._set(video_id, state="error", error=str(ex))
                try:
                    part.unlink()
                except OSError:
                    pass
