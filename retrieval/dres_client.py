"""
DRES (Distributed Retrieval Evaluation Server) client for the AIC 2026 final.

Flow (see HD-ChungKet-2026.pdf):
    1. POST /api/v2/login                       -> sessionId
    2. GET  /api/v2/client/evaluation/list      -> ACTIVE evaluationId
    3. POST /api/v2/submit/{evaluationId}       -> one answer per call

Only ONE answer is ever sent per call. Wrong answers cost 10 points each and
an identical answer must not be sent twice for the same query, so this module
also owns the duplicate guard and an append-only audit log of everything that
was actually sent.

The payload builders are pure functions so they can be unit-tested against the
formats in the rules without any network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

__all__ = [
    "DresError", "DresClient", "normalize_item",
    "build_kis", "build_qa", "build_trake", "payload_summary", "task_kind",
]


class DresError(Exception):
    """A problem the operator can act on (bad login, no active task, ...)."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- #
# Payload builders -- formats are fixed by the contest rules.
# --------------------------------------------------------------------------- #

_VIDEO_EXT = re.compile(r"\.(mp4|avi|mov|mkv|webm|flv|wmv)$", re.IGNORECASE)


def normalize_item(name: str) -> str:
    """'D:/x/N043-V001.mp4' -> 'N043-V001'. DRES items are the file name
    without directory or extension."""
    n = str(name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    n = _VIDEO_EXT.sub("", n)
    if not n:
        raise DresError("Thiếu video_id")
    return n


def _ms(v: Any, what: str) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise DresError(f"{what} không phải số: {v!r}")
    if f != f or f in (float("inf"), float("-inf")):
        raise DresError(f"{what} không hợp lệ: {v!r}")
    return max(0, int(round(f)))


def build_kis(video_id: str, start_ms: Any, end_ms: Any = None, collection: Optional[str] = None) -> Dict[str, Any]:
    """Textual / Video KIS: a video and a time range in milliseconds.

    `collection` (DRES `mediaItemCollectionName`) is only added when given: some DRES servers answer
    "Could not find media item ... Maybe collection name is required" without it."""
    start = _ms(start_ms, "start")
    end = start if end_ms is None else _ms(end_ms, "end")
    if end < start:
        start, end = end, start
    ans: Dict[str, Any] = {"mediaItemName": normalize_item(video_id), "start": start, "end": end}
    if collection and collection.strip():
        ans["mediaItemCollectionName"] = collection.strip()
    return {"answerSets": [{"answers": [ans]}]}


def _clean_answer(answer: str) -> str:
    a = re.sub(r"\s+", " ", str(answer or "")).strip()
    if not a:
        raise DresError("Chưa nhập câu trả lời cho Q&A")
    return a


def build_qa(answer: str, video_id: str, time_ms: Any) -> Dict[str, Any]:
    """Q&A: QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>."""
    text = f"QA-{_clean_answer(answer)}-{normalize_item(video_id)}-{_ms(time_ms, 'time')}"
    return {"answerSets": [{"answers": [{"text": text}]}]}


def build_trake(video_id: str, frame_ids: Sequence[Any]) -> Dict[str, Any]:
    """TRAKE: TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,... (one per event, in order)."""
    ids = [_ms(f, "frame") for f in frame_ids]
    if not ids:
        raise DresError("TRAKE cần ít nhất một frame")
    if any(b <= a for a, b in zip(ids, ids[1:])):
        raise DresError("Các frame TRAKE phải tăng dần theo thời gian: " + ",".join(map(str, ids)))
    text = f"TR-{normalize_item(video_id)}-{','.join(map(str, ids))}"
    return {"answerSets": [{"answers": [{"text": text}]}]}


def task_kind(task: Dict[str, Any]) -> str:
    """Which tab a DRES task belongs to: "tkis" | "vkis" | "qa" | "trake" | "" (unknown).

    Same idea as the organisers' contestant portal (taskType, else the name: QA / TRAKE|ALIGN / KIS), plus the
    Textual-vs-Video split that decides the 5-minute or 4-minute clock.
    """
    parts = [str(task.get(k) or "") for k in ("task_type", "task_group", "name")]
    for text in parts:
        t = text.upper()
        if not t.strip():
            continue
        if "TRAKE" in t or "ALIGN" in t:
            return "trake"
        if re.search(r"Q\s*&\s*A|\bQA\b|QUESTION", t):
            return "qa"
        if "KIS" in t or "KNOWN ITEM" in t:
            allp = " ".join(parts).upper()
            return "vkis" if ("VIDEO" in allp or "VISUAL" in allp) and "TEXT" not in t else "tkis"
    return ""


def payload_summary(payload: Dict[str, Any]) -> str:
    """One-line human readable form of a payload (for logs / the UI)."""
    try:
        a = payload["answerSets"][0]["answers"][0]
    except (KeyError, IndexError, TypeError):
        return json.dumps(payload, ensure_ascii=False)
    if "text" in a:
        return str(a["text"])
    return f"{a.get('mediaItemName')} [{a.get('start')}..{a.get('end')}] ms"


def _canon(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

@dataclass
class _Sent:
    payload_summary: str
    verdict: str
    at: float


class DresClient:
    """Holds one DRES session for the whole app (single operator)."""

    def __init__(self, base_url: str, timeout: float = 10.0, log_path: Optional[Path] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None, state_path: Optional[Path] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.log_path = Path(log_path) if log_path else None
        # A session saved here (NEVER the password) lets a restart of this server resume without a new login.
        self.state_path = Path(state_path) if state_path else None
        self._creds: Optional[Tuple[str, str]] = None      # memory only: used to log in again when the session expires
        self._relogins: List[float] = []                   # when we logged in again on our own (see the fuse in _relogin)
        self._transport = transport
        self._http: Optional[httpx.AsyncClient] = None
        self.username: str = ""
        self.session_id: str = ""
        self.evaluation_id: str = ""
        self.evaluations: List[Dict[str, Any]] = []
        self.collection: str = ""          # optional mediaItemCollectionName sent with KIS answers
        self.query_epoch: int = 0
        self._sent: Dict[str, _Sent] = {}
        self._lock = asyncio.Lock()

    # ---- plumbing -------------------------------------------------------- #

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, transport=self._transport,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=120),
                headers={"Accept": "application/json"},
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def set_base_url(self, base_url: str) -> None:
        base_url = (base_url or "").strip().rstrip("/")
        if not re.match(r"^https?://", base_url):
            raise DresError("Địa chỉ DRES phải bắt đầu bằng http:// hoặc https://")
        if base_url != self.base_url:
            self.base_url = base_url
            self._http = None          # rebuilt lazily; the old one is dropped with its session
            self.session_id = self.evaluation_id = ""
            self.evaluations = []
            self._creds = None

    def _audit(self, record: Dict[str, Any]) -> None:
        """Append-only log. Never contains the password or the session id."""
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass                       # logging must never block a submission

    @staticmethod
    def _describe(resp: httpx.Response) -> str:
        try:
            j = resp.json()
            if isinstance(j, dict) and j.get("description"):
                return str(j["description"])
        except Exception:
            pass
        return (resp.text or "").strip()[:300] or f"HTTP {resp.status_code}"

    # ---- session --------------------------------------------------------- #

    def state(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "logged_in": bool(self.session_id),
            "username": self.username,
            "evaluation_id": self.evaluation_id,
            "evaluations": self.evaluations,
            "collection": self.collection,
            "query_epoch": self.query_epoch,
            "sent_this_query": [
                {"payload": s.payload_summary, "verdict": s.verdict} for s in self._sent.values()
            ],
        }

    async def login(self, username: str, password: str) -> Dict[str, Any]:
        if not username or not password:
            raise DresError("Cần nhập username và password")
        try:
            r = await self._client().post("/api/v2/login", json={"username": username, "password": password})
        except httpx.HTTPError as e:
            raise DresError(f"Không kết nối được DRES ({self.base_url}): {type(e).__name__}")
        if r.status_code != 200:
            raise DresError(f"Đăng nhập thất bại: {self._describe(r)}", r.status_code)
        try:
            sid = r.json().get("sessionId")
        except Exception:
            sid = None
        if not sid:
            raise DresError("DRES không trả về sessionId")
        self.session_id, self.username = sid, username
        self._creds = (username, password)
        await self.refresh_evaluations(_allow_relogin=False)
        self._audit({"ts": time.time(), "event": "login", "user": username, "base_url": self.base_url,
                     "evaluation_id": self.evaluation_id})
        self._save()
        return self.state()

    async def use_session(self, session_id: str) -> Dict[str, Any]:
        """Use a sessionId copied from the DRES user page instead of logging in."""
        sid = (session_id or "").strip()
        if not sid:
            raise DresError("Thiếu sessionId")
        self.session_id, self.username = sid, self.username or "(sessionId)"
        try:
            await self.refresh_evaluations()
        except DresError:
            self.session_id = ""
            raise
        self._audit({"ts": time.time(), "event": "use_session", "base_url": self.base_url,
                     "evaluation_id": self.evaluation_id})
        self._save()
        return self.state()

    def logout(self) -> None:
        self.session_id = self.username = self.evaluation_id = ""
        self.evaluations = []
        self._creds = None
        self._sent.clear()
        self._save()

    async def refresh_evaluations(self, _allow_relogin: bool = True) -> List[Dict[str, Any]]:
        if not self.session_id:
            raise DresError("Chưa đăng nhập DRES")
        try:
            r = await self._client().get("/api/v2/client/evaluation/list", params={"session": self.session_id})
        except httpx.HTTPError as e:
            raise DresError(f"Không kết nối được DRES: {type(e).__name__}")
        if r.status_code == 401:
            self.session_id = ""
            if _allow_relogin and await self._relogin():
                return self.evaluations
            self._save()
            raise DresError("Session hết hạn hoặc sai — đăng nhập lại", 401)
        if r.status_code != 200:
            raise DresError(f"Không lấy được danh sách evaluation: {self._describe(r)}", r.status_code)
        try:
            data = r.json()
        except Exception:
            raise DresError("Danh sách evaluation không phải JSON")
        self.evaluations = [
            {"id": e.get("id"), "name": e.get("name", ""), "status": e.get("status", ""), "type": e.get("type", "")}
            for e in (data if isinstance(data, list) else []) if isinstance(e, dict) and e.get("id")
        ]
        keep = next((e for e in self.evaluations if e["id"] == self.evaluation_id and e["status"] == "ACTIVE"), None)
        if keep is None:
            active = [e for e in self.evaluations if e["status"] == "ACTIVE"]
            self.evaluation_id = active[0]["id"] if active else ""
        self._save()
        return self.evaluations

    async def current_task(self, evaluation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """The task DRES is running for the selected evaluation, or None between tasks / when unknown.

        Read-only; the same call the organisers' contestant portal polls:
        GET /api/v2/client/evaluation/currentTask/{evaluationId}?session=
        """
        eid = evaluation_id or self.evaluation_id
        if not (self.session_id and eid):
            return None
        try:
            r = await self._client().get(f"/api/v2/client/evaluation/currentTask/{eid}",
                                         params={"session": self.session_id})
        except httpx.HTTPError as e:
            raise DresError(f"Không kết nối được DRES: {type(e).__name__}")
        if r.status_code == 401:
            raise DresError("Session hết hạn hoặc sai — đăng nhập lại", 401)
        if r.status_code != 200:
            return None                                   # 400 / 404: no task is running right now
        try:
            d = r.json()
        except Exception:
            return None
        if not isinstance(d, dict) or not d.get("name"):
            return None
        try:
            duration = int(float(d.get("duration") or 0))
        except (TypeError, ValueError):
            duration = 0
        if duration > 3600:          # the test evaluations report 300000 for a 5-minute task: that is milliseconds
            duration //= 1000
        t = {"name": str(d.get("name")), "task_group": str(d.get("taskGroup") or ""), "task_type": str(d.get("taskType") or ""),
             "duration": duration, "template_id": str(d.get("templateId") or d.get("id") or "")}
        t["kind"] = task_kind(t)
        return t

    # ---- staying logged in ------------------------------------------------ #

    def _save(self) -> None:
        """Persist the session (not the password) so a server restart does not log the operator out."""
        if self.state_path is None:
            return
        try:
            if not self.session_id:
                self.state_path.unlink(missing_ok=True)
                return
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(self.state_path.name + ".tmp")
            tmp.write_text(json.dumps({"base_url": self.base_url, "session_id": self.session_id, "username": self.username,
                                       "evaluation_id": self.evaluation_id, "collection": self.collection}), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except Exception:
            pass

    async def restore(self) -> bool:
        """Resume the session saved by a previous run. False (and the file is dropped) if DRES no longer accepts it."""
        if self.state_path is None or not self.state_path.exists():
            return False
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            base = (d.get("base_url") or self.base_url).rstrip("/")
            if base != self.base_url:
                self.set_base_url(base)
            self.session_id, self.username = d["session_id"], d.get("username", "")
            self.evaluation_id, self.collection = d.get("evaluation_id", ""), d.get("collection", "")
        except Exception:
            return False
        try:
            await self.refresh_evaluations(_allow_relogin=False)
        except DresError:
            self.session_id = self.username = self.evaluation_id = ""
            self._save()
            return False
        self._audit({"ts": time.time(), "event": "restore_session", "base_url": self.base_url,
                     "evaluation_id": self.evaluation_id})
        return True

    async def _relogin(self) -> bool:
        if not self._creds:
            return False
        # Fuse: if DRES allows only one session per account, logging in again here would kick the operator out of the
        # DRES web page (where the task is read), and that page's login would kick us out -- a fight. So renewing on
        # our own is allowed twice in 10 minutes; beyond that we stop and leave the login to the operator.
        now = time.time()
        self._relogins = [t for t in self._relogins if now - t < 600]
        if len(self._relogins) >= 2:
            self._audit({"ts": now, "event": "relogin_disabled", "reason": "session dropped again and again"})
            self._creds = None
            return False
        self._relogins.append(now)
        user, pw = self._creds
        try:
            await self.login(user, pw)
        except DresError:
            self.session_id = ""
            return False
        self._audit({"ts": time.time(), "event": "relogin", "user": user, "evaluation_id": self.evaluation_id})
        return True

    async def startup(self, username: str = "", password: str = "") -> None:
        """Resume a saved session; optional credentials (from the environment) enable login now and re-login later."""
        if username and password:
            self._creds = (username, password)
        if await self.restore():
            return
        if self._creds:
            await self._relogin()

    async def keepalive(self, interval: float = 30.0) -> None:
        """Ping DRES so the session stays alive, log in again if it died, and follow the ACTIVE evaluation."""
        while True:
            await asyncio.sleep(interval)
            try:
                if self.session_id:
                    await self.refresh_evaluations()
                elif self._creds:
                    await self._relogin()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    def select_evaluation(self, evaluation_id: str) -> None:
        if not any(e["id"] == evaluation_id for e in self.evaluations):
            raise DresError("Evaluation không có trong danh sách")
        self.evaluation_id = evaluation_id

    # ---- query lifecycle ------------------------------------------------- #

    def new_query(self) -> int:
        """Start a fresh query: forget what was sent (a new question may legitimately
        have the same answer text as the previous one)."""
        self.query_epoch += 1
        self._sent.clear()
        return self.query_epoch

    # ---- submit ---------------------------------------------------------- #

    async def submit(self, payload: Dict[str, Any], force: bool = False, evaluation_id: Optional[str] = None) -> Dict[str, Any]:
        """Send exactly one answer. Returns a dict the UI can render as-is.

        `evaluation_id` lets each open page submit to the evaluation of its own tab without changing the one other
        pages use; without it the client's selected evaluation is used."""
        if not self.session_id:
            raise DresError("Chưa đăng nhập DRES")
        if evaluation_id:
            if not any(e["id"] == evaluation_id for e in self.evaluations):
                await self.refresh_evaluations()
                if not any(e["id"] == evaluation_id for e in self.evaluations):
                    raise DresError("Evaluation không có trong danh sách của DRES — bấm chip DRES để làm mới", 404)
        elif not self.evaluation_id:
            await self.refresh_evaluations()
            if not self.evaluation_id:
                raise DresError("Chưa có evaluation nào ở trạng thái ACTIVE", 412)

        key = _canon(payload)
        summary = payload_summary(payload)
        async with self._lock:                      # a double click cannot race past the guard
            prev = self._sent.get(key)
            if prev is not None and not (force and prev.verdict == "UNKNOWN"):
                return {"ok": False, "duplicate": True, "verdict": prev.verdict, "payload": payload,
                        "summary": summary,
                        "description": f"Đã nộp đáp án này rồi ({prev.verdict}). DRES không cho nộp trùng."}
            t0 = time.perf_counter()
            eid = evaluation_id or self.evaluation_id
            try:
                r = await self._client().post(
                    f"/api/v2/submit/{eid}", params={"session": self.session_id}, json=payload)
                if r.status_code == 401 and self._creds and await self._relogin():      # nothing was processed: safe to replay
                    eid = evaluation_id or self.evaluation_id
                    if not eid:
                        raise DresError("Đã đăng nhập lại nhưng chưa có evaluation ACTIVE", 412)
                    r = await self._client().post(
                        f"/api/v2/submit/{eid}", params={"session": self.session_id}, json=payload)
            except httpx.HTTPError as e:
                # We cannot know whether DRES received it. Block a blind retry.
                self._sent[key] = _Sent(summary, "UNKNOWN", time.time())
                self._audit({"ts": time.time(), "epoch": self.query_epoch, "evaluation_id": eid, "payload": payload,
                             "http_status": None, "error": type(e).__name__})
                return {"ok": False, "duplicate": False, "verdict": "UNKNOWN", "payload": payload,
                        "summary": summary, "http_status": None,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "description": f"Mất kết nối khi nộp ({type(e).__name__}). Chưa rõ DRES đã nhận chưa — "
                                       f"kiểm tra trên web DRES trước khi nộp lại."}
            latency = round((time.perf_counter() - t0) * 1000, 1)

            verdict, desc, ok = "ERROR", self._describe(r), False
            if r.status_code == 200:
                try:
                    j = r.json()
                except Exception:
                    j = {}
                verdict = str(j.get("submission") or "UNKNOWN").upper()
                desc = str(j.get("description") or desc)
                ok = bool(j.get("status", True))
            # Only a judged answer counts as "sent"; rejected requests (401/412/...) can be retried.
            if r.status_code == 200:
                self._sent[key] = _Sent(summary, verdict, time.time())
            if r.status_code == 401:
                self.session_id = ""
                self._save()
            elif r.status_code == 404 and "Evaluation" in desc:         # the evaluation id we hold is gone: re-read the list
                try:
                    await self.refresh_evaluations()
                except DresError:
                    pass
            self._audit({"ts": time.time(), "epoch": self.query_epoch, "evaluation_id": eid, "payload": payload,
                         "http_status": r.status_code, "verdict": verdict, "description": desc,
                         "latency_ms": latency})
            return {"ok": ok, "duplicate": False, "verdict": verdict, "payload": payload, "summary": summary,
                    "http_status": r.status_code, "latency_ms": latency, "description": desc}
