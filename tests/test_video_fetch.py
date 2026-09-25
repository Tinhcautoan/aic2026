"""VideoFetcher: one video out of a big zip via HTTP Range, against an in-memory fake of the Hugging Face server."""
import json
import os
import re
import time
import zipfile

import httpx
import pytest

from retrieval.video_fetch import FetchError, VideoFetcher

URL = "https://huggingface.co/datasets/x/y/resolve/main/videos.zip"
CDN = "https://cdn.example.com/signed/abc"


def make_zip(tmp_path):
    payloads = {
        "videos/L26_V001.mp4": (os.urandom(200_000) + b"A" * 2_500_000 + os.urandom(100_000), zipfile.ZIP_DEFLATED),
        "videos/L26_V002.mp4": (os.urandom(300_000), zipfile.ZIP_STORED),
        "videos/L21_V009.mp4": (os.urandom(700_000), zipfile.ZIP_DEFLATED),
    }
    zp = tmp_path / "videos.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for name, (data, method) in payloads.items():
            z.writestr(zipfile.ZipInfo(name), data, compress_type=method)
    idx = []
    with zipfile.ZipFile(zp) as z:
        for i in z.infolist():
            idx.append({"name": i.filename, "off": i.header_offset, "csize": i.compress_size, "size": i.file_size,
                        "crc": i.CRC, "method": i.compress_type})
    (tmp_path / "index.json").write_text(json.dumps(idx))
    return zp.read_bytes(), {n: d for n, (d, _) in payloads.items()}, tmp_path / "index.json"


class Cut(httpx.SyncByteStream):
    """Yields part of the body, then drops the connection."""

    def __init__(self, body, keep):
        self.body, self.keep = body, keep

    def __iter__(self):
        yield self.body[:self.keep]
        raise httpx.ReadError("connection reset")


class FakeHF:
    def __init__(self, blob, cut_first_stream_at=None, status=None):
        self.blob, self.cut, self.status, self.calls = blob, cut_first_stream_at, status, []

    def __call__(self, request: httpx.Request):
        self.calls.append((str(request.url), request.headers.get("range")))
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"location": CDN})
        if self.status:
            return httpx.Response(self.status)
        m = re.match(r"bytes=(\d+)-(\d+)", request.headers.get("range", ""))
        if not m:
            return httpx.Response(200, content=self.blob)               # the server ignored Range
        a, b = int(m.group(1)), min(int(m.group(2)), len(self.blob) - 1)
        body = self.blob[a:b + 1]
        hdr = {"content-range": f"bytes {a}-{b}/{len(self.blob)}"}
        if self.cut is not None and len(body) > 100_000:                  # only a data request, not the tiny header read
            keep, self.cut = self.cut, None
            return httpx.Response(206, headers=hdr, stream=Cut(body, keep))
        return httpx.Response(206, headers=hdr, content=body)


def wait(f, vid, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = f.status(vid)
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.02)
    raise AssertionError(f"timeout: {f.status(vid)}")


def fetcher(tmp_path, hf, **kw):
    blob, originals, idx = make_zip(tmp_path)
    hf.blob = blob
    done = []
    f = VideoFetcher(idx, tmp_path / "dest", URL, transport=httpx.MockTransport(hf), retry_sleep=0,
                     on_done=lambda v, p: done.append((v, p)), **kw)
    return f, originals, done


def test_fetches_a_deflated_video_through_a_redirect(tmp_path):
    hf = FakeHF(b"")
    f, orig, done = fetcher(tmp_path, hf)
    assert f.can_fetch("L26_V001") and not f.can_fetch("L26_V999")
    st = f.start("L26_V001")
    assert st["state"] in ("queued", "running") and st["total"] > 0
    st = wait(f, "L26_V001")
    assert st["state"] == "done", st
    out = tmp_path / "dest" / "L26_V001.mp4"
    assert out.read_bytes() == orig["videos/L26_V001.mp4"]
    assert not (tmp_path / "dest" / "L26_V001.mp4.part").exists()
    assert done and done[0][0] == "L26_V001" and done[0][1] == out
    assert f.status("L26_V001")["state"] == "done"
    assert all(rng and rng.startswith("bytes=") for _, rng in hf.calls)
    assert any(u.startswith(CDN) for u, _ in hf.calls)                    # the redirect was followed


def test_stored_entry_and_second_start_is_a_no_op(tmp_path):
    hf = FakeHF(b"")
    f, orig, _ = fetcher(tmp_path, hf)
    f.start("L26_V002")
    f.start("L26_V002")                                                    # already running: must not start a second thread
    assert wait(f, "L26_V002")["state"] == "done"
    assert (tmp_path / "dest" / "L26_V002.mp4").read_bytes() == orig["videos/L26_V002.mp4"]
    n = len(hf.calls)
    assert f.start("L26_V002")["state"] == "done" and len(hf.calls) == n   # local file exists -> no request


def test_dropped_connection_resumes_where_it_stopped(tmp_path):
    hf = FakeHF(b"", cut_first_stream_at=400_000)
    f, orig, _ = fetcher(tmp_path, hf)
    f.start("L21_V009")
    st = wait(f, "L21_V009")
    assert st["state"] == "done", st
    assert (tmp_path / "dest" / "L21_V009.mp4").read_bytes() == orig["videos/L21_V009.mp4"]
    assert f.status("L21_V009")["attempt"] >= 2
    data_reqs = [r for _, r in hf.calls if r and int(r.split("=")[1].split("-")[0]) > 0]
    starts = sorted({int(r.split("=")[1].split("-")[0]) for r in data_reqs})
    assert len(starts) >= 3                                                # header read, first data request, the resumed one


def test_wrong_crc_is_rejected_and_leaves_nothing_behind(tmp_path):
    hf = FakeHF(b"")
    blob, orig, idx = make_zip(tmp_path)
    hf.blob = blob
    entries = json.loads(idx.read_text())
    for e in entries:
        e["crc"] ^= 1
    idx.write_text(json.dumps(entries))
    f = VideoFetcher(idx, tmp_path / "dest", URL, transport=httpx.MockTransport(hf), retry_sleep=0)
    f.start("L26_V001")
    st = wait(f, "L26_V001")
    assert st["state"] == "error" and "CRC32" in st["error"]
    assert not (tmp_path / "dest" / "L26_V001.mp4").exists() and not (tmp_path / "dest" / "L26_V001.mp4.part").exists()


def test_permanent_http_errors_do_not_retry_25_times(tmp_path):
    hf = FakeHF(b"", status=404)
    f, _, _ = fetcher(tmp_path, hf)
    f.start("L26_V001")
    st = wait(f, "L26_V001")
    assert st["state"] == "error" and "404" in st["error"]
    assert len(hf.calls) <= 4
    assert f.status("L26_V001")["state"] == "error"
    hf.status = None                                                       # fixed on the server side: a new click works
    f.start("L26_V001")
    assert wait(f, "L26_V001")["state"] == "done"


def test_server_that_ignores_range_is_reported(tmp_path):
    class NoRange(FakeHF):
        def __call__(self, request):
            if request.url.host == "huggingface.co":
                return httpx.Response(302, headers={"location": CDN})
            return httpx.Response(200, content=self.blob)
    hf = NoRange(b"")
    f, _, _ = fetcher(tmp_path, hf)
    f.start("L26_V001")
    st = wait(f, "L26_V001")
    assert st["state"] == "error" and "200" in st["error"]


def test_unknown_video_and_missing_index(tmp_path):
    hf = FakeHF(b"")
    f, _, _ = fetcher(tmp_path, hf)
    with pytest.raises(FetchError):
        f.start("L99_V999")
    assert f.status("L99_V999") == {"state": "idle", "done": 0, "total": 0, "fetchable": False}
    g = VideoFetcher(tmp_path / "missing.json", tmp_path / "d")
    assert not g.can_fetch("L26_V001")


# ------------------------------------------------------------------ API endpoints
def make_api(tmp_path, with_fetcher=True):
    import httpx as _httpx
    from fastapi.testclient import TestClient
    from PIL import Image
    from api.app import Engine, create_app
    from retrieval.thumbs import ThumbService
    from tests.test_lean_engine import FakeText, FakeTranslator, basis, trake_index

    idx = trake_index(tmp_path / "idx")
    text = FakeText()
    text.vectors = {"e0": basis(0)}
    kf = tmp_path / "kf"; kf.mkdir()
    Image.new("RGB", (64, 36), "purple").save(kf / "GOOD-V1_frame_000100.jpg")
    vids = tmp_path / "videos"; vids.mkdir()
    (vids / "GOOD-V1.mov").write_bytes(bytes(range(256)) * 40)
    dest = tmp_path / "fetched"
    thumbs = ThumbService([kf], vids, width=200, index=idx, extra_video_dirs=[dest])
    eng = Engine(index=idx, text=text, translator=FakeTranslator(), ready=True, thumbs=thumbs)
    hf = FakeHF(b"")
    blob, originals, zidx = make_zip(tmp_path)
    hf.blob = blob
    if with_fetcher:
        eng.fetcher = VideoFetcher(zidx, dest, URL, transport=_httpx.MockTransport(hf), retry_sleep=0,
                                   has_local=lambda v: thumbs.video_path(v) is not None,
                                   on_done=lambda v, p: thumbs.register_video(v, p))
    return TestClient(create_app(engine=eng, autoload=False)), originals, hf


def test_videoinfo_tells_local_indexed_and_fetchable_videos_apart(tmp_path):
    c, _, _ = make_api(tmp_path)
    with c:
        j = c.get("/api/videoinfo/GOOD-V1").json()
        assert j["in_index"] and j["local"] and not j["fetchable"] and j["fps"] > 0 and j["frames"] > 0 and "last_ms" in j
        j = c.get("/api/videoinfo/L26_V001").json()
        assert not j["in_index"] and not j["local"] and j["fetchable"] and j["size_mb"] > 2 and j["fetch"]["state"] == "idle"
        j = c.get("/api/videoinfo/NOPE-V9")
        assert j.status_code == 200 and j.json()["in_index"] is False and j.json()["local"] is False and j.json()["fetchable"] is False


def test_fetch_endpoint_downloads_then_the_video_is_served(tmp_path):
    c, orig, _ = make_api(tmp_path)
    with c:
        assert c.get("/api/video/L26_V001").status_code == 404
        r = c.post("/api/fetch_video/L26_V001", json={})
        assert r.status_code == 200 and r.json()["state"] in ("queued", "running", "done")
        t0 = time.time()
        while c.get("/api/fetch_video/L26_V001").json()["state"] not in ("done", "error") and time.time() - t0 < 20:
            time.sleep(0.05)
        assert c.get("/api/fetch_video/L26_V001").json()["state"] == "done"
        v = c.get("/api/video/L26_V001", headers={"Range": "bytes=0-9"})
        assert v.status_code == 206 and v.content == orig["videos/L26_V001.mp4"][:10]
        j = c.get("/api/videoinfo/L26_V001").json()
        assert j["local"] is True and j["fetch"]["state"] == "done"


def test_fetch_endpoint_guards(tmp_path):
    c, _, _ = make_api(tmp_path)
    with c:
        assert c.post("/api/fetch_video/L99_V999", json={}).status_code == 404            # not in the zip
        assert c.post("/api/fetch_video/L26_V001", content=b"{}", headers={"content-type": "text/plain"}).status_code == 415
        assert c.post("/api/fetch_video/L26_V001", json={}, headers={"origin": "https://evil.example"}).status_code == 403
    c2, _, _ = make_api(tmp_path / "b", with_fetcher=False)
    with c2:
        assert c2.post("/api/fetch_video/L26_V001", json={}).status_code == 404
        assert c2.get("/api/fetch_video/L26_V001").status_code == 404
        j = c2.get("/api/videoinfo/L26_V001").json()
        assert j["fetchable"] is False
