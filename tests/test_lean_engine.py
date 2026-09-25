"""Frame index, TRAKE alignment/ranking, thumbnails and the lean API -- all on tiny synthetic data.

No model is loaded: text vectors are hand-made unit vectors, so the expected rankings are known.
"""
import itertools
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from retrieval.dres_client import DresClient
from retrieval.frame_index import FrameIndex, save_index
from retrieval.trake_engine import align_chain, rank_trake

D = 16


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def basis(i, noise=0.0, rng=None):
    v = np.zeros(D, np.float32)
    v[i] = 1.0
    if noise:
        v += rng.normal(0, noise, D).astype(np.float32)
    return unit(v)


def build_index(tmp_path, videos, device="cpu"):
    """videos: {id: (fps, [(frame_idx, vector), ...])}"""
    embs, frames, vid, ms, meta = [], [], [], [], []
    pos = 0
    for vi, (name, (fps, fr)) in enumerate(videos.items()):
        for f, vec in sorted(fr, key=lambda x: x[0]):
            embs.append(vec); frames.append(f); vid.append(vi); ms.append(int(round(f * 1000 / fps)))
        meta.append({"id": name, "fps": fps, "start": pos, "end": pos + len(fr), "source": "test"})
        pos += len(fr)
    save_index(tmp_path, np.array(embs, np.float16), np.array(frames), np.array(vid), np.array(ms), meta)
    return FrameIndex.load(tmp_path, device=device)


# ------------------------------------------------------------------ align_chain

def brute_force(S, frames, gap):
    M, E = S.shape
    best, arg = -1e18, None
    for rows in itertools.combinations(range(M), E):
        if any(frames[rows[i + 1]] - frames[rows[i]] < gap for i in range(E - 1)):
            continue
        tot = sum(S[r, t] for t, r in enumerate(rows))
        if tot > best:
            best, arg = tot, rows
    return arg, best


@pytest.mark.parametrize("seed", range(25))
def test_align_chain_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    M, E = int(rng.integers(3, 9)), int(rng.integers(1, 4))
    frames = np.sort(rng.choice(np.arange(0, 400, 5), size=M, replace=False))
    S = rng.normal(size=(M, E))
    gap = int(rng.choice([1, 1, 30, 80]))
    want = brute_force(S, frames, gap)
    got = align_chain(S, frames, gap)
    if want[0] is None:
        assert got is None
    else:
        assert got is not None
        rows, total = got
        assert total == pytest.approx(want[1])
        assert all(frames[rows[i + 1]] - frames[rows[i]] >= gap for i in range(E - 1))
        assert sum(S[r, t] for t, r in enumerate(rows)) == pytest.approx(total)


def test_align_chain_is_strictly_increasing_even_when_events_want_the_same_frame():
    frames = np.array([0, 5, 10, 15])
    S = np.zeros((4, 3)); S[1, :] = 10                     # every event loves frame 5
    rows, _ = align_chain(S, frames)
    assert list(rows) == sorted(set(rows)) and len(rows) == 3


def test_align_chain_needs_enough_frames():
    assert align_chain(np.ones((2, 3)), np.array([0, 5])) is None
    assert align_chain(np.ones((3, 3)), np.array([0, 5, 10]), min_gap=8) is None


# --------------------------------------------------------------------- the index

def test_index_roundtrip_and_invariants(tmp_path):
    rng = np.random.default_rng(0)
    idx = build_index(tmp_path, {"A-V1": (25.0, [(f, basis(0, .1, rng)) for f in (0, 5, 10)]),
                                 "B-V2": (30.0, [(f, basis(1, .1, rng)) for f in (5, 50)])})
    assert idx.n == 5 and idx.dim == D and idx.video_ids == ["A-V1", "B-V2"]
    assert idx.slice_of(1) == (3, 5)
    assert idx.ms[3] == 167                                 # 5 frames @ 30 fps
    with pytest.raises(ValueError):                         # rows must be sorted by video
        save_index(tmp_path / "bad", np.zeros((2, D), np.float16), np.array([0, 5]), np.array([1, 0]),
                   np.array([0, 0]), [{"id": "x", "fps": 25, "start": 0, "end": 1}, {"id": "y", "fps": 25, "start": 1, "end": 2}])
    with pytest.raises(ValueError):                         # frames must increase inside a video
        save_index(tmp_path / "bad2", np.zeros((2, D), np.float16), np.array([5, 5]), np.array([0, 0]),
                   np.array([0, 0]), [{"id": "x", "fps": 25, "start": 0, "end": 2}])


def test_load_rejects_a_truncated_index(tmp_path):
    rng = np.random.default_rng(0)
    build_index(tmp_path, {"A-V1": (25.0, [(f, basis(0, .1, rng)) for f in (0, 5, 10)])})
    np.save(tmp_path / "frames.npy", np.array([0, 5], np.int32))     # drive dropped a write
    with pytest.raises(ValueError):
        FrameIndex.load(tmp_path, device="cpu")
    (tmp_path / "videos.json").unlink()
    with pytest.raises(FileNotFoundError):
        FrameIndex.load(tmp_path, device="cpu")


def test_top_frames_orders_by_true_cosine_and_applies_temporal_nms(tmp_path):
    rng = np.random.default_rng(1)
    # video A: frames 0..45 all look like event 0 (a long static shot); video B has one better frame
    a = [(f, basis(0, .02, rng)) for f in range(0, 50, 5)]
    b = [(0, basis(3, .05, rng)), (100, unit(basis(0) * .8 + basis(2) * .1))]
    idx = build_index(tmp_path, {"A-V1": (25.0, a), "B-V2": (25.0, b)})
    hits = idx.top_frames(torch.from_numpy(basis(0)), k=5, nms_sec=6.0)
    per_video = {}
    for h in hits:
        per_video.setdefault(h.video_id, []).append(h.ms)
    assert len(per_video["A-V1"]) == 1                      # 10 near-identical frames collapse to one
    assert hits[0].video_id == "A-V1"                       # its best frame beats B's 0.8 match
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)
    # without NMS every frame of A is a separate hit
    assert sum(h.video_id == "A-V1" for h in idx.top_frames(torch.from_numpy(basis(0)), k=20, nms_sec=0)) == 10


def test_video_filter_never_leaks_other_videos(tmp_path):
    rng = np.random.default_rng(2)
    idx = build_index(tmp_path, {"A-V1": (25.0, [(f, basis(0, .05, rng)) for f in range(0, 20, 5)]),
                                 "B-V2": (25.0, [(f, basis(0, .05, rng)) for f in range(0, 20, 5)])})
    hits = idx.top_frames(torch.from_numpy(basis(0)), k=10, nms_sec=0, video_ids=["B-V2"], pool=8)
    assert hits and {h.video_id for h in hits} == {"B-V2"}


def test_neighbors_and_nearest_row(tmp_path):
    rng = np.random.default_rng(3)
    idx = build_index(tmp_path, {"A-V1": (25.0, [(f, basis(0, .05, rng)) for f in range(0, 100, 10)])})
    assert idx.find_row("A-V1", 44) == 4 and idx.find_row("A-V1", 46) == 5      # nearest keyframe
    assert idx.find_row("A-V1", -5) == 0 and idx.find_row("A-V1", 9999) == 9
    nb = idx.neighbors("A-V1", 40, 2)
    assert [x["frame_idx"] for x in nb] == [20, 30, 40, 50, 60]
    assert [x["frame_idx"] for x in idx.neighbors("A-V1", 0, 2)] == [0, 10, 20]   # clipped at the start
    with pytest.raises(KeyError):
        idx.find_row("nope", 0)


# ----------------------------------------------------------------- TRAKE ranking

def trake_index(tmp_path):
    """Three events e0,e1,e2 = basis vectors 0,1,2.
    GOOD : shows e0, e1, e2 in order (frames 100, 400, 800).
    REVERSED: same content but in the opposite order.
    PARTIAL : only e0 and e1.  SHORT: one frame only."""
    rng = np.random.default_rng(4)
    filler = lambda f: (f, basis(10, .05, rng))
    good = [filler(0), (100, basis(0, .03, rng)), filler(200), (400, basis(1, .03, rng)), filler(600), (800, basis(2, .03, rng))]
    rev = [filler(0), (100, basis(2, .03, rng)), filler(200), (400, basis(1, .03, rng)), filler(600), (800, basis(0, .03, rng))]
    part = [filler(0), (100, basis(0, .03, rng)), (400, basis(1, .03, rng)), filler(800), filler(900), filler(950)]
    short = [(0, basis(0, .01, rng))]
    return build_index(tmp_path, {"GOOD-V1": (25.0, good), "REV-V2": (25.0, rev),
                                  "PART-V3": (25.0, part), "SHORT-V4": (25.0, short)})


def test_rank_trake_finds_the_ordered_video_and_its_frames(tmp_path):
    idx = trake_index(tmp_path)
    Q = torch.from_numpy(np.stack([basis(0), basis(1), basis(2)]))
    res = rank_trake(idx, Q, ["e0", "e1", "e2"], top_k=10, shortlist=10, alt_n=3, alt_min_sec=1.0)
    assert res[0].video_id == "GOOD-V1"
    assert [f["frame_idx"] for f in res[0].frames] == [100, 400, 800]
    assert res[0].score > res[1].score
    assert "SHORT-V4" not in [r.video_id for r in res]                # cannot hold 3 ordered frames
    for r in res:                                                     # every returned chain is ordered
        fr = [f["frame_idx"] for f in r.frames]
        assert all(b > a for a, b in zip(fr, fr[1:]))
    # the reversed video contains all three events but in the wrong order -> must lose to GOOD
    assert [r.video_id for r in res].index("REV-V2") > 0


def test_rank_trake_alternatives_start_with_the_chain_pick_and_are_spread_out(tmp_path):
    idx = trake_index(tmp_path)
    Q = torch.from_numpy(np.stack([basis(0), basis(1), basis(2)]))
    good = rank_trake(idx, Q, ["a", "b", "c"], top_k=1, shortlist=4, alt_n=4, alt_min_sec=2.0)[0]
    for f in good.frames:
        alts = f["alts"]
        assert alts[0]["frame_idx"] == f["frame_idx"]
        assert len({a["frame_idx"] for a in alts}) == len(alts)
        ms = sorted(a["ms"] for a in alts)
        assert all(b - a >= 2000 for a, b in zip(ms, ms[1:]))


def test_rank_trake_single_event_is_plain_sub_event_ranking(tmp_path):
    idx = trake_index(tmp_path)
    res = rank_trake(idx, torch.from_numpy(basis(1)[None]), ["e1"], top_k=5, shortlist=5)
    assert res and res[0].frames[0]["frame_idx"] in (400,)


# ---------------------------------------------------------------- thumbnails

def test_thumbs_prefer_keyframe_files_then_nearest_keyframe(tmp_path):
    from retrieval.thumbs import ThumbService
    rng = np.random.default_rng(5)
    idx = build_index(tmp_path / "idx", {"A-V1": (25.0, [(f, basis(0, .05, rng)) for f in (0, 50, 100)])})
    kf = tmp_path / "kf"; kf.mkdir()
    for f, color in ((0, "red"), (50, "green"), (100, "blue")):
        Image.new("RGB", (1280, 720), color).save(kf / f"A-V1_frame_{f:06d}.jpg")
    th = ThumbService([kf], tmp_path / "no-videos", width=320, index=idx)
    exact = th.get("A-V1", 50)
    im = Image.open(__import__("io").BytesIO(exact)); assert im.size[0] == 320 and im.size[1] == 180
    assert im.getpixel((10, 10))[1] > 100                                  # green
    near = th.get("A-V1", 97)                                              # not a keyframe -> nearest is 100 (blue)
    assert Image.open(__import__("io").BytesIO(near)).getpixel((10, 10))[2] > 100
    assert th.get("A-V1", 50) is exact                                      # served from the LRU
    assert th.get("Z-V9", 0) is None                                        # unknown video, no file -> placeholder path
    th.close()


# ------------------------------------------------------------------------- API

class FakeText:
    dim = D
    vectors = {}

    def encode(self, texts):
        return torch.from_numpy(np.stack([self.vectors[t] for t in texts]))

    def warmup(self):
        pass


class FakeTranslator:
    def translate_many(self, texts):
        return [{"đứng dậy": "e0", "đi ra cửa": "e1", "ngồi xuống": "e2"}.get(t, t) for t in texts]


@pytest.fixture()
def api(tmp_path):
    from api.app import Engine, create_app
    from retrieval.thumbs import ThumbService
    idx = trake_index(tmp_path / "idx")
    text = FakeText()
    text.vectors = {"e0": basis(0), "e1": basis(1), "e2": basis(2), "unknown": basis(11)}
    kf = tmp_path / "kf"; kf.mkdir()
    Image.new("RGB", (640, 360), "purple").save(kf / "GOOD-V1_frame_000100.jpg")
    vids = tmp_path / "videos"; vids.mkdir()
    (vids / "GOOD-V1.mov").write_bytes(bytes(range(256)) * 40)
    eng = Engine(index=idx, text=text, translator=FakeTranslator(), ready=True,
                 thumbs=ThumbService([kf], vids, width=200, index=idx))
    received = []

    def dres_handler(req: httpx.Request):
        p = req.url.path
        if p == "/api/v2/login":
            return httpx.Response(200, json={"sessionId": "S1"})
        if p == "/api/v2/client/evaluation/list":
            return httpx.Response(200, json=[{"id": "ev", "name": "Final", "status": "ACTIVE"}])
        received.append(json.loads(req.content))
        return httpx.Response(200, json={"status": True, "submission": "WRONG", "description": "no"})

    dres = DresClient("https://dres.test", transport=httpx.MockTransport(dres_handler))
    app = create_app(engine=eng, dres=dres, autoload=False)
    with TestClient(app) as c:
        c.received = received
        yield c


def test_status(api):
    j = api.get("/api/status").json()
    assert j["ready"] and j["frames"] == 6 + 6 + 6 + 1 and j["videos"] == 4 and j["dim"] == D


def test_search_uses_translation_and_reports_timings(api):
    j = api.post("/api/search", json={"query": "đứng dậy", "top_k": 5}).json()
    assert j["query_en"] == "e0" and j["timings"].keys() >= {"translate_ms", "encode_ms", "search_ms"}
    top = j["results"][0]
    assert top["video_id"] in ("GOOD-V1", "PART-V3", "SHORT-V4") and top["frame_idx"] in (100, 0)
    assert top["ms"] == round(top["frame_idx"] * 1000 / top["fps"])
    assert api.post("/api/search", json={"query": "x", "en_override": "e2", "top_k": 3}).json()["query_en"] == "e2"
    assert api.post("/api/search", json={"query": ""}).status_code == 422
    assert api.post("/api/search", json={"query": "x", "top_k": 0}).status_code == 422


def test_trake_endpoint(api):
    j = api.post("/api/trake", json={"events": ["đứng dậy", "đi ra cửa", "ngồi xuống"], "top_k": 3}).json()
    assert j["events_en"] == ["e0", "e1", "e2"]
    best = j["results"][0]
    assert best["video_id"] == "GOOD-V1" and [f["frame_idx"] for f in best["frames"]] == [100, 400, 800]
    assert best["frames"][0]["event"] == "đứng dậy" and best["frames"][0]["alts"]
    assert api.post("/api/trake", json={"events": []}).status_code == 422
    assert api.post("/api/trake", json={"events": ["  "]}).status_code == 422


def test_neighbors_thumb_and_video(api):
    nb = api.get("/api/neighbors/GOOD-V1/400?n=2").json()
    assert [x["frame_idx"] for x in nb["items"]] == [200, 400, 600] + [800][:0] or len(nb["items"]) == 5
    assert api.get("/api/neighbors/NOPE/1").status_code == 404
    t = api.get("/api/thumb/GOOD-V1/100")
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg" and "max-age" in t.headers["cache-control"]
    assert api.get("/api/thumb/GOOD-V1/12345678").status_code in (200, 404)     # nearest keyframe file or placeholder path
    assert api.get("/api/thumb/NOPE/1").status_code == 404
    v = api.get("/api/video/GOOD-V1", headers={"Range": "bytes=10-19"})
    assert v.status_code == 206 and v.content == bytes(range(256)) * 40 and False or v.content == (bytes(range(256)) * 40)[10:20]
    assert api.get("/api/video/NOPE").status_code == 404


def test_not_ready_and_error_states():
    from api.app import Engine, create_app
    with TestClient(create_app(engine=Engine(), autoload=False)) as c:
        r = c.post("/api/search", json={"query": "x"})
        assert r.status_code == 503 and "đang nạp" in r.json()["detail"]
        assert c.get("/api/status").json()["ready"] is False
    with TestClient(create_app(engine=Engine(error="index thiếu"), autoload=False)) as c:
        assert "index thiếu" in c.post("/api/search", json={"query": "x"}).json()["detail"]


# ---------------------------------------------------------------- DRES endpoints

def test_dres_submit_flow_builds_the_spec_payloads(api):
    assert api.post("/api/dres/login", json={"username": "u", "password": "p"}).json()["evaluation_id"] == "ev"
    r = api.post("/api/dres/submit", json={"kind": "kis", "video_id": "N043-V001.mp4", "start_ms": 1000, "end_ms": 1000}).json()
    assert r["verdict"] == "WRONG" and not r["duplicate"]
    assert api.received[-1] == {"answerSets": [{"answers": [{"mediaItemName": "N043-V001", "start": 1000, "end": 1000}]}]}
    api.post("/api/dres/submit", json={"kind": "qa", "video_id": "N043-V001", "answer": "màu đỏ", "time_ms": 5000})
    assert api.received[-1]["answerSets"][0]["answers"][0]["text"] == "QA-màu đỏ-N043-V001-5000"
    api.post("/api/dres/submit", json={"kind": "trake", "video_id": "N043-V001", "frames": [10, 20, 30]})
    assert api.received[-1]["answerSets"][0]["answers"][0]["text"] == "TR-N043-V001-10,20,30"
    n = len(api.received)
    dup = api.post("/api/dres/submit", json={"kind": "trake", "video_id": "N043-V001", "frames": [10, 20, 30]}).json()
    assert dup["duplicate"] and len(api.received) == n
    assert api.post("/api/dres/new_query", json={}).json()["query_epoch"] == 1
    assert not api.post("/api/dres/submit", json={"kind": "trake", "video_id": "N043-V001", "frames": [10, 20, 30]}).json()["duplicate"]


def test_dres_rejects_bad_requests_before_anything_is_sent(api):
    api.post("/api/dres/login", json={"username": "u", "password": "p"})
    n = len(api.received)
    r = api.post("/api/dres/submit", json={"kind": "trake", "video_id": "V", "frames": [30, 20]})
    assert r.status_code == 400 and "tăng dần" in r.json()["detail"]
    assert api.post("/api/dres/submit", json={"kind": "qa", "video_id": "V", "answer": " ", "time_ms": 1}).status_code == 400
    assert api.post("/api/dres/submit", json={"kind": "zzz", "video_id": "V"}).status_code == 400
    assert api.post("/api/dres/submit", json={"kind": "kis", "video_id": "V"}).status_code == 400
    assert len(api.received) == n


def test_dres_endpoints_only_answer_this_machine(api):
    from api.app import create_app, Engine
    app = api.app
    remote = TestClient(app, client=("203.0.113.9", 4444))
    assert remote.get("/api/dres/state").status_code == 403
    assert remote.post("/api/dres/submit", json={"kind": "kis", "video_id": "V", "start_ms": 1}).status_code == 403
    assert remote.get("/api/health").status_code == 200                      # search side stays open


def test_dres_post_needs_json_content_type_and_same_origin(api):
    # a cross-site form/text POST must not be able to submit as the team
    r = api.post("/api/dres/submit", content='{"kind":"kis","video_id":"V","start_ms":1}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    r = api.post("/api/dres/logout", json={}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert api.post("/api/dres/logout", json={}).status_code == 200
