"""DRES client: payload formats from HD-ChungKet-2026.pdf, duplicate guard, error paths.

No real network: every call goes through httpx.MockTransport.
"""
import asyncio
import json

import httpx
import pytest

from retrieval.dres_client import (
    DresClient, DresError, build_kis, build_qa, build_trake, normalize_item, payload_summary,
)


# ----------------------------- payload formats ----------------------------- #

def test_kis_payload_matches_spec():
    p = build_kis("N043-V001.mp4", 12345, 12345)
    assert p == {"answerSets": [{"answers": [{"mediaItemName": "N043-V001", "start": 12345, "end": 12345}]}]}


def test_kis_swaps_reversed_range_and_clamps_negative():
    a = build_kis("S01-V003", -50, 900)["answerSets"][0]["answers"][0]
    assert (a["start"], a["end"]) == (0, 900)
    b = build_kis("S01-V003", 9000, 1000)["answerSets"][0]["answers"][0]
    assert (b["start"], b["end"]) == (1000, 9000)


def test_kis_single_time_means_point():
    a = build_kis("L21_V002", 7000)["answerSets"][0]["answers"][0]
    assert a["start"] == a["end"] == 7000


def test_qa_payload_matches_spec():
    p = build_qa("  màu  đỏ ", "N043-V001", 61000.4)
    assert p == {"answerSets": [{"answers": [{"text": "QA-màu đỏ-N043-V001-61000"}]}]}


def test_qa_requires_answer():
    with pytest.raises(DresError):
        build_qa("   ", "N043-V001", 1000)


def test_trake_payload_matches_spec():
    p = build_trake("N043-V001", [120, 480, 1500])
    assert p == {"answerSets": [{"answers": [{"text": "TR-N043-V001-120,480,1500"}]}]}


def test_trake_rejects_out_of_order_frames():
    # an out-of-order row scores zero, so it must never leave the machine
    with pytest.raises(DresError):
        build_trake("N043-V001", [480, 120])
    with pytest.raises(DresError):
        build_trake("N043-V001", [120, 120])
    with pytest.raises(DresError):
        build_trake("N043-V001", [])


def test_item_name_is_file_name_without_extension():
    assert normalize_item(r"D:\videos\N043-V001.mp4") == "N043-V001"
    assert normalize_item("S01-V011.MKV") == "S01-V011"
    assert normalize_item("L21_V002") == "L21_V002"
    with pytest.raises(DresError):
        normalize_item("  ")


def test_summary_is_readable():
    assert payload_summary(build_trake("A-V1", [1, 2])) == "TR-A-V1-1,2"
    assert payload_summary(build_kis("A-V1", 1000, 2000)) == "A-V1 [1000..2000] ms"


# ------------------------------- fake DRES -------------------------------- #

class FakeDres:
    """Minimal DRES v2 stand-in that records every submission it receives."""

    def __init__(self, active=True, correct=lambda payload: False):
        self.active = active
        self.correct = correct
        self.received = []
        self.logins = 0
        self.session = "sess-123"

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/v2/login":
            body = json.loads(req.content)
            self.logins += 1
            if body == {"username": "team_x", "password": "pw"}:
                return httpx.Response(200, json={"id": "u1", "username": "team_x", "role": "PARTICIPANT",
                                                 "sessionId": self.session})
            return httpx.Response(401, json={"status": False, "description": "Invalid credentials"})
        if req.url.params.get("session") != self.session:
            return httpx.Response(401, json={"status": False, "description": "Unauthorized"})
        if path == "/api/v2/client/evaluation/list":
            return httpx.Response(200, json=[
                {"id": "old", "name": "Warmup", "type": "SYNCHRONOUS", "status": "TERMINATED"},
                {"id": "ev1", "name": "Final", "type": "SYNCHRONOUS", "status": "ACTIVE" if self.active else "CREATED"},
            ])
        if path.startswith("/api/v2/submit/"):
            if not self.active:
                return httpx.Response(412, json={"status": False, "description": "There is currently no active task"})
            payload = json.loads(req.content)
            self.received.append((path, payload))
            ok = self.correct(payload)
            return httpx.Response(200, json={"status": True, "submission": "CORRECT" if ok else "WRONG",
                                             "description": "Submission correct" if ok else "Submission incorrect"})
        return httpx.Response(404)


def make(fake: FakeDres, tmp_path=None) -> DresClient:
    return DresClient("https://dres.test", log_path=(tmp_path / "log.jsonl") if tmp_path else None,
                      transport=httpx.MockTransport(fake.handler))


def run(coro):
    return asyncio.run(coro)


# ------------------------------ session flow ------------------------------ #

def test_login_picks_the_active_evaluation():
    async def go():
        fake = FakeDres(); c = make(fake)
        st = await c.login("team_x", "pw")
        assert st["logged_in"] and st["username"] == "team_x"
        assert st["evaluation_id"] == "ev1"          # the ACTIVE one, not the first listed
        await c.close()
    run(go())


def test_bad_login_raises_and_stays_logged_out():
    async def go():
        c = make(FakeDres())
        with pytest.raises(DresError) as e:
            await c.login("team_x", "wrong")
        assert e.value.status == 401 and not c.state()["logged_in"]
        await c.close()
    run(go())


def test_use_session_id_directly():
    async def go():
        fake = FakeDres(); c = make(fake)
        st = await c.use_session("sess-123")
        assert st["logged_in"] and st["evaluation_id"] == "ev1" and fake.logins == 0
        await c.close()
        c2 = make(fake)
        with pytest.raises(DresError):
            await c2.use_session("bogus")
        assert not c2.state()["logged_in"]
        await c2.close()
    run(go())


def test_password_and_session_never_reach_the_audit_log(tmp_path):
    async def go():
        fake = FakeDres(); c = make(fake, tmp_path)
        await c.login("team_x", "pw")
        await c.submit(build_kis("N043-V001", 1000))
        await c.close()
    run(go())
    text = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert "pw" not in text.replace("payload", "") and "sess-123" not in text
    assert '"mediaItemName": "N043-V001"' in text


# --------------------------------- submit --------------------------------- #

def test_submit_sends_exactly_one_answer_to_the_active_evaluation():
    async def go():
        fake = FakeDres(correct=lambda p: True); c = make(fake)
        await c.login("team_x", "pw")
        res = await c.submit(build_kis("N043-V001", 5000))
        assert res["ok"] and res["verdict"] == "CORRECT" and not res["duplicate"]
        assert len(fake.received) == 1
        path, payload = fake.received[0]
        assert path == "/api/v2/submit/ev1"
        assert len(payload["answerSets"]) == 1 and len(payload["answerSets"][0]["answers"]) == 1
        await c.close()
    run(go())


def test_identical_answer_is_not_sent_twice():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")
        first = await c.submit(build_trake("N043-V001", [10, 20]))
        second = await c.submit(build_trake("N043-V001", [10, 20]))
        assert first["verdict"] == "WRONG" and not first["duplicate"]
        assert second["duplicate"] and second["verdict"] == "WRONG"
        assert len(fake.received) == 1                       # the guard stopped it locally
        await c.submit(build_trake("N043-V001", [10, 25]))    # a different answer is fine
        assert len(fake.received) == 2
        await c.close()
    run(go())


def test_double_click_race_sends_once():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")
        p = build_kis("N043-V001", 1000)
        results = await asyncio.gather(*[c.submit(p) for _ in range(5)])
        assert sum(1 for r in results if not r["duplicate"]) == 1
        assert len(fake.received) == 1
        await c.close()
    run(go())


def test_new_query_allows_the_same_answer_again():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")
        p = build_kis("N043-V001", 1000)
        await c.submit(p)
        assert (await c.submit(p))["duplicate"]
        c.new_query()
        assert not (await c.submit(p))["duplicate"]
        assert len(fake.received) == 2
        await c.close()
    run(go())


def test_no_active_task_is_reported_and_retryable():
    async def go():
        fake = FakeDres(active=False); c = make(fake)
        await c.login("team_x", "pw")
        with pytest.raises(DresError) as e:
            await c.submit(build_kis("N043-V001", 1000))       # no ACTIVE evaluation at all
        assert e.value.status == 412
        await c.close()
    run(go())


def test_task_not_running_yet_does_not_burn_the_answer():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")
        fake.active = True
        c.evaluation_id = "ev1"
        # DRES answers 412 when the evaluation is ACTIVE but no task is running
        orig = fake.handler
        state = {"n": 0}

        def flaky(req):
            if req.url.path.startswith("/api/v2/submit/") and state["n"] == 0:
                state["n"] += 1
                return httpx.Response(412, json={"status": False, "description": "There is currently no active task"})
            return orig(req)

        c._transport = httpx.MockTransport(flaky); c._http = None
        p = build_kis("N043-V001", 1000)
        r1 = await c.submit(p)
        assert r1["verdict"] == "ERROR" and r1["http_status"] == 412 and "no active task" in r1["description"]
        r2 = await c.submit(p)                                  # retry is allowed: it was never judged
        assert not r2["duplicate"] and r2["verdict"] == "WRONG"
        await c.close()
    run(go())


def test_lost_connection_blocks_blind_resubmit_but_allows_forced():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")

        def boom(req):
            if req.url.path.startswith("/api/v2/submit/"):
                raise httpx.ConnectTimeout("timeout", request=req)
            return fake.handler(req)

        c._transport = httpx.MockTransport(boom); c._http = None
        p = build_kis("N043-V001", 1000)
        r1 = await c.submit(p)
        assert r1["verdict"] == "UNKNOWN" and not r1["ok"]
        assert (await c.submit(p))["duplicate"]                 # do not guess: it may have arrived
        c._transport = httpx.MockTransport(fake.handler); c._http = None
        r3 = await c.submit(p, force=True)                      # operator confirmed on the DRES page
        assert not r3["duplicate"] and r3["verdict"] == "WRONG"
        await c.close()
    run(go())


def test_expired_pasted_session_is_cleared():
    # a sessionId pasted from the DRES page has no password behind it, so it cannot be renewed
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.use_session("sess-123")
        fake.session = "rotated"                                # server no longer accepts ours
        r = await c.submit(build_kis("N043-V001", 1000))
        assert r["http_status"] == 401 and not c.state()["logged_in"]
        await c.close()
    run(go())


def test_base_url_must_be_http():
    c = DresClient("https://dres.test")
    with pytest.raises(DresError):
        c.set_base_url("ftp://x")
    with pytest.raises(DresError):
        c.set_base_url("eventretrieval.one")
    c.set_base_url("https://eventretrieval.one/")
    assert c.base_url == "https://eventretrieval.one"


def test_kis_collection_name_is_optional_and_only_sent_when_given():
    assert "mediaItemCollectionName" not in build_kis("N043-V001", 1000)["answerSets"][0]["answers"][0]
    assert "mediaItemCollectionName" not in build_kis("N043-V001", 1000, collection="  ")["answerSets"][0]["answers"][0]
    a = build_kis("N043-V001.mov", 1000, 2000, collection=" AIC2026 ")["answerSets"][0]["answers"][0]
    assert a == {"mediaItemName": "N043-V001", "start": 1000, "end": 2000, "mediaItemCollectionName": "AIC2026"}


# ------------------------------ staying logged in ------------------------------ #

def test_expired_session_is_renewed_with_the_password_and_the_answer_replayed_once():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")
        fake.session = "sess-456"                               # DRES dropped our session
        r = await c.submit(build_kis("N043-V001", 1000))
        assert r["http_status"] == 200 and r["verdict"] == "WRONG" and c.state()["logged_in"]
        assert c.session_id == "sess-456" and fake.logins == 2
        assert len(fake.received) == 1                          # a 401 processes nothing, so exactly one judged copy
        await c.close()
    run(go())


def test_keepalive_logs_in_again_when_the_session_dies():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")
        task = asyncio.create_task(c.keepalive(0.05))
        fake.session = "sess-789"
        await asyncio.sleep(0.4)
        task.cancel()
        assert c.session_id == "sess-789" and c.state()["evaluation_id"] == "ev1"
        await c.close()
    run(go())


def test_keepalive_follows_the_active_evaluation():
    async def go():
        fake = FakeDres(active=False); c = make(fake)
        await c.login("team_x", "pw")
        assert c.state()["evaluation_id"] == ""                 # nothing ACTIVE yet
        task = asyncio.create_task(c.keepalive(0.05))
        fake.active = True                                      # BTC starts the evaluation
        await asyncio.sleep(0.3)
        task.cancel()
        assert c.state()["evaluation_id"] == "ev1"
        await c.close()
    run(go())


def test_session_survives_a_server_restart_and_the_password_is_never_written(tmp_path):
    async def go():
        fake = FakeDres()
        state = tmp_path / "dres_session.json"
        c1 = DresClient("https://dres.test", state_path=state, transport=httpx.MockTransport(fake.handler))
        await c1.login("team_x", "pw")
        c1.collection = "AIC2026"; c1._save()
        await c1.close()
        text = state.read_text(encoding="utf-8")
        assert "sess-123" in text and "pw" not in text.replace("session", "") and "password" not in text
        c2 = DresClient("https://dres.test", state_path=state, transport=httpx.MockTransport(fake.handler))   # "restart"
        assert await c2.restore()
        st = c2.state()
        assert st["logged_in"] and st["username"] == "team_x" and st["evaluation_id"] == "ev1" and st["collection"] == "AIC2026"
        assert fake.logins == 1                                 # resumed, not logged in again
        await c2.close()
    run(go())


def test_a_dead_saved_session_is_dropped_on_restart(tmp_path):
    async def go():
        fake = FakeDres()
        state = tmp_path / "dres_session.json"
        c1 = DresClient("https://dres.test", state_path=state, transport=httpx.MockTransport(fake.handler))
        await c1.login("team_x", "pw"); await c1.close()
        fake.session = "someone-else"
        c2 = DresClient("https://dres.test", state_path=state, transport=httpx.MockTransport(fake.handler))
        assert not await c2.restore()
        assert not c2.state()["logged_in"] and not state.exists()
        await c2.close()
    run(go())


def test_startup_credentials_from_the_environment_enable_login_and_relogin(tmp_path):
    async def go():
        fake = FakeDres()
        c = DresClient("https://dres.test", state_path=tmp_path / "s.json", transport=httpx.MockTransport(fake.handler))
        await c.startup("team_x", "pw")                         # nothing saved -> logs in
        assert c.state()["logged_in"]
        fake.session = "sess-999"
        r = await c.submit(build_kis("N043-V001", 1000))        # session died meanwhile -> renewed transparently
        assert r["http_status"] == 200
        await c.close()
    run(go())


def test_logout_removes_the_saved_session_and_forgets_the_password(tmp_path):
    async def go():
        fake = FakeDres()
        state = tmp_path / "s.json"
        c = DresClient("https://dres.test", state_path=state, transport=httpx.MockTransport(fake.handler))
        await c.login("team_x", "pw")
        assert state.exists()
        c.logout()
        assert not state.exists() and c._creds is None
        await c.close()
    run(go())


def test_a_vanished_evaluation_is_refreshed_so_the_next_try_uses_the_new_one():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")
        c.evaluation_id = "gone-evaluation"                     # BTC recreated the evaluation under a new id

        def handler(req):
            if req.url.path == "/api/v2/submit/gone-evaluation":
                return httpx.Response(404, json={"status": False, "description": "Evaluation with ID 'gone-evaluation' could not be found."})
            return fake.handler(req)
        c._transport = httpx.MockTransport(handler); c._http = None
        r1 = await c.submit(build_kis("N043-V001", 1000))
        assert r1["http_status"] == 404 and "could not be found" in r1["description"]
        assert c.state()["evaluation_id"] == "ev1"              # refreshed by itself
        r2 = await c.submit(build_kis("N043-V001", 1000))       # the 404 was never judged, so the same answer may be retried
        assert r2["http_status"] == 200 and not r2["duplicate"]
        await c.close()
    run(go())


def test_relogin_fuse_stops_a_fight_with_another_login_of_the_same_account():
    async def go():
        fake = FakeDres(); c = make(fake)
        await c.login("team_x", "pw")
        # every time we log in, "the other login" (e.g. the DRES web page) takes the session away again
        for n in range(1, 5):
            fake.session = f"stolen-{n}"
            try:
                await c.refresh_evaluations()
            except DresError:
                pass
        assert fake.logins <= 3                                  # the first login + at most two automatic renewals
        assert c._creds is None                                  # then we give up instead of fighting
        assert not c.state()["logged_in"]
        await c.close()
    run(go())
