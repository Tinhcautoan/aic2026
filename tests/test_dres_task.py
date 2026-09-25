"""DresClient.current_task / task_kind and the /api/dres/task endpoint (what the UI follows to pick the tab and the clock)."""
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from retrieval.dres_client import DresClient, DresError, task_kind


@pytest.mark.parametrize("task,kind", [
    ({"task_type": "KIS", "task_group": "Textual", "name": "Task 03 - Textual KIS"}, "tkis"),
    ({"task_type": "Textual KIS", "name": "Task 03"}, "tkis"),
    ({"task_type": "Video KIS", "name": "Task 01"}, "vkis"),
    ({"task_type": "KIS", "name": "Task 01 - Video KIS"}, "vkis"),
    ({"task_type": "Textual Known Item Search", "task_group": "Textual KIS", "name": "tkis-test"}, "tkis"),      # real test evaluation
    ({"task_type": "Video Known Item Search", "name": "vkis-test"}, "vkis"),
    ({"task_type": "Visual Known Item Search", "task_group": "Video KIS", "name": "vkis-test"}, "vkis"),   # real test evaluation
    ({"task_type": "Visual Known Item Search", "name": "vkis-test"}, "vkis"),
    ({"task_type": "TRAKE", "task_group": "TRAKE", "name": "trake-test"}, "trake"),
    ({"task_type": "Question Answering", "name": "qa-test"}, "qa"),
    ({"task_type": "QA", "name": "Task 07"}, "qa"),
    ({"task_type": "Q&A", "name": "Task 07"}, "qa"),
    ({"task_type": "", "name": "Task 09 - TRAKE"}, "trake"),
    ({"task_type": "Event alignment", "name": "Task 09"}, "trake"),
    ({"task_type": "Standard", "task_group": "General", "name": "Task 05 - Textual KIS"}, "tkis"),   # unhelpful type, the name decides
    ({"task_type": "Standard", "name": "Task 05"}, ""),
    ({}, ""),
])
def test_task_kind(task, kind):
    assert task_kind(task) == kind


def make(handler):
    c = DresClient("https://dres.test", transport=httpx.MockTransport(handler))
    c.session_id, c.evaluation_id = "S1", "EV1"
    return c


def test_current_task_reads_the_portal_endpoint():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, dict(req.url.params)))
        return httpx.Response(200, json={"templateId": "T9", "name": "Task 03 - Textual KIS", "taskGroup": "Round 1",
                                         "taskType": "KIS", "duration": 300})
    t = asyncio.run(make(handler).current_task())
    assert seen == [("GET", "/api/v2/client/evaluation/currentTask/EV1", {"session": "S1"})]
    assert t == {"name": "Task 03 - Textual KIS", "task_group": "Round 1", "task_type": "KIS", "duration": 300,
                 "template_id": "T9", "kind": "tkis"}


def test_duration_in_milliseconds_is_normalised_to_seconds():
    d = asyncio.run(make(lambda r: httpx.Response(200, json={"name": "trake-test", "taskType": "TRAKE", "duration": 300000})).current_task())
    assert d["duration"] == 300 and d["kind"] == "trake" and d["template_id"] == ""
    d = asyncio.run(make(lambda r: httpx.Response(200, json={"name": "t", "taskType": "KIS", "duration": 240})).current_task())
    assert d["duration"] == 240


@pytest.mark.parametrize("status,body", [(404, {"description": "no task"}), (400, {}), (500, {}), (200, {}), (200, []), (200, {"name": ""})])
def test_no_running_task_is_none_not_an_error(status, body):
    assert asyncio.run(make(lambda r: httpx.Response(status, json=body)).current_task()) is None


def test_nothing_is_sent_without_session_or_evaluation():
    calls = []
    c = make(lambda r: calls.append(r) or httpx.Response(200, json={"name": "x"}))
    c.evaluation_id = ""
    assert asyncio.run(c.current_task()) is None and not calls
    c.evaluation_id, c.session_id = "EV1", ""
    assert asyncio.run(c.current_task()) is None and not calls


def test_expired_session_and_network_errors_raise():
    with pytest.raises(DresError) as e:
        asyncio.run(make(lambda r: httpx.Response(401, json={})).current_task())
    assert e.value.status == 401

    def boom(req):
        raise httpx.ConnectError("down")
    with pytest.raises(DresError):
        asyncio.run(make(boom).current_task())


def test_endpoint_wraps_task_and_errors():
    from api.app import Engine, create_app
    state = {"resp": httpx.Response(200, json={"name": "Task 09 - TRAKE", "taskType": "TRAKE", "duration": 300, "templateId": "T1"})}
    c = make(lambda r: state["resp"])
    with TestClient(create_app(engine=Engine(), dres=c, autoload=False)) as api:
        j = api.get("/api/dres/task").json()
        assert j["task"]["kind"] == "trake" and j["task"]["duration"] == 300
        state["resp"] = httpx.Response(404, json={})
        assert api.get("/api/dres/task").json()["task"] is None
        state["resp"] = httpx.Response(401, json={})
        assert api.get("/api/dres/task").status_code == 401
        c.session_id = ""
        assert api.get("/api/dres/task").json()["task"] is None



def test_current_task_of_another_evaluation_and_submit_to_it_leave_the_selection_alone():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        if req.url.path.endswith("/evaluation/list"):
            return httpx.Response(200, json=[{"id": "EV1", "name": "A", "status": "ACTIVE"}, {"id": "EV2", "name": "B", "status": "ACTIVE"}])
        if "currentTask" in req.url.path:
            return httpx.Response(200, json={"name": "t-" + req.url.path.rsplit("/", 1)[1], "taskType": "TRAKE", "duration": 300})
        return httpx.Response(200, json={"status": True, "submission": "CORRECT", "description": "ok"})
    c = make(handler)
    c.evaluations = [{"id": "EV1", "name": "A", "status": "ACTIVE", "type": ""}, {"id": "EV2", "name": "B", "status": "ACTIVE", "type": ""}]
    assert asyncio.run(c.current_task("EV2"))["name"] == "t-EV2"
    assert asyncio.run(c.current_task())["name"] == "t-EV1"
    r = asyncio.run(c.submit({"answerSets": [{"answers": [{"text": "TR-V-1,2"}]}]}, evaluation_id="EV2"))
    assert r["verdict"] == "CORRECT" and ("POST", "/api/v2/submit/EV2") in seen
    assert c.evaluation_id == "EV1"                                   # the shared selection did not move


def test_submit_to_an_unknown_evaluation_is_refused_before_sending():
    sent = []

    def handler(req):
        if req.url.path.endswith("/evaluation/list"):
            return httpx.Response(200, json=[{"id": "EV1", "name": "A", "status": "ACTIVE"}])
        sent.append(req.url.path)
        return httpx.Response(200, json={"status": True, "submission": "WRONG"})
    c = make(handler)
    c.evaluations = [{"id": "EV1", "name": "A", "status": "ACTIVE", "type": ""}]
    with pytest.raises(DresError) as e:
        asyncio.run(c.submit({"answerSets": [{"answers": [{"text": "x"}]}]}, evaluation_id="GONE"))
    assert e.value.status == 404 and not sent


def test_endpoints_pass_the_page_evaluation_through():
    from api.app import Engine, create_app
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if "currentTask" in req.url.path:
            return httpx.Response(200, json={"name": "vkis-test", "taskType": "Visual Known Item Search", "duration": 240})
        return httpx.Response(200, json={"status": True, "submission": "WRONG", "description": "no"})
    c = make(handler)
    c.evaluations = [{"id": "EV1", "name": "A", "status": "ACTIVE", "type": ""}, {"id": "EV2", "name": "B", "status": "ACTIVE", "type": ""}]
    with TestClient(create_app(engine=Engine(), dres=c, autoload=False)) as api:
        j = api.get("/api/dres/task?evaluation_id=EV2").json()
        assert j["task"]["kind"] == "vkis" and j["evaluation_id"] == "EV2" and seen[-1].endswith("/currentTask/EV2")
        r = api.post("/api/dres/submit", json={"kind": "kis", "video_id": "L21_V013", "start_ms": 1, "end_ms": 1, "evaluation_id": "EV2"})
        assert r.status_code == 200 and seen[-1] == "/api/v2/submit/EV2"
        r = api.post("/api/dres/submit", json={"kind": "kis", "video_id": "L21_V014", "start_ms": 1, "end_ms": 1})
        assert seen[-1] == "/api/v2/submit/EV1"                        # no evaluation given: the server's selection
        assert api.get("/api/dres/state").json()["evaluation_id"] == "EV1"
