from fastapi.testclient import TestClient

from app.database import Database
from app.web import create_app


def test_root_page_title(tmp_path):
    client = TestClient(create_app(Database(str(tmp_path / "api.db"))))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "晶格候选室" in resp.text


def test_demo_flow_and_freeze_conflict(tmp_path):
    client = TestClient(create_app(Database(str(tmp_path / "api2.db"))))
    created = client.post("/api/demo").json()
    aid = created["analysis_id"]
    assert created["candidates_created"] >= 2

    data = client.get(f"/api/analyses/{aid}").json()
    assert data["analysis"]["revision"] == 1
    cid = next(c["id"] for c in data["candidates"] if c["status"] == "active")
    detail = client.get(f"/api/candidates/{cid}").json()
    assert detail["scores"]["outlier_budget"]["budget"] == 2

    # 错误修订号冻结 -> 409
    resp = client.post(f"/api/analyses/{aid}/freeze",
                       json={"expected_revision": 999})
    assert resp.status_code == 409

    # 正确冻结
    rev = data["analysis"]["revision"]
    resp = client.post(f"/api/analyses/{aid}/freeze",
                       json={"expected_revision": rev})
    assert resp.status_code == 200
