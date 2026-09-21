"""API 层测试：导入、局部过期、冻结冲突、分叉、校准失效。"""
import os
import tempfile

import numpy as np
import pytest

_tmp = tempfile.mkdtemp()
os.environ["LATTICE_DB"] = os.path.join(_tmp, "test.db")

from fastapi.testclient import TestClient

import app as app_module
import lattice_core as core
import sample_data
from store import Store

client = TestClient(app_module.app)


@pytest.fixture()
def analysis():
    aid = client.post("/analyses", params={"name": "t"}).json()["id"]
    yield aid


def _import_sample(aid):
    r = client.post(f"/analyses/{aid}/load_sample")
    assert r.status_code == 200
    return r.json()


def test_import_and_generate(analysis):
    _import_sample(analysis)
    st = client.get(f"/analyses/{analysis}/state").json()
    assert len(st["observations"]) > 200
    assert st["calibration_version"] == sample_data.CALIB_VERSION
    # 原始观测已保存（含协方差与校准版本）
    obs0 = st["observations"][0]
    assert len(obs0["q"]) == 3 and len(obs0["cov"]) == 6
    r = client.post(f"/analyses/{analysis}/generate")
    assert r.status_code == 200
    assert r.json()["n_families"] >= 2  # 双晶域
    st = client.get(f"/analyses/{analysis}/state").json()
    assert len(st["candidates"]) >= 2
    for c in st["candidates"]:
        s = c["scores"]
        # 四个评分分量分别展示
        for key in ("indexed_fraction", "weighted_residual",
                    "complexity", "outlier_used", "outlier_budget"):
            assert key in s
    # 候选详情：每个反射有 hkl/预测位置/残差/离群标记，变换链完整
    det = client.get(f"/candidates/{st['candidates'][0]['id']}").json()
    a0 = det["assignments"][0]
    assert len(a0["hkl"]) == 3 and len(a0["predicted"]) == 3
    assert len(a0["residual"]) == 3 and "is_outlier" in a0
    assert isinstance(det["reduction_steps"], list)
    assert det["family_transform"] is not None


def test_local_staleness_on_exclude(analysis):
    _import_sample(analysis)
    client.post(f"/analyses/{analysis}/generate")
    st = client.get(f"/analyses/{analysis}/state").json()
    # 找一个在某些候选中被指标化、在另一些中是离群的反射
    details = {c["id"]: client.get(f"/candidates/{c['id']}").json()
               for c in st["candidates"]}
    target = None
    for obs in st["observations"]:
        flags = []
        for cid, det in details.items():
            a = next((x for x in det["assignments"]
                      if x["observation_id"] == obs["id"]), None)
            flags.append(a["indexed"] if a else None)
        if any(f is True for f in flags) and any(f is False for f in flags):
            target = obs["id"]
            break
    assert target is not None
    client.post(f"/observations/{target}/exclude", json={"value": True})
    st2 = client.get(f"/analyses/{analysis}/state").json()
    stale_by_id = {c["id"]: c["stale"] for c in st2["candidates"]}
    for cid, det in details.items():
        a = next((x for x in det["assignments"]
                  if x["observation_id"] == target), None)
        if a and a["indexed"]:
            assert stale_by_id[cid] is True   # 依赖该反射的候选过期
        else:
            assert stale_by_id[cid] is False  # 其余保持有效
    assert any(stale_by_id.values()) and not all(stale_by_id.values())


def test_calibration_change_stales_all(analysis):
    _import_sample(analysis)
    client.post(f"/analyses/{analysis}/generate")
    r = client.post(f"/analyses/{analysis}/calibration",
                    json={"version": "cal-2026-09b", "bias": [0.001, 0.0, -0.001]})
    assert r.status_code == 200
    st = client.get(f"/analyses/{analysis}/state").json()
    assert all(c["stale"] for c in st["candidates"])
    assert st["calibration_version"] == "cal-2026-09b"


def test_freeze_conflict(analysis):
    _import_sample(analysis)
    st = client.get(f"/analyses/{analysis}/state").json()
    v = st["draft_version"]
    # 错误版本 -> 409 冲突
    r = client.post(f"/analyses/{analysis}/freeze",
                    json={"draft_version": v + 99})
    assert r.status_code == 409
    # 正确版本 -> 冻结成功，钉住校准/容差/人工标记
    r = client.post(f"/analyses/{analysis}/freeze", json={"draft_version": v})
    assert r.status_code == 200
    pinned = r.json()["pinned"]
    assert pinned["calibration_version"] == sample_data.CALIB_VERSION
    assert pinned["tolerances"]["family_tol"] == core.FAMILY_TOL
    assert str(st["observations"][0]["id"]) in pinned["marks"]
    # 并发冻结旧草案 -> 冲突
    r = client.post(f"/analyses/{analysis}/freeze", json={"draft_version": v})
    assert r.status_code == 409


def test_fork_inherits_and_isolates(analysis):
    _import_sample(analysis)
    obs_id = client.get(f"/analyses/{analysis}/state").json()["observations"][0]["id"]
    client.post(f"/observations/{obs_id}/lock", json={"value": True})
    r = client.post(f"/analyses/{analysis}/fork", json={"name": "branch"})
    new_id = r.json()["id"]
    assert r.json()["parent_id"] == analysis
    st_new = client.get(f"/analyses/{new_id}/state").json()
    st_old = client.get(f"/analyses/{analysis}/state").json()
    assert len(st_new["observations"]) == len(st_old["observations"])
    assert st_new["observations"][0]["locked"] is True
    assert st_new["calibration_version"] == st_old["calibration_version"]
    # 分叉后修改互不影响
    client.post(f"/observations/{st_new['observations'][1]['id']}/exclude",
                json={"value": True})
    st_old2 = client.get(f"/analyses/{analysis}/state").json()
    assert st_old2["observations"][1]["excluded"] is False


def test_lock_prevents_outlier(analysis):
    _import_sample(analysis)
    client.post(f"/analyses/{analysis}/generate")
    st = client.get(f"/analyses/{analysis}/state").json()
    det = client.get(f"/candidates/{st['candidates'][0]['id']}").json()
    outlier = next(a for a in det["assignments"] if a["is_outlier"])
    client.post(f"/observations/{outlier['observation_id']}/lock",
                json={"value": True})
    # 锁定后该候选过期（锁定改变离群判定）
    st2 = client.get(f"/analyses/{analysis}/state").json()
    c0 = next(c for c in st2["candidates"] if c["id"] == det["id"])
    assert c0["stale"] is True
