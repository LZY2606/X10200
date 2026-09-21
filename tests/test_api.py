"""API / persistence tests: import, invalidation, fork, freeze conflicts."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from db import ConflictError, Store
from lattice.synthetic import lattice_peaks

BASIS_A = np.array([[0.52, 0.0, 0.0], [0.0, 0.61, 0.0], [0.0, 0.0, 0.71]])


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


BASIS_B = np.array([[0.40, 0.0, 0.10], [0.0, 0.40, 0.0], [0.0, 0.0, 0.45]])


@pytest.fixture()
def seeded(store):
    qs_a, covs_a, _ = lattice_peaks(BASIS_A, h_max=2, noise=0.001, seed=11)
    qs_b, covs_b, _ = lattice_peaks(BASIS_B, h_max=2, noise=0.001, seed=12)
    qs = np.vstack([qs_a, qs_b])
    covs = np.vstack([covs_a, covs_b])
    peaks = [{"q": q.tolist(), "intensity": 1.0, "cov": c.tolist()}
             for q, c in zip(qs, covs)]
    store.import_observations(peaks, "cal-v1")
    aid = store.create_analysis("main", {}, {})
    store.run_analysis(aid)
    return store, aid


def test_raw_observations_preserved(seeded):
    store, _ = seeded
    obs = store.list_observations()
    assert obs
    for o in obs:
        assert o["calibration_version"] == "cal-v1"
        assert o["raw"]["q"] == o["q"]
        assert "cov" in o["raw"]


def test_candidates_have_reflections_and_chains(seeded):
    store, aid = seeded
    state = store.get_analysis_state(aid)
    assert state["candidates"]
    cand = state["candidates"][0]
    assert cand["valid"]
    assert cand["reflections"]
    r = cand["reflections"][0]
    assert len(r["hkl"]) == 3
    assert len(r["predicted_q"]) == 3
    assert len(r["residual"]) == 3
    assert isinstance(r["is_outlier"], bool)
    assert isinstance(cand["transform_chain"], list)
    assert cand["transform_to_family"]


def test_partial_invalidation_only_dependent(seeded):
    store, aid = seeded
    state = store.get_analysis_state(aid)
    # Pick an observation that some candidates use and others ignore.
    usage = {}
    for c in state["candidates"]:
        for r in c["reflections"]:
            if r["status"] != "unindexed":
                usage.setdefault(r["observation_id"], set()).add(c["id"])
    all_ids = {c["id"] for c in state["candidates"]}
    obs_id = next(oid for oid, users in usage.items()
                  if 0 < len(users) < len(all_ids))
    dependent = usage[obs_id]
    independent = all_ids - dependent
    store.set_mark(aid, obs_id, locked=True)
    after = store.get_analysis_state(aid)
    for c in after["candidates"]:
        if c["id"] in dependent:
            assert not c["valid"]
        else:
            assert c["valid"]
    # Unrelated analyses are untouched by the mark.
    assert store.get_analysis_state(aid)["version"] == 2


def test_calibration_change_expires_all(seeded):
    store, aid = seeded
    store.update_calibration(aid, {"scale": 1.001})
    state = store.get_analysis_state(aid)
    assert all(not c["valid"] for c in state["candidates"])


def test_fork_copies_marks_and_runs(seeded):
    store, aid = seeded
    state = store.get_analysis_state(aid)
    obs_id = state["candidates"][0]["reflections"][0]["observation_id"]
    store.set_mark(aid, obs_id, excluded=True)
    fork_id = store.fork_analysis(aid, "branch")
    fork_state = store.get_analysis_state(fork_id)
    assert fork_state["parent_id"] == aid
    assert fork_state["marks"][str(obs_id)]["excluded"] is True
    assert fork_state["candidates"]
    # Excluded observation must not appear in the fork's reflections.
    used = {r["observation_id"] for c in fork_state["candidates"]
            for r in c["reflections"]}
    assert obs_id not in used


def test_freeze_conflict_on_stale_version(seeded):
    store, aid = seeded
    v1 = store.get_analysis_state(aid)["version"]
    store.freeze(aid, v1)  # first freeze fine
    obs_id = store.get_analysis_state(aid)["candidates"][0][
        "reflections"][0]["observation_id"]
    store.set_mark(aid, obs_id, locked=True)  # bumps version
    with pytest.raises(ConflictError):
        store.freeze(aid, v1)  # stale draft must conflict
    frozen = store.list_freezes(aid)
    assert len(frozen) == 1
    assert frozen[0]["calibration"] == {}
    assert "tolerances" in frozen[0] and "marks" in frozen[0]


def test_freeze_pins_marks(seeded):
    store, aid = seeded
    state = store.get_analysis_state(aid)
    obs_id = state["candidates"][0]["reflections"][0]["observation_id"]
    store.set_mark(aid, obs_id, locked=True)
    v = store.get_analysis_state(aid)["version"]
    result = store.freeze(aid, v)
    assert result["marks"][obs_id]["locked"] is True


def test_api_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "store", Store(str(tmp_path / "api.db")))
    client = TestClient(app_module.app)
    qs, covs, _ = lattice_peaks(BASIS_A, h_max=2, noise=0.001, seed=12)
    peaks = [{"q": q.tolist(), "intensity": 1.0, "cov": c.tolist()}
             for q, c in zip(qs, covs)]
    r = client.post("/api/observations/import",
                    json={"peaks": peaks, "calibration_version": "cal-v1"})
    assert r.status_code == 200
    r = client.post("/api/analyses", json={"name": "t", "calibration": {},
                                           "tolerances": {}})
    assert r.status_code == 200
    aid = r.json()["analysis_id"]
    state = client.get(f"/api/analyses/{aid}").json()
    assert state["families"] and state["candidates"]
    v = state["version"]
    assert client.post(f"/api/analyses/{aid}/freeze",
                       json={"expected_version": v}).status_code == 200
    obs_id = state["candidates"][0]["reflections"][0]["observation_id"]
    client.post(f"/api/analyses/{aid}/marks",
                json={"observation_id": obs_id, "locked": True})
    # Concurrent freeze of the stale draft now conflicts.
    assert client.post(f"/api/analyses/{aid}/freeze",
                       json={"expected_version": v}).status_code == 409
    r = client.post(f"/api/analyses/{aid}/fork", json={"name": "b"})
    assert r.status_code == 200
    assert client.get("/").status_code == 200
