"""晶格候选室 — FastAPI 入口。

导入峰坐标/强度/协方差/校准版本，生成多个晶格候选，
提供族分组、锁定/排除、校准调整、分叉与冻结接口。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import lattice_core as core
import sample_data
from store import Store

DB_PATH = os.environ.get("LATTICE_DB", "lattice.db")
store = Store(DB_PATH)
app = FastAPI(title="晶格候选室")


# ---------- 请求模型 ----------
class PeakIn(BaseModel):
    q: List[float]
    intensity: float
    cov: List[float]  # 上三角 6 元素
    calibration_version: Optional[str] = None


class ImportIn(BaseModel):
    calibration_version: str
    peaks: List[PeakIn]


class CalibrationIn(BaseModel):
    version: str
    bias: List[float]


class FlagIn(BaseModel):
    value: bool


class FreezeIn(BaseModel):
    draft_version: int
    note: Optional[str] = ""


class ForkIn(BaseModel):
    name: Optional[str] = "fork"


# ---------- 工具 ----------
def _analysis_or_404(aid: int):
    row = store.get_analysis(aid)
    if row is None:
        raise HTTPException(404, "analysis not found")
    return row


def _active_obs(aid: int):
    rows = store.list_observations(aid)
    ana = _analysis_or_404(aid)
    bias = np.array(json.loads(ana["bias"]), dtype=float)
    qs, covs, locked, ids = [], [], [], []
    for r in rows:
        if r["excluded"]:
            continue
        qs.append(core.correct_q(np.array(json.loads(r["q"])), bias))
        covs.append(sample_data.cov6_to_matrix(json.loads(r["cov"])))
        locked.append(bool(r["locked"]))
        ids.append(r["id"])
    return rows, qs, covs, locked


def _candidate_json(row, with_assignments=False):
    out = {
        "id": row["id"],
        "family_id": row["family_id"],
        "seq": row["seq"],
        "stale": bool(row["stale"]),
        "scores": json.loads(row["scores"]),
        "basis": json.loads(row["basis"]),
        "seed_basis": json.loads(row["seed_basis"]),
        "reduction_steps": json.loads(row["reduction_steps"]),
        "family_transform": json.loads(row["family_transform"]),
    }
    if with_assignments:
        out["assignments"] = [
            {
                "observation_id": a["observation_id"],
                "hkl": [a["h"], a["k"], a["l"]],
                "predicted": json.loads(a["predicted"]),
                "residual": json.loads(a["residual"]),
                "d2": a["d2"],
                "is_outlier": bool(a["is_outlier"]),
                "indexed": bool(a["indexed"]),
                "overlap": bool(a["overlap"]),
            }
            for a in store.list_assignments(row["id"])
        ]
    return out


# ---------- 页面 ----------
@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


# ---------- 分析 ----------
@app.post("/analyses")
def create_analysis(name: str = "analysis"):
    aid = store.create_analysis(name)
    return {"id": aid}


@app.post("/analyses/{aid}/import")
def import_peaks(aid: int, body: ImportIn):
    _analysis_or_404(aid)
    for p in body.peaks:
        store.add_observation(
            aid, p.q, p.intensity, p.cov,
            p.calibration_version or body.calibration_version)
    store.bump_draft(aid)
    return {"imported": len(body.peaks)}


@app.post("/analyses/{aid}/load_sample")
def load_sample(aid: int):
    _analysis_or_404(aid)
    data = sample_data.build_sample()
    store.set_calibration(aid, data["calibration_version"], data["bias"])
    for p in data["peaks"]:
        store.add_observation(aid, p["q"], p["intensity"], p["cov"],
                              data["calibration_version"])
    store.bump_draft(aid)
    return {"imported": len(data["peaks"]),
            "calibration_version": data["calibration_version"]}


@app.post("/analyses/{aid}/calibration")
def set_calibration(aid: int, body: CalibrationIn):
    _analysis_or_404(aid)
    if len(body.bias) != 3:
        raise HTTPException(400, "bias must have 3 components")
    store.set_calibration(aid, body.version, body.bias)
    return {"ok": True}


@app.post("/analyses/{aid}/generate")
def generate(aid: int, max_candidates: int = 8):
    _analysis_or_404(aid)
    rows, qs, covs, locked = _active_obs(aid)
    if len(qs) < 3:
        raise HTTPException(400, "need at least 3 active observations")
    candidates = core.generate_candidates(qs, covs, locked, max_candidates)
    core.group_families(candidates)
    store.replace_candidates(aid, candidates)
    return {"n_candidates": len(candidates),
            "n_families": len({c.family_id for c in candidates})}


@app.get("/analyses/{aid}/state")
def state(aid: int):
    ana = _analysis_or_404(aid)
    obs = [
        {
            "id": r["id"], "q": json.loads(r["q"]), "intensity": r["intensity"],
            "cov": json.loads(r["cov"]),
            "calibration_version": r["calibration_version"],
            "locked": bool(r["locked"]), "excluded": bool(r["excluded"]),
        }
        for r in store.list_observations(aid)
    ]
    cands = [_candidate_json(c) for c in store.list_candidates(aid)]
    families = {}
    for c in cands:
        families.setdefault(c["family_id"], []).append(c["id"])
    return {
        "id": aid,
        "name": ana["name"],
        "parent_id": ana["parent_id"],
        "calibration_version": ana["calibration_version"],
        "bias": json.loads(ana["bias"]),
        "draft_version": ana["draft_version"],
        "observations": obs,
        "candidates": cands,
        "families": [{"family_id": k, "candidate_ids": v}
                     for k, v in sorted(families.items())],
        "freezes": [
            {"id": f["id"], "draft_version": f["draft_version"],
             "pinned": json.loads(f["pinned"]), "created_at": f["created_at"]}
            for f in store.list_freezes(aid)
        ],
    }


@app.get("/candidates/{cid}")
def candidate_detail(cid: int):
    row = store.get_candidate(cid)
    if row is None:
        raise HTTPException(404, "candidate not found")
    return _candidate_json(row, with_assignments=True)


# ---------- 人工标记 ----------
@app.post("/observations/{obs_id}/lock")
def lock_observation(obs_id: int, body: FlagIn):
    aid = store.set_obs_flag(obs_id, "locked", body.value)
    if aid < 0:
        raise HTTPException(404, "observation not found")
    return {"ok": True, "analysis_id": aid}


@app.post("/observations/{obs_id}/exclude")
def exclude_observation(obs_id: int, body: FlagIn):
    aid = store.set_obs_flag(obs_id, "excluded", body.value)
    if aid < 0:
        raise HTTPException(404, "observation not found")
    return {"ok": True, "analysis_id": aid}


# ---------- 分叉与冻结 ----------
@app.post("/analyses/{aid}/fork")
def fork(aid: int, body: ForkIn):
    _analysis_or_404(aid)
    new_id = store.fork(aid, body.name or "fork")
    return {"id": new_id, "parent_id": aid}


@app.post("/analyses/{aid}/freeze")
def freeze(aid: int, body: FreezeIn):
    ana = _analysis_or_404(aid)
    obs = store.list_observations(aid)
    pinned = {
        "calibration_version": ana["calibration_version"],
        "bias": json.loads(ana["bias"]),
        "tolerances": {
            "family_tol": core.FAMILY_TOL,
            "outlier_threshold": core.OUTLIER_THRESHOLD,
            "singular_tol": core.SINGULAR_TOL,
        },
        "marks": {
            str(r["id"]): {"locked": bool(r["locked"]),
                           "excluded": bool(r["excluded"])}
            for r in obs
        },
        "note": body.note,
    }
    status, val = store.freeze(aid, body.draft_version, pinned)
    if status == "not_found":
        raise HTTPException(404, "analysis not found")
    if status == "conflict":
        raise HTTPException(409, f"draft conflict: current version is {val}")
    return {"freeze_id": val, "pinned": pinned}


app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(__file__), "static")), name="static")
