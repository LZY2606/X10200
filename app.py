"""晶格候选室 —— FastAPI 应用。

导入衍射反射观测（原始保存），生成多晶格候选，分项展示指标，
支持锁定/排除反射、校准调整、分叉分析与冻结冲突检测。
"""
import json
import os
import sqlite3
import threading
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import lattice_core as core

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations(
  id INTEGER PRIMARY KEY,
  qx REAL, qy REAL, qz REAL,
  intensity REAL,
  c00 REAL, c01 REAL, c02 REAL, c11 REAL, c12 REAL, c22 REAL,
  calibration_version TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS analyses(
  id INTEGER PRIMARY KEY,
  parent_id INTEGER,
  name TEXT,
  calibration_version TEXT,
  calib_dx REAL DEFAULT 0, calib_dy REAL DEFAULT 0, calib_dz REAL DEFAULT 0,
  tolerance REAL DEFAULT 9.0,
  outlier_budget INTEGER DEFAULT 2,
  version INTEGER DEFAULT 1,
  generation INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS marks(
  analysis_id INTEGER, reflection_id INTEGER, mark TEXT,
  PRIMARY KEY(analysis_id, reflection_id)
);
CREATE TABLE IF NOT EXISTS candidates(
  id INTEGER PRIMARY KEY,
  analysis_id INTEGER, generation INTEGER, domain INTEGER, serial INTEGER,
  basis TEXT, reduced_basis TEXT, transform_chain TEXT,
  family_id INTEGER, family_transform TEXT, metrics TEXT,
  stale INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS assignments(
  candidate_id INTEGER, reflection_id INTEGER,
  h INTEGER, k INTEGER, l INTEGER,
  px REAL, py REAL, pz REAL,
  residual REAL, weighted REAL, is_outlier INTEGER,
  PRIMARY KEY(candidate_id, reflection_id)
);
CREATE TABLE IF NOT EXISTS freezes(
  id INTEGER PRIMARY KEY, analysis_id INTEGER, analysis_version INTEGER,
  snapshot TEXT, created_at TEXT DEFAULT (datetime('now'))
);
"""

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ---------- 请求模型 ----------
class PeakIn(BaseModel):
    q: List[float]
    intensity: float = 1.0
    cov: Optional[List[float]] = None  # 6 个独立分量；缺省各向同性


class ImportIn(BaseModel):
    calibration_version: str
    peaks: List[PeakIn]


class AnalysisIn(BaseModel):
    name: str = "analysis"
    parent_id: Optional[int] = None


class MarkIn(BaseModel):
    reflection_id: int
    mark: str  # locked | excluded | none


class CalibIn(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


class GenerateIn(BaseModel):
    outlier_budget: int = 2
    tolerance: float = core.TOL_WEIGHTED_DEFAULT


class FreezeIn(BaseModel):
    expected_version: int


# ---------- 工具 ----------
def _cov_matrix(c):
    return np.array([[c[0], c[1], c[2]],
                     [c[1], c[3], c[4]],
                     [c[2], c[4], c[5]]], dtype=float)


def _load_observations(conn):
    rows = conn.execute(
        "SELECT * FROM observations ORDER BY id").fetchall()
    ids, qs, covs, intens, vers = [], [], [], [], []
    for r in rows:
        ids.append(r["id"])
        qs.append([r["qx"], r["qy"], r["qz"]])
        covs.append([r["c00"], r["c01"], r["c02"], r["c11"], r["c12"], r["c22"]])
        intens.append(r["intensity"])
        vers.append(r["calibration_version"])
    return ids, np.array(qs, dtype=float), covs, intens, vers


def _get_analysis(conn, aid):
    row = conn.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    if row is None:
        raise HTTPException(404, "analysis not found")
    return row


def _marks(conn, aid):
    return {r["reflection_id"]: r["mark"] for r in conn.execute(
        "SELECT * FROM marks WHERE analysis_id=?", (aid,))}


def _bump_version(conn, aid):
    conn.execute("UPDATE analyses SET version=version+1 WHERE id=?", (aid,))


def create_app(db_path="lattice.db"):
    app = FastAPI(title="晶格候选室")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    app.state.db = conn
    app.state.lock = threading.Lock()

    # ---------- 页面 ----------
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    # ---------- 观测导入（原始保存） ----------
    @app.post("/api/observations/import")
    def import_observations(body: ImportIn):
        with app.state.lock:
            for p in body.peaks:
                if len(p.q) != 3:
                    raise HTTPException(400, "q must have 3 components")
                cov = p.cov if p.cov is not None else [1e-6, 0, 0, 1e-6, 0, 1e-6]
                if len(cov) != 6:
                    raise HTTPException(400, "cov must have 6 components")
                conn.execute(
                    "INSERT INTO observations"
                    "(qx,qy,qz,intensity,c00,c01,c02,c11,c12,c22,calibration_version)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (p.q[0], p.q[1], p.q[2], p.intensity,
                     cov[0], cov[1], cov[2], cov[3], cov[4], cov[5],
                     body.calibration_version))
            conn.commit()
        return {"imported": len(body.peaks),
                "calibration_version": body.calibration_version}

    @app.get("/api/observations")
    def list_observations():
        rows = conn.execute(
            "SELECT * FROM observations ORDER BY id").fetchall()
        return {"observations": [dict(r) for r in rows]}

    # ---------- 分析（含分叉） ----------
    @app.post("/api/analyses")
    def create_analysis(body: AnalysisIn):
        with app.state.lock:
            calib_version = None
            row = conn.execute(
                "SELECT calibration_version FROM observations"
                " ORDER BY id DESC LIMIT 1").fetchone()
            if row:
                calib_version = row["calibration_version"]
            dx = dy = dz = 0.0
            if body.parent_id is not None:
                parent = _get_analysis(conn, body.parent_id)
                calib_version = parent["calibration_version"]
                dx, dy, dz = (parent["calib_dx"], parent["calib_dy"],
                              parent["calib_dz"])
            cur = conn.execute(
                "INSERT INTO analyses(parent_id,name,calibration_version,"
                "calib_dx,calib_dy,calib_dz) VALUES(?,?,?,?,?,?)",
                (body.parent_id, body.name, calib_version, dx, dy, dz))
            aid = cur.lastrowid
            if body.parent_id is not None:  # 分叉继承人工标记
                for rid, mark in _marks(conn, body.parent_id).items():
                    conn.execute(
                        "INSERT INTO marks(analysis_id,reflection_id,mark)"
                        " VALUES(?,?,?)", (aid, rid, mark))
            conn.commit()
        return {"id": aid, "parent_id": body.parent_id, "name": body.name}

    @app.get("/api/analyses")
    def list_analyses():
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id").fetchall()
        return {"analyses": [dict(r) for r in rows]}

    # ---------- 候选生成 ----------
    @app.post("/api/analyses/{aid}/generate")
    def generate(aid: int, body: GenerateIn):
        with app.state.lock:
            ana = _get_analysis(conn, aid)
            ids, qs, covs6, _, _ = _load_observations(conn)
            if len(ids) < 3:
                raise HTTPException(400, "need at least 3 observations")
            offset = np.array([ana["calib_dx"], ana["calib_dy"],
                               ana["calib_dz"]], dtype=float)
            qs_corr = qs - offset
            covs = np.array([_cov_matrix(c) for c in covs6])
            marks = _marks(conn, aid)
            idx_of = {rid: i for i, rid in enumerate(ids)}
            excluded = [idx_of[r] for r, m in marks.items()
                        if m == "excluded" and r in idx_of]
            locked = [idx_of[r] for r, m in marks.items()
                      if m == "locked" and r in idx_of]
            result = core.generate_candidates(
                qs_corr, covs, excluded=excluded, locked=locked,
                outlier_budget=body.outlier_budget, tol=body.tolerance)
            generation = ana["generation"] + 1
            conn.execute(
                "UPDATE analyses SET generation=?, outlier_budget=?,"
                " tolerance=? WHERE id=?",
                (generation, body.outlier_budget, body.tolerance, aid))
            for cand in result["candidates"]:
                cur = conn.execute(
                    "INSERT INTO candidates(analysis_id,generation,domain,"
                    "serial,basis,reduced_basis,transform_chain,family_id,"
                    "family_transform,metrics,stale)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,0)",
                    (aid, generation, cand["domain"], cand["serial"],
                     json.dumps(cand["basis"]),
                     json.dumps(cand["reduced_basis"]),
                     json.dumps(cand["transform_chain"]),
                     cand["family_id"],
                     json.dumps(cand["family_transform"]),
                     json.dumps(cand["metrics"])))
                cid = cur.lastrowid
                for a in cand["assignments"]:
                    conn.execute(
                        "INSERT INTO assignments(candidate_id,reflection_id,"
                        "h,k,l,px,py,pz,residual,weighted,is_outlier)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (cid, ids[a["index"]], a["hkl"][0], a["hkl"][1],
                         a["hkl"][2], a["pred"][0], a["pred"][1],
                         a["pred"][2], a["residual"], a["weighted"],
                         1 if a["is_outlier"] else 0))
            conn.commit()
        return {"generation": generation,
                "n_candidates": len(result["candidates"]),
                "singular_skipped": result["singular_skipped"]}

    @app.get("/api/analyses/{aid}/candidates")
    def list_candidates(aid: int):
        _get_analysis(conn, aid)
        rows = conn.execute(
            "SELECT * FROM candidates WHERE analysis_id=?"
            " ORDER BY generation DESC, serial", (aid,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"])
            d["basis"] = json.loads(d["basis"])
            d["family_transform"] = json.loads(d["family_transform"])
            out.append(d)
        return {"candidates": out}

    @app.get("/api/candidates/{cid}")
    def candidate_detail(cid: int):
        row = conn.execute(
            "SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
        if row is None:
            raise HTTPException(404, "candidate not found")
        d = dict(row)
        for key in ("basis", "reduced_basis", "transform_chain",
                    "family_transform", "metrics"):
            d[key] = json.loads(d[key])
        marks = _marks(conn, d["analysis_id"])
        assigns = []
        for a in conn.execute(
                "SELECT * FROM assignments WHERE candidate_id=?"
                " ORDER BY reflection_id", (cid,)):
            ad = dict(a)
            ad["mark"] = marks.get(a["reflection_id"], "none")
            assigns.append(ad)
        d["assignments"] = assigns
        return d

    # ---------- hkl -> 观测与变换链回溯 ----------
    @app.get("/api/candidates/{cid}/reflections/{rid}")
    def trace_reflection(cid: int, rid: int):
        cand = conn.execute(
            "SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
        if cand is None:
            raise HTTPException(404, "candidate not found")
        obs = conn.execute(
            "SELECT * FROM observations WHERE id=?", (rid,)).fetchone()
        if obs is None:
            raise HTTPException(404, "observation not found")
        assign = conn.execute(
            "SELECT * FROM assignments WHERE candidate_id=?"
            " AND reflection_id=?", (cid, rid)).fetchone()
        return {
            "observation": dict(obs),
            "assignment": dict(assign) if assign else None,
            "transform_chain": json.loads(cand["transform_chain"]),
            "family_transform": json.loads(cand["family_transform"]),
        }

    # ---------- 人工标记：局部过期 ----------
    @app.put("/api/analyses/{aid}/marks")
    def set_mark(aid: int, body: MarkIn):
        if body.mark not in ("locked", "excluded", "none"):
            raise HTTPException(400, "invalid mark")
        with app.state.lock:
            _get_analysis(conn, aid)
            if body.mark == "none":
                conn.execute(
                    "DELETE FROM marks WHERE analysis_id=?"
                    " AND reflection_id=?", (aid, body.reflection_id))
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO marks"
                    "(analysis_id,reflection_id,mark) VALUES(?,?,?)",
                    (aid, body.reflection_id, body.mark))
            # 只有依赖该反射（已索引它）的候选过期
            cands = conn.execute(
                "SELECT id FROM candidates WHERE analysis_id=? AND stale=0",
                (aid,)).fetchall()
            staled = 0
            for c in cands:
                dep = conn.execute(
                    "SELECT 1 FROM assignments WHERE candidate_id=?"
                    " AND reflection_id=? AND is_outlier=0",
                    (c["id"], body.reflection_id)).fetchone()
                if dep:
                    conn.execute(
                        "UPDATE candidates SET stale=1 WHERE id=?", (c["id"],))
                    staled += 1
            _bump_version(conn, aid)
            conn.commit()
        return {"staled_candidates": staled}

    # ---------- 校准调整：全部候选过期 ----------
    @app.put("/api/analyses/{aid}/calibration")
    def set_calibration(aid: int, body: CalibIn):
        with app.state.lock:
            _get_analysis(conn, aid)
            conn.execute(
                "UPDATE analyses SET calib_dx=?, calib_dy=?, calib_dz=?"
                " WHERE id=?", (body.dx, body.dy, body.dz, aid))
            cur = conn.execute(
                "UPDATE candidates SET stale=1"
                " WHERE analysis_id=? AND stale=0", (aid,))
            _bump_version(conn, aid)
            conn.commit()
        return {"staled_candidates": cur.rowcount}

    # ---------- 冻结（乐观并发） ----------
    @app.post("/api/analyses/{aid}/freeze")
    def freeze(aid: int, body: FreezeIn):
        with app.state.lock:
            ana = _get_analysis(conn, aid)
            if ana["version"] != body.expected_version:
                raise HTTPException(
                    409, "analysis changed since draft; reload and refreeze")
            snapshot = {
                "calibration_version": ana["calibration_version"],
                "calibration_offset": [ana["calib_dx"], ana["calib_dy"],
                                       ana["calib_dz"]],
                "tolerance": ana["tolerance"],
                "outlier_budget": ana["outlier_budget"],
                "marks": _marks(conn, aid),
            }
            conn.execute(
                "INSERT INTO freezes(analysis_id,analysis_version,snapshot)"
                " VALUES(?,?,?)", (aid, ana["version"], json.dumps(snapshot)))
            conn.commit()
        return {"frozen": True, "analysis_version": ana["version"],
                "snapshot": snapshot}

    @app.get("/api/analyses/{aid}/freezes")
    def list_freezes(aid: int):
        _get_analysis(conn, aid)
        rows = conn.execute(
            "SELECT * FROM freezes WHERE analysis_id=? ORDER BY id",
            (aid,)).fetchall()
        return {"freezes": [dict(r) for r in rows]}

    # ---------- 倒易空间投影 ----------
    @app.get("/api/analyses/{aid}/projection")
    def projection(aid: int):
        ana = _get_analysis(conn, aid)
        ids, qs, _, intens, _ = _load_observations(conn)
        if len(ids) == 0:
            return {"points": []}
        offset = np.array([ana["calib_dx"], ana["calib_dy"],
                           ana["calib_dz"]], dtype=float)
        q = qs - offset
        q = q - q.mean(axis=0)
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        basis2 = vt[:2]
        for j in range(2):  # 固定符号，保证确定性
            col = basis2[j]
            if col[np.argmax(np.abs(col))] < 0:
                basis2[j] = -col
        pts = q @ basis2.T
        marks = _marks(conn, aid)
        return {"points": [
            {"id": ids[i], "x": float(pts[i, 0]), "y": float(pts[i, 1]),
             "intensity": intens[i], "mark": marks.get(ids[i], "none")}
            for i in range(len(ids))]}

    return app


app = create_app(os.environ.get("LATTICE_DB", "lattice.db"))
