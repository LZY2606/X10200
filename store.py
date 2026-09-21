"""SQLite 存储层：保存原始观测、候选、族、冻结与版本。"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import List, Optional

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER,
    name TEXT,
    calibration_version TEXT,
    bias TEXT DEFAULT '[0,0,0]',
    draft_version INTEGER DEFAULT 0,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    q TEXT, intensity REAL, cov TEXT,
    calibration_version TEXT,
    locked INTEGER DEFAULT 0, excluded INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    family_id INTEGER,
    seq INTEGER,
    seed_basis TEXT, basis TEXT,
    reduction_steps TEXT, family_transform TEXT,
    scores TEXT, stale INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER,
    observation_id INTEGER,
    h INTEGER, k INTEGER, l INTEGER,
    predicted TEXT, residual TEXT, d2 REAL,
    is_outlier INTEGER, indexed INTEGER, overlap INTEGER
);
CREATE TABLE IF NOT EXISTS freezes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    draft_version INTEGER,
    pinned TEXT,
    created_at REAL
);
"""


class Store:
    def __init__(self, path: str = "lattice.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---- analyses ----
    def create_analysis(self, name: str, parent_id: Optional[int] = None,
                        calibration_version: str = "uncalibrated",
                        bias=(0.0, 0.0, 0.0)) -> int:
        cur = self.conn.execute(
            "INSERT INTO analyses(parent_id,name,calibration_version,bias,created_at)"
            " VALUES(?,?,?,?,?)",
            (parent_id, name, calibration_version, json.dumps(list(bias)), time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_analysis(self, aid: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()

    def bump_draft(self, aid: int):
        self.conn.execute(
            "UPDATE analyses SET draft_version=draft_version+1 WHERE id=?", (aid,))
        self.conn.commit()

    def set_calibration(self, aid: int, version: str, bias):
        self.conn.execute(
            "UPDATE analyses SET calibration_version=?, bias=? WHERE id=?",
            (version, json.dumps([float(x) for x in bias]), aid))
        # 校准影响全部候选的预测位置 -> 全部过期
        self.conn.execute("UPDATE candidates SET stale=1 WHERE analysis_id=?", (aid,))
        self.bump_draft(aid)

    # ---- observations ----
    def add_observation(self, aid: int, q, intensity: float, cov, calib_version: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO observations(analysis_id,q,intensity,cov,calibration_version)"
            " VALUES(?,?,?,?,?)",
            (aid, json.dumps([float(x) for x in q]), float(intensity),
             json.dumps([float(x) for x in cov]), calib_version),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_observations(self, aid: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM observations WHERE analysis_id=? ORDER BY id", (aid,)).fetchall()

    def set_obs_flag(self, obs_id: int, field: str, value: bool) -> int:
        assert field in ("locked", "excluded")
        row = self.conn.execute("SELECT * FROM observations WHERE id=?", (obs_id,)).fetchone()
        if row is None:
            return -1
        self.conn.execute(
            f"UPDATE observations SET {field}=? WHERE id=?", (int(value), obs_id))
        # 局部过期：只有依赖该反射的候选过期。
        # 依赖定义：候选把该观测指标化（非离群），或锁定状态改变影响其离群判定。
        if field == "excluded":
            self.conn.execute(
                """UPDATE candidates SET stale=1 WHERE id IN (
                       SELECT candidate_id FROM assignments
                       WHERE observation_id=? AND indexed=1)""",
                (obs_id,))
        else:  # locked 改变离群判定，影响所有含该观测指派的候选
            self.conn.execute(
                """UPDATE candidates SET stale=1 WHERE id IN (
                       SELECT candidate_id FROM assignments
                       WHERE observation_id=?)""",
                (obs_id,))
        self.bump_draft(row["analysis_id"])
        return row["analysis_id"]

    # ---- candidates ----
    def replace_candidates(self, aid: int, candidates):
        self.conn.execute("DELETE FROM assignments WHERE candidate_id IN "
                          "(SELECT id FROM candidates WHERE analysis_id=?)", (aid,))
        self.conn.execute("DELETE FROM candidates WHERE analysis_id=?", (aid,))
        obs_rows = self.list_observations(aid)
        obs_ids = [r["id"] for r in obs_rows]
        active = [i for i, r in enumerate(obs_rows) if not r["excluded"]]
        for c in candidates:
            cur = self.conn.execute(
                "INSERT INTO candidates(analysis_id,family_id,seq,seed_basis,basis,"
                "reduction_steps,family_transform,scores,stale)"
                " VALUES(?,?,?,?,?,?,?,?,0)",
                (aid, c.family_id, c.seq,
                 json.dumps(np.asarray(c.seed_basis).tolist()),
                 json.dumps(np.asarray(c.basis).tolist()),
                 json.dumps(c.reduction_steps),
                 json.dumps(c.family_transform),
                 json.dumps(c.scores)))
            cid = cur.lastrowid
            c.db_id = cid
            for a in c.assignments:
                obs_id = obs_ids[active[a.obs_index]]
                self.conn.execute(
                    "INSERT INTO assignments(candidate_id,observation_id,h,k,l,"
                    "predicted,residual,d2,is_outlier,indexed,overlap)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, obs_id, a.hkl[0], a.hkl[1], a.hkl[2],
                     json.dumps(np.asarray(a.predicted).tolist()),
                     json.dumps(np.asarray(a.residual).tolist()),
                     a.d2, int(a.is_outlier), int(a.indexed), int(a.overlap)))
        self.conn.commit()

    def list_candidates(self, aid: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM candidates WHERE analysis_id=? ORDER BY seq", (aid,)).fetchall()

    def get_candidate(self, cid: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()

    def list_assignments(self, cid: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM assignments WHERE candidate_id=? ORDER BY id", (cid,)).fetchall()

    # ---- freeze ----
    def freeze(self, aid: int, expected_draft: int, pinned: dict):
        row = self.get_analysis(aid)
        if row is None:
            return ("not_found", None)
        if row["draft_version"] != expected_draft:
            return ("conflict", row["draft_version"])
        cur = self.conn.execute(
            "INSERT INTO freezes(analysis_id,draft_version,pinned,created_at)"
            " VALUES(?,?,?,?)",
            (aid, expected_draft, json.dumps(pinned), time.time()))
        # 冻结本身推进草案版本：并发冻结旧草案将冲突
        self.conn.execute(
            "UPDATE analyses SET draft_version=draft_version+1 WHERE id=?", (aid,))
        self.conn.commit()
        return ("ok", cur.lastrowid)

    def list_freezes(self, aid: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM freezes WHERE analysis_id=? ORDER BY id", (aid,)).fetchall()

    # ---- fork ----
    def fork(self, aid: int, name: str) -> Optional[int]:
        src = self.get_analysis(aid)
        if src is None:
            return None
        new_id = self.create_analysis(
            name=name, parent_id=aid,
            calibration_version=src["calibration_version"],
            bias=json.loads(src["bias"]))
        for r in self.list_observations(aid):
            cur = self.conn.execute(
                "INSERT INTO observations(analysis_id,q,intensity,cov,"
                "calibration_version,locked,excluded) VALUES(?,?,?,?,?,?,?)",
                (new_id, r["q"], r["intensity"], r["cov"],
                 r["calibration_version"], r["locked"], r["excluded"]))
        self.conn.commit()
        return new_id
