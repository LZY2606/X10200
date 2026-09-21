"""SQLite 持久化层。所有数值对象以 JSON 文本保存，保证可审计、可移植。"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    created_at TEXT NOT NULL,
    instrument_version TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    calibration_version INTEGER NOT NULL,
    digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    external_id TEXT,
    x_raw REAL NOT NULL,
    y_raw REAL NOT NULL,
    intensity REAL,
    cov_json TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS marks (
    analysis_id TEXT NOT NULL,
    observation_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('locked','excluded','normal')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (analysis_id, observation_id)
);

CREATE TABLE IF NOT EXISTS calibration_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    point_id TEXT,
    raw_json TEXT NOT NULL,
    ref_json TEXT NOT NULL,
    cov_json TEXT NOT NULL,
    excluded INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS calibration_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    A_json TEXT NOT NULL,
    t_json TEXT NOT NULL,
    rms REAL,
    n_points INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS families (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    family_seq INTEGER NOT NULL,
    representative_candidate_id INTEGER
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    family_seq INTEGER,
    basis_json TEXT NOT NULL,
    reduction_json TEXT NOT NULL,
    score_json TEXT NOT NULL,
    extinction_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    stale_reason TEXT,
    digest TEXT NOT NULL,
    singular INTEGER NOT NULL DEFAULT 0,
    singular_json TEXT
);

CREATE TABLE IF NOT EXISTS candidate_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    observation_id INTEGER NOT NULL,
    hkl_json TEXT NOT NULL,
    pred_json TEXT NOT NULL,
    residual_json TEXT NOT NULL,
    raw_dist REAL NOT NULL,
    mahal REAL NOT NULL,
    role TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS freezes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    calibration_version INTEGER NOT NULL,
    settings_json TEXT NOT NULL,
    marks_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    client_digest TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
"""


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str = "lattice_lab.db"):
        self.path = path
        self._lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(SCHEMA)

    # -- 通用辅助 ---------------------------------------------------------
    @staticmethod
    def j(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def d(text: str) -> Any:
        return json.loads(text)

    def event(self, analysis_id: str, kind: str, detail: Dict[str, Any]
              ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO events(analysis_id, ts, kind, detail_json) "
                "VALUES (?,?,?,?)",
                (analysis_id, _now(), kind, self.j(detail)))

    # -- 分析与观测 -------------------------------------------------------
    def create_analysis(self, analysis_id: str, instrument_version: str,
                        settings: Dict[str, Any], parent_id: Optional[str],
                        digest: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO analyses(id, parent_id, created_at, "
                "instrument_version, settings_json, calibration_version, "
                "digest) VALUES (?,?,?,?,?,?,?)",
                (analysis_id, parent_id, _now(), instrument_version,
                 self.j(settings), 1, digest))

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["settings"] = self.d(d.pop("settings_json"))
            return d

    def list_analyses(self) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, parent_id, created_at, instrument_version, "
                "calibration_version FROM analyses ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def update_settings(self, analysis_id: str, settings: Dict[str, Any],
                        digest: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE analyses SET settings_json=?, digest=? WHERE id=?",
                (self.j(settings), digest, analysis_id))

    def add_observations(self, analysis_id: str,
                         obs: List[Dict[str, Any]]) -> List[int]:
        ids: List[int] = []
        with self._lock, self._conn() as conn:
            for o in obs:
                cur = conn.execute(
                    "INSERT INTO observations(analysis_id, external_id, "
                    "x_raw, y_raw, intensity, cov_json, raw_json) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (analysis_id, o.get("external_id"),
                     float(o["x_raw"]), float(o["y_raw"]),
                     float(o.get("intensity", 0.0)),
                     self.j(o.get("cov", [[1.0, 0.0], [0.0, 1.0]])),
                     self.j(o)))
                ids.append(cur.lastrowid)
        return ids

    def list_observations(self, analysis_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM observations WHERE analysis_id=? ORDER BY id",
                (analysis_id,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["cov"] = self.d(d.pop("cov_json"))
                d["raw"] = self.d(d.pop("raw_json"))
                out.append(d)
            return out

    def set_mark(self, analysis_id: str, observation_id: int,
                 kind: str) -> None:
        assert kind in ("locked", "excluded", "normal")
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO marks(analysis_id, observation_id, kind, "
                "updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(analysis_id, observation_id) DO UPDATE SET "
                "kind=excluded.kind, updated_at=excluded.updated_at",
                (analysis_id, observation_id, kind, _now()))

    def marks(self, analysis_id: str) -> Dict[int, str]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT observation_id, kind FROM marks WHERE analysis_id=?",
                (analysis_id,)).fetchall()
            return {r["observation_id"]: r["kind"] for r in rows}

    # -- 标定 -------------------------------------------------------------
    def replace_calibration_points(self, analysis_id: str,
                                   points: List[Dict[str, Any]]) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "DELETE FROM calibration_points WHERE analysis_id=?",
                (analysis_id,))
            for p in points:
                conn.execute(
                    "INSERT INTO calibration_points(analysis_id, point_id, "
                    "raw_json, ref_json, cov_json, excluded) "
                    "VALUES (?,?,?,?,?,?)",
                    (analysis_id, p.get("id"), self.j(p["raw"]),
                     self.j(p["ref"]),
                     self.j(p.get("cov", [[1.0, 0.0], [0.0, 1.0]])),
                     1 if p.get("excluded") else 0))

    def calibration_points(self, analysis_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM calibration_points WHERE analysis_id=? "
                "ORDER BY id", (analysis_id,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["raw"] = self.d(d.pop("raw_json"))
                d["ref"] = self.d(d.pop("ref_json"))
                d["cov"] = self.d(d.pop("cov_json"))
                d["excluded"] = bool(d["excluded"])
                out.append(d)
            return out

    def add_calibration_version(self, analysis_id: str, version: int,
                                fit: Dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO calibration_versions(analysis_id, version, "
                "created_at, A_json, t_json, rms, n_points) "
                "VALUES (?,?,?,?,?,?,?)",
                (analysis_id, version, _now(), self.j(fit["A"]),
                 self.j(fit["t"]), fit.get("rms"), fit.get("n_points", 0)))
            conn.execute(
                "UPDATE analyses SET calibration_version=? WHERE id=?",
                (version, analysis_id))

    def calibration_version(self, analysis_id: str,
                            version: Optional[int] = None
                            ) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            if version is None:
                row = conn.execute(
                    "SELECT * FROM calibration_versions WHERE analysis_id=? "
                    "ORDER BY version DESC LIMIT 1", (analysis_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM calibration_versions WHERE analysis_id=? "
                    "AND version=?", (analysis_id, version)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["A"] = self.d(d.pop("A_json"))
            d["t"] = self.d(d.pop("t_json"))
            return d

    # -- 候选与反射对应 ---------------------------------------------------
    def next_run_id(self, analysis_id: str) -> int:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(run_id),0)+1 AS n FROM candidates "
                "WHERE analysis_id=?", (analysis_id,)).fetchone()
            return int(row["n"])

    def insert_candidate(self, analysis_id: str, run_id: int, rank: int,
                         family_seq: Optional[int], basis: Any,
                         reduction: Dict[str, Any], score: Dict[str, Any],
                         extinction: Dict[str, Any], digest: str,
                         singular: bool = False,
                         singular_json: Optional[str] = None) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO candidates(analysis_id, run_id, rank, "
                "family_seq, basis_json, reduction_json, score_json, "
                "extinction_json, status, digest, singular, singular_json) "
                "VALUES (?,?,?,?,?,?,?,?,'active',?,?,?)",
                (analysis_id, run_id, rank, family_seq,
                 self.j(np_tolist(basis)), self.j(reduction),
                 self.j(score), self.j(extinction), digest,
                 1 if singular else 0, singular_json))
            return cur.lastrowid

    def insert_reflection(self, candidate_id: int, observation_id: int,
                          hkl: Any, pred: Any, residual: Any,
                          raw_dist: float, mahal: float, role: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO candidate_reflections(candidate_id, "
                "observation_id, hkl_json, pred_json, residual_json, "
                "raw_dist, mahal, role) VALUES (?,?,?,?,?,?,?,?)",
                (candidate_id, observation_id, self.j(np_tolist(hkl)),
                 self.j(np_tolist(pred)), self.j(np_tolist(residual)),
                 float(raw_dist), float(mahal), role))

    def mark_stale(self, analysis_id: str, digest: str, reason: str) -> int:
        """把所有依赖摘要不匹配的活动候选标记为 stale（局部过期）。"""
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE candidates SET status='stale', stale_reason=? "
                "WHERE analysis_id=? AND status='active' AND digest<>?",
                (reason, analysis_id, digest))
            return cur.rowcount

    def list_candidates(self, analysis_id: str, run_id: Optional[int] = None
                        ) -> List[Dict[str, Any]]:
        sql = ("SELECT * FROM candidates WHERE analysis_id=? AND singular=0")
        args: List[Any] = [analysis_id]
        if run_id is not None:
            sql += " AND run_id=?"
            args.append(run_id)
        sql += " ORDER BY run_id DESC, rank"
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["basis"] = self.d(d.pop("basis_json"))
                d["reduction"] = self.d(d.pop("reduction_json"))
                d["score"] = self.d(d.pop("score_json"))
                d["extinction"] = self.d(d.pop("extinction_json"))
                out.append(d)
            return out

    def latest_run_id(self, analysis_id: str) -> Optional[int]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(run_id) AS m FROM candidates WHERE analysis_id=? "
                "AND singular=0", (analysis_id,)).fetchone()
            return None if row["m"] is None else int(row["m"])

    def get_candidate(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            r = conn.execute("SELECT * FROM candidates WHERE id=?",
                             (candidate_id,)).fetchone()
            if r is None:
                return None
            d = dict(r)
            d["basis"] = self.d(d.pop("basis_json"))
            d["reduction"] = self.d(d.pop("reduction_json"))
            d["score"] = self.d(d.pop("score_json"))
            d["extinction"] = self.d(d.pop("extinction_json"))
            return d

    def candidate_reflections(self, candidate_id: int) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT cr.*, o.external_id, o.x_raw, o.y_raw, o.intensity, "
                "o.cov_json, o.raw_json FROM candidate_reflections cr "
                "JOIN observations o ON o.id=cr.observation_id "
                "WHERE cr.candidate_id=? ORDER BY cr.id",
                (candidate_id,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                for k in ("hkl_json", "pred_json", "residual_json",
                          "cov_json", "raw_json"):
                    d[k.replace("_json", "")] = self.d(d.pop(k))
                out.append(d)
            return out

    def singular_candidates(self, analysis_id: str, run_id: int
                            ) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, rank, singular_json FROM candidates "
                "WHERE analysis_id=? AND run_id=? AND singular=1 ORDER BY rank",
                (analysis_id, run_id)).fetchall()
            return [{"id": r["id"], "rank": r["rank"],
                     **(self.d(r["singular_json"]) if r["singular_json"]
                        else {})} for r in rows]

    # -- 冻结 -------------------------------------------------------------
    def add_freeze(self, analysis_id: str, calibration_version: int,
                   settings: Dict[str, Any], marks: Dict[int, str],
                   digest: str, client_digest: Optional[str]) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO freezes(analysis_id, created_at, "
                "calibration_version, settings_json, marks_json, digest, "
                "client_digest) VALUES (?,?,?,?,?,?,?)",
                (analysis_id, _now(), calibration_version,
                 self.j(settings),
                 self.j({str(k): v for k, v in marks.items()}),
                 digest, client_digest))
            return cur.lastrowid

    def latest_freeze(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            r = conn.execute(
                "SELECT * FROM freezes WHERE analysis_id=? ORDER BY id DESC "
                "LIMIT 1", (analysis_id,)).fetchone()
            if r is None:
                return None
            d = dict(r)
            d["settings"] = self.d(d.pop("settings_json"))
            d["marks"] = self.d(d.pop("marks_json"))
            return d


def np_tolist(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    return obj
