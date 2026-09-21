"""SQLite 持久层。

设计原则
--------
* 原始观测（raw point / 强度 / 协方差 / 校准版本）一旦导入永不修改、永不删除；
* 候选不删除，只在依赖失效时置为 ``expired``，重新生成时置为 ``superseded``；
* 所有写入经单一连接串行化，配合 WAL 与乐观版本号实现并发冲突检测。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_versions (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    offset_json TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    calibration_version_id TEXT NOT NULL,
    active_offset_json TEXT NOT NULL,
    index_tol REAL NOT NULL,
    outlier_budget INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    frozen_at_seq INTEGER,
    created_seq INTEGER NOT NULL,
    FOREIGN KEY (calibration_version_id) REFERENCES calibration_versions(id)
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    peak_index INTEGER NOT NULL,
    raw_point_json TEXT NOT NULL,
    calibrated_point_json TEXT NOT NULL,
    intensity REAL NOT NULL,
    covariance_json TEXT NOT NULL,
    calibration_version_id TEXT NOT NULL,
    is_overlap INTEGER NOT NULL DEFAULT 0,
    mark TEXT NOT NULL DEFAULT 'normal',
    created_seq INTEGER NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    seq_in_analysis INTEGER NOT NULL,
    basis_json TEXT NOT NULL,
    reduced_basis_json TEXT NOT NULL,
    domain_label INTEGER,
    family_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    indexed_count INTEGER NOT NULL,
    eligible_count INTEGER NOT NULL,
    outlier_count INTEGER NOT NULL,
    outlier_budget INTEGER NOT NULL,
    outlier_budget_exceeded INTEGER NOT NULL,
    weighted_residual REAL NOT NULL,
    complexity REAL NOT NULL,
    basis_condition REAL NOT NULL,
    anchor_peak_indexes_json TEXT NOT NULL,
    revision_created INTEGER NOT NULL,
    created_seq INTEGER NOT NULL,
    expired_seq INTEGER,
    expiry_reason TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS candidate_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    hkl_json TEXT NOT NULL,
    predicted_json TEXT NOT NULL,
    residual_json TEXT NOT NULL,
    residual_norm REAL NOT NULL,
    mahalanobis REAL NOT NULL,
    role TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id),
    FOREIGN KEY (observation_id) REFERENCES observations(id)
);

CREATE TABLE IF NOT EXISTS transforms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT,
    family_id TEXT,
    step_index INTEGER NOT NULL,
    operation TEXT NOT NULL,
    matrix_json TEXT NOT NULL,
    basis_before_json TEXT,
    basis_after_json TEXT,
    note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS families (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    representative_candidate_id TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS family_members (
    family_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    relative_transform_json TEXT NOT NULL,
    PRIMARY KEY (family_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS candidate_dependencies (
    candidate_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    is_anchor INTEGER NOT NULL,
    PRIMARY KEY (candidate_id, observation_id)
);

CREATE TABLE IF NOT EXISTS freezes (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_obs_analysis ON observations(analysis_id);
CREATE INDEX IF NOT EXISTS idx_cand_analysis ON candidates(analysis_id);
CREATE INDEX IF NOT EXISTS idx_refl_candidate ON candidate_reflections(candidate_id);
CREATE INDEX IF NOT EXISTS idx_dep_obs ON candidate_dependencies(observation_id);
CREATE INDEX IF NOT EXISTS idx_events_analysis ON event_log(analysis_id, seq);
"""


class Database:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = str(Path(path or config.DB_PATH))
        self._lock = threading.RLock()
        self._seq = 0
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

    # -- 基础工具 ----------------------------------------------------------
    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        return list(self._conn.execute(sql, params))

    def query_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchone()

    def log_event(self, analysis_id: str, kind: str,
                  payload: Optional[Dict[str, Any]] = None) -> None:
        seq = self.next_seq()
        self.execute(
            "INSERT INTO event_log(analysis_id, seq, kind, payload_json) "
            "VALUES (?,?,?,?)",
            (analysis_id, seq, kind,
             json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))),
        )

    def close(self) -> None:
        self._conn.close()
