"""SQLite persistence for the lattice candidate room.

Stores raw observations, analyses, candidates (with full transform chains),
per-reflection assignments, researcher marks and freezes. Partial
invalidation: marking a reflection expires only candidates that depend on
it; a calibration change expires all candidates of the analysis.
"""
from __future__ import annotations

import json
import sqlite3
import time

import numpy as np

from lattice.families import group_families
from lattice.indexing import generate_candidates

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qx REAL NOT NULL, qy REAL NOT NULL, qz REAL NOT NULL,
    intensity REAL NOT NULL,
    cov_json TEXT NOT NULL,
    calibration_version TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    parent_id INTEGER,
    calibration_json TEXT NOT NULL,
    tolerances_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    family_id INTEGER NOT NULL,
    seed_peaks_json TEXT NOT NULL,
    seed_basis_json TEXT NOT NULL,
    basis_json TEXT NOT NULL,
    reduced_basis_json TEXT NOT NULL,
    transform_chain_json TEXT NOT NULL,
    refinement_json TEXT NOT NULL,
    metric_json TEXT NOT NULL,
    transform_to_family_json TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    valid INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    observation_id INTEGER NOT NULL,
    h INTEGER NOT NULL, k INTEGER NOT NULL, l INTEGER NOT NULL,
    pred_json TEXT NOT NULL,
    residual_json TEXT NOT NULL,
    weighted_residual REAL NOT NULL,
    status TEXT NOT NULL,
    is_outlier INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS marks (
    analysis_id INTEGER NOT NULL,
    observation_id INTEGER NOT NULL,
    locked INTEGER NOT NULL DEFAULT 0,
    excluded INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (analysis_id, observation_id)
);
CREATE TABLE IF NOT EXISTS freezes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    analysis_version INTEGER NOT NULL,
    calibration_json TEXT NOT NULL,
    tolerances_json TEXT NOT NULL,
    marks_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class ConflictError(Exception):
    """Raised when a freeze targets a stale analysis version."""


class Store:
    def __init__(self, path: str = "lattice.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- observations ----------------------------------------------------
    def import_observations(self, peaks, calibration_version: str) -> int:
        cur = self.conn.cursor()
        for p in peaks:
            cov = p.get("cov") or (np.eye(3) * 1e-4).tolist()
            raw = {"q": p["q"], "intensity": p.get("intensity", 1.0),
                   "cov": cov, "calibration_version": calibration_version}
            cur.execute(
                "INSERT INTO observations (qx,qy,qz,intensity,cov_json,"
                "calibration_version,raw_json) VALUES (?,?,?,?,?,?,?)",
                (p["q"][0], p["q"][1], p["q"][2], p.get("intensity", 1.0),
                 json.dumps(cov), calibration_version, json.dumps(raw)))
        self.conn.commit()
        return cur.lastrowid or 0

    def list_observations(self):
        rows = self.conn.execute(
            "SELECT * FROM observations ORDER BY id").fetchall()
        return [self._obs_dict(r) for r in rows]

    @staticmethod
    def _obs_dict(r):
        return {"id": r["id"], "q": [r["qx"], r["qy"], r["qz"]],
                "intensity": r["intensity"], "cov": json.loads(r["cov_json"]),
                "calibration_version": r["calibration_version"],
                "raw": json.loads(r["raw_json"])}

    # -- analyses --------------------------------------------------------
    def create_analysis(self, name, calibration=None, tolerances=None,
                        parent_id=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO analyses (name,parent_id,calibration_json,"
            "tolerances_json,version,created_at) VALUES (?,?,?,?,1,?)",
            (name, parent_id, json.dumps(calibration or {}),
             json.dumps(tolerances or {}), time.time()))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_analysis_row(self, analysis_id: int):
        row = self.conn.execute(
            "SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        if row is None:
            raise KeyError(f"analysis {analysis_id} not found")
        return row

    def list_analyses(self):
        rows = self.conn.execute(
            "SELECT * FROM analyses ORDER BY id").fetchall()
        return [{"id": r["id"], "name": r["name"], "parent_id": r["parent_id"],
                 "calibration": json.loads(r["calibration_json"]),
                 "tolerances": json.loads(r["tolerances_json"]),
                 "version": r["version"]} for r in rows]

    def get_marks(self, analysis_id: int):
        rows = self.conn.execute(
            "SELECT * FROM marks WHERE analysis_id=?", (analysis_id,)).fetchall()
        return {r["observation_id"]: {"locked": bool(r["locked"]),
                                      "excluded": bool(r["excluded"])}
                for r in rows}

    def run_analysis(self, analysis_id: int) -> dict:
        """Generate candidates from non-excluded observations and persist."""
        row = self.get_analysis_row(analysis_id)
        calibration = json.loads(row["calibration_json"])
        tolerances = json.loads(row["tolerances_json"])
        marks = self.get_marks(analysis_id)
        obs = [o for o in self.list_observations()
               if not marks.get(o["id"], {}).get("excluded")]
        locked_ids = {oid for oid, m in marks.items() if m.get("locked")}
        qs = np.array([o["q"] for o in obs])
        covs = np.array([o["cov"] for o in obs])
        candidates = generate_candidates(qs, covs, calibration, tolerances)
        families, assignments = group_families(candidates)
        cur = self.conn.cursor()
        for cand, assign in zip(candidates, assignments):
            # Locked reflections are protected from the outlier flag.
            for refl in cand["reflections"]:
                obs_id = obs[refl["peak_index"]]["id"]
                if obs_id in locked_ids and refl["is_outlier"]:
                    refl["is_outlier"] = False
                    refl["status"] = "locked"
            cur.execute(
                "INSERT INTO candidates (analysis_id,family_id,"
                "seed_peaks_json,seed_basis_json,basis_json,"
                "reduced_basis_json,transform_chain_json,refinement_json,"
                "metric_json,transform_to_family_json,scores_json,valid)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,1)",
                (analysis_id, assign["family_id"],
                 json.dumps(cand["seed_peaks"]), json.dumps(cand["seed_basis"]),
                 json.dumps(cand["basis"]), json.dumps(cand["reduced_basis"]),
                 json.dumps(cand["transform_chain"]),
                 json.dumps(cand["refinement_steps"]), json.dumps(cand["metric"]),
                 json.dumps(assign["transform_to_family"]),
                 json.dumps(cand["scores"])))
            cand_id = cur.lastrowid
            for refl in cand["reflections"]:
                cur.execute(
                    "INSERT INTO reflections (candidate_id,observation_id,h,k,l,"
                    "pred_json,residual_json,weighted_residual,status,is_outlier)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (cand_id, obs[refl["peak_index"]]["id"],
                     *refl["hkl"], json.dumps(refl["predicted_q"]),
                     json.dumps(refl["residual"]), refl["weighted_residual"],
                     refl["status"], int(refl["is_outlier"])))
        self.conn.commit()
        return {"families": families, "n_candidates": len(candidates)}

    def get_analysis_state(self, analysis_id: int) -> dict:
        row = self.get_analysis_row(analysis_id)
        cands = self.conn.execute(
            "SELECT * FROM candidates WHERE analysis_id=? ORDER BY id",
            (analysis_id,)).fetchall()
        families = {}
        out_cands = []
        for c in cands:
            refl_rows = self.conn.execute(
                "SELECT * FROM reflections WHERE candidate_id=? ORDER BY id",
                (c["id"],)).fetchall()
            reflections = [{
                "id": r["id"], "observation_id": r["observation_id"],
                "hkl": [r["h"], r["k"], r["l"]],
                "predicted_q": json.loads(r["pred_json"]),
                "residual": json.loads(r["residual_json"]),
                "weighted_residual": r["weighted_residual"],
                "status": r["status"], "is_outlier": bool(r["is_outlier"])}
                for r in refl_rows]
            out_cands.append({
                "id": c["id"], "family_id": c["family_id"],
                "valid": bool(c["valid"]),
                "seed_peaks": json.loads(c["seed_peaks_json"]),
                "seed_basis": json.loads(c["seed_basis_json"]),
                "basis": json.loads(c["basis_json"]),
                "reduced_basis": json.loads(c["reduced_basis_json"]),
                "transform_chain": json.loads(c["transform_chain_json"]),
                "refinement_steps": json.loads(c["refinement_json"]),
                "metric": json.loads(c["metric_json"]),
                "transform_to_family": json.loads(c["transform_to_family_json"]),
                "scores": json.loads(c["scores_json"]),
                "reflections": reflections})
            fam = families.setdefault(c["family_id"], {
                "id": c["family_id"], "canonical_basis": None,
                "metric": json.loads(c["metric_json"]), "candidates": []})
            fam["candidates"].append(c["id"])
        for c in out_cands:
            fam = families[c["family_id"]]
            if fam["canonical_basis"] is None:
                fam["canonical_basis"] = c["reduced_basis"]
        return {"id": row["id"], "name": row["name"],
                "parent_id": row["parent_id"], "version": row["version"],
                "calibration": json.loads(row["calibration_json"]),
                "tolerances": json.loads(row["tolerances_json"]),
                "marks": {str(k): v for k, v in self.get_marks(analysis_id).items()},
                "families": sorted(families.values(), key=lambda f: f["id"]),
                "candidates": out_cands}

    def get_candidate(self, candidate_id: int) -> dict:
        row = self.conn.execute(
            "SELECT analysis_id FROM candidates WHERE id=?",
            (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(f"candidate {candidate_id} not found")
        state = self.get_analysis_state(row["analysis_id"])
        for cand in state["candidates"]:
            if cand["id"] == candidate_id:
                cand["analysis_id"] = row["analysis_id"]
                return cand
        raise KeyError(f"candidate {candidate_id} not found")

    # -- marks / calibration / invalidation ------------------------------
    def set_mark(self, analysis_id: int, observation_id: int,
                 locked=None, excluded=None) -> dict:
        self.get_analysis_row(analysis_id)
        marks = self.get_marks(analysis_id)
        current = marks.get(observation_id, {"locked": False, "excluded": False})
        new_locked = current["locked"] if locked is None else bool(locked)
        new_excluded = current["excluded"] if excluded is None else bool(excluded)
        self.conn.execute(
            "INSERT INTO marks (analysis_id,observation_id,locked,excluded)"
            " VALUES (?,?,?,?) ON CONFLICT(analysis_id,observation_id)"
            " DO UPDATE SET locked=excluded.locked, excluded=excluded.excluded",
            (analysis_id, observation_id, int(new_locked), int(new_excluded)))
        # Partial invalidation: only candidates referencing this observation.
        self.conn.execute(
            "UPDATE candidates SET valid=0 WHERE analysis_id=? AND id IN"
            " (SELECT DISTINCT candidate_id FROM reflections"
            "  WHERE observation_id=? AND status != 'unindexed')",
            (analysis_id, observation_id))
        self._bump_version(analysis_id)
        self.conn.commit()
        return {"observation_id": observation_id, "locked": new_locked,
                "excluded": new_excluded}

    def update_calibration(self, analysis_id: int, calibration: dict) -> None:
        self.get_analysis_row(analysis_id)
        self.conn.execute(
            "UPDATE analyses SET calibration_json=? WHERE id=?",
            (json.dumps(calibration), analysis_id))
        # Calibration shifts every predicted position: expire all candidates.
        self.conn.execute(
            "UPDATE candidates SET valid=0 WHERE analysis_id=?", (analysis_id,))
        self._bump_version(analysis_id)
        self.conn.commit()

    def _bump_version(self, analysis_id: int) -> None:
        self.conn.execute(
            "UPDATE analyses SET version=version+1 WHERE id=?", (analysis_id,))

    # -- fork / freeze ---------------------------------------------------
    def fork_analysis(self, analysis_id: int, name: str = None) -> int:
        row = self.get_analysis_row(analysis_id)
        new_id = self.create_analysis(
            name or f"{row['name']}-fork",
            calibration=json.loads(row["calibration_json"]),
            tolerances=json.loads(row["tolerances_json"]),
            parent_id=analysis_id)
        for oid, mark in self.get_marks(analysis_id).items():
            self.conn.execute(
                "INSERT INTO marks (analysis_id,observation_id,locked,excluded)"
                " VALUES (?,?,?,?)",
                (new_id, oid, int(mark["locked"]), int(mark["excluded"])))
        self.conn.commit()
        self.run_analysis(new_id)
        return new_id

    def freeze(self, analysis_id: int, expected_version: int) -> dict:
        row = self.get_analysis_row(analysis_id)
        if int(expected_version) != row["version"]:
            raise ConflictError(
                f"stale draft: expected version {expected_version}, "
                f"current version {row['version']}")
        marks = self.get_marks(analysis_id)
        cur = self.conn.execute(
            "INSERT INTO freezes (analysis_id,analysis_version,"
            "calibration_json,tolerances_json,marks_json,created_at)"
            " VALUES (?,?,?,?,?,?)",
            (analysis_id, row["version"], row["calibration_json"],
             row["tolerances_json"],
             json.dumps({str(k): v for k, v in marks.items()}), time.time()))
        self.conn.commit()
        return {"freeze_id": int(cur.lastrowid), "analysis_id": analysis_id,
                "analysis_version": row["version"],
                "calibration": json.loads(row["calibration_json"]),
                "tolerances": json.loads(row["tolerances_json"]),
                "marks": marks}

    def list_freezes(self, analysis_id: int):
        rows = self.conn.execute(
            "SELECT * FROM freezes WHERE analysis_id=? ORDER BY id",
            (analysis_id,)).fetchall()
        return [{"freeze_id": r["id"], "analysis_version": r["analysis_version"],
                 "calibration": json.loads(r["calibration_json"]),
                 "tolerances": json.loads(r["tolerances_json"]),
                 "marks": json.loads(r["marks_json"]),
                 "created_at": r["created_at"]} for r in rows]
