"""编排层：把数值核心、标定与 SQLite 串成“分析”的完整生命周期。"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import numeric
from .calibration import (apply_calibration, fit_affine_calibration,
                          identity_calibration)
from .db import Database

DEFAULT_SETTINGS: Dict[str, Any] = {
    "tol_mahal": numeric.TOL_MAHALANOBIS,
    "outlier_budget": 3,
    "outlier_span": numeric.MAHAL_OUTLIER_SPAN,
    "min_inliers": numeric.MIN_INLIERS,
}

ROLE_INLIER = "inlier"
ROLE_OUTLIER = "outlier"
ROLE_UNINDEXED = "unindexed"
ROLE_EXCLUDED = "excluded"


def _canon(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=_default)


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def compute_digest(instrument_version: str, settings: Dict[str, Any],
                   calib_version: int, marks: Dict[int, str],
                   calib_points: List[Dict[str, Any]],
                   observations: List[Dict[str, Any]]) -> str:
    """
    候选依赖摘要：仪器版本 + 容差/预算设置 + 标定版本 + 人工标记
    + 标定参考点 + 原始观测坐标/协方差。
    """
    payload = {
        "instrument_version": instrument_version,
        "settings": {k: round(float(v), 10) if isinstance(v, float) else v
                     for k, v in sorted(settings.items())},
        "calib_version": calib_version,
        "marks": {str(k): marks[k] for k in sorted(marks)},
        "calib_points": [
            {"raw": [round(float(x), 9) for x in p["raw"]],
             "ref": [round(float(x), 9) for x in p["ref"]],
             "excluded": bool(p.get("excluded", False))}
            for p in sorted(calib_points, key=lambda p: p.get("id", ""))],
        "observations": [
            {"id": o["id"],
             "x_raw": round(float(o["x_raw"]), 9),
             "y_raw": round(float(o["y_raw"]), 9),
             "cov": [[round(float(v), 12) for v in row]
                     for row in o["cov"]]}
            for o in sorted(observations, key=lambda o: o["id"])],
    }
    return hashlib.sha256(_canon(payload).encode("utf-8")).hexdigest()[:16]


class LabService:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    def create_analysis(self, payload: Dict[str, Any],
                        analysis_id: Optional[str] = None,
                        parent_id: Optional[str] = None) -> str:
        analysis_id = analysis_id or ("an_" + uuid.uuid4().hex[:10])
        settings = dict(DEFAULT_SETTINGS)
        settings.update(payload.get("settings", {}))
        self.db.create_analysis(
            analysis_id, payload.get("instrument_version", "unknown"),
            settings, parent_id, digest="")
        self.db.add_observations(analysis_id, payload["observations"])
        self.db.replace_calibration_points(
            analysis_id, payload.get("calibration_points", []))
        fit = fit_affine_calibration(payload.get("calibration_points", []))
        self.db.add_calibration_version(analysis_id, 1, fit)
        digest = self._digest(analysis_id)
        self.db.update_settings(
            analysis_id, self.db.get_analysis(analysis_id)["settings"],
            digest)
        self.db.event(analysis_id, "create",
                      {"parent_id": parent_id,
                       "n_observations": len(payload["observations"])})
        return analysis_id

    def _digest(self, analysis_id: str) -> str:
        an = self.db.get_analysis(analysis_id)
        return compute_digest(
            an["instrument_version"], an["settings"],
            an["calibration_version"], self.db.marks(analysis_id),
            self.db.calibration_points(analysis_id),
            self.db.list_observations(analysis_id))

    def _prepared(self, analysis_id: str
                  ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray, np.ndarray, str]:
        an = self.db.get_analysis(analysis_id)
        obs = self.db.list_observations(analysis_id)
        marks = self.db.marks(analysis_id)
        calib = self.db.calibration_version(analysis_id)
        q_raw = np.array([[o["x_raw"], o["y_raw"]] for o in obs], dtype=float)
        covs = np.array([o["cov"] for o in obs], dtype=float)
        q = apply_calibration(q_raw, calib)
        locked = np.array([marks.get(o["id"]) == "locked" for o in obs])
        excluded = np.array(
            [marks.get(o["id"]) == "excluded" for o in obs])
        digest = self._digest(analysis_id)
        return an, q, q_raw, covs, locked, excluded, digest
