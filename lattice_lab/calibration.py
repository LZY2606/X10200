"""仪器标定版本与加权仿射标定拟合。"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def fit_affine_calibration(points: Sequence[Dict[str, Any]]
                           ) -> Dict[str, Any]:
    """
    由标定参考点 (raw -> ref) 做加权最小二乘仿射拟合：
        ref ≈ A @ raw + t
    权重 w = 1 / tr(Sigma)。点少于 3 时退回单位变换。
    返回 2x2 的 A 与平移 t、残差 RMS、参与点数。
    """
    pts = [p for p in points if not p.get("excluded", False)]
    if len(pts) < 3:
        return {
            "A": np.eye(2).tolist(),
            "t": [0.0, 0.0],
            "rms": None,
            "n_points": len(pts),
            "underdetermined": True,
        }
    n = len(pts)
    X = np.array([p["raw"] for p in pts], dtype=float)
    Y = np.array([p["ref"] for p in pts], dtype=float)
    w = np.array([1.0 / max(float(np.trace(np.asarray(p.get(
        "cov", [[1.0, 0.0], [0.0, 1.0]]), dtype=float))), 1.0e-12)
        for p in pts], dtype=float)
    w /= np.sum(w)
    W = np.diag(w)
    M = np.hstack([X, np.ones((n, 1))])
    # 加权最小二乘：(M' W M) θ = M' W Y
    MtWM = M.T @ W @ M + np.eye(3) * 1.0e-14
    MtWY = M.T @ W @ Y
    theta = np.linalg.solve(MtWM, MtWY)
    A = theta[:2, :].T          # y = A x + t
    t = theta[2, :]
    pred = (A @ X.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - Y) ** 2, axis=1))))
    return {
        "A": A.tolist(),
        "t": t.tolist(),
        "rms": rms,
        "n_points": n,
        "underdetermined": False,
    }


def apply_calibration(q_raw: np.ndarray, calib: Dict[str, Any]) -> np.ndarray:
    A = np.asarray(calib["A"], dtype=float)
    t = np.asarray(calib["t"], dtype=float)
    return (A @ np.asarray(q_raw, dtype=float).T).T + t


def identity_calibration(version: int = 1) -> Dict[str, Any]:
    return {"version": version, "A": np.eye(2).tolist(),
            "t": [0.0, 0.0], "rms": None, "n_points": 0,
            "underdetermined": True, "points": []}
