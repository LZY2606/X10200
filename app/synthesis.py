"""固定合成衍射数据集：测试与演示共用，完全确定性。

包含：两个晶域、系统消光（缺失反射）、峰重叠、离群点、
常数标定偏差以及各向异性测量协方差。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


SYNTH_SEED = 20260919

# 两个晶域的正空间列基（埃）
DIRECT_A = np.array([
    [6.0, 0.3, 0.1],
    [0.0, 5.0, -0.2],
    [0.2, 0.0, 4.5],
])
DIRECT_B = np.array([
    [4.8, 0.1, 0.0],
    [0.2, 5.4, 0.3],
    [0.0, -0.1, 6.2],
])

# 固定的仪器标定偏差（常数平移）
TRUE_OFFSET = np.array([0.04, -0.03, 0.02])

# 固定的缺失反射（系统消光），(domain, h,k,l)
SYSTEMATIC_ABSENCES = {
    (0, 0, 0, 2),
    (0, 0, 0, 4),
    (0, 0, 2, 0),
    (1, 0, 0, 2),
}

# 固定离群点（索引计数内但明显偏离格点）
OUTLIER_SHIFTS = {
    3: np.array([0.35, 0.0, 0.0]),
    17: np.array([0.0, -0.40, 0.0]),
}


def _cov(rng: np.random.Generator, anisotropic: bool) -> np.ndarray:
    base = 0.008 ** 2
    if not anisotropic:
        return np.eye(3) * base
    axes = np.diag([1.0, 1.8, 0.7])
    angle = rng.uniform(0, 2 * np.pi)
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return base * rot @ axes @ rot.T


def generate_synthetic_dataset() -> Dict[str, Any]:
    """返回峰坐标（含标定偏差）、强度、协方差与校准版本元数据。"""
    rng = np.random.default_rng(SYNTH_SEED)
    points: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    intensities: List[float] = []
    truth: List[Dict[str, Any]] = []

    bases = [np.linalg.inv(DIRECT_A).T, np.linalg.inv(DIRECT_B).T]

    for domain, basis in enumerate(bases):
        for h in range(-2, 3):
            for k in range(-2, 3):
                for ell in range(-2, 3):
                    if h == 0 and k == 0 and ell == 0:
                        continue
                    if (domain, h, k, ell) in SYSTEMATIC_ABSENCES:
                        continue
                    ideal = basis @ np.array([h, k, ell], dtype=float)
                    cov = _cov(rng, anisotropic=(domain == 0))
                    noise = rng.multivariate_normal(np.zeros(3), cov)
                    points.append(ideal + noise + TRUE_OFFSET)
                    covariances.append(cov)
                    intensities.append(
                        float(1000.0 * np.exp(-(h*h + k*k + ell*ell) / 6.0)
                              + rng.uniform(5, 20)))
                    truth.append({"domain": domain, "hkl": [h, k, ell]})

    # 一个不属于任何晶域的离群点
    points.append(np.array([1.9, -2.1, 0.8]) + TRUE_OFFSET)
    covariances.append(np.eye(3) * 0.008 ** 2)
    intensities.append(42.0)
    truth.append({"domain": None, "hkl": None})

    # 固定的“域内但偏移”的离群点
    for idx, shift in OUTLIER_SHIFTS.items():
        points[idx] = points[idx] + shift

    # 峰重叠：取前 5 个峰与另外 5 个峰重合到测量精度内
    overlap_groups = [(4, 30), (9, 55)]
    overlap_ids = set()
    for a, b in overlap_groups:
        points[b] = points[a] + np.array([0.002, 0.0, 0.0])
        overlap_ids.add(a)
        overlap_ids.add(b)

    peaks = []
    for i, (pt, cov, inten) in enumerate(zip(points, covariances, intensities)):
        peaks.append({
            "peak_index": i,
            "point": [float(x) for x in pt],
            "intensity": float(inten),
            "covariance": [[float(v) for v in row] for row in cov],
            "is_overlap": i in overlap_ids,
            "truth": truth[i],
        })

    return {
        "calibration_version": {
            "id": "CAL-SYNTH-1",
            "label": "synthetic-instrument-v1",
            "offset": [0.0, 0.0, 0.0],
        },
        "true_offset": [float(x) for x in TRUE_OFFSET],
        "peaks": peaks,
    }
