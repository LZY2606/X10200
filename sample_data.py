"""固定合成数据集：双晶域 + 系统消光 + 峰重叠 + 标定偏差 + 离群峰。

使用固定随机种子，保证测试与演示可复现。
"""
from __future__ import annotations

import numpy as np

CALIB_VERSION = "cal-2026-09a"
TRUE_BIAS = np.array([0.0020, -0.0012, 0.0015])


def _domain_reflections(B, max_hkl, q_max, systematic=None):
    B = np.asarray(B, dtype=float)
    out = []
    for h in range(-max_hkl, max_hkl + 1):
        for k in range(-max_hkl, max_hkl + 1):
            for l in range(-max_hkl, max_hkl + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if systematic == "bcc" and (h + k + l) % 2 != 0:
                    continue  # 系统消光：体心格子 h+k+l 为奇数时缺失
                q = B @ np.array([h, k, l], dtype=float)
                if 0.05 < np.linalg.norm(q) < q_max:
                    out.append((h, k, l, q))
    return out


def build_sample(seed: int = 20260919):
    rng = np.random.default_rng(seed)
    # 晶域 1：近立方，带系统消光
    B1 = np.array([[0.250, 0.004, 0.000],
                   [0.000, 0.248, 0.003],
                   [0.002, 0.000, 0.252]])
    # 晶域 2：六方状，旋转过
    ang = np.deg2rad(37.0)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                   [np.sin(ang), np.cos(ang), 0.0],
                   [0.0, 0.0, 1.0]])
    B2 = Rz @ np.array([[0.180, -0.090, 0.000],
                        [0.000, 0.156, 0.000],
                        [0.000, 0.000, 0.210]])
    peaks = []
    for domain, (B, max_hkl, q_max, sys) in enumerate(
            [(B1, 3, 0.95, "bcc"), (B2, 2, 0.75, None)]):
        for h, k, l, q in _domain_reflections(B, max_hkl, q_max, sys):
            sigma = 0.0012 * (1.0 + 0.4 * np.linalg.norm(q))
            cov = np.eye(3) * sigma ** 2
            cov[0, 0] *= 1.8  # 各向异性协方差
            cov[2, 2] *= 0.6
            noise = rng.multivariate_normal(np.zeros(3), cov)
            q_meas = q + noise + TRUE_BIAS  # 仪器标定偏差混入观测
            intensity = 1200.0 / (1.0 + h * h + k * k + l * l) + rng.normal(0, 8)
            peaks.append({
                "q": q_meas.tolist(),
                "intensity": float(max(intensity, 1.0)),
                "cov": cov[np.triu_indices(3)].tolist(),
                "domain": domain, "true_hkl": [h, k, l],
            })
    # 峰重叠：与晶域1的 (1,0,0) 几乎重合的第二个观测
    q_ov = B1 @ np.array([1.0, 0.0, 0.0]) + TRUE_BIAS
    peaks.append({"q": q_ov.tolist(), "intensity": 640.0,
                  "cov": (np.eye(3) * 0.0012 ** 2)[np.triu_indices(3)].tolist(),
                  "domain": 0, "true_hkl": [1, 0, 0]})
    # 离群峰（不属于任何晶域）
    for v in [[0.41, 0.13, 0.07], [0.09, 0.44, 0.21], [0.23, 0.05, 0.47]]:
        peaks.append({"q": (np.array(v) + TRUE_BIAS).tolist(), "intensity": 55.0,
                      "cov": (np.eye(3) * 0.0015 ** 2)[np.triu_indices(3)].tolist(),
                      "domain": -1, "true_hkl": None})
    return {
        "calibration_version": CALIB_VERSION,
        "bias": TRUE_BIAS.tolist(),
        "peaks": peaks,
    }


def cov6_to_matrix(cov6):
    c = np.asarray(cov6, dtype=float)
    i, j = np.triu_indices(3)
    C = np.zeros((3, 3))
    C[i, j] = c
    C[j, i] = c
    return C
