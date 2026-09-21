"""
固定（无随机数）合成衍射数据（探测器 2D 平面上的倒数网格）。

  - 晶域 A：C 心网（h+k 为偶）；另确定性删除一部分允许反射 = 缺失反射。
  - 晶域 B：整体旋转 ~18° 的原始斜格，与 A 共存（多晶域）。
  - 峰重叠、逐峰 2x2 测量协方差、三个离群峰、轻微已知标定偏差。
观测保存 raw（未标定）坐标；true_* 仅供测试断言，不参与算法。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

TRUE_A = [[1.012, 0.006], [-0.004, 0.994]]
TRUE_T = [0.018, -0.011]

BASIS_A = np.array([[0.100, 0.000], [0.000, 0.082]], dtype=float)
ANGLE_B = math.radians(18.0)
ROT2 = np.array([[math.cos(ANGLE_B), -math.sin(ANGLE_B)],
                 [math.sin(ANGLE_B), math.cos(ANGLE_B)]], dtype=float)
BASIS_B = ROT2 @ np.array([[0.092, 0.012], [0.000, 0.086]], dtype=float)

QCUT = 0.30
MISSING_FRACTION = 0.18
NOISE_TABLE = (0.0040, 0.0055, 0.0070, 0.0062, 0.0048)
SHEAR_COV = 0.45


def _shell_hkls(radius_sq_max: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    lim = int(math.ceil(math.sqrt(radius_sq_max))) + 1
    for h in range(-lim, lim + 1):
        for k in range(-lim, lim + 1):
            if h == 0 and k == 0:
                continue
            if h * h + k * k <= radius_sq_max:
                out.append((h, k))
    out.sort(key=lambda t: (t[0] * t[0] + t[1] * t[1], t[0], t[1]))
    return out


def _noise(i: int) -> Tuple[float, float]:
    a = NOISE_TABLE[i % len(NOISE_TABLE)]
    b = NOISE_TABLE[(i * 3 + 1) % len(NOISE_TABLE)]
    return (a if i % 2 == 0 else -a), (b if i % 3 != 0 else -b)


def make_dataset() -> Dict[str, Any]:
    hkls_a = _shell_hkls(int((QCUT / 0.082) ** 2))
    hkls_b = _shell_hkls(int((QCUT / 0.078) ** 2))

    observations: List[Dict[str, Any]] = []
    intensity_seq = (980.0, 760.0, 540.0, 430.0, 610.0, 350.0)
    seq = 0

    def push(q_corr: np.ndarray, domain: str, hkl: Tuple[int, int],
             kind: str, shift: Tuple[float, float] = (0.0, 0.0)) -> None:
        nonlocal seq
        ainv = np.linalg.inv(np.asarray(TRUE_A))
        q_raw = ainv @ (q_corr + np.asarray(shift) - np.asarray(TRUE_T))
        sx, sy = _noise(seq)
        cov = [[sx * sx, SHEAR_COV * sx * sy],
               [SHEAR_COV * sx * sy, sy * sy]]
        observations.append({
            "external_id": "P%03d" % (seq + 1),
            "x_raw": float(q_raw[0]),
            "y_raw": float(q_raw[1]),
            "x_corr_true": float(q_corr[0]),
            "y_corr_true": float(q_corr[1]),
            "intensity": intensity_seq[seq % len(intensity_seq)],
            "cov": cov,
            "domain_true": domain,
            "hkl_true": list(hkl),
            "kind_true": kind,
        })
        seq += 1

    # 晶域 A：C 心（h+k 为偶），再确定性删除允许反射 = 缺失反射
    missing_a = 0
    a_index = 0
    for (h, k) in hkls_a:
        if (h + k) % 2 != 0:
            continue
        q = BASIS_A @ np.array([h, k], dtype=float)
        if float(np.linalg.norm(q)) > QCUT:
            continue
        if (a_index * 7) % 100 < int(MISSING_FRACTION * 100):
            missing_a += 1
            a_index += 1
            continue
        kind, shift = "inlier", (0.0, 0.0)
        if a_index in (6, 21):
            kind, shift = "outlier", (
                0.075 * (1 if a_index % 2 == 0 else -1), 0.060)
        push(q, "A", (h, k), kind, shift)
        a_index += 1

    # 晶域 B：旋转斜基
    for i, (h, k) in enumerate(hkls_b):
        q = BASIS_B @ np.array([h, k], dtype=float)
        if float(np.linalg.norm(q)) > QCUT:
            continue
        kind, shift = "inlier", (0.0, 0.0)
        if i == 9:
            kind, shift = "outlier", (-0.065, 0.085)
        push(q, "B", (h, k), kind, shift)

    # 峰重叠：B 域一个峰落在第 5 个 A 域观测附近
    t = observations[4]
    push(np.array([t["x_corr_true"], t["y_corr_true"]], dtype=float),
         "B", (1, 1), "overlap", shift=(0.011, -0.009))

    refs = [
        ([-0.30, -0.24], [-0.287, -0.251]),
        ([0.26, -0.20], [0.283, -0.208]),
        ([-0.18, 0.22], [-0.166, 0.206]),
        ([0.24, 0.26], [0.263, 0.245]),
    ]
    points = []
    for j, (raw, ref) in enumerate(refs):
        s = 0.003 + 0.0005 * j
        points.append({"id": "CP%d" % (j + 1), "raw": raw, "ref": ref,
                       "cov": [[s * s, 0.0], [0.0, s * s]],
                       "excluded": False})

    return {
        "instrument_version": "LAB-DIFF-2026.09",
        "observations": observations,
        "calibration_points": points,
        "true_transform": {"A": TRUE_A, "t": TRUE_T,
                           "basis_a": BASIS_A.tolist(),
                           "basis_b": BASIS_B.tolist()},
        "meta": {"n_missing_a": missing_a, "qcut": QCUT},
    }
