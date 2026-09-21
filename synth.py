"""固定随机种子的合成晶格数据，用于演示与测试。"""
from __future__ import annotations

import itertools

import numpy as np


def hkls_range(n):
    return [h for h in itertools.product(range(-n, n + 1), repeat=3) if any(h)]


def make_peaks(B, hkls, noise=0.0, seed=0, cov_scale=1e-8):
    """由倒易基 B（列基）与 hkl 列表生成峰，强度随 |q| 衰减。"""
    rng = np.random.default_rng(seed)
    Q = np.array(hkls, dtype=float) @ np.asarray(B, dtype=float).T
    if noise:
        Q = Q + rng.normal(0.0, noise, Q.shape)
    peaks = []
    for q in Q:
        c = cov_scale
        peaks.append({
            "q": q.tolist(),
            "intensity": float(1000.0 / (1.0 + q @ q)),
            "cov": [c, 0.0, 0.0, c, 0.0, c],
        })
    return peaks


def make_outliers(n, seed=99, box=1.2, cov_scale=1e-8):
    rng = np.random.default_rng(seed)
    peaks = []
    for _ in range(n):
        q = rng.uniform(-box, box, 3)
        c = cov_scale
        peaks.append({
            "q": q.tolist(),
            "intensity": 5.0,
            "cov": [c, 0.0, 0.0, c, 0.0, c],
        })
    return peaks


def with_ids(peaks, start=1):
    out = []
    for i, p in enumerate(peaks):
        q = dict(p)
        q["id"] = start + i
        q.setdefault("excluded", False)
        q.setdefault("locked_hkl", None)
        out.append(q)
    return out
