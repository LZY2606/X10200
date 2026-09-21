"""Deterministic synthetic data for tests and the demo seed endpoint."""
from __future__ import annotations

import itertools

import numpy as np


def lattice_peaks(basis: np.ndarray, h_max: int = 3, extinction=None,
                  noise: float = 0.0, seed: int = 0):
    """Generate (qs, covs, hkls) for a reciprocal basis (columns)."""
    rng = np.random.default_rng(seed)
    qs, hkls = [], []
    for h in itertools.product(range(-h_max, h_max + 1), repeat=3):
        if h == (0, 0, 0):
            continue
        if extinction and extinction(h):
            continue
        q = basis @ np.array(h, dtype=float)
        q = q + rng.normal(0.0, noise, size=3) if noise else q
        qs.append(q)
        hkls.append(h)
    qs = np.array(qs)
    covs = np.repeat((1e-4 * np.eye(3))[None, :, :], len(qs), axis=0)
    return qs, covs, hkls


def overlap_pair(q: np.ndarray, delta: float = 0.02):
    """A peak plus a near-duplicate to emulate peak overlap."""
    return q + np.array([delta, -delta, delta])
