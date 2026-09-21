"""Core lattice mathematics for the Lattice Candidate Chamber.

All tolerances are fixed module constants and every sort is stable so
that results are deterministic for a fixed input.
"""
from __future__ import annotations

import itertools

import numpy as np

INDEX_TOL = 0.18        # max fractional deviation of hkl from integers
SINGULAR_RTOL = 1e-6    # relative determinant threshold for basis triples
FAMILY_RTOL = 0.05      # relative tolerance on reduced-metric eigenvalues
TRANSFORM_TOL = 0.1     # max deviation of a basis transform from integers
REDUCE_MAX_ITER = 200
MAX_BASIS_PEAKS = 16
MAX_CANDIDATES = 60
NO_SCORE = 1e18


def correct_peaks(q, scale, offset):
    """Apply instrument calibration: q_corr = scale * (q - offset)."""
    return scale * (np.asarray(q, dtype=float) - np.asarray(offset, dtype=float))


def reduce_basis(basis, max_iter=REDUCE_MAX_ITER):
    """Greedy lattice reduction of a 3x3 basis (columns are b1, b2, b3).

    Returns (reduced_basis, transform, steps) where
    ``reduced_basis == basis @ transform`` and ``steps`` is the ordered
    list of integer unimodular matrices whose product is ``transform``.
    """
    b = np.array(basis, dtype=float, copy=True)
    m = np.eye(3, dtype=int)
    steps = []
    for _ in range(max_iter):
        changed = False
        norms = np.linalg.norm(b, axis=0)
        order = np.argsort(norms, kind="stable")
        if list(order) != [0, 1, 2]:
            p = np.eye(3, dtype=int)[:, order]
            b = b @ p
            m = m @ p
            steps.append(p.copy())
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                denom = float(b[:, j] @ b[:, j])
                if denom <= 0.0:
                    continue
                n = int(round(float(b[:, i] @ b[:, j]) / denom))
                if n != 0:
                    t = np.eye(3, dtype=int)
                    t[j, i] = -n
                    b = b @ t
                    m = m @ t
                    steps.append(t.copy())
                    changed = True
        if not changed:
            break
    return b, m, steps


def metric_tensor(basis):
    return basis.T @ basis


def _sorted_eigs(metric):
    return np.sort(np.linalg.eigvalsh(np.asarray(metric, dtype=float)))


def metrics_close(g1, g2, rtol=FAMILY_RTOL):
    """Family test: sorted metric eigenvalues agree within fixed rtol."""
    e1 = _sorted_eigs(g1)
    e2 = _sorted_eigs(g2)
    scale = max(float(np.abs(e1).max()), float(np.abs(e2).max()), 1e-30)
    return bool(np.abs(e1 - e2).max() <= rtol * scale)


def basis_transform(basis_from, basis_to, tol=TRANSFORM_TOL):
    """Integer unimodular T with basis_to ~= basis_from @ T, else None."""
    t = np.linalg.inv(np.asarray(basis_from, float)) @ np.asarray(basis_to, float)
    ti = np.rint(t).astype(int)
    if np.abs(t - ti).max() > tol:
        return None
    det = int(round(float(np.linalg.det(ti))))
    if abs(det) != 1:
        return None
    return ti


def index_peaks(basis, Q, covs, index_tol=INDEX_TOL):
    """Assign hkl, predicted position, residual and outlier flag per peak."""
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    binv = np.linalg.inv(basis)
    H = binv @ Q.T
    Hr = np.rint(H)
    frac = np.abs(H - Hr).max(axis=0) if n else np.zeros(0)
    assignments = []
    for i in range(n):
        h = Hr[:, i].astype(int)
        pred = basis @ h
        r = Q[i] - pred
        cov = np.asarray(covs[i], dtype=float)
        try:
            cinv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cinv = np.linalg.pinv(cov)
        weighted = float(r @ cinv @ r)
        assignments.append({
            "index": i,
            "hkl": [int(h[0]), int(h[1]), int(h[2])],
            "pred": [float(v) for v in pred],
            "residual": float(np.linalg.norm(r)),
            "weighted": weighted,
            "outlier": bool(frac[i] > index_tol),
        })
    return assignments


def score_candidate(assignments, outlier_budget):
    """Score components kept separate: no single merged total score."""
    n = len(assignments)
    indexed = [a for a in assignments if not a["outlier"]]
    outliers = n - len(indexed)
    fraction = len(indexed) / n if n else 0.0
    wres = float(np.mean([a["weighted"] for a in indexed])) if indexed else NO_SCORE
    complexity = max((max(abs(v) for v in a["hkl"]) for a in indexed), default=0)
    return {
        "indexed_fraction": fraction,
        "weighted_residual": wres,
        "complexity": complexity,
        "outliers": outliers,
        "outlier_budget": outlier_budget,
        "within_budget": outliers <= outlier_budget,
    }


def generate_candidates(Q, covs, outlier_budget, index_tol=INDEX_TOL,
                        max_basis_peaks=MAX_BASIS_PEAKS,
                        max_candidates=MAX_CANDIDATES):
    """Enumerate basis triples from the shortest observed Q vectors."""
    Q = np.asarray(Q, dtype=float)
    n = len(Q)
    if n < 3:
        return []
    norms = np.linalg.norm(Q, axis=1)
    order = np.argsort(norms, kind="stable")
    short = [int(i) for i in order[:max_basis_peaks]]
    candidates = []
    for combo in itertools.combinations(short, 3):
        basis = Q[list(combo)].T
        scale = float(np.prod(norms[list(combo)]))
        det = abs(float(np.linalg.det(basis)))
        if scale <= 0.0 or det <= SINGULAR_RTOL * scale:
            continue  # singular (near-collinear / coplanar) basis triple
        reduced, transform, steps = reduce_basis(basis)
        assignments = index_peaks(basis, Q, covs, index_tol)
        scores = score_candidate(assignments, outlier_budget)
        candidates.append({
            "basis": basis,
            "basis_ids": list(combo),
            "reduced_basis": reduced,
            "transform": transform,
            "steps": steps,
            "metric": metric_tensor(reduced),
            "assignments": assignments,
            "scores": scores,
        })
    # Stable sort: ties keep deterministic combination order.
    candidates.sort(key=lambda c: (-c["scores"]["indexed_fraction"],
                                   c["scores"]["weighted_residual"],
                                   c["scores"]["complexity"]))
    return candidates[:max_candidates]
