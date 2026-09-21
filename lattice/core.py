"""Core reciprocal-lattice math: cell reduction, transform chains, metric compare.

All tolerances are fixed module constants so results are deterministic.
Every basis transform applied during reduction is recorded as an integer
unimodular 3x3 matrix; chains are stored as lists of row-major matrices.
"""
from __future__ import annotations

import numpy as np

# Fixed numerical tolerances (no runtime configuration).
DET_MIN = 1e-8          # below this |det| a basis is singular
REDUCE_EPS = 1e-10      # convergence epsilon for the reduction loop
REDUCE_MAX_STEPS = 200  # hard cap on reduction iterations
FAMILY_RTOL = 1e-3      # relative tolerance on the metric tensor for family merge
INDEX_TOL = 0.35        # max |fractional hkl - integer| to accept an index
H_MAX = 12              # max |h|,|k|,|l| considered during indexing
REFINE_ITERS = 3        # weighted least-squares refinement passes


def metric_tensor(basis: np.ndarray) -> np.ndarray:
    """G = B^T B for a basis whose columns are the reciprocal basis vectors."""
    return basis.T @ basis


def is_singular(basis: np.ndarray) -> bool:
    return abs(float(np.linalg.det(basis))) < DET_MIN


def _norms(basis: np.ndarray) -> np.ndarray:
    return np.linalg.norm(basis, axis=0)


def reduce_cell(basis: np.ndarray):
    """Greedy norm-reduction of a reciprocal basis.

    Returns (reduced_basis, chain) where chain is a list of integer
    unimodular matrices M_k applied on the right:
        reduced = basis @ M_0 @ M_1 @ ... @ M_n
    The chain fully records the coordinate transform process.
    """
    reduced = np.array(basis, dtype=float)
    chain = []
    steps = 0
    improved = True
    while improved and steps < REDUCE_MAX_STEPS:
        improved = False
        # 1) order columns by ascending norm (permutation, det +/-1)
        order = np.argsort(_norms(reduced), kind="stable")
        if list(order) != [0, 1, 2]:
            perm = np.zeros((3, 3), dtype=int)
            for new_i, old_i in enumerate(order):
                perm[old_i, new_i] = 1
            reduced = reduced @ perm
            chain.append(perm.tolist())
            improved = True
        # 2) shear: b_i <- b_i - round(<b_i,b_j>/<b_j,b_j>) b_j
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                denom = float(reduced[:, j] @ reduced[:, j])
                if denom < DET_MIN:
                    continue
                n = int(np.round(float(reduced[:, i] @ reduced[:, j]) / denom))
                if n != 0:
                    shear = np.eye(3, dtype=int)
                    shear[j, i] = -n
                    candidate = reduced @ shear
                    if _norms(candidate)[i] < _norms(reduced)[i] - REDUCE_EPS:
                        reduced = candidate
                        chain.append(shear.tolist())
                        improved = True
        steps += 1
    # 3) force positive determinant with a sign flip (det -1 unimodular)
    if np.linalg.det(reduced) < 0:
        flip = np.diag([-1, 1, 1]).astype(int)
        reduced = reduced @ flip
        chain.append(flip.tolist())
    return reduced, chain


def compose_chain(chain) -> np.ndarray:
    """Compose a recorded chain into a single integer matrix."""
    total = np.eye(3, dtype=int)
    for mat in chain:
        total = total @ np.array(mat, dtype=int)
    return total


def metrics_equivalent(g1: np.ndarray, g2: np.ndarray, rtol: float = FAMILY_RTOL) -> bool:
    """True iff two metric tensors agree within the fixed relative tolerance.

    Near-identical cells outside the tolerance deliberately return False so
    their candidates are never merged.
    """
    scale = max(float(np.abs(g1).max()), float(np.abs(g2).max()), DET_MIN)
    return bool(np.all(np.abs(g1 - g2) <= rtol * scale))


def metric_key(metric: np.ndarray, rtol: float = FAMILY_RTOL):
    """Deterministic bucket key for a metric tensor (stable ordering aid)."""
    scale = max(float(np.abs(metric).max()), DET_MIN)
    snapped = np.round(metric / (rtol * scale)).astype(int)
    return tuple(int(v) for v in snapped[np.triu_indices(3)])
