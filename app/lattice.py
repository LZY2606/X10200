"""Deterministic reciprocal-lattice indexing, reduction and scoring.

Network-free numerical core.  All thresholds are module constants, ordering is
explicit/stable, and every change of basis records its integer matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TOL_RESIDUAL = 2.5          # Mahalanobis threshold for an indexed reflection
TOL_GRID = 1.0e-7           # canonical metric grid
TOL_MATCH = 1.0e-6          # metric cells differing below this are equal
TOL_SINGULAR = 1.0e-10      # minimum |det B| in reciprocal units
TOL_RCOND = 1.0e-10
REDUCE_SWEEPS = 80
MAX_HKL_RADIUS = 6
MISSING_RADIUS = 3
MIN_CANDIDATE_INLIERS = 4
TOP_SEEDS = 24
SHORTLIST = 12
ENUM_RANGE = 5

I3 = np.eye(3, dtype=float)


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def sym_spd(matrix: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite copy of a 3x3 covariance."""
    m = np.asarray(matrix, dtype=float)
    m = 0.5 * (m + m.T)
    eigvals, eigvecs = np.linalg.eigh(m)
    eigvals = np.maximum(eigvals, 1.0e-12)
    fixed = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (fixed + fixed.T)


def whiten(covariance: np.ndarray) -> np.ndarray:
    """Symmetric W with W^2 = C^-1 (Mahalanobis whitening)."""
    eigvals, eigvecs = np.linalg.eigh(sym_spd(covariance))
    inv = (eigvecs * np.power(eigvals, -0.5)) @ eigvecs.T
    return 0.5 * (inv + inv.T)


def mahalanobis_sq(delta: np.ndarray, covariance: np.ndarray) -> float:
    z = whiten(covariance) @ np.asarray(delta, dtype=float)
    return float(z @ z)


def apply_calibration(point: np.ndarray, calibration: Optional[Dict[str, Any]]) -> np.ndarray:
    """Map measured y=R x+t into the analysis frame x=R^T(y-t)."""
    y = np.asarray(point, dtype=float)
    if not calibration:
        return y
    rotation = np.asarray(calibration.get("rotation", I3), dtype=float)
    offset = np.asarray(calibration.get("offset", np.zeros(3)), dtype=float)
    return rotation.T @ (y - offset)


def basis_volume(basis: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(basis, dtype=float))))


def is_unimodular(matrix: np.ndarray) -> bool:
    m = np.asarray(matrix)
    if m.shape != (3, 3):
        return False
    if not np.allclose(m, np.rint(m), atol=1.0e-8):
        return False
    return abs(int(round(np.linalg.det(m)))) == 1


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def nearest_hkl(point: np.ndarray, basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest integer Miller indices and predicted reciprocal point."""
    coords = np.linalg.solve(basis, np.asarray(point, dtype=float))
    hkl = np.rint(coords).astype(int)
    predicted = basis @ hkl.astype(float)
    return hkl, predicted


def index_point(point, covariance, basis):
    """Return one correspondence dict (without overlap/domain decoration)."""
    point = np.asarray(point, dtype=float)
    hkl, predicted = nearest_hkl(point, basis)
    residual = point - predicted
    chi_sq = mahalanobis_sq(residual, covariance)
    return {
        "hkl": tuple(int(v) for v in hkl),
        "predicted": [float(v) for v in predicted],
        "residual_vector": [float(v) for v in residual],
        "residual": float(np.sqrt(max(chi_sq, 0.0))),
        "chi_sq": chi_sq,
        "outlier": bool(chi_sq > TOL_RESIDUAL * TOL_RESIDUAL),
    }


def refine_basis(points: np.ndarray, covariances: np.ndarray,
                 basis: np.ndarray, hkls: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Weighted least-squares refinement B @ h = p over inlier rows.

    Each Cartesian component is solved separately with W=C^-1.  Singular or
    rank-deficient systems leave the basis unchanged.
    """
    h = np.asarray(hkls, dtype=float)
    p = np.asarray(points, dtype=float)
    # Generalized least squares for p_i = B h_i.  The stacked Cartesian model
    # is p = (H kron I) b and block weight is blockdiag(C_i^-1).
    if h.shape[0] < 3:
        return basis, False
    try:
        lhs_sum = np.zeros((9, 9))
        rhs_sum = np.zeros((9, 1))
        for point_i, hkl_i, cov_i in zip(points, hkls, covariances):
            precision = np.linalg.inv(sym_spd(cov_i))
            outer = np.outer(hkl_i, hkl_i)
            lhs_sum += np.kron(outer, precision)
            rhs_sum += np.kron(hkl_i.reshape(3, 1),
                               (precision @ point_i).reshape(3, 1))
        result_flat, _, rank, _ = np.linalg.lstsq(
            lhs_sum, rhs_sum, rcond=TOL_RCOND)
        result = result_flat.reshape(3, 3, order="F")
    except np.linalg.LinAlgError:
        return basis, False
    if rank < 3 or basis_volume(result) < TOL_SINGULAR:
        return basis, False
    return result, True


# ---------------------------------------------------------------------------
# Deterministic lattice reduction
# ---------------------------------------------------------------------------
def _grid(value: float) -> float:
    return float(np.round(value / TOL_GRID) * TOL_GRID)


def _vec_key(vector: np.ndarray) -> Tuple[float, float, float, float]:
    norm = _grid(float(vector @ vector))
    return (norm, _grid(float(vector[0])), _grid(float(vector[1])),
            _grid(float(vector[2])))


def _step_record(basis: np.ndarray, cumulative: np.ndarray,
                 operation: str, local_matrix: np.ndarray,
                 vectors: Optional[np.ndarray] = None) -> Dict[str, Any]:
    return {
        "operation": operation,
        "matrix": [[int(v) for v in row] for row in np.asarray(local_matrix)],
        "cumulative": [[int(v) for v in row] for row in np.asarray(cumulative)],
        "basis": [[float(v) for v in col] for col in np.asarray(basis).T],
        "vectors": (None if vectors is None
                    else [[float(v) for v in item] for item in vectors]),
    }


def _size_reduce(basis: np.ndarray, coefficient: np.ndarray,
                 steps: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-plane coefficient reduction using fixed rounding (half-to-even)."""
    changed = True
    sweeps = 0
    while changed and sweeps < REDUCE_SWEEPS:
        changed = False
        sweeps += 1
        for i in range(2, -1, -1):
            for j in range(i - 1, -1, -1):
                # Gram-Schmidt coefficient of b_i on b_j in current vectors.
                if basis[:, j] @ basis[:, j] <= TOL_SINGULAR:
                    continue
                mu = float((basis[:, i] @ basis[:, j]) /
                           (basis[:, j] @ basis[:, j]))
                amount = int(round(mu))
                if amount == 0:
                    continue
                local = np.eye(3, dtype=int)
                local[i, j] = -amount
                basis = basis @ local
                coefficient = coefficient @ local
                steps.append(_step_record(basis, coefficient,
                                          f"size_reduce b{i+1} -= {amount}*b{j+1}",
                                          local))
                changed = True
    return basis, coefficient


def _shortest_independent(basis: np.ndarray, used: List[np.ndarray],
                          used_norms: List[float]) -> Optional[np.ndarray]:
    best = None
    best_key = None
    for n1 in range(-ENUM_RANGE, ENUM_RANGE + 1):
        for n2 in range(-ENUM_RANGE, ENUM_RANGE + 1):
            for n3 in range(-ENUM_RANGE, ENUM_RANGE + 1):
                coeffs = np.array([n1, n2, n3], dtype=int)
                if not np.any(coeffs):
                    continue
                vector = basis @ coeffs.astype(float)
                norm = float(vector @ vector)
                if norm < TOL_SINGULAR:
                    continue
                independent = True
                for old, old_norm in zip(used, used_norms):
                    if abs(float(old @ vector)) / np.sqrt(old_norm * norm) > 0.999999:
                        independent = False
                        break
                if not independent:
                    continue
                key = (norm, _grid(vector[0]), _grid(vector[1]), _grid(vector[2]))
                if best_key is None or key < best_key:
                    best_key = key
                    best = coeffs.copy()
    return best


def _all_lattice_vectors(basis: np.ndarray, bound: int):
    """All nonzero lattice vectors in a fixed integer box, stable order."""
    vectors = []
    for n1 in range(-bound, bound + 1):
        for n2 in range(-bound, bound + 1):
            for n3 in range(-bound, bound + 1):
                coeffs = np.array([n1, n2, n3], dtype=int)
                if not np.any(coeffs):
                    continue
                physical = basis @ coeffs.astype(float)
                norm = float(physical @ physical)
                if norm < TOL_SINGULAR:
                    continue
                key = (_grid(norm),
                       _grid(float(physical[0])),
                       _grid(float(physical[1])),
                       _grid(float(physical[2])))
                vectors.append((key, coeffs, physical, norm))
    vectors.sort(key=lambda item: item[0])
    return vectors


def _independent(matrix: np.ndarray, vector: np.ndarray) -> bool:
    if matrix.shape[1] == 0:
        return True
    if matrix.shape[1] == 1:
        cross = np.cross(matrix[:, 0], vector)
        return float(cross @ cross) > TOL_SINGULAR
    volume = abs(float(np.linalg.det(np.column_stack([matrix, vector]))))
    return volume > TOL_SINGULAR


def _metric_values(basis: np.ndarray):
    a = float(basis[:, 0] @ basis[:, 0])
    b = float(basis[:, 1] @ basis[:, 1])
    c = float(basis[:, 2] @ basis[:, 2])
    d = 2.0 * float(basis[:, 1] @ basis[:, 2])
    e = 2.0 * float(basis[:, 0] @ basis[:, 2])
    f = 2.0 * float(basis[:, 0] @ basis[:, 1])
    return a, b, c, d, e, f


def _apply_int(basis, coefficient, steps, matrix, operation):
    basis = basis @ matrix
    coefficient = coefficient @ matrix
    steps.append(_step_record(basis, coefficient, operation, matrix))
    return basis, coefficient


def reduce_basis(basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                             List[Dict[str, Any]]]:
    """Krivy-Gruber Niggli reduction with a complete integer transform log."""
    basis = np.asarray(basis, dtype=float).copy()
    coefficient = np.eye(3, dtype=int)
    steps: List[Dict[str, Any]] = []
    eps = TOL_GRID

    def swap12():
        nonlocal basis, coefficient
        m = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=int)
        basis, coefficient = _apply_int(basis, coefficient, steps, m, "N1 swap")

    def swap23():
        nonlocal basis, coefficient
        m = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=int)
        basis, coefficient = _apply_int(basis, coefficient, steps, m, "N2 swap")

    def swap13():
        nonlocal basis, coefficient
        m = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=int)
        basis, coefficient = _apply_int(basis, coefficient, steps, m, "N5 swap")

    for iteration in range(REDUCE_SWEEPS * 10):
        a, b, c, d, e, f = _metric_values(basis)
        # N1
        if a > b + eps or (abs(a - b) <= eps and abs(d) > abs(e) + eps):
            swap12()
            continue
        # N2
        if b > c + eps or (abs(b - c) <= eps and abs(e) > abs(f) + eps):
            swap23()
            continue
        # N3: sign changes to non-positive d,e,f.
        if d > eps or e > eps or f > eps:
            signs = np.eye(3, dtype=int)
            if d > eps:
                signs[1, 1] = -1
                signs[2, 2] = -1
            if e > eps:
                signs[0, 0] = -1
                signs[2, 2] = -1
            if f > eps:
                signs[0, 0] = -1
                signs[1, 1] = -1
            basis, coefficient = _apply_int(basis, coefficient, steps, signs,
                                            "N3 sign normalization")
            continue
        if d < -eps or e < -eps or f < -eps:
            # Standard sign cases from Krivy-Gruber.
            if (d > eps and e > eps) or (d < -eps and e < -eps and f < -eps):
                m = np.diag([1, -1, -1])
            elif (d > eps and f > eps) or (e < -eps and f < -eps and d < -eps):
                m = np.diag([-1, 1, -1])
            elif (e > eps and f > eps) or (d < -eps and f < -eps and e < -eps):
                m = np.diag([-1, -1, 1])
            elif d < -eps and e > eps and f > eps:
                m = np.array([[1, 0, 0], [0, 1, -1], [0, 0, -1]])
            elif e < -eps and d > eps and f > eps:
                m = np.array([[1, 0, -1], [0, 1, 0], [0, 0, -1]])
            elif f < -eps and d > eps and e > eps:
                m = np.array([[1, -1, 0], [0, -1, 0], [0, 0, 1]])
            elif d < -eps and e < -eps and f > eps:
                m = np.array([[-1, 0, 0], [0, -1, -1], [0, 0, 1]])
            elif e < -eps and f < -eps and d > eps:
                m = np.array([[-1, 0, -1], [0, 1, 0], [0, 0, -1]])
            elif d < -eps and f < -eps and e > eps:
                m = np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]])
            else:
                m = np.eye(3, dtype=int)
            if not np.allclose(m, np.eye(3)):
                basis, coefficient = _apply_int(
                    basis, coefficient, steps, m.astype(int), "N4 sign case")
                continue
        # N5/N8 integer shears.
        if abs(d) > b + eps or (abs(d) <= eps and b < -eps) or \
           (abs(d - b) <= eps and 2 * e < f - eps) or \
           (abs(d + b) <= eps and f < -eps):
            # Translate c onto b using integer m=round(d/2b); here map columns.
            m = int(round(d / (2.0 * b))) if abs(b) > eps else 0
            mat = np.array([[1, 0, 0], [0, 1, -m], [0, 0, 1]], dtype=int)
            basis, coefficient = _apply_int(basis, coefficient, steps, mat,
                                            "N5 shear")
            continue
        if abs(e) > a + eps or (abs(e) <= eps and a < -eps) or \
           (abs(e - a) <= eps and 2 * d < f - eps) or \
           (abs(e + a) <= eps and f < -eps):
            m = int(round(e / (2.0 * a))) if abs(a) > eps else 0
            mat = np.array([[1, 0, -m], [0, 1, 0], [0, 0, 1]], dtype=int)
            basis, coefficient = _apply_int(basis, coefficient, steps, mat,
                                            "N6 shear")
            continue
        if abs(f) > a + eps or (abs(f) <= eps and a < -eps) or \
           (abs(f - a) <= eps and 2 * d < e - eps) or \
           (abs(f + a) <= eps and e < -eps):
            m = int(round(f / (2.0 * a))) if abs(a) > eps else 0
            mat = np.array([[1, -m, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
            basis, coefficient = _apply_int(basis, coefficient, steps, mat,
                                            "N7 shear")
            continue
        if d + e + f + a + c < -eps or \
           (abs(d + e + f + a + c) <= eps and 2 * (a + e) + f > eps):
            mat = np.array([[1, 0, 1], [0, 1, 1], [0, 0, -1]], dtype=int)
            basis, coefficient = _apply_int(basis, coefficient, steps, mat,
                                            "N8 shear")
            continue
        break

    basis, coefficient, sign_steps = _canonical_signs(basis, coefficient)
    steps.extend(sign_steps)
    return basis, coefficient, steps


def _canonical_signs(basis: np.ndarray, coefficient: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Unique sign convention: lex-min metric, then lex-min column values."""
    best_basis = basis
    best_coeff = coefficient
    best_key = None
    best_sign = None
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for s3 in (-1, 1):
                sign = np.diag([s1, s2, s3]).astype(int)
                candidate = basis @ sign
                dots = [candidate[:, i] @ candidate[:, j]
                        for i in range(3) for j in range(i + 1, 3)]
                key = (tuple(_grid(d) for d in dots),
                       tuple(_grid(float(v)) for column in
                             (candidate[:, 0], candidate[:, 1], candidate[:, 2])
                             for v in column))
                if best_key is None or key < best_key:
                    best_key = key
                    best_basis = candidate
                    best_coeff = coefficient @ sign
                    best_sign = sign
    steps = []
    if best_sign is not None and not np.allclose(best_sign, np.eye(3)):
        steps.append(_step_record(best_basis, best_coeff,
                                  "canonical sign convention", best_sign))
    return best_basis, best_coeff, steps


def metric_matrix(basis: np.ndarray) -> np.ndarray:
    return np.asarray(basis).T @ np.asarray(basis)


def canonical_signature(basis: np.ndarray) -> Tuple[Tuple[int, ...], List[List[float]]]:
    """Quantized upper-triangular metric of the reduced cell; family key."""
    reduced, _, _ = reduce_basis(np.asarray(basis, dtype=float))
    metric = metric_matrix(reduced)
    values = []
    for i in range(3):
        for j in range(i, 3):
            values.append(int(round(metric[i, j] / TOL_MATCH)))
    return tuple(values), [[float(v) for v in col] for col in reduced.T]


def equivalent_basis(basis_a: np.ndarray, basis_b: np.ndarray) -> Optional[np.ndarray]:
    """Integer unimodular P with B_a P == B_b within tolerance, else None."""
    try:
        candidate = np.linalg.solve(np.asarray(basis_a, dtype=float),
                                    np.asarray(basis_b, dtype=float))
    except np.linalg.LinAlgError:
        return None
    rounded = np.rint(candidate)
    if not np.allclose(candidate, rounded, atol=1.0e-5, rtol=1.0e-7):
        return None
    matrix = rounded.astype(int)
    if not is_unimodular(matrix):
        return None
    if not np.allclose(np.asarray(basis_a) @ matrix,
                       np.asarray(basis_b), atol=1.0e-5, rtol=1.0e-7):
        return None
    return matrix
