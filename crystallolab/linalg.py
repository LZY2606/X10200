"""Numerical core for the Lattice Candidate Chamber.

Conventions:
* vectors are columns; a reciprocal basis is a 3x3 matrix ``B`` whose columns
  are ``a*, b*, c*`` (so a lattice vector predicted for hkl is ``B @ h``).
* ``a* . a = 1`` (reciprocal vectors lie in reciprocal space), hence the
  direct basis is ``A = inv(B).T``.

All tolerances are module-level fixed constants; sorting is always performed
on explicit deterministic keys (stable sort).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Fixed numerical tolerances -------------------------------------------
SINGULAR_RCOND = 1e-6          # cutoff for reciprocal-condition / inversion
INTEGER_TOL = 1.0e-2           # distance from nearest integer to count as indexed
OUTLIER_TAU = 3.0              # Mahalanobis cutoff (in sigma)
REDUCE_TOL = 1.0e-9            # Niggli comparison slack
FAMILY_REL_TOL = 1.0e-2        # reduced bases equal within this relative gap
FAMILY_INTEGER_TOL = 2.0e-2    # slack for the unimodular change of basis
ORIGIN_ITERS = 10              # fixed-point origin refinement iterations
SHORT_NORM_CUT = 4.0           # deltas accepted up to 4x the shortest norm
MAX_SHORT_VECTORS = 12         # stable cap on quantised short vectors
MAX_REDUCTION_ITERS = 100
COMPLEXITY_CLIP = 6.0          # direct-cell relative volume cap for scoring

EPS = 1.0e-12


# --- Basic linear algebra helpers -----------------------------------------
def is_singular_basis(mat: np.ndarray) -> bool:
    m = np.asarray(mat, dtype=float)
    if m.shape != (3, 3):
        return True
    sv = np.linalg.svd(m, compute_uv=False)
    if sv[0] <= EPS:
        return True
    return bool(sv[2] / sv[0] < SINGULAR_RCOND)


def nearest_int(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    rounded = np.rint(v)
    # Normalise -0.0 so signatures compare cleanly.
    return rounded + 0.0


def mahalanobis_sq(residual: np.ndarray, cov: np.ndarray) -> float:
    r = np.asarray(residual, dtype=float)
    c = np.asarray(cov, dtype=float)
    inv = np.linalg.inv(c)
    return float(r @ inv @ r)


# --- Deterministic lattice reduction --------------------------------------
def reduce_basis(basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Reduce a reciprocal basis greedily (Niggli-style 3D adaptation).

    Returns ``(reduced_basis, U, steps)`` where ``reduced = basis @ U`` and
    ``U`` is an integer unimodular matrix accumulated by the recorded steps.
    The reduction is deterministic and fixed-iteration; it is not claimed to
    be a full Niggli implementation, but every transformation is recorded.
    """
    b = np.asarray(basis, dtype=float).copy()
    u = np.eye(3, dtype=float)
    steps: List[dict] = []

    def record(kind: str, matrix: np.ndarray, note: str) -> None:
        before = b.copy()
        steps.append({
            "kind": kind,
            "note": note,
            "matrix": np.asarray(matrix, dtype=float).tolist(),
            "before_gram": _gram(before).tolist(),
        })

    seen: Dict[str, int] = {}
    for _ in range(MAX_REDUCTION_ITERS):
        gram = _gram(b)
        sig = _gram_signature(gram)
        if sig in seen:
            record("stop", np.eye(3), "state repeated; reduction converged/oscillation guard")
            break
        seen[sig] = len(steps)

        norms = np.diag(gram)
        lengths = np.sqrt(np.maximum(norms, 0.0))

        # Swap: keep the shortest column first, then second shortest.
        order = sorted(range(3), key=lambda i: (round(norms[i], 12), i))
        if order[0] != 0:
            p = _swap_perm(0, order[0])
            b, u = _apply(b, u, p)
            record("swap", p, "shortest vector to position 1")
            continue
        if order[1] != 1:
            p = _swap_perm(1, order[1])
            b, u = _apply(b, u, p)
            record("swap", p, "second-shortest vector to position 2")
            continue

        # Shear: remove multiples of the first vector from the others.
        g = _gram(b)
        progressed = False
        for j in (1, 2):
            m = int(nearest_int(g[0, j] / g[0, 0]))
            if m != 0:
                shear = np.eye(3)
                shear[0, j] = -m
                nb = b @ shear
                if np.trace(_gram(nb)) <= np.trace(g) + REDUCE_TOL:
                    b, u = _apply(b, u, shear)
                    record("shear", shear, "b%d <- b%d - %d*b1" % (j + 1, j + 1, m))
                    progressed = True
                    break
        if progressed:
            continue

        # Shear between vectors 2 and 3.
        m = int(nearest_int(g[1, 2] / g[1, 1]))
        if m != 0:
            shear = np.eye(3)
            shear[1, 2] = -m
            nb = b @ shear
            if np.trace(_gram(nb)) <= np.trace(g) + REDUCE_TOL:
                b, u = _apply(b, u, shear)
                record("shear", shear, "b3 <- b3 - %d*b2" % m)
                continue

        # Sign convention: g12 <= 0 and g13 <= 0.
        g = _gram(b)
        neg = None
        if g[0, 1] > REDUCE_TOL:
            neg = 1
        elif g[0, 2] > REDUCE_TOL:
            neg = 2
        if neg is not None:
            s = np.eye(3)
            s[neg, neg] = -1.0
            b, u = _apply(b, u, s)
            record("sign", s, "flip b%d to enforce negative off-diagonal" % (neg + 1))
            continue

        record("stop", np.eye(3), "reduction criteria satisfied")
        break

    return b, u, steps


def _gram(b: np.ndarray) -> np.ndarray:
    return b.T @ b


def _gram_signature(g: np.ndarray) -> str:
    rounded = np.round(g, 9) + 0.0
    return "|".join("%.9f" % v for v in rounded.flatten())


def _swap_perm(i: int, j: int) -> np.ndarray:
    p = np.eye(3)
    p[:, [i, j]] = p[:, [j, i]]
    return p


def _apply(b: np.ndarray, u: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return b @ m, u @ m


def direct_basis(reciprocal: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(reciprocal, dtype=float)).T


def cell_parameters(basis: np.ndarray) -> dict:
    """Return lengths/angles of a basis (works for direct or reciprocal)."""
    b = np.asarray(basis, dtype=float)
    vecs = [b[:, i] for i in range(3)]
    lens = [float(np.linalg.norm(v)) for v in vecs]
    angles = []
    pairs = [(1, 2), (0, 2), (0, 1)]
    for i, j in pairs:
        cosang = float(np.dot(vecs[i], vecs[j]) / (lens[i] * lens[j] + EPS))
        cosang = max(-1.0, min(1.0, cosang))
        angles.append(float(np.degrees(np.arccos(cosang))))
    return {"lengths": lens, "angles": angles, "volume": abs(float(np.linalg.det(b)))}


# --- Equivalence / families -----------------------------------------------
def same_reduced_basis(r1: np.ndarray, r2: np.ndarray, tol: float = FAMILY_REL_TOL) -> bool:
    """Two reduced bases match if their column sets agree up to sign/permutation."""
    a = [np.asarray(r1, dtype=float)[:, i] for i in range(3)]
    b = [np.asarray(r2, dtype=float)[:, i] for i in range(3)]
    scale = max(np.linalg.norm(a[0]), np.linalg.norm(a[1]), np.linalg.norm(a[2]), EPS)
    used = [False, False, False]
    for va in a:
        found = -1
        for j in range(3):
            if used[j]:
                continue
            vb = b[j]
            if np.linalg.norm(va - vb) <= tol * scale or np.linalg.norm(va + vb) <= tol * scale:
                found = j
                break
        if found < 0:
            return False
        used[found] = True
    return True


def family_transform(reference_basis: np.ndarray, member_basis: np.ndarray,
                     int_tol: float = FAMILY_INTEGER_TOL) -> Optional[np.ndarray]:
    """Return integer unimodular M with member = reference @ M, else None."""
    ref = np.asarray(reference_basis, dtype=float)
    mem = np.asarray(member_basis, dtype=float)
    if is_singular_basis(ref):
        return None
    m = np.linalg.inv(ref) @ mem
    mi = nearest_int(m)
    if np.max(np.abs(m - mi)) > int_tol:
        return None
    det = round(float(np.linalg.det(mi)))
    if abs(det) != 1:
        return None
    return mi + 0.0


def permutation_sign_columns(r1: np.ndarray, r2: np.ndarray,
                             tol: float = FAMILY_REL_TOL) -> Optional[np.ndarray]:
    """Signed permutation Q with r2 = r1 @ Q, if column sets match."""
    a = [np.asarray(r1, dtype=float)[:, i] for i in range(3)]
    b = [np.asarray(r2, dtype=float)[:, i] for i in range(3)]
    scale = max([np.linalg.norm(v) for v in a] + [EPS])
    q = np.zeros((3, 3))
    used = [False, False, False]
    for i, va in enumerate(a):
        match = None
        for j, vb in enumerate(b):
            if not used[j] and np.linalg.norm(va - vb) <= tol * scale:
                match = (j, 1.0)
                break
            if not used[j] and np.linalg.norm(va + vb) <= tol * scale:
                match = (j, -1.0)
                break
        if match is None:
            return None
        j, sign = match
        used[j] = True
        q[j, i] = sign
    return q


# --- Indexing --------------------------------------------------------------
def assign_reflections(basis: np.ndarray, origin: np.ndarray,
                       points: np.ndarray, covariances: np.ndarray,
                       locks: Optional[Sequence[Optional[Sequence[int]]]] = None,
                       tau: float = OUTLIER_TAU) -> List[dict]:
    """Assign hkl / predicted position / residual / outlier flag per reflection.

    Locks are given as pre-assigned integer hkl (or None); a locked reflection
    is evaluated at its pinned hkl and is never marked an outlier.
    """
    b = np.asarray(basis, dtype=float)
    o = np.asarray(origin, dtype=float)
    results: List[dict] = []
    for idx, p in enumerate(points):
        p = np.asarray(p, dtype=float)
        cov = np.asarray(covariances[idx], dtype=float)
        if locks is not None and locks[idx] is not None:
            h = np.asarray(locks[idx], dtype=float)
            locked = True
        else:
            h = nearest_int(np.linalg.solve(b, p - o))
            locked = False
        predicted = o + b @ h
        residual = p - predicted
        mh_sq = mahalanobis_sq(residual, cov)
        outlier = (not locked) and (mh_sq > tau * tau)
        results.append({
            "hkl": [int(v) for v in h],
            "predicted": predicted.tolist(),
            "residual": residual.tolist(),
            "mahal_sq": mh_sq,
            "outlier": bool(outlier),
            "locked": locked,
        })
    return results


def refine_origin(basis: np.ndarray, points: np.ndarray,
                  covariances: np.ndarray, start: Optional[np.ndarray] = None,
                  iters: int = ORIGIN_ITERS) -> np.ndarray:
    """Weighted fixed-point estimate of the origin using inlier reflections."""
    b = np.asarray(basis, dtype=float)
    o = np.zeros(3) if start is None else np.asarray(start, dtype=float).copy()
    for _ in range(iters):
        weights = []
        shifts = []
        for idx, p in enumerate(points):
            p = np.asarray(p, dtype=float)
            cov = np.asarray(covariances[idx], dtype=float)
            h = nearest_int(np.linalg.solve(b, p - o))
            residual = p - (o + b @ h)
            mh_sq = mahalanobis_sq(residual, cov)
            if mh_sq <= OUTLIER_TAU * OUTLIER_TAU:
                weights.append(np.linalg.inv(cov))
                shifts.append(p - b @ h)
        if not weights:
            break
        wsum = np.zeros((3, 3))
        wmu = np.zeros(3)
        for w, s in zip(weights, shifts):
            wsum += w
            wmu += w @ s
        new_o = np.linalg.solve(wsum, wmu)
        if np.linalg.norm(new_o - o) < 1e-12:
            o = new_o
            break
        o = new_o
    return o


# --- Candidate generation --------------------------------------------------
def _quantise_short_vectors(points: np.ndarray) -> List[np.ndarray]:
    """Cluster pairwise difference vectors (with +/-) by fixed tolerance."""
    deltas = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.asarray(points[j] - points[i], dtype=float)
            norm = float(np.linalg.norm(d))
            if norm > EPS:
                deltas.append((norm, d))
    deltas.sort(key=lambda item: (round(item[0], 12),
                                  round(float(item[1][0]), 9),
                                  round(float(item[1][1]), 9),
                                  round(float(item[1][2]), 9)))
    if not deltas:
        return []
    base = deltas[0][0]
    cut = base * SHORT_NORM_CUT
    short = [d for norm, d in deltas if norm <= cut][:64]

    clusters: List[Tuple[np.ndarray, float, List[np.ndarray]]] = []
    tol = max(base * 1e-3, 1e-9)
    for d in short:
        best = -1
        best_gap = None
        for k, (mean, norm, members) in enumerate(clusters):
            gap = min(np.linalg.norm(d - mean), np.linalg.norm(d + mean))
            if gap <= tol and (best_gap is None or gap < best_gap):
                best = k
                best_gap = gap
        if best < 0:
            clusters.append((d.copy(), float(np.linalg.norm(d)), [d]))
        else:
            mean, norm, members = clusters[best]
            members.append(d)
            mean *= 0.0  # mean is recomputed below; keep list stable
            clusters[best] = (np.mean(members, axis=0), norm, members)

    clusters.sort(key=lambda c: (round(c[1], 12),
                                 round(float(c[2][0][0]), 9),
                                 round(float(c[2][0][1]), 9),
                                 round(float(c[2][0][2]), 9)))
    return [c[0] for c in clusters[:MAX_SHORT_VECTORS]]


def generate_candidate_bases(points: np.ndarray) -> List[np.ndarray]:
    """Enumerate deterministic lattice basis candidates from peak differences."""
    vectors = _quantise_short_vectors(points)
    if len(vectors) < 3:
        return []
    candidates: List[np.ndarray] = []
    signatures = set()
    m = len(vectors)
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                choices = ((i, 1.0), (i, -1.0))
                for ai, sa in choices:
                    for bj, sb in ((j, 1.0), (j, -1.0)):
                        for ck, sc in ((k, 1.0), (k, -1.0)):
                            mat = np.column_stack([sa * vectors[ai],
                                                   sb * vectors[bj],
                                                   sc * vectors[ck]])
                            if is_singular_basis(mat):
                                continue
                            h_all = np.linalg.solve(mat, points.T).T
                            frac = np.abs(h_all - nearest_int(h_all))
                            if float(np.max(frac)) <= INTEGER_TOL:
                                continue  # a supercell would index everything trivially
                            close = float(np.sum(np.max(frac, axis=1) <= INTEGER_TOL))
                            if close < 3:
                                continue
                            sig = _basis_signature(mat)
                            if sig not in signatures:
                                signatures.add(sig)
                                candidates.append(mat)
    return candidates


def _basis_signature(mat: np.ndarray) -> Tuple:
    reduced, _, _ = reduce_basis(mat)
    cols = sorted(
        (round(float(np.linalg.norm(reduced[:, c])), 8) for c in range(3))
    )
    gram = _gram(reduced)
    off = tuple(sorted(round(float(v), 8) for v in (gram[0, 1], gram[0, 2], gram[1, 2])))
    return tuple(cols) + off
