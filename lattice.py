"""Core lattice mathematics for the lattice candidate chamber.

All numerical algorithms use fixed tolerances and stable (deterministic)
ordering so results are reproducible. No network access happens here.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# Fixed tolerances (never adapted at runtime).
SINGULAR_TOL = 1e-8        # determinant threshold for degenerate bases
REDUCE_TOL = 1e-10         # numerical zero inside basis reduction
LENGTH_REL_TOL = 5e-3      # relative tolerance for reduced-cell edge lengths
ANGLE_TOL_DEG = 0.5        # absolute tolerance (degrees) for reduced-cell angles
INDEX_CHI2_TOL = 9.0       # chi^2 threshold for an indexed reflection
HMAX = 8                   # largest |h|,|k|,|l| considered
OUTLIER_BUDGET = 0.25      # allowed outlier fraction of active reflections
MAX_TRIPLES = 3000         # deterministic cap on seed triples examined
MAX_CANDIDATES = 40        # deterministic cap on accepted candidates
MIN_INDEXED = 3            # a generated candidate must index at least this many
LLL_DELTA = 0.75
DEDUP_TOL = 1e-9           # exact-duplicate filter for reduced bases


def lll_reduce(basis, delta=LLL_DELTA, tol=REDUCE_TOL):
    """LLL-reduce the rows of ``basis``.

    Returns ``(reduced, chain)`` where ``chain`` is the ordered list of
    unimodular integer matrices applied on the left, so that
    ``reduced == chain[-1] @ ... @ chain[0] @ basis``. The full chain is
    kept so the cell-reduction / coordinate-transform process is auditable.
    """
    rows = [np.asarray(r, dtype=float).copy() for r in basis]
    chain: List[np.ndarray] = []
    n = len(rows)

    def gram_schmidt():
        stars = []
        mu = np.zeros((n, n))
        norms = np.zeros(n)
        for i in range(n):
            v = rows[i].copy()
            for j in range(i):
                if norms[j] > tol:
                    mu[i, j] = float(np.dot(rows[i], stars[j]) / norms[j])
                v = v - mu[i, j] * stars[j]
            stars.append(v)
            norms[i] = float(np.dot(v, v))
        return stars, mu, norms

    _, mu, norms = gram_schmidt()
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k, j]))
            if q:
                m = np.eye(n, dtype=int)
                m[k, j] = -q
                rows[k] = rows[k] - q * rows[j]
                chain.append(m)
                _, mu, norms = gram_schmidt()
        if norms[k] >= (delta - mu[k, k - 1] ** 2) * norms[k - 1] - tol:
            k += 1
        else:
            m = np.eye(n, dtype=int)
            m[k, k] = 0
            m[k - 1, k - 1] = 0
            m[k, k - 1] = 1
            m[k - 1, k] = 1
            rows[k], rows[k - 1] = rows[k - 1], rows[k]
            chain.append(m)
            _, mu, norms = gram_schmidt()
            k = max(k - 1, 1)
    return np.array(rows), chain


def chain_product(chain):
    """Collapse a recorded transform chain into a single integer matrix."""
    total = np.eye(3, dtype=int)
    for m in chain:
        total = m @ total
    return total


def cell_params(ub):
    """Reduced reciprocal-cell parameters (a*, b*, c*, alpha*, beta*, gamma*)."""
    g = ub @ ub.T
    lengths = np.sqrt(np.clip(np.diag(g), 0.0, None))

    def angle(i, j):
        denom = lengths[i] * lengths[j]
        if denom <= REDUCE_TOL:
            return 90.0
        cos = float(np.clip(g[i, j] / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    return (
        float(lengths[0]), float(lengths[1]), float(lengths[2]),
        angle(1, 2), angle(0, 2), angle(0, 1),
    )


def cells_equivalent(p, q, length_tol=LENGTH_REL_TOL, angle_tol=ANGLE_TOL_DEG):
    """Fixed-tolerance comparison of reduced-cell parameters.

    Cells that are close but outside tolerance are NOT merged.
    """
    for a, b in zip(p[:3], q[:3]):
        if abs(a - b) > length_tol * max(abs(a), abs(b), 1e-12):
            return False
    for a, b in zip(p[3:], q[3:]):
        if abs(a - b) > angle_tol:
            return False
    return True


@dataclass
class Assignment:
    reflection_index: int
    hkl: Tuple[int, int, int]
    predicted: np.ndarray
    residual: np.ndarray
    chi2: float
    indexed: bool
    is_outlier: bool


@dataclass
class Candidate:
    ub_raw: np.ndarray
    ub_reduced: np.ndarray
    chain: List[np.ndarray]
    params: Tuple[float, ...]
    assignments: List[Assignment]
    indexed_fraction: float
    weighted_residual: float
    complexity: float
    outlier_count: int
    outlier_budget: float
    over_budget: bool
    depends_on: List[int]


def index_reflections(ub, qs, covs, active, hmax=HMAX, chi2_tol=INDEX_CHI2_TOL):
    """Assign hkl, predicted position, residual and outlier flag per reflection."""
    ub_inv = np.linalg.inv(ub)
    assignments: List[Assignment] = []
    for idx in range(len(qs)):
        if not active[idx]:
            continue
        h = ub @ qs[idx]
        hr = np.rint(h)
        if np.max(np.abs(hr)) > hmax or np.all(hr == 0):
            continue  # not indexable by this cell: no assignment recorded
        r = h - hr
        cov_h = ub @ covs[idx] @ ub.T + np.eye(3) * REDUCE_TOL
        chi2 = float(r @ np.linalg.solve(cov_h, r))
        predicted = ub_inv @ hr
        residual = qs[idx] - predicted
        indexed = chi2 <= chi2_tol
        assignments.append(Assignment(
            reflection_index=idx,
            hkl=tuple(int(x) for x in hr),
            predicted=predicted,
            residual=residual,
            chi2=chi2,
            indexed=indexed,
            is_outlier=not indexed,
        ))
    return assignments


def make_candidate(ub_raw, qs, covs, active, outlier_budget=OUTLIER_BUDGET,
                   hmax=HMAX, chi2_tol=INDEX_CHI2_TOL):
    """Reduce a basis, index reflections and compute separate score components."""
    ub_reduced, chain = lll_reduce(ub_raw)
    params = cell_params(ub_reduced)
    assignments = index_reflections(ub_reduced, qs, covs, active, hmax, chi2_tol)
    indexed = [a for a in assignments if a.indexed]
    outliers = [a for a in assignments if a.is_outlier]
    n_active = int(np.sum(active))
    indexed_fraction = len(indexed) / n_active if n_active else 0.0
    weighted_residual = (
        float(np.mean([a.chi2 for a in indexed])) if indexed else float("inf")
    )
    det = abs(float(np.linalg.det(ub_reduced)))
    complexity = 1.0 / det if det > SINGULAR_TOL else float("inf")
    outlier_count = len(outliers)
    over_budget = n_active > 0 and (outlier_count / n_active) > outlier_budget
    depends_on = sorted({a.reflection_index for a in assignments})
    return Candidate(
        ub_raw=np.asarray(ub_raw, dtype=float),
        ub_reduced=ub_reduced,
        chain=chain,
        params=params,
        assignments=assignments,
        indexed_fraction=indexed_fraction,
        weighted_residual=weighted_residual,
        complexity=complexity,
        outlier_count=outlier_count,
        outlier_budget=outlier_budget,
        over_budget=over_budget,
        depends_on=depends_on,
    )


def generate_candidates(qs, covs, active, max_triples=MAX_TRIPLES,
                        max_candidates=MAX_CANDIDATES, **kwargs):
    """Deterministically seed candidate bases from reflection triples.

    Singular (degenerate) triples are skipped and counted. Accepted
    candidates are de-duplicated only when their reduced bases are
    numerically identical; near-duplicates outside tolerance are kept.
    """
    idxs = [i for i in range(len(qs)) if active[i]]
    candidates: List[Candidate] = []
    seen: List[np.ndarray] = []
    skipped_singular = 0
    tried = 0
    stop = False
    for a, b, c in itertools.combinations(idxs, 3):
        if tried >= max_triples or len(candidates) >= max_candidates:
            stop = True
            break
        tried += 1
        basis = np.column_stack([qs[a], qs[b], qs[c]])
        if abs(float(np.linalg.det(basis))) < SINGULAR_TOL:
            skipped_singular += 1
            continue
        ub = np.linalg.inv(basis)
        cand = make_candidate(ub, qs, covs, active, **kwargs)
        if len([x for x in cand.assignments if x.indexed]) < MIN_INDEXED:
            continue
        if any(np.max(np.abs(cand.ub_reduced - s)) < DEDUP_TOL for s in seen):
            continue
        seen.append(cand.ub_reduced)
        candidates.append(cand)
    return candidates, {"triples_tried": tried, "skipped_singular": skipped_singular,
                        "stopped_early": stop}


def sort_candidates(candidates):
    """Stable, deterministic ordering; ties keep generation order."""
    return sorted(candidates, key=lambda c: (-c.indexed_fraction,
                                             c.weighted_residual))


@dataclass
class Family:
    representative: Candidate
    members: List[Tuple[Candidate, np.ndarray]] = field(default_factory=list)


def group_families(candidates):
    """Group equivalent cells into families via allowed (unimodular) transforms.

    The transform matrix of every member relative to the family
    representative is recorded, and every original candidate is kept.
    Candidates outside the fixed tolerance are never merged.
    """
    families: List[Family] = []
    for cand in candidates:
        placed = False
        for fam in families:
            rep = fam.representative
            if not cells_equivalent(cand.params, rep.params):
                continue
            m = cand.ub_reduced @ np.linalg.inv(rep.ub_reduced)
            m_round = np.rint(m)
            if np.max(np.abs(m - m_round)) < 1e-6 and \
                    abs(abs(float(np.linalg.det(m_round)))) == 1:
                fam.members.append((cand, m_round.astype(int)))
                placed = True
                break
        if not placed:
            fam = Family(representative=cand)
            fam.members.append((cand, np.eye(3, dtype=int)))
            families.append(fam)
    return families


def demo_dataset(seed=7):
    """Fixed synthetic dataset: two domains, systematic absences, outliers,
    peak overlap and a known calibration bias. Deterministic by seed."""
    rng = np.random.default_rng(seed)
    offset = np.array([0.0015, -0.0020, 0.0008])

    def domain(rec, hkls, sigma, intensity):
        peaks = []
        for h in hkls:
            q = h @ rec
            q = q + rng.normal(0.0, sigma, 3)
            peaks.append((q, intensity * (0.7 + 0.6 * rng.random()),
                          np.eye(3) * sigma ** 2))
        return peaks

    # Domain A: pseudo-cubic with bcc-like systematic absences (h+k+l odd).
    rec_a = np.array([[0.250, 0.006, 0.000],
                      [0.000, 0.250, 0.004],
                      [0.003, 0.000, 0.250]])
    hkls_a = [h for h in itertools.product(range(-3, 4), repeat=3)
              if h != (0, 0, 0) and sum(h) % 2 == 0]
    # Domain B: smaller tetragonal-like lattice, fewer reflections.
    rec_b = np.array([[0.200, 0.000, 0.000],
                      [0.000, 0.200, 0.005],
                      [0.000, 0.000, 0.111]])
    hkls_b = [h for h in itertools.product(range(-2, 3), repeat=2)
              for l in (-2, -1, 1, 2) if h != (0, 0)]
    peaks = domain(rec_a, hkls_a, 0.0015, 1.0) + domain(rec_b, hkls_b, 0.0015, 0.8)
    # Peak overlap: one extra peak nearly coincident with a domain-A peak.
    q_ov = hkls_a and (np.array([2, 0, 0]) @ rec_a) or np.zeros(3)
    peaks.append((q_ov + np.array([0.004, 0.0, 0.0]), 0.5, np.eye(3) * 0.0015 ** 2))
    # Outliers that belong to no lattice.
    for _ in range(3):
        peaks.append((rng.uniform(-0.9, 0.9, 3), 0.3, np.eye(3) * 0.0015 ** 2))

    reflections = []
    for q, inten, cov in peaks:
        q_obs = q + offset  # calibration bias applied to the raw observation
        reflections.append({
            "position": [float(x) for x in q_obs],
            "intensity": float(inten),
            "covariance": [[float(v) for v in row] for row in cov],
        })
    order = sorted(range(len(reflections)),
                   key=lambda i: -reflections[i]["intensity"])
    reflections = [reflections[i] for i in order]
    return {
        "name": "demo-two-domain",
        "calibration_version": "cal-2026.09",
        "calibration_offset": [float(x) for x in offset],
        "reflections": reflections,
    }
