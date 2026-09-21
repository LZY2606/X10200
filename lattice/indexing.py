"""Candidate generation, hkl indexing, residual/outlier analysis, scoring.

Scoring is deliberately multi-dimensional: indexed fraction, weighted
residual, complexity and outlier budget are reported separately and never
collapsed into a single number.
"""
from __future__ import annotations

import itertools

import numpy as np

from .core import (H_MAX, INDEX_TOL, REFINE_ITERS, is_singular, metric_tensor,
                   reduce_cell)

# Fixed generation / scoring constants.
SEED_PEAKS = 16         # shortest-|q| peaks used to seed basis triples
MAX_CANDIDATES = 40     # kept after scoring, stable-sorted
MAX_PER_FAMILY = 8      # representative candidates kept per lattice family
DEFAULT_OUTLIER_BUDGET = 0.25   # allowed outlier fraction
DEFAULT_WEIGHT_TOL = 9.0        # inlier threshold on weighted residual (3-sigma^2)
OUTLIER_WINDOW = 100.0          # weighted peaks above this weighted residual are "unindexed", else not outliers


def apply_calibration(q: np.ndarray, calibration: dict) -> np.ndarray:
    """q_corr = q * scale + offset (defaults: identity)."""
    scale = float(calibration.get("scale", 1.0))
    offset = np.asarray(calibration.get("offset", [0.0, 0.0, 0.0]), dtype=float)
    return q * scale + offset


def _scalar_weights(covs: np.ndarray) -> np.ndarray:
    """Per-peak scalar weights from covariance diagonals (1/mean variance)."""
    diag = np.diagonal(covs, axis1=1, axis2=2)
    var = np.maximum(diag.mean(axis=1), 1e-12)
    return 1.0 / var


def weighted_residual(residual: np.ndarray, cov: np.ndarray) -> float:
    """r^T Sigma^-1 r with a small diagonal jitter for stability."""
    prec = np.linalg.inv(cov + 1e-12 * np.eye(3))
    return float(residual @ prec @ residual)


def index_peaks(basis: np.ndarray, qs: np.ndarray):
    """Assign integer hkl to each peak. Returns (hkls, frac_err, residuals)."""
    inv = np.linalg.inv(basis)
    frac = (inv @ qs.T).T                      # (n,3) fractional hkl
    hkls = np.rint(frac).astype(int)
    frac_err = np.abs(frac - hkls).max(axis=1)
    predicted = (basis @ hkls.T).T
    residuals = qs - predicted
    too_big = np.abs(hkls).max(axis=1) > H_MAX
    frac_err = np.where(too_big, INDEX_TOL + 1.0, frac_err)
    return hkls, frac_err, residuals


def refine_basis(basis: np.ndarray, hkls: np.ndarray, qs: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    """Weighted least squares: min_B sum_i w_i |q_i - B h_i|^2.

    Rank-deficient hkl sets (e.g. collinear inliers) fall back to the
    pseudo-inverse so degenerate candidates are dropped, not crashed on.
    """
    h = hkls.astype(float)
    gram = h.T @ (weights[:, None] * h)
    reg = 1e-9 * max(float(np.trace(gram)) / 3.0, 1e-12)
    gram = gram + reg * np.eye(3)
    rhs = h.T @ (weights[:, None] * qs)
    try:
        return np.linalg.solve(gram, rhs).T   # rows->columns transpose
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(gram) @ rhs).T


def classify(hkls, frac_err, residuals, covs, weight_tol):
    """Split peaks into inlier / outlier / unindexed with weighted residuals.

    outlier: a plausible integer hkl exists (small fractional error) but the
    weighted residual exceeds the inlier tolerance (e.g. overlapped peak).
    unindexed: no plausible hkl at all.
    """
    prec = np.linalg.inv(covs + 1e-12 * np.eye(3))
    wres = np.einsum("ni,nij,nj->n", residuals, prec, residuals)
    status = []
    for i in range(len(hkls)):
        if frac_err[i] > INDEX_TOL:
            status.append("unindexed")
        elif wres[i] <= weight_tol:
            status.append("inlier")
        elif wres[i] <= OUTLIER_WINDOW:
            status.append("outlier")
        else:
            status.append("unindexed")
    return status, wres


def score_candidate(hkls, status, wres, outlier_budget):
    """Separate score components; never merged into one number."""
    n = len(status)
    inliers = [i for i, s in enumerate(status) if s == "inlier"]
    outliers = [i for i, s in enumerate(status) if s == "outlier"]
    indexed_fraction = len(inliers) / n if n else 0.0
    weighted_res = float(np.mean([wres[i] for i in inliers])) if inliers else float("inf")
    complexity = (float(np.mean([np.abs(hkls[i]).sum() for i in inliers]))
                  if inliers else float("inf"))
    outlier_fraction = len(outliers) / n if n else 0.0
    return {
        "indexed_fraction": indexed_fraction,
        "weighted_residual": weighted_res,
        "complexity": complexity,
        "outliers": len(outliers),
        "outlier_fraction": outlier_fraction,
        "outlier_budget": outlier_budget,
        "outlier_budget_exceeded": outlier_fraction > outlier_budget,
        "n_inliers": len(inliers),
    }


def _sort_key(cand):
    s = cand["scores"]
    # Deterministic total order: fraction desc, residual asc, complexity asc,
    # fewer outliers asc, then basis bytes for a fully stable tie-break.
    return (-s["indexed_fraction"], s["weighted_residual"], s["complexity"],
            s["outliers"], np.asarray(cand["reduced_basis"]).round(9).tobytes())


def generate_candidates(qs: np.ndarray, covs: np.ndarray, calibration: dict,
                        tolerances: dict):
    """Generate, reduce, index, refine and score lattice candidates.

    Multi-domain data: after a first pass seeded by the shortest-|q| peaks,
    peaks left unexplained by the best candidate seed a second pass, so
    each crystalline domain gets its own candidates. Returns a
    deterministically sorted list; singular bases are skipped.
    """
    q = apply_calibration(np.asarray(qs, dtype=float), calibration)
    covs = np.asarray(covs, dtype=float)
    weight_tol = float(tolerances.get("weight_tol", DEFAULT_WEIGHT_TOL))
    outlier_budget = float(tolerances.get("outlier_budget", DEFAULT_OUTLIER_BUDGET))

    order = np.argsort(np.linalg.norm(q, axis=1), kind="stable")
    first_seeds = [int(i) for i in order[:min(SEED_PEAKS, len(q))]]
    candidates = _generate_pass(q, covs, first_seeds, weight_tol,
                                outlier_budget)
    if candidates:
        unexplained = [r["peak_index"] for r in candidates[0]["reflections"]
                       if r["status"] == "unindexed"]
        if len(unexplained) >= 6:
            sub_order = sorted(unexplained,
                               key=lambda i: float(np.linalg.norm(q[i])))
            second_seeds = sub_order[:min(SEED_PEAKS, len(sub_order))]
            candidates += _generate_pass(q, covs, second_seeds, weight_tol,
                                         outlier_budget)
    candidates.sort(key=_sort_key)
    return _diversify(candidates)


def _diversify(candidates):
    """Keep the ranking order but cap representatives per lattice family so
    one domain cannot crowd out another. Family test = unimodular basis
    transform within the fixed tolerance (same rule as family grouping)."""
    from .families import unimodular_transform
    kept = []
    family_bases = []   # list of canonical bases, index = family slot
    family_counts = []
    for cand in candidates:
        basis = np.asarray(cand["basis"], dtype=float)
        slot = None
        for i, fam_basis in enumerate(family_bases):
            if unimodular_transform(basis, fam_basis) is not None:
                slot = i
                break
        if slot is None:
            family_bases.append(basis)
            family_counts.append(0)
            slot = len(family_bases) - 1
        if family_counts[slot] >= MAX_PER_FAMILY:
            continue
        family_counts[slot] += 1
        kept.append(cand)
        if len(kept) >= MAX_CANDIDATES:
            break
    return kept


def _generate_pass(q, covs, seed_idx, weight_tol, outlier_budget):
    """One seed-triple sweep; every peak is still indexed and scored."""
    weights = _scalar_weights(covs)
    candidates = []
    for combo in itertools.combinations(seed_idx, 3):
        seed_basis = q[list(combo), :].T.copy()   # columns = chosen peak vectors
        if is_singular(seed_basis):
            continue
        reduced, chain = reduce_cell(seed_basis)
        if is_singular(reduced):
            continue
        basis = reduced
        refinement_steps = []
        hkls = status = wres = None
        for _ in range(REFINE_ITERS):
            hkls, frac_err, residuals = index_peaks(basis, q)
            status, wres = classify(hkls, frac_err, residuals, covs, weight_tol)
            in_idx = [i for i, s in enumerate(status) if s == "inlier"]
            if len(in_idx) < 3:
                break
            refined = refine_basis(basis, hkls[in_idx], q[in_idx], weights[in_idx])
            if not np.all(np.isfinite(refined)) or is_singular(refined):
                break
            basis = refined
            refinement_steps.append(basis.tolist())
        if hkls is None:
            continue
        hkls, frac_err, residuals = index_peaks(basis, q)
        status, wres = classify(hkls, frac_err, residuals, covs, weight_tol)
        scores = score_candidate(hkls, status, wres, outlier_budget)
        if scores["n_inliers"] < 3:
            continue
        reflections = []
        for i in range(len(q)):
            reflections.append({
                "peak_index": int(i),
                "hkl": [int(v) for v in hkls[i]],
                "predicted_q": (basis @ hkls[i]).tolist(),
                "residual": residuals[i].tolist(),
                "weighted_residual": wres[i],
                "status": status[i],
                "is_outlier": status[i] == "outlier",
            })
        candidates.append({
            "seed_basis": seed_basis.tolist(),
            "seed_peaks": [int(c) for c in combo],
            "basis": basis.tolist(),
            "reduced_basis": reduced.tolist(),
            "transform_chain": chain,
            "refinement_steps": refinement_steps,
            "metric": metric_tensor(basis).tolist(),
            "scores": scores,
            "reflections": reflections,
        })
    return candidates
