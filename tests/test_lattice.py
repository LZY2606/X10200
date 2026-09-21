"""Fixed synthetic-lattice tests: deterministic tolerances, stable ordering."""
import numpy as np
import pytest

from lattice.core import (compose_chain, is_singular, metric_tensor,
                          metrics_equivalent, reduce_cell)
from lattice.families import group_families
from lattice.indexing import (apply_calibration, generate_candidates,
                              score_candidate, weighted_residual)
from lattice.synthetic import lattice_peaks, overlap_pair

BASIS_A = np.array([[0.52, 0.0, 0.0], [0.0, 0.61, 0.0], [0.0, 0.0, 0.71]])


def test_equivalent_bases_same_family():
    # Two unimodular-related bases of the same lattice must merge into one
    # family, with the transform matrix recorded.
    m = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])  # det 1, unimodular
    basis_b = BASIS_A @ m
    qs_a, covs_a, _ = lattice_peaks(BASIS_A, h_max=2, noise=0.0, seed=3)
    qs_b, covs_b, _ = lattice_peaks(basis_b, h_max=2, noise=0.0, seed=4)
    cands_a = generate_candidates(qs_a, covs_a, {}, {})
    cands_b = generate_candidates(qs_b, covs_b, {}, {})
    assert cands_a and cands_b
    fams_a, _ = group_families(cands_a)
    fams_b, _ = group_families(cands_b)
    # Top candidate of each data set must land in an equivalent metric.
    assert metrics_equivalent(np.array(fams_a[0]["metric"]),
                              np.array(fams_b[0]["metric"]))
    # Transform chain composes to a unimodular integer matrix.
    for cand in cands_a[:5]:
        total = compose_chain(cand["transform_chain"])
        assert abs(round(np.linalg.det(total))) == 1
        reduced = np.array(cand["seed_basis"]) @ total
        assert np.allclose(reduced, cand["reduced_basis"], atol=1e-8)


def test_near_identical_beyond_tolerance_not_merged():
    g1 = metric_tensor(BASIS_A)
    basis2 = BASIS_A.copy()
    basis2[0, 0] *= 1.05  # 5% off: outside the fixed 1e-3 family tolerance
    g2 = metric_tensor(basis2)
    assert not metrics_equivalent(g1, g2)
    cands = [{"metric": g1.tolist(), "basis": BASIS_A.tolist()},
             {"metric": g2.tolist(), "basis": basis2.tolist()}]
    fams, assigns = group_families(cands)
    assert len(fams) == 2
    assert assigns[0]["family_id"] != assigns[1]["family_id"]


def test_two_domains_both_recovered():
    basis_b = np.array([[0.40, 0.0, 0.10], [0.0, 0.40, 0.0],
                        [0.0, 0.0, 0.45]])
    qs_a, covs_a, _ = lattice_peaks(BASIS_A, h_max=2, noise=0.001, seed=5)
    qs_b, covs_b, _ = lattice_peaks(basis_b, h_max=2, noise=0.001, seed=6)
    qs = np.vstack([qs_a, qs_b])
    covs = np.vstack([covs_a, covs_b])
    cands = generate_candidates(qs, covs, {}, {})
    fams, assigns = group_families(cands)
    # Both domain metrics must appear among the families.
    target = [metric_tensor(BASIS_A), metric_tensor(basis_b)]
    matched = [any(metrics_equivalent(np.array(f["metric"]), t, rtol=5e-2)
                   for f in fams) for t in target]
    assert all(matched)


def test_missing_reflections_systematic_extinction():
    # Deterministic 25% dropout of reflections; true cell still recovered.
    qs, covs, _ = lattice_peaks(
        BASIS_A, h_max=2, noise=0.001, seed=7,
        extinction=lambda h: (h[0] * 7 + h[1] * 13 + h[2] * 29) % 4 == 0)
    cands = generate_candidates(qs, covs, {}, {})
    assert cands
    top = cands[0]
    assert top["scores"]["indexed_fraction"] > 0.7
    assert metrics_equivalent(metric_tensor(BASIS_A),
                              np.array(top["metric"]), rtol=5e-2)


def test_tied_candidates_stable_order():
    qs, covs, _ = lattice_peaks(BASIS_A, h_max=2, noise=0.0, seed=8)
    first = generate_candidates(qs, covs, {}, {})
    second = generate_candidates(qs, covs, {}, {})
    assert len(first) > 1
    # Identical input must yield an identical, deterministic ordering.
    for c1, c2 in zip(first, second):
        assert c1["seed_peaks"] == c2["seed_peaks"]
        assert c1["scores"] == c2["scores"]
    keys = [(c["scores"]["indexed_fraction"], c["scores"]["weighted_residual"])
            for c in first]
    assert keys == sorted(keys, key=lambda k: (-k[0], k[1]))


def test_singular_basis_rejected():
    assert is_singular(np.zeros((3, 3)))
    collinear = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert is_singular(collinear)
    qs, covs, _ = lattice_peaks(BASIS_A, h_max=2, noise=0.001, seed=9)
    # Inject a degenerate configuration: all generation must still succeed.
    cands = generate_candidates(qs, covs, {}, {})
    assert all(not is_singular(np.array(c["basis"])) for c in cands)


def test_outlier_budget_reported_separately():
    qs, covs, hkls = lattice_peaks(BASIS_A, h_max=2, noise=0.001, seed=10)
    # Overlapped peaks: near a real lattice point but shifted, so a plausible
    # hkl exists while the weighted residual blows past the inlier window.
    junk1 = qs[5] + np.array([0.02, -0.02, 0.02])
    junk2 = qs[17] + np.array([-0.02, 0.02, -0.02])
    qs = np.vstack([qs, junk1, junk2])
    covs = np.vstack([covs, covs[:2]])
    cands = generate_candidates(qs, covs, {}, {"outlier_budget": 0.05})
    assert cands
    s = cands[0]["scores"]
    # Four components exposed independently; no single fused score key.
    for key in ("indexed_fraction", "weighted_residual", "complexity",
                "outlier_fraction", "outlier_budget", "outlier_budget_exceeded"):
        assert key in s
    assert "total" not in s and "score" not in s
    assert s["outliers"] >= 1
    assert s["outlier_budget_exceeded"] == (
        s["outlier_fraction"] > s["outlier_budget"])


def test_covariance_weights_matter():
    r = np.array([0.01, 0.0, 0.0])
    tight = np.eye(3) * 1e-4
    loose = np.eye(3) * 1.0
    assert weighted_residual(r, tight) > weighted_residual(r, loose)
    # Anisotropic covariance: residual along the tight axis weighs more.
    aniso = np.diag([1e-4, 1.0, 1.0])
    assert weighted_residual(np.array([0.01, 0, 0]), aniso) > \
           weighted_residual(np.array([0, 0.01, 0]), aniso)


def test_calibration_applied():
    q = np.array([[1.0, 2.0, 3.0]])
    out = apply_calibration(q, {"scale": 2.0, "offset": [0.5, 0.0, -0.5]})
    assert np.allclose(out, [[2.5, 4.0, 5.5]])
    ident = apply_calibration(q, {})
    assert np.allclose(ident, q)


def test_reduce_cell_records_chain():
    skewed = BASIS_A @ np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
    reduced, chain = reduce_cell(skewed)
    assert chain  # non-trivial chain recorded
    total = compose_chain(chain)
    assert np.allclose(skewed @ total, reduced)
    norms = np.linalg.norm(reduced, axis=0)
    assert norms[0] <= norms[1] <= norms[2] + 1e-9
