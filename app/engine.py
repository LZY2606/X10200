"""Candidate generation, scoring, domains, extinction/overlap and fixtures."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import lattice as L

I3 = L.I3


def calibrated_observations(peaks: Sequence[Dict[str, Any]],
                            calibration: Optional[Dict[str, Any]]
                            ) -> Tuple[np.ndarray, np.ndarray]:
    points = np.vstack([
        L.apply_calibration(np.asarray(p["position"], dtype=float), calibration)
        for p in peaks
    ])
    covariances = np.asarray([np.asarray(p["covariance"], dtype=float)
                              for p in peaks])
    return points, covariances


def quick_inliers(points, covariances, basis):
    result = []
    for idx, (point, cov) in enumerate(zip(points, covariances)):
        item = L.index_point(point, cov, basis)
        if not item["outlier"]:
            result.append((idx, item))
    return result


def _direction_groups(points, covariances, locked_indices):
    """Stable groups of peaks lying on one reciprocal-lattice line.

    The shortest representative of each line is a candidate basis vector.
    This naturally separates domains whose vectors merely coincide as points.
    """
    order = sorted(range(len(points)),
                   key=lambda i: (float(points[i] @ points[i]), i))
    shortlist = order[: max(L.SHORTLIST * 2, L.SHORTLIST + len(locked_indices))]
    locked = sorted(locked_indices)
    groups: List[Dict[str, Any]] = []
    cos_tol = 0.985
    for idx in shortlist:
        vector = points[idx]
        norm = float(vector @ vector)
        if norm < 1.0e-12:
            continue
        unit = vector / np.sqrt(norm)
        matched = None
        for group in groups:
            representative = group["unit"]
            if abs(float(representative @ unit)) >= cos_tol:
                matched = group
                break
        if matched is None:
            groups.append({"unit": unit, "members": [idx], "norm": norm})
        else:
            matched["members"].append(idx)
            matched["norm"] = min(matched["norm"], norm)
    for idx in locked:
        if not any(idx in group["members"] for group in groups):
            vector = points[idx]
            groups.append({"unit": vector / np.sqrt(max(float(vector @ vector),
                                                        1e-12)),
                           "members": [idx], "norm": float(vector @ vector)})
    groups.sort(key=lambda g: (g["norm"], min(g["members"])))
    return groups


def seed_triplets(points, covariances, locked_indices, tolerance=L.TOL_RESIDUAL):
    """Deterministic basis-vector triplets from stable direction groups."""
    groups = _direction_groups(points, covariances, locked_indices)
    representatives = [sorted(group["members"],
                              key=lambda i: (float(points[i] @ points[i]), i))[0]
                       for group in groups]
    seeds = []
    seen = set()
    for triplet in itertools.combinations(range(len(representatives)), 3):
        peak_triplet = tuple(representatives[i] for i in triplet)
        key = tuple(sorted(peak_triplet))
        if key in seen:
            continue
        seen.add(key)
        basis = np.column_stack([points[i] for i in peak_triplet])
        if L.basis_volume(basis) < L.TOL_SINGULAR:
            continue
        inliers = quick_inliers(points, covariances, basis)
        if len(inliers) < L.MIN_CANDIDATE_INLIERS:
            continue
        seeds.append((len(inliers), peak_triplet, basis,
                      np.eye(3, dtype=int), peak_triplet,
                      "direction_group_triplet"))
    seeds.sort(key=lambda row: (-row[0], row[1]))
    return seeds[:L.TOP_SEEDS]


def transform_chain_for_seed(basis, refinement, refined_basis):
    chain = [{
        "operation": "seed from three observed reciprocal vectors",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "cumulative": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "basis": [[float(v) for v in col] for col in basis.T],
        "vectors": None,
    }]
    if refinement:
        local = np.linalg.solve(basis, refined_basis)
        chain.append({
            "operation": "weighted least-squares refinement",
            "matrix": [[float(v) for v in row] for row in local],
            "cumulative": [[float(v) for v in row] for row in local],
            "basis": [[float(v) for v in col] for col in refined_basis.T],
            "vectors": None,
        })
    reduced, integer_matrix, reduction_steps = L.reduce_basis(refined_basis)
    chain.extend(reduction_steps)
    return chain, reduced, integer_matrix


def _domain_components(inliers):
    """Split seed inliers into at most two one-to-one HKL components.

    A mixed two-domain seed can give two different physical points the same
    provisional HKL.  Components are built stably by choosing, for each HKL,
    the closest previously placed point; component A receives the smallest
    peak index at the first conflict so ordering is deterministic.
    """
    components: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {0: [], 1: []}
    anchors: Dict[int, Dict[Tuple[int, int, int], np.ndarray]] = {0: {}, 1: {}}
    grouped: Dict[Tuple[int, int, int], List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, item in inliers:
        grouped.setdefault(tuple(int(v) for v in item["hkl"]), []).append(
            (idx, item))
    for hkl in sorted(grouped):
        rows = sorted(grouped[hkl], key=lambda row: (row[0],))
        if len(rows) == 1:
            idx, item = rows[0]
            distances = []
            observed = np.asarray(item["predicted"]) + np.asarray(
                item["residual_vector"])
            for component in (0, 1):
                anchor = anchors[component].get(hkl)
                if anchor is None:
                    distances.append((1, component, 0.0))
                else:
                    distances.append(
                        (0, component, float(np.sum((observed - anchor) ** 2))))
            component = sorted(distances, key=lambda x: (x[0], x[2], x[1]))[0][1]
            components[component].append((idx, item))
            anchors[component][hkl] = observed
        else:
            # Explicit overlap/conflict: each distinct physical assignment
            # seeds a separate component in stable peak-index order.
            order = sorted(rows, key=lambda row: row[0])
            for component, (idx, item) in enumerate(order[:2]):
                components[component].append((idx, item))
                observed = np.asarray(item["predicted"]) + np.asarray(
                    item["residual_vector"])
                anchors[component][tuple(int(v) for v in item["hkl"])] = observed
    return [c for c in (components[0], components[1])
            if len(c) >= L.MIN_CANDIDATE_INLIERS]


def _build_one_component(seed_basis, first, points, covariances, excluded,
                         seed_indices, source):
    active = [(i, item) for i, item in first if i not in excluded]
    hkls = np.vstack([np.asarray(item["hkl"], dtype=int)
                      for _, item in active])
    active_points = np.vstack([points[i] for i, _ in active])
    active_cov = np.asarray([covariances[i] for i, _ in active])
    refined, ok = L.refine_basis(active_points, active_cov, seed_basis, hkls)
    second = []
    for i, cov in enumerate(covariances):
        item = L.index_point(points[i], cov, refined)
        if not item["outlier"] and i not in excluded:
            second.append(i)
    if len(second) >= L.MIN_CANDIDATE_INLIERS:
        hkls2 = np.vstack([np.rint(np.linalg.solve(refined, points[i]))
                           for i in second]).astype(int)
        refined2, ok2 = L.refine_basis(
            points[np.asarray(second)],
            np.asarray([covariances[i] for i in second]), refined, hkls2)
        if ok2:
            refined = refined2
    chain, reduced, integer_matrix = transform_chain_for_seed(
        seed_basis, ok, refined)
    signature, reduced_columns = L.canonical_signature(refined)
    return {
        "seed_basis": seed_basis,
        "basis": refined,
        "reduced_basis": np.column_stack(reduced_columns),
        "seed_indices": seed_indices,
        "source": source,
        "signature": signature,
        "transform_chain": chain,
        "integer_reduction": integer_matrix,
    }


def build_candidate_components(seed, points, covariances, excluded):
    _, _, seed_basis, _, seed_indices, source = seed
    first = quick_inliers(points, covariances, seed_basis)
    components = _domain_components(first)
    return [_build_one_component(seed_basis, component, points, covariances,
                                excluded, seed_indices, source)
            for component in components]


def candidate_correspondences(candidate, points, covariances, excluded,
                              locks):
    rows = []
    for idx, (point, cov) in enumerate(zip(points, covariances)):
        item = L.index_point(point, cov, candidate["basis"])
        item["peak_id"] = idx
        item["manually_excluded"] = idx in excluded
        item["locked"] = idx in locks
        item["overlap"] = False
        item["domain"] = None
        if idx in excluded:
            item["outlier"] = True
            item["status"] = "excluded"
        elif item["outlier"]:
            item["status"] = "outlier"
        else:
            item["status"] = "indexed"
        rows.append(item)
    return rows


def missing_reflections(candidate, correspondences):
    indexed_hkl = {tuple(r["hkl"]) for r in correspondences
                   if r["status"] == "indexed"}
    basis = candidate["reduced_basis"]
    expected = {}
    for h in range(-L.MISSING_RADIUS, L.MISSING_RADIUS + 1):
        for k in range(-L.MISSING_RADIUS, L.MISSING_RADIUS + 1):
            for ell in range(-L.MISSING_RADIUS, L.MISSING_RADIUS + 1):
                hkl = (h, k, ell)
                radius = h * h + k * k + ell * ell
                if radius == 0 or radius > L.MISSING_RADIUS * L.MISSING_RADIUS:
                    continue
                expected[hkl] = basis @ np.asarray(hkl, dtype=float)
    # A hole is a missing point whose direct integer-lattice neighbours exist.
    missing = []
    for hkl, position in expected.items():
        if hkl in indexed_hkl:
            continue
        neighbours = 0
        for axis in range(3):
            for sign in (-1, 1):
                probe = list(hkl)
                probe[axis] += sign
                if tuple(probe) in indexed_hkl:
                    neighbours += 1
        supported = neighbours >= 4
        if supported:
            missing.append({"hkl": hkl,
                            "predicted": [float(v) for v in position]})
    missing.sort(key=lambda row: (sum(v * v for v in row["hkl"]), row["hkl"]))
    return missing


def score_candidate(correspondences, peak_count, outlier_budget):
    indexed = [r for r in correspondences if r["status"] == "indexed"]
    outliers = [r for r in correspondences
                if r["status"] in ("outlier", "excluded")]
    chi = [r["chi_sq"] for r in indexed]
    weighted_rms = float(np.sqrt(np.mean(chi))) if chi else None
    # Volume is filled by the caller after reduced-basis construction.
    used = len(outliers)
    return {
        "indexed_ratio": len(indexed) / float(peak_count),
        "indexed_count": len(indexed),
        "weighted_rms": weighted_rms,
        "complexity": {
            "continuous_parameters": 9,
            "reciprocal_cell_volume": 0.0,
            "log_volume": None,
        },
        "outlier_budget": {
            "budget": int(outlier_budget),
            "used": used,
            "remaining": int(outlier_budget) - used,
            "over_budget": used > int(outlier_budget),
        },
    }


def union_find_groups(candidate_rows, jaccard=0.30):
    parents = list(range(len(candidate_rows)))

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[max(ra, rb)] = min(ra, rb)

    sets = []
    for rows in candidate_rows:
        sets.append({r["peak_id"] for r in rows if r["status"] == "indexed"})
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union_set = sets[i] | sets[j]
            if union_set and len(sets[i] & sets[j]) / len(union_set) >= jaccard:
                union(i, j)
    groups = {}
    for i in range(len(candidate_rows)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def rank_key(metric):
    rms = metric["weighted_rms"]
    return (
        1 if metric["outlier_budget"]["over_budget"] else 0,
        -metric["indexed_count"],
        float("inf") if rms is None else rms,
        metric["complexity"]["log_volume"],
    )


def generate_candidates(peaks, calibration, excluded, locked,
                        outlier_budget=3, limit=6):
    """Produce distinct, stably ordered lattice candidates and decorations."""
    points, covariances = calibrated_observations(peaks, calibration)
    seeds = seed_triplets(points, covariances, sorted(locked))
    candidates = []
    seen_signatures = set()
    for seed in seeds:
        for candidate in build_candidate_components(
                seed, points, covariances, excluded):
            dedup_key = (candidate["signature"],
                         tuple(sorted(candidate["seed_indices"])),
                         tuple(int(round(x / L.TOL_MATCH)) for x in
                               np.asarray(candidate["basis"]).ravel()))
            if dedup_key in seen_signatures:
                continue
            seen_signatures.add(dedup_key)
            correspondences = candidate_correspondences(
                candidate, points, covariances, excluded, locked)
            volume = L.basis_volume(candidate["reduced_basis"])
            candidate["correspondences"] = correspondences
            candidate["missing"] = missing_reflections(candidate, correspondences)
            candidate["metric"] = score_candidate(
                correspondences, len(peaks), outlier_budget)
            candidate["metric"]["complexity"]["reciprocal_cell_volume"] = volume
            candidate["metric"]["complexity"]["log_volume"] = float(np.log10(volume))
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break

    # Stable preliminary ordering before domains/families are assigned.
    candidates.sort(key=lambda c: rank_key(c["metric"]))

    groups = union_find_groups([c["correspondences"] for c in candidates])
    groups.sort(key=lambda members: min(rank_key(candidates[i]["metric"])
                                        for i in members))
    domain_by_candidate = {}
    for domain_number, members in enumerate(groups, start=1):
        for i in members:
            domain_by_candidate[i] = f"D{domain_number}"
    for idx, candidate in enumerate(candidates):
        domain = domain_by_candidate[idx]
        candidate["domain"] = domain
        for row in candidate["correspondences"]:
            if row["status"] == "indexed":
                row["domain"] = domain

    # Overlap: the same observation explains inliers from different domains.
    by_peak: Dict[int, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        for row in candidate["correspondences"]:
            if row["status"] == "indexed":
                by_peak.setdefault(row["peak_id"], []).append(row)
    for rows in by_peak.values():
        if len({r["domain"] for r in rows}) > 1:
            for row in rows:
                row["overlap"] = True

    # Families are created in final stable candidate order.
    family_index = {}
    for candidate in candidates:
        sig = candidate["signature"]
        if sig not in family_index:
            family_index[sig] = f"F{len(family_index) + 1}"
        candidate["family"] = family_index[sig]

    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return candidates


# ---------------------------------------------------------------------------
# Fixed synthetic lattice fixture
# ---------------------------------------------------------------------------
def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def synthetic_lattice(seed=1157):
    """Fixed two-domain cubic reciprocal lattice with controlled effects.

    Domain 1 misses +-(1,0,0), demonstrating systematic missing reflections.
    Domain 2 is rotated and deliberately shares one observed peak (overlap).
    Anisotropic covariances and two non-indexed noise peaks exercise weights
    and the outlier budget.  Raw points include a fixed calibration offset;
    calibration v2 supplies the known correction.
    """
    rng = np.random.RandomState(seed)
    true_basis = np.diag([1.0, 1.5, 0.8])
    rotate = rotation_z(np.pi / 2.0)
    absent = {(1, 0, 0), (-1, 0, 0)}
    peaks = []

    def add_peak(position, intensity, covariance, domain, hkl, note=""):
        peaks.append({
            "position": [float(v) for v in position],
            "intensity": float(intensity),
            "covariance": [[float(v) for v in row] for row in covariance],
            "true_domain": domain,
            "true_hkl": hkl,
            "note": note,
        })

    radius_sq = 5.0
    for h in range(-2, 3):
        for k in range(-2, 3):
            for ell in range(-2, 3):
                hkl = (h, k, ell)
                if sum(v * v for v in hkl) == 0 or hkl in absent:
                    continue
                base = true_basis @ np.asarray(hkl, dtype=float)
                if float(base @ base) > radius_sq:
                    continue
                cov = np.diag([0.0009, 0.0009, 0.0016])
                add_peak(base, 100.0 / (1 + sum(v * v for v in hkl)),
                         cov, "D1", list(hkl))

    # Domain 2: rotated.  Use (0,1,1) so its predicted point coincides with a
    # domain-1 vector and creates a deterministic overlap at (0,-1,1).
    for h in range(-2, 3):
        for k in range(-2, 3):
            for ell in range(-2, 3):
                hkl = (h, k, ell)
                norm = sum(v * v for v in hkl)
                if norm == 0:
                    continue
                vector = rotate @ (true_basis @ np.asarray(hkl, dtype=float))
                if float(vector @ vector) > radius_sq:
                    continue
                cov = np.diag([0.0012, 0.0012, 0.0008])
                add_peak(vector, 80.0 / (1 + norm), cov, "D2", list(hkl),
                         "second crystalline domain")

    # Deterministic noise/outlier observations.
    noise = [np.array([0.42, 0.88, -0.31]),
             np.array([-0.71, 0.37, 0.53])]
    for i, vector in enumerate(noise):
        add_peak(vector, 8.0 + i, np.diag([0.002, 0.002, 0.002]),
                 None, None, "noise")

    offset = np.array([0.015, -0.010, 0.000])
    calibration_v1 = {"version": "v1", "rotation": I3.tolist(),
                      "offset": [0.0, 0.0, 0.0], "notes": "import default"}
    calibration_v2 = {"version": "v2", "rotation": I3.tolist(),
                      "offset": [float(v) for v in offset],
                      "notes": "known detector translation correction"}
    for peak in peaks:
        raw = np.asarray(peak["position"]) + offset
        peak["position"] = [float(v) for v in raw]
    return {
        "peaks": peaks,
        "calibrations": [calibration_v1, calibration_v2],
        "true_basis": [[float(v) for v in col] for col in true_basis.T],
        "domain_rotation": [[float(v) for v in col] for col in rotate.T],
        "absent": [list(v) for v in sorted(absent)],
        "outlier_budget": 2,
    }
