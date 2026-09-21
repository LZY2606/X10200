"""Group equivalent cells into families via allowed (unimodular) basis
transforms. Original candidates and their transform matrices are kept;
candidates related by no unimodular transform -- including near-identical
ones beyond the fixed tolerance -- are never merged.
"""
from __future__ import annotations

import numpy as np

# Fixed tolerance: fractional-hkl matrix entries must be this close to
# integers for two cells to count as the same lattice.
UNIMODULAR_ATOL = 0.02


def unimodular_transform(b_from: np.ndarray, b_to: np.ndarray,
                         atol: float = UNIMODULAR_ATOL):
    """Integer unimodular M with b_to ~= b_from @ M, or None."""
    m = np.linalg.inv(b_from) @ b_to
    rounded = np.rint(m).astype(int)
    if not np.all(np.abs(m - rounded) <= atol):
        return None
    if abs(int(round(np.linalg.det(rounded)))) != 1:
        return None
    return rounded


def group_families(candidates):
    """Assign family ids. Returns (families, assignments).

    families: list of {id, canonical_basis, metric}
    assignments: candidate index -> {family_id, transform_to_family}
    where canonical_basis ~= candidate_basis @ transform_to_family.
    """
    families = []
    assignments = []
    for cand in candidates:
        basis = np.asarray(cand["basis"], dtype=float)
        placed = None
        transform = None
        for fam in families:
            transform = unimodular_transform(
                basis, np.asarray(fam["canonical_basis"], dtype=float))
            if transform is not None:
                placed = fam
                break
        if placed is None:
            placed = {"id": len(families), "canonical_basis": basis.tolist(),
                      "metric": cand["metric"]}
            families.append(placed)
            transform = np.eye(3, dtype=int)
        assignments.append({"family_id": placed["id"],
                            "transform_to_family": transform.tolist()})
    return families, assignments
