"""晶格几何核心：倒易基、Miller 索引、加权残差、基约化与允许基变换。

约定
----
* 倒易点阵向量写作列，列基矩阵 ``B`` 满足格点 ``q = B @ h``（``h`` 为整数 hkl）。
* 所有容差来自 :mod:`app.config`，排序均为稳定、字典序，结果可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as cfg


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def as_vec3(value: Sequence[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"期望长度为 3 的向量，实际 shape={arr.shape}")
    return arr


def vec_key(vec: np.ndarray) -> Tuple[float, float, float]:
    """用于稳定排序/去重的确定性键。"""
    return (round(float(vec[0]), 12),
            round(float(vec[1]), 12),
            round(float(vec[2]), 12))


def nearest_integer(arr: np.ndarray) -> np.ndarray:
    """统一 floor(x+0.5) 向最近整数取整，保证跨平台一致。"""
    return np.floor(np.asarray(arr, dtype=float) + 0.5).astype(int)


def reciprocal_from_direct(direct: np.ndarray) -> np.ndarray:
    """正空间列基 A -> 倒易列基 B = A^{-T}（晶体学约定）。"""
    return np.linalg.inv(direct).T


def is_singular(basis: np.ndarray, tol: float = cfg.SINGULAR_DET_TOL) -> bool:
    return abs(float(np.linalg.det(basis))) <= tol


def is_unimodular(matrix: np.ndarray, tol: float = cfg.UNIMODULAR_TOL) -> bool:
    """允许的基变换：整数矩阵且行列式绝对值为 1。"""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        return False
    if np.max(np.abs(m - np.rint(m))) > tol:
        return False
    return abs(abs(round(float(np.linalg.det(m)))) - 1) <= tol


def cell_parameters(basis: np.ndarray) -> Dict[str, float]:
    """倒易列基对应的正晶胞参数 a,b,c(埃), alpha,beta,gamma(度)。"""
    direct = np.linalg.inv(basis).T
    cols = [direct[:, i] for i in range(3)]
    lengths = [float(np.linalg.norm(c)) for c in cols]

    def angle(u: np.ndarray, v: np.ndarray) -> float:
        cosang = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    return {
        "a": lengths[0],
        "b": lengths[1],
        "c": lengths[2],
        "alpha": angle(cols[1], cols[2]),
        "beta": angle(cols[0], cols[2]),
        "gamma": angle(cols[0], cols[1]),
        "volume": abs(float(np.linalg.det(direct))),
    }


# ---------------------------------------------------------------------------
# 索引与加权残差
# ---------------------------------------------------------------------------
@dataclass
class IndexResult:
    hkl: Tuple[int, int, int]
    predicted: np.ndarray
    residual: np.ndarray
    residual_norm: float          # 几何残差（倒易单位）
    mahalanobis: float            # 加权残差 sqrt(r^T C^-1 r)
    within_index_tol: bool


def whiten_residual(residual: np.ndarray, cov: np.ndarray) -> float:
    """sqrt(r^T C^-1 r)；协方差奇异时退化为几何范数。"""
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(residual))
    solved = np.linalg.solve(chol, residual)
    return float(np.linalg.norm(solved))


def index_point(point: np.ndarray,
                basis: np.ndarray,
                cov: np.ndarray,
                index_tol: float = cfg.INDEX_TOL) -> IndexResult:
    """把单个倒易点 q 索引到最近整数 hkl，并计算预测与残差。"""
    coeffs = np.linalg.solve(basis, point)
    hkl = nearest_integer(coeffs)
    predicted = basis @ hkl
    residual = point - predicted
    norm = float(np.linalg.norm(residual))
    mahal = whiten_residual(residual, cov)
    return IndexResult(
        hkl=tuple(int(v) for v in hkl),
        predicted=predicted,
        residual=residual,
        residual_norm=norm,
        mahalanobis=mahal,
        within_index_tol=norm <= index_tol,
    )


def fit_calibration_offset(points: np.ndarray,
                           covs: np.ndarray,
                           basis: np.ndarray,
                           hkls: np.ndarray) -> np.ndarray:
    """在给定 (B, hkl) 下求最佳常数标定偏移 d = q - B h 的加权最小二乘解。"""
    ata = np.zeros((3, 3))
    atb = np.zeros(3)
    for i in range(points.shape[0]):
        try:
            chol = np.linalg.cholesky(covs[i])
            winv = np.linalg.solve(chol, np.eye(3))
            weight = winv.T @ winv  # C^-1
        except np.linalg.LinAlgError:
            weight = np.eye(3)
        ata += weight
        atb += weight @ (points[i] - basis @ hkls[i])
    return np.linalg.solve(ata, atb)


# ---------------------------------------------------------------------------
# 基约化：确定性贪心格点约化（3D）
# ---------------------------------------------------------------------------
@dataclass
class ReductionRecord:
    """一次基变换的完整审计记录。"""
    operation: str
    matrix: np.ndarray
    basis_before: np.ndarray
    basis_after: np.ndarray
    note: str = ""


@dataclass
class ReductionResult:
    reduced: np.ndarray
    transform: np.ndarray              # B_reduced = B @ transform
    chain: List[ReductionRecord] = field(default_factory=list)


def _basis_sort_key(col: np.ndarray) -> Tuple:
    return (round(float(np.linalg.norm(col)), 12),) + vec_key(col)


def reduce_basis(basis: np.ndarray) -> ReductionResult:
    """对倒易列基做确定性约化。

    交替执行两类允许（幺模）操作：

    * 剪切 b_j <- b_j - round(b_i.b_j/|b_i|^2) b_i；
    * 交换：把最短列稳定换到前面。

    返回累计变换 T 与完整操作链，满足 B_reduced = B @ T。
    """
    b = np.array(basis, dtype=float)
    if is_singular(b):
        raise ValueError("奇异基无法约化：行列式为零")
    total = np.eye(3)
    chain: List[ReductionRecord] = []

    for _ in range(cfg.REDUCE_MAX_STEPS):
        moved = False

        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                bi, bj = b[:, i], b[:, j]
                coeff = int(nearest_integer(
                    np.array([np.dot(bi, bj) / np.dot(bi, bi)]))[0])
                if coeff != 0:
                    m = np.eye(3)
                    m[i, j] = -coeff
                    before = b.copy()
                    b = b @ m
                    total = total @ m
                    chain.append(ReductionRecord(
                        operation=f"shear b{j+1} <- b{j+1} - {coeff}*b{i+1}",
                        matrix=m,
                        basis_before=before,
                        basis_after=b.copy(),
                    ))
                    moved = True

        order = sorted(range(3), key=lambda k: _basis_sort_key(b[:, k]))
        if order != [0, 1, 2]:
            perm = np.zeros((3, 3))
            for new_pos, old_pos in enumerate(order):
                perm[old_pos, new_pos] = 1.0
            before = b.copy()
            b = b @ perm
            total = total @ perm
            chain.append(ReductionRecord(
                operation="swap columns to shortest-first order",
                matrix=perm,
                basis_before=before,
                basis_after=b.copy(),
            ))
            moved = True

        if not moved:
            break
    else:  # pragma: no cover
        raise RuntimeError("基约化未在固定步数内收敛")

    return ReductionResult(reduced=b, transform=total, chain=chain)


# ---------------------------------------------------------------------------
# 同族判定
# ---------------------------------------------------------------------------
def family_transform(basis_a: np.ndarray,
                     basis_b: np.ndarray,
                     tol: float = cfg.FAMILY_TRANSFORM_TOL
                     ) -> Optional[np.ndarray]:
    """若 B_a 与 B_b 相差允许的整数幺模变换，返回 T（B_b = B_a @ T）。

    近似相同但超出 tol 时返回 None —— 绝不合并。
    """
    try:
        candidate = np.linalg.solve(basis_a, basis_b)
    except np.linalg.LinAlgError:
        return None
    rounded = np.rint(candidate)
    if np.max(np.abs(candidate - rounded)) > tol:
        return None
    matrix = rounded.astype(float)
    if not is_unimodular(matrix, tol=tol):
        return None
    return matrix


def transformation_chain(basis_from: np.ndarray,
                         basis_to: np.ndarray) -> List[ReductionRecord]:
    """构造从候选基到同族另一候选基的变换链（经各自约化基中转）。"""
    red_from = reduce_basis(basis_from)
    red_to = reduce_basis(basis_to)
    bridge = family_transform(red_from.reduced, red_to.reduced)
    if bridge is None:
        return []

    chain: List[ReductionRecord] = list(red_from.chain)
    chain.append(ReductionRecord(
        operation="reduced-cell bridge (unimodular, 同族中转)",
        matrix=bridge,
        basis_before=red_from.reduced,
        basis_after=red_from.reduced @ bridge,
    ))
    for step in reversed(red_to.chain):
        inv = np.linalg.inv(step.matrix)
        chain.append(ReductionRecord(
            operation="inverse: " + step.operation,
            matrix=inv,
            basis_before=step.basis_after,
            basis_after=step.basis_before,
        ))
    return chain
