"""晶格候选室核心数值算法。

所有数值比较使用固定容差，所有排序使用稳定排序，保证结果可复现。
计算过程不访问外网，仅依赖 NumPy。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---- 固定容差（不得改为自适应值，测试依赖这些常量） ----
FAMILY_TOL = 1e-6          # 等价晶胞归族容差（变换矩阵接近整数的程度）
REDUCE_TOL = 1e-9          # 晶胞约化迭代容差
SINGULAR_TOL = 1e-8        # 奇异基判定：|det| < SINGULAR_TOL * prod(|bi|)
MIN_ANGLE_COS = 0.96       # 基向量夹角过小时拒绝（约 16 度）
MIN_NORM_DET = 0.02        # 归一化体积下限，排除近共面三重轴
OUTLIER_THRESHOLD = 9.0    # 马氏距离平方阈值，超过则判为离群
OUTLIER_BUDGET_FRAC = 0.15 # 离群预算占观测数的比例
MAX_REDUCE_STEPS = 64
MAX_SEED_VECTORS = 36
MAX_TRIPLES = 60
MAX_EVAL_TRIPLES = 8000


@dataclass
class Assignment:
    obs_index: int
    hkl: Tuple[int, int, int]
    predicted: np.ndarray
    residual: np.ndarray
    d2: float                 # 加权残差 r^T C^-1 r
    is_outlier: bool
    indexed: bool
    overlap: bool = False


@dataclass
class Candidate:
    basis: np.ndarray                       # 约化后的倒易基（列为 b1,b2,b3）
    seed_basis: np.ndarray                  # 生成时的原始基
    reduction_steps: List[list]             # 约化过程记录的整数单模矩阵序列
    assignments: List[Assignment] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    family_id: Optional[int] = None
    family_transform: Optional[list] = None  # 与族代表之间的整数变换矩阵
    seq: int = 0                            # 生成顺序，用于稳定排序


def correct_q(q: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """应用校准偏差，得到校正后的倒易坐标。"""
    return np.asarray(q, dtype=float) - np.asarray(bias, dtype=float)


def is_valid_triple(B: np.ndarray) -> bool:
    """排除奇异基与近共线/共面基。"""
    B = np.asarray(B, dtype=float)
    norms = np.linalg.norm(B, axis=0)
    if np.any(norms < REDUCE_TOL):
        return False
    det = abs(float(np.linalg.det(B)))
    if det < SINGULAR_TOL * float(np.prod(norms)):
        return False
    if det < MIN_NORM_DET * float(np.prod(norms)):
        return False
    for i, j in itertools.combinations(range(3), 2):
        cos = abs(float(B[:, i] @ B[:, j]) / (norms[i] * norms[j]))
        if cos > MIN_ANGLE_COS:
            return False
    return True


def reduce_basis(B: np.ndarray, tol: float = REDUCE_TOL) -> Tuple[np.ndarray, List[list]]:
    """迭代式 Minkowski 约化；返回 (约化基, 变换矩阵序列)。

    每一步记录整数单模矩阵 M，满足 B_new = B_old @ M，
    完整保留晶胞约化与坐标变换过程。
    """
    B = np.asarray(B, dtype=float).reshape(3, 3).copy()
    steps: List[list] = []
    if float(np.linalg.det(B)) < 0:
        B[:, 0] *= -1.0
        steps.append([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for _ in range(MAX_REDUCE_STEPS):
        changed = False
        # 逐对缩短：若 b_i ± b_j 比 b_i 更短则替换
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                ni = float(B[:, i] @ B[:, i])
                for s in (1.0, -1.0):
                    v = B[:, i] + s * B[:, j]
                    if float(v @ v) < ni - tol:
                        M = np.eye(3, dtype=int)
                        M[j, i] = int(s)
                        B = B @ M
                        steps.append(M.tolist())
                        ni = float(B[:, i] @ B[:, i])
                        changed = True
        lengths = np.linalg.norm(B, axis=0)
        order = np.argsort(lengths, kind="stable")
        if not np.array_equal(order, np.arange(3)):
            P = np.zeros((3, 3), dtype=int)
            for new_i, old_i in enumerate(order):
                P[old_i, new_i] = 1
            B = B @ P
            steps.append(P.tolist())
            changed = True
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                nj = float(B[:, j] @ B[:, j])
                if nj < tol:
                    continue
                bij = float(B[:, i] @ B[:, j])
                if abs(bij) > 0.5 * nj + tol:
                    k = int(np.round(bij / nj))
                    if k != 0:
                        M = np.eye(3, dtype=int)
                        M[j, i] = -k
                        B = B @ M
                        steps.append(M.tolist())
                        changed = True
        if not changed:
            break
    return B, steps


def index_reflections(
    B: np.ndarray,
    qs: Sequence[np.ndarray],
    covs: Sequence[np.ndarray],
    locked: Sequence[bool],
    outlier_threshold: float = OUTLIER_THRESHOLD,
) -> List[Assignment]:
    """为每个反射给出 hkl、预测位置、残差与离群标记。"""
    Binv = np.linalg.inv(B)
    assignments: List[Assignment] = []
    for idx, (q, C, lk) in enumerate(zip(qs, covs, locked)):
        q = np.asarray(q, dtype=float)
        C = np.asarray(C, dtype=float).reshape(3, 3)
        hkl_real = Binv @ q
        hkl = np.round(hkl_real).astype(int)
        pred = B @ hkl
        r = q - pred
        d2 = float(r @ np.linalg.inv(C) @ r)
        is_outlier = bool(d2 > outlier_threshold and not lk)
        indexed = bool((not is_outlier) and np.any(hkl != 0))
        assignments.append(
            Assignment(
                obs_index=idx,
                hkl=(int(hkl[0]), int(hkl[1]), int(hkl[2])),
                predicted=pred,
                residual=r,
                d2=d2,
                is_outlier=is_outlier,
                indexed=indexed,
            )
        )
    # 峰重叠：多个观测被指派到同一 hkl
    seen = {}
    for a in assignments:
        if a.indexed:
            seen.setdefault(a.hkl, []).append(a)
    for group in seen.values():
        if len(group) > 1:
            for a in group:
                a.overlap = True
    return assignments


def score_candidate(
    assignments: List[Assignment],
    B: np.ndarray,
    outlier_budget_frac: float = OUTLIER_BUDGET_FRAC,
) -> dict:
    """候选评分：四个分量分别展示，不合成单一总分。"""
    n = len(assignments)
    indexed = [a for a in assignments if a.indexed]
    outliers = [a for a in assignments if a.is_outlier]
    budget = max(1, int(np.floor(outlier_budget_frac * n))) if n else 0
    weighted_residual = (
        float(np.mean([a.d2 for a in indexed])) if indexed else float("inf")
    )
    return {
        "indexed_fraction": (len(indexed) / n) if n else 0.0,
        "weighted_residual": weighted_residual,
        "complexity": abs(float(np.linalg.det(B))),
        "outlier_used": len(outliers),
        "outlier_budget": budget,
        "over_budget": len(outliers) > budget,
        "n_indexed": len(indexed),
        "n_total": n,
    }


def _candidate_sort_key(c: Candidate):
    # 稳定排序：相等键保持生成顺序（seq），保证并列候选次序确定
    return (
        -c.scores["indexed_fraction"],
        c.scores["weighted_residual"],
        c.scores["complexity"],
        c.seq,
    )


def generate_candidates(
    qs: Sequence[np.ndarray],
    covs: Sequence[np.ndarray],
    locked: Sequence[bool],
    max_candidates: int = 8,
    outlier_threshold: float = OUTLIER_THRESHOLD,
) -> List[Candidate]:
    """从观测向量中生成多个晶格候选。"""
    qs = [np.asarray(q, dtype=float) for q in qs]
    n = len(qs)
    if n < 3:
        return []
    order = sorted(range(n), key=lambda i: (float(np.linalg.norm(qs[i])), i))
    seed_idx = order[:MAX_SEED_VECTORS]
    triples = []
    for combo in itertools.combinations(seed_idx, 3):
        B0 = np.column_stack([qs[i] for i in combo])
        if is_valid_triple(B0):
            triples.append((combo, B0))
            if len(triples) >= MAX_EVAL_TRIPLES:
                break

    # 快速指标化计数：每个三重轴对全部观测试指标化，按可解释反射数筛选
    Q = np.array(qs)
    Winv = np.array([np.linalg.inv(np.asarray(C, dtype=float).reshape(3, 3))
                     for C in covs])
    scored = []
    for combo, B0 in triples:
        Binv = np.linalg.inv(B0)
        H = Q @ Binv.T
        R = Q - np.round(H) @ B0.T
        d2 = np.einsum("ni,nij,nj->n", R, Winv, R)
        cnt = int(np.sum(d2 <= outlier_threshold))
        scored.append((-cnt, abs(float(np.linalg.det(B0))), combo, B0))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))  # 稳定排序
    triples = [(combo, B0) for _, _, combo, B0 in scored[:MAX_TRIPLES]]

    candidates: List[Candidate] = []
    for seq, (_, B0) in enumerate(triples):
        B_red, steps = reduce_basis(B0)
        assignments = index_reflections(B_red, qs, covs, locked, outlier_threshold)
        scores = score_candidate(assignments, B_red)
        if scores["n_indexed"] < 3:
            continue
        candidates.append(
            Candidate(
                basis=B_red,
                seed_basis=B0,
                reduction_steps=steps,
                assignments=assignments,
                scores=scores,
                seq=seq,
            )
        )
    candidates.sort(key=_candidate_sort_key)
    return candidates[:max_candidates]


def group_families(
    candidates: List[Candidate], tol: float = FAMILY_TOL
) -> List[List[Candidate]]:
    """等价晶胞经允许的基变换归为同族。

    变换矩阵与原候选都保留；近似相同却超出容差的候选不合并。
    返回族列表，每个族的第一个元素为代表。
    """
    families: List[List[Candidate]] = []
    for c in candidates:
        placed = False
        for fam in families:
            rep = fam[0]
            M_real = np.linalg.inv(rep.basis) @ c.basis
            M = np.round(M_real)
            if (
                np.all(np.abs(M_real - M) < tol)
                and abs(int(round(float(np.linalg.det(M))))) == 1
            ):
                c.family_transform = M.astype(int).tolist()
                fam.append(c)
                placed = True
                break
        if not placed:
            c.family_transform = np.eye(3, dtype=int).tolist()
            families.append([c])
    for fid, fam in enumerate(families):
        for c in fam:
            c.family_id = fid
    return families
