"""晶格候选室核心数值算法。

所有容差为固定常数；所有排序使用显式键，保证结果确定、可复现。
"""
from __future__ import annotations

import itertools

import numpy as np

# ---- 固定容差（测试依赖这些值，不得随意调整） ----
TOL_SINGULAR = 1e-10          # 基行列式奇异阈值（相对）
INLIER_CHI2 = 9.0             # 内点阈值：3 sigma, 3 自由度卡方
MIN_INDEXED_FRACTION = 0.3    # 候选最低已索引比例
OUTLIER_BUDGET_FRACTION = 0.6  # 离群预算（占总反射比例）
FAMILY_DECIMALS = 4           # 等价晶胞归族舍入精度（超出此误差不合并）
CANON_DECIMALS = 6            # 候选去重舍入精度
MAX_BASIS_PEAKS = 12          # 参与三重组合枚举的最强峰数
MAX_CANDIDATES = 64
LLL_DELTA = 0.75
MINKOWSKI_BOUND = 3           # 最短矢量搜索的整数系数范围

FIXED_TOLERANCES = {
    "tol_singular": TOL_SINGULAR,
    "inlier_chi2": INLIER_CHI2,
    "min_indexed_fraction": MIN_INDEXED_FRACTION,
    "outlier_budget_fraction": OUTLIER_BUDGET_FRACTION,
    "family_decimals": FAMILY_DECIMALS,
    "canon_decimals": CANON_DECIMALS,
    "lll_delta": LLL_DELTA,
}


def cov_inv(cov6):
    """由 6 分量 (xx,xy,xz,yy,yz,zz) 构造协方差逆矩阵。"""
    c = np.array(
        [[cov6[0], cov6[1], cov6[2]],
         [cov6[1], cov6[3], cov6[4]],
         [cov6[2], cov6[4], cov6[5]]],
        dtype=float,
    )
    return np.linalg.inv(c)


def index_peaks(B, Q, covs, locked=None):
    """对基 B 指标化所有峰。locked: {行号: (h,k,l)} 强制指定 hkl。"""
    Hf = np.linalg.solve(B, Q.T).T
    H = np.rint(Hf).astype(int)
    if locked:
        for i, hkl in locked.items():
            H[i] = np.asarray(hkl, dtype=int)
    R = Q - H @ B.T
    chi2 = np.einsum("ni,nij,nj->n", R, covs, R)
    return H, R, chi2


def refine_basis(Q, H, covs):
    """加权最小二乘精化基 B，权重为协方差逆（全 3x3）。"""
    A = np.zeros((9, 9))
    b = np.zeros(9)
    for q, h, ci in zip(Q, H, covs):
        A += np.kron(np.outer(h, h), ci)
        b += np.kron(h, ci @ q)
    v = np.linalg.solve(A, b)
    return v.reshape((3, 3), order="F")


def _gram_schmidt(B):
    n = B.shape[1]
    Bs = np.zeros_like(B)
    mu = np.zeros((n, n))
    norm = np.zeros(n)
    for i in range(n):
        Bs[:, i] = B[:, i].copy()
        for j in range(i):
            if norm[j] > 0:
                mu[i, j] = float(B[:, i] @ Bs[:, j]) / norm[j]
                Bs[:, i] -= mu[i, j] * Bs[:, j]
        norm[i] = float(Bs[:, i] @ Bs[:, i])
    return mu, norm


def lll_reduce(B, delta=LLL_DELTA):
    """对列基做 LLL 约化。返回 (约化基, 变换矩阵 M, 步骤记录)，B_red = B @ M。"""
    B = np.array(B, dtype=float)
    n = B.shape[1]
    M = np.eye(n, dtype=int)
    steps = []
    k = 1
    while k < n:
        mu, norm = _gram_schmidt(B)
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k, j]))
            if q != 0:
                B[:, k] -= q * B[:, j]
                M[:, k] -= q * M[:, j]
                T = np.eye(n, dtype=int)
                T[j, k] = -q
                steps.append({"kind": "size_reduce", "matrix": T.tolist()})
                mu, norm = _gram_schmidt(B)
        if norm[k] >= (delta - mu[k, k - 1] ** 2) * norm[k - 1]:
            k += 1
        else:
            B[:, [k - 1, k]] = B[:, [k, k - 1]]
            M[:, [k - 1, k]] = M[:, [k, k - 1]]
            S = np.eye(n, dtype=int)
            S[k - 1, k - 1] = 0
            S[k, k] = 0
            S[k - 1, k] = 1
            S[k, k - 1] = 1
            steps.append({"kind": "swap", "matrix": S.tolist()})
            k = max(k - 1, 1)
    return B, M, steps


def minkowski_canonical(B):
    """Minkowski 意义下的规范基：最短独立矢量组，符号/顺序确定化。

    返回 (规范基 Bc, 幺模变换 U)，Bc = B @ U，det(U) = +1。
    """
    rng = range(-MINKOWSKI_BOUND, MINKOWSKI_BOUND + 1)
    H = np.array([h for h in itertools.product(rng, repeat=3) if any(h)], dtype=int)
    V = H @ B.T
    norms = np.einsum("ij,ij->i", V, V)
    eps = 1e-9 * float(norms.max())

    def pick(idxs, sign_of=None):
        best = None
        for i in idxs:
            for s in (1, -1):
                if sign_of is not None and s * sign_of[i] <= 0:
                    continue
                key = tuple(np.round(s * V[i], 9))
                if best is None or key < best[0]:
                    best = (key, i, s)
        return best[1], best[2]

    n1 = norms.min()
    i1, s1 = pick(np.nonzero(norms <= n1 + eps)[0])
    v1 = s1 * V[i1]
    h1 = s1 * H[i1]

    cross = np.linalg.norm(
        np.cross(V, np.broadcast_to(v1, V.shape)), axis=1
    )
    indep2 = np.nonzero(cross > eps)[0]
    if len(indep2) == 0:
        return B.copy(), np.eye(3, dtype=int)
    n2 = norms[indep2].min()
    i2, s2 = pick(indep2[norms[indep2] <= n2 + eps])
    v2 = s2 * V[i2]
    h2 = s2 * H[i2]

    triple = np.cross(np.broadcast_to(v2, V.shape), V) @ v1
    indep3 = np.nonzero(np.abs(triple) > eps)[0]
    if len(indep3) == 0:
        return B.copy(), np.eye(3, dtype=int)
    n3 = norms[indep3].min()
    i3, s3 = pick(indep3[norms[indep3] <= n3 + eps], sign_of=triple)
    v3 = s3 * V[i3]
    h3 = s3 * H[i3]

    Bc = np.column_stack([v1, v2, v3])
    U = np.column_stack([h1, h2, h3]).astype(int)
    return Bc, U


def canonical_of(B):
    """便捷函数：LLL + Minkowski 规范形。"""
    Bred, _, _ = lll_reduce(B)
    Bc, _ = minkowski_canonical(Bred)
    return Bc


def family_key_of(Bc):
    """归族键：按 FAMILY_DECIMALS 舍入；超出此误差的近似晶胞不合并。"""
    return tuple(np.round(np.asarray(Bc, dtype=float).flatten(), FAMILY_DECIMALS))


def sort_candidates(cands):
    """确定性排序：分数并列时按归族键与基矩阵字典序，稳定可复现。"""
    def key(c):
        s = c["scores"]
        return (
            -s["indexed_fraction"],
            s["weighted_residual"],
            s["complexity"],
            c["family_key"],
            tuple(np.round(np.asarray(c["basis"]).flatten(), CANON_DECIMALS)),
        )

    return sorted(cands, key=key)


def generate_candidates(peaks, outlier_budget=OUTLIER_BUDGET_FRACTION,
                        min_indexed=MIN_INDEXED_FRACTION):
    """从反射峰生成晶格候选。

    peaks: [{id, q:[3], intensity, cov:[6], excluded, locked_hkl}]
    返回按确定顺序排列的候选列表。
    """
    used = [p for p in peaks if not p.get("excluded")]
    if len(used) < 3:
        return []
    Q = np.array([p["q"] for p in used], dtype=float)
    inten = np.array([p.get("intensity", 1.0) for p in used], dtype=float)
    covs = np.array([cov_inv(p["cov"]) for p in used])
    locked = {i: tuple(p["locked_hkl"]) for i, p in enumerate(used)
              if p.get("locked_hkl")}
    n_total = len(used)
    locked_mask = np.zeros(n_total, dtype=bool)
    if locked:
        locked_mask[sorted(locked)] = True

    order = sorted(range(n_total), key=lambda i: (-inten[i], i))
    pool = order[:MAX_BASIS_PEAKS]
    allowed_outliers = int(outlier_budget * n_total)

    cands = []
    seen = set()
    for combo in itertools.combinations(pool, 3):
        B0 = np.column_stack([Q[i] for i in combo])
        scale = float(np.mean(np.linalg.norm(B0, axis=0)))
        if scale <= 0 or abs(np.linalg.det(B0)) < TOL_SINGULAR * scale ** 3:
            continue  # 奇异基，跳过
        try:
            H, _, chi2 = index_peaks(B0, Q, covs, locked)
            ref = (chi2 <= INLIER_CHI2) | locked_mask
            if ref.sum() < 3:
                continue
            B1 = refine_basis(Q[ref], H[ref], covs[ref])
            H, _, chi2 = index_peaks(B1, Q, covs, locked)
            ref = (chi2 <= INLIER_CHI2) | locked_mask
            if ref.sum() < 3:
                continue
            B2 = refine_basis(Q[ref], H[ref], covs[ref])
        except np.linalg.LinAlgError:
            continue
        det = abs(np.linalg.det(B2))
        if det < TOL_SINGULAR:
            continue
        H, R, chi2 = index_peaks(B2, Q, covs, locked)
        inl = (chi2 <= INLIER_CHI2) | locked_mask
        n_in = int(inl.sum())
        frac = n_in / n_total
        n_out = n_total - n_in
        if frac < min_indexed or n_out > allowed_outliers:
            continue

        Bred, Mlll, steps = lll_reduce(B2)
        Bc, U = minkowski_canonical(Bred)
        Mtot = Mlll @ U
        canon_key = tuple(np.round(Bc.flatten(), CANON_DECIMALS))
        if canon_key in seen:
            continue
        seen.add(canon_key)

        max_hkl = int(np.abs(H[inl]).max()) if n_in else 0
        complexity = round(max_hkl + float(np.log10(1.0 / det)), 6)
        wres = round(float(np.mean(chi2[inl])), 6) if n_in else 0.0
        assignments = []
        deps = []
        for i, p in enumerate(used):
            is_in = bool(inl[i])
            if is_in:
                deps.append(p["id"])
            assignments.append({
                "observation_id": p["id"],
                "hkl": [int(v) for v in H[i]],
                "predicted": (B2 @ H[i]).tolist(),
                "residual": R[i].tolist(),
                "chi2": float(chi2[i]),
                "is_outlier": not is_in,
            })
        cands.append({
            "basis": B2,
            "canonical": Bc,
            "transform": Mtot,
            "steps": [{"kind": "candidate_basis", "matrix": B2.tolist()}]
                     + steps
                     + [{"kind": "minkowski", "matrix": U.tolist()}],
            "family_key": family_key_of(Bc),
            "scores": {
                "indexed_fraction": round(frac, 6),
                "weighted_residual": wres,
                "complexity": complexity,
                "outlier_budget": {
                    "used": n_out,
                    "allowed": allowed_outliers,
                    "within": n_out <= allowed_outliers,
                },
            },
            "assignments": assignments,
            "dependencies": sorted(deps),
        })
        if len(cands) >= MAX_CANDIDATES:
            break
    return sort_candidates(cands)
