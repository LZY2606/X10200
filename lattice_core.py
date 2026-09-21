"""确定性晶格候选生成、约化、族归并与分项评分。

所有数值算法使用固定容差与稳定排序，保证同一输入产生同一输出。
计算全程不访问外网。
"""
from itertools import combinations

import numpy as np

# ---- 固定容差（不得改为运行时随机值） ----
TOL_WEIGHTED_DEFAULT = 9.0      # 加权残差 chi^2 阈值（约 3 sigma）
FAMILY_RTOL = 1e-3              # 等价晶胞度量张量相对容差
SINGULAR_RTOL = 1e-3            # 基三重积相对阈值，低于则视为奇异基
MIN_VOL = 1e-4                  # 倒易晶胞体积下限
MAX_VOL = 1e4                   # 倒易晶胞体积上限
LLL_DELTA = 0.75                # LLL 约化参数（固定）
POOL_SIZE = 8                   # 参与基组合枚举的最短向量数
MAX_KEEP = 24                   # 每个域最多保留的候选数
ROUND = 9                       # 指标输出统一舍入位数


def weighted_residual(r, inv_cov):
    """单个反射的 Mahalanobis 加权残差 r^T C^{-1} r。"""
    return float(r @ inv_cov @ r)


def invert_covariances(covs):
    """covs: (n,3,3)。奇异协方差加固定抖动后求逆，保证确定性。"""
    invs = []
    for cov in covs:
        cov = np.asarray(cov, dtype=float)
        try:
            invs.append(np.linalg.inv(cov))
        except np.linalg.LinAlgError:
            invs.append(np.linalg.inv(cov + 1e-12 * np.eye(3)))
    return invs


def index_reflections(basis, qs, inv_covs, tol):
    """用给定倒易基为全部反射定 hkl、预测位置与残差。

    返回 assignment 列表，元素为 dict：
    index, hkl, pred, residual(笛卡尔), weighted, is_outlier
    """
    inv_b = np.linalg.inv(basis)
    out = []
    for i, q in enumerate(qs):
        frac = inv_b @ q
        hkl = np.rint(frac).astype(int)
        pred = basis @ hkl
        r = q - pred
        w = weighted_residual(r, inv_covs[i])
        out.append({
            "index": i,
            "hkl": [int(hkl[0]), int(hkl[1]), int(hkl[2])],
            "pred": [float(pred[0]), float(pred[1]), float(pred[2])],
            "residual": float(np.sqrt(r @ r)),
            "weighted": w,
            "is_outlier": bool(w > tol),
        })
    return out


def _gram_schmidt(b):
    n = b.shape[1]
    bs = np.zeros_like(b)
    mu = np.zeros((n, n))
    norm = np.zeros(n)
    for i in range(n):
        bs[:, i] = b[:, i].copy()
        for j in range(i):
            mu[i, j] = float(b[:, i] @ bs[:, j] / norm[j])
            bs[:, i] -= mu[i, j] * bs[:, j]
        norm[i] = float(bs[:, i] @ bs[:, i])
    return bs, mu, norm


def lll_reduce(basis, delta=LLL_DELTA):
    """对倒易基做确定性 LLL 约化。

    返回 (约化基, 幺模变换 U, 步骤记录)。约化基 = basis @ U。
    步骤记录完整保存，供坐标变换链回放。
    """
    b = np.array(basis, dtype=float)
    n = b.shape[1]
    u = np.eye(n, dtype=int)
    steps = []
    bs, mu, norm = _gram_schmidt(b)
    k = 1
    guard = 0
    while k < n:
        guard += 1
        if guard > 10000:
            raise RuntimeError("LLL 未收敛")
        for j in range(k - 1, -1, -1):
            r = int(round(mu[k, j]))
            if r != 0:
                b[:, k] -= r * b[:, j]
                u[:, k] -= r * u[:, j]
                steps.append({"type": "reduce", "k": k, "j": j, "r": r})
                bs, mu, norm = _gram_schmidt(b)
        if norm[k] >= (delta - mu[k, k - 1] ** 2) * norm[k - 1]:
            k += 1
        else:
            b[:, [k - 1, k]] = b[:, [k, k - 1]]
            u[:, [k - 1, k]] = u[:, [k, k - 1]]
            steps.append({"type": "swap", "a": k - 1, "b": k})
            bs, mu, norm = _gram_schmidt(b)
            k = max(k - 1, 1)
    return b, u, steps


def metric_tensor(basis):
    return basis.T @ basis


def same_family(g1, g2, rtol=FAMILY_RTOL):
    """度量张量相对容差内视为同族；超出容差的近似晶胞不合并。"""
    scale = max(float(np.abs(g1).max()), float(np.abs(g2).max()), 1e-12)
    return bool(np.abs(g1 - g2).max() <= rtol * scale)


def assign_families(reduced_bases):
    """按约化基的度量张量把候选归入族。

    返回 (family_ids, family_canonical, transforms)。
    transforms[i] 为候选原始基 -> 族规范基的整数变换矩阵。
    """
    family_ids = []
    canonical_g = []
    canonical_u = []
    transforms = []
    for red, u in reduced_bases:
        g = metric_tensor(red)
        fid = None
        for k, cg in enumerate(canonical_g):
            if same_family(g, cg):
                fid = k
                break
        if fid is None:
            fid = len(canonical_g)
            canonical_g.append(g)
            canonical_u.append(u)
        # B_can = B_fam @ U_fam ; B_red = B @ U ; B_red ~= B_can
        # => B_can ~= B @ U @ inv(U_fam)
        m = u @ np.linalg.inv(canonical_u[fid])
        m_int = np.rint(m).astype(int)
        family_ids.append(fid)
        transforms.append(m_int)
    return family_ids, canonical_g, transforms


def compute_metrics(assignments, active, basis, outlier_budget):
    """四项指标分别输出，绝不合并为单一总分。"""
    active_set = set(active)
    indexed = [a for a in assignments if a["index"] in active_set and not a["is_outlier"]]
    outliers = [a for a in assignments if a["index"] in active_set and a["is_outlier"]]
    n_active = len(active)
    n_indexed = len(indexed)
    wres = float(np.mean([a["weighted"] for a in indexed])) if indexed else float("inf")
    vol_recip = abs(float(np.linalg.det(basis)))
    return {
        "indexed": n_indexed,
        "total_active": n_active,
        "indexed_fraction": round(n_indexed / n_active, ROUND) if n_active else 0.0,
        "weighted_residual": round(wres, ROUND) if np.isfinite(wres) else None,
        "complexity": round(1.0 / vol_recip, ROUND),  # 正晶胞体积越大越复杂
        "outliers": len(outliers),
        "outlier_budget": outlier_budget,
        "within_budget": bool(len(outliers) <= outlier_budget),
    }


def _candidate_sort_key(c):
    m = c["metrics"]
    wres = m["weighted_residual"] if m["weighted_residual"] is not None else float("inf")
    # 稳定排序键：已索引数降序、加权残差升序、复杂度升序、生成序号升序
    return (-m["indexed"], round(wres, 6), round(m["complexity"], 6), c["serial"])


def _enumerate_bases(qs, pool_indices):
    """从向量池枚举非奇异三重基。返回 (bases, singular_skipped)。"""
    bases = []
    singular = 0
    for combo in combinations(pool_indices, 3):
        b = qs[list(combo)].T  # 列为三个基矢量
        det = float(np.linalg.det(b))
        norms = np.linalg.norm(qs[list(combo)], axis=1)
        scale = float(norms[0] * norms[1] * norms[2])
        if scale <= 0.0 or abs(det) / scale < SINGULAR_RTOL:
            singular += 1  # 奇异基：跳过并计数
            continue
        vol = abs(det)
        if not (MIN_VOL <= vol <= MAX_VOL):
            continue
        bases.append((combo, b))
    return bases, singular


def generate_candidates(qs, covs, excluded=(), locked=(), outlier_budget=2,
                        tol=TOL_WEIGHTED_DEFAULT, pool_size=POOL_SIZE,
                        max_keep=MAX_KEEP, n_domains=2):
    """主入口：从倒易空间峰坐标生成多个晶格候选。

    qs: (n,3) 已按当前分析校准修正的倒易向量；covs: (n,3,3)。
    返回 dict：candidates（含每反射 hkl/残差/离群标记、约化与变换链、
    分项指标、族归属）、singular_skipped。
    """
    qs = np.asarray(qs, dtype=float)
    covs = np.asarray(covs, dtype=float)
    inv_covs = invert_covariances(covs)
    excluded = set(excluded)
    locked = set(locked)
    active = [i for i in range(len(qs)) if i not in excluded]

    candidates = []
    singular_skipped = 0
    serial = 0

    # 域拆分：第一遍用全部活跃反射；之后用最优候选的离群反射播种下一域
    seed_sets = [active]
    for domain in range(1, n_domains + 1):
        if domain > 1:
            if not candidates:
                break
            best = sorted(candidates, key=_candidate_sort_key)[0]
            rest = [a["index"] for a in best["assignments"]
                    if a["index"] in set(active) and a["is_outlier"]]
            if len(rest) < 3:
                break
            seed_sets.append(rest)
        seeds = seed_sets[domain - 1]
        if len(seeds) < 3:
            continue
        order = sorted(seeds, key=lambda i: (float(qs[i] @ qs[i]), i))
        pool = order[:pool_size]
        bases, singular = _enumerate_bases(qs, pool)
        singular_skipped += singular
        domain_cands = []
        for combo, b in bases:
            assignments = index_reflections(b, qs, inv_covs, tol)
            # 锁定反射必须被索引，否则候选无效
            if any(a["index"] in locked and a["is_outlier"] for a in assignments):
                continue
            red, u, steps = lll_reduce(b)
            metrics = compute_metrics(assignments, active, b, outlier_budget)
            chain = [
                {"step": 0, "type": "observation_basis",
                 "reflection_indices": [int(c) for c in combo],
                 "matrix": np.round(b, ROUND).tolist()},
                {"step": 1, "type": "lll_reduction", "delta": LLL_DELTA,
                 "operations": steps},
                {"step": 2, "type": "reduction_transform",
                 "U": u.tolist(),
                 "U_inv": np.rint(np.linalg.inv(u)).astype(int).tolist()},
                {"step": 3, "type": "reduced_basis",
                 "matrix": np.round(red, ROUND).tolist()},
            ]
            domain_cands.append({
                "serial": serial,
                "domain": domain,
                "basis": np.round(b, ROUND).tolist(),
                "reduced_basis": np.round(red, ROUND).tolist(),
                "transform_U": u.tolist(),
                "assignments": assignments,
                "metrics": metrics,
                "transform_chain": chain,
            })
            serial += 1
        domain_cands.sort(key=_candidate_sort_key)
        candidates.extend(domain_cands[:max_keep])

    # 族归并：等价晶胞同族，变换矩阵与原候选都保留
    reds = [(np.array(c["reduced_basis"], dtype=float),
             np.array(c["transform_U"], dtype=int)) for c in candidates]
    if reds:
        fam_ids, _, transforms = assign_families(reds)
        for c, fid, m in zip(candidates, fam_ids, transforms):
            c["family_id"] = int(fid)
            c["family_transform"] = m.tolist()
            c["transform_chain"].append({
                "step": 4, "type": "family_transform",
                "family_id": int(fid), "M": m.tolist()})
    candidates.sort(key=_candidate_sort_key)
    return {"candidates": candidates, "singular_skipped": singular_skipped}
