"""
晶格候选室 —— 数值核心（探测器 2D 投影上的倒数基，hkl 以 (h,k,0) 展示）。

全部算法使用固定容差、确定性顺序与稳定排序；不使用任何随机数。
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 固定容差（数值算法必须可复现）
# ---------------------------------------------------------------------------
TOL_MAHALANOBIS = 10.0          # 内点的马氏距离门限
MAHAL_OUTLIER_SPAN = 2.5        # 可纳入离群预算的最大马氏倍数
DET_SINGULAR = 1.0e-8           # |det| 小于此值视为奇异基
BASIS_COND_LIMIT = 1.0e6        # 基矩阵条件数上限
INT_FRAC_TOL = 2.0e-2           # 判定“整数矩阵”的分数容差
FAMILY_SNAP_TOL = 5.0e-3        # 同族归并的残差门限（倒数空间单位）
NEAR_CELL_TOL = 5.0e-2          # “近似相同但超误差”的展示门限
REDUCTION_STEP_LIMIT = 4096     # 约化迭代上限
MIN_INLIERS = 5                 # 候选存活所需的最少内点
MAX_PAIRS = 20000               # 单次索引最多尝试的峰对数（确定性截断）
MAX_SINGULAR_LOG = 50           # 保留的奇异/退化基记录数


def inv_spd_2x2(cov: np.ndarray) -> Optional[np.ndarray]:
    """对 2x2 对称测量协方差求逆。"""
    cov = np.asarray(cov, dtype=float)
    a, b, c, d = cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
    det = a * d - b * c
    if not np.isfinite(det) or det <= 1.0e-14:
        return None
    return np.array([[d, -b], [-c, a]], dtype=float) / det


def _weights(covs: np.ndarray, active: np.ndarray) -> np.ndarray:
    """逐峰标量权重 w = 1 / tr(Sigma)，在活跃峰上归一化。"""
    n = covs.shape[0]
    w = np.zeros(n)
    for i in range(n):
        w[i] = 1.0 / max(float(np.trace(covs[i])), 1.0e-12)
    denom = float(np.sum(w[active]))
    if denom > 0.0:
        w /= denom
    return w


# ---------------------------------------------------------------------------
# 候选评估：hkl、预测位置、残差、内点/离群/未索引、加权残差
# ---------------------------------------------------------------------------
def evaluate_basis(
    basis: np.ndarray,
    q: np.ndarray,
    covs: np.ndarray,
    *,
    excluded: np.ndarray,
    locked: np.ndarray,
    outlier_budget: int,
    tol_mahal: float = TOL_MAHALANOBIS,
    outlier_span: float = MAHAL_OUTLIER_SPAN,
) -> Optional[Dict[str, Any]]:
    """
    h = B^{-1} q；hkl = round(h)；预测 = B hkl；
    残差按每个峰自身的 2x2 协方差白化。
      马氏距离 <= tol           -> 内点 (inlier)
      tol < 马氏 <= span*tol    -> 离群候选，按预算确定性选取 (outlier)
      其余                      -> 未索引 (unindexed)
    排除峰不参与拟合；锁定峰必须为内点，否则该候选不可行。
    """
    basis = np.asarray(basis, dtype=float)
    q = np.asarray(q, dtype=float)
    n = q.shape[0]
    det = float(np.linalg.det(basis))
    if not np.isfinite(det) or abs(det) < DET_SINGULAR:
        return None
    cond = float(np.linalg.cond(basis))
    if not np.isfinite(cond) or cond > BASIS_COND_LIMIT:
        return None

    Binv = np.linalg.inv(basis)
    h_cont = q @ Binv.T
    hkl2 = np.rint(h_cont).astype(int)
    predicted = hkl2 @ basis.T
    residual = q - predicted
    raw_dist = np.linalg.norm(residual, axis=1)

    mahal = np.full(n, np.inf)
    for i in range(n):
        Sinv = inv_spd_2x2(covs[i])
        if Sinv is not None:
            r = residual[i]
            mahal[i] = float(np.sqrt(max(r @ Sinv @ r, 0.0)))
        else:
            mahal[i] = raw_dist[i] / 0.05

    within_tol = mahal <= tol_mahal
    in_band = (mahal > tol_mahal) & (mahal <= tol_mahal * outlier_span)

    band_ids = [i for i in range(n) if not excluded[i] and in_band[i]]
    band_ids.sort(key=lambda i: (round(float(mahal[i]), 10), i))
    outlier = np.zeros(n, dtype=bool)
    for i in band_ids[: max(0, int(outlier_budget))]:
        outlier[i] = True
    inlier = within_tol & ~excluded
    indexed = (inlier | outlier) & ~excluded

    lock_violation = bool(np.any(locked & excluded)) or bool(
        np.any(locked & ~within_tol))
    active = ~excluded
    w = _weights(covs, active)

    inl_wss = float(np.sum(w[inlier] * raw_dist[inlier] ** 2))
    inl_wrms = float(np.sqrt(inl_wss)) if np.any(inlier) else float("inf")
    all_wrms = (
        float(np.sqrt(np.sum(w[indexed] * raw_dist[indexed] ** 2)))
        if np.any(indexed) else float("inf"))
    inl_rms = (
        float(np.sqrt(np.mean(raw_dist[inlier] ** 2)))
        if np.any(inlier) else float("inf"))
    inl_mahal_rms = (
        float(np.sqrt(np.mean(mahal[inlier] ** 2)))
        if np.any(inlier) else float("inf"))
    n_active = int(np.sum(active))
    n_in = int(np.sum(inlier))
    n_out = int(np.sum(outlier))
    indexed_ratio = (n_in + n_out) / n_active if n_active > 0 else 0.0

    hkl3 = np.column_stack(
        [hkl2[:, 0], hkl2[:, 1], np.zeros(n, dtype=int)])
    return {
        "basis": basis.copy(),
        "h_cont": h_cont,
        "hkl": hkl3,
        "predicted": predicted,
        "residual": residual,
        "raw_dist": raw_dist,
        "mahal": mahal,
        "inlier": inlier,
        "outlier": outlier,
        "unindexed": (~indexed) & active,
        "excluded": excluded.copy(),
        "lock_violation": lock_violation,
        "weighted_rms": inl_wrms,
        "all_indexed_weighted_rms": all_wrms,
        "rms": inl_rms,
        "mahal_rms": inl_mahal_rms,
        "indexed_ratio": indexed_ratio,
        "n_inlier": n_in,
        "n_outlier": n_out,
        "n_active": n_active,
        "n_indexed": n_in + n_out,
        "complexity": abs(det),
        "condition": cond,
        "inlier_key": "".join("1" if x else "0" for x in inlier),
    }


# ---------------------------------------------------------------------------
# 确定性二维整数尺寸约化 + 列排序 + 符号规范化
# 每一步记录整数幺模矩阵 P（B_new = B_old @ P），形成完整变换链。
# ---------------------------------------------------------------------------
def _step_record(P: np.ndarray, basis: np.ndarray, op: str, detail: str,
                 norms: np.ndarray) -> Dict[str, Any]:
    return {
        "op": op,
        "detail": detail,
        "P": P.astype(int).tolist(),
        "column_norms": [round(float(x), 10) for x in norms],
        "basis": np.asarray(basis, dtype=float).tolist(),
    }


def _sign_canonicalize_2d(b: np.ndarray, P: np.ndarray,
                          steps: List[Dict[str, Any]]
                          ) -> Tuple[np.ndarray, np.ndarray, bool]:
    changed = False
    for j in range(2):
        k = j
        while k < 2 and abs(b[k, j]) < 1.0e-12:
            k += 1
        if k < 2 and b[k, j] < 0.0:
            Pj = np.eye(2, dtype=int)
            Pj[j, j] = -1
            b = b @ Pj
            P = P @ Pj
            steps.append(_step_record(Pj, b, "sign_flip",
                                      "flip column %d" % j,
                                      np.linalg.norm(b, axis=0)))
            changed = True
    return b, P, changed


def reduce_basis(basis: np.ndarray) -> Dict[str, Any]:
    basis = np.asarray(basis, dtype=float)
    b = basis.copy()
    P = np.eye(2, dtype=int)
    steps: List[Dict[str, Any]] = [
        _step_record(np.eye(2, dtype=int), b, "start", "input basis",
                     np.linalg.norm(b, axis=0))]
    truncated = False
    used = 0
    while used < REDUCTION_STEP_LIMIT:
        changed = False

        # 尺寸约化：反复令两列互相减去整数倍，直到 |2 dot| <= 另一列长²
        pair_changed = True
        while pair_changed and used < REDUCTION_STEP_LIMIT:
            pair_changed = False
            for i, j in ((0, 1), (1, 0)):
                n2 = float(b[:, i] @ b[:, i])
                if n2 <= 0.0:
                    continue
                k = int(np.rint(float(b[:, i] @ b[:, j]) / n2))
                if k != 0:
                    Pj = np.eye(2, dtype=int)
                    Pj[i, j] -= k
                    b = b @ Pj
                    P = P @ Pj
                    steps.append(_step_record(
                        Pj, b, "size_reduce",
                        "column %d -= %d * column %d" % (j, k, i),
                        np.linalg.norm(b, axis=0)))
                    used += 1
                    pair_changed = changed = True

        # 列排序：第一列必须不更长（严格才交换，避免循环）
        n0 = float(np.linalg.norm(b[:, 0]))
        n1 = float(np.linalg.norm(b[:, 1]))
        if n0 > n1 + LENGTH_EPS:
            Pj = np.array([[0, 1], [1, 0]], dtype=int)
            b = b @ Pj
            P = P @ Pj
            steps.append(_step_record(Pj, b, "order",
                                      "swap columns",
                                      np.linalg.norm(b, axis=0)))
            used += 1
            changed = True

        # 符号规范化（原地修改数组需要重绑，见下）
        before = b.copy()
        b, P, schanged = _sign_canonicalize_2d(b, P, steps)
        if schanged or not np.allclose(before, b, atol=1.0e-13):
            changed = True

        if not changed:
            break
    else:
        truncated = True

    return {
        "red_basis": b,
        "P": P,
        "P_inv": _unimod_inverse(P),
        "steps": steps,
        "truncated": truncated,
        "n_steps": len(steps) - 1,
    }


def _unimod_inverse(P: np.ndarray) -> np.ndarray:
    """整数幺模 2x2 矩阵的整数逆。"""
    P = np.asarray(P, dtype=int)
    a, b, c, d = P[0, 0], P[0, 1], P[1, 0], P[1, 1]
    det = int(a * d - b * c)
    if det not in (-1, 1):
        raise ValueError("matrix is not unimodular: det=%s" % det)
    return (np.array([[d, -b], [-c, a]], dtype=int) * det)


# ---------------------------------------------------------------------------
# 同族判定：允许的整数幺模基变换 U（B2 ≈ B1 @ U，|det U| = 1）
# ---------------------------------------------------------------------------
def integer_unimod_map(b1: np.ndarray, b2: np.ndarray,
                       int_tol: float = INT_FRAC_TOL,
                       snap_tol: float = FAMILY_SNAP_TOL
                       ) -> Optional[np.ndarray]:
    """
    若 B2 = B1 U 中的 U 是（带误差的）整数幺模矩阵且几何残差很小，
    返回整数 U；否则 None。近似相同但超出误差的候选不会被合并。
    """
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    d1, d2 = abs(float(np.linalg.det(b1))), abs(float(np.linalg.det(b2)))
    if abs(d1 - d2) > 1.0e-6 * max(1.0, d1):
        return None
    U = np.linalg.solve(b1, b2)
    if not np.all(np.isfinite(U)):
        return None
    Ur = np.rint(U)
    if np.max(np.abs(U - Ur)) > int_tol:
        return None
    Ui = Ur.astype(int)
    if abs(int(round(float(np.linalg.det(Ui))))) != 1:
        return None
    if float(np.max(np.abs(b1 @ Ui - b2))) > snap_tol:
        return None
    return Ui


def group_families(bases: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
    """确定性同族归并；保留每个成员相对族代表的变换矩阵与残差。"""
    families: List[Dict[str, Any]] = []
    for idx, basis in enumerate(bases):
        basis = np.asarray(basis, dtype=float)
        for fam in families:
            rep = np.asarray(fam["representative_basis"], dtype=float)
            U = integer_unimod_map(rep, basis)
            if U is not None:
                fam["members"].append({
                    "index": idx,
                    "U_from_rep": U.tolist(),
                    "max_resid": float(np.max(np.abs(rep @ U - basis))),
                })
                break
        else:
            families.append({
                "family_id": len(families),
                "representative_basis": basis.tolist(),
                "representative_index": idx,
                "members": [{"index": idx,
                             "U_from_rep": np.eye(2, dtype=int).tolist(),
                             "max_resid": 0.0}],
            })
    return families


# ---------------------------------------------------------------------------
# 系统消光（二维有心网）分析
# ---------------------------------------------------------------------------
EXTINCTION_RULES = [
    {"id": "p", "name": "原始网（无系统消光）",
     "test": lambda h, k: False},
    {"id": "c", "name": "C 心网：h+k 为偶（奇数反射缺失）",
     "test": lambda h, k: (h + k) % 2 != 0},
    {"id": "h_only", "name": "h 为偶（测试规则）",
     "test": lambda h, k: h % 2 != 0},
    {"id": "k_only", "name": "k 为偶（测试规则）",
     "test": lambda h, k: k % 2 != 0},
]


def extinction_analysis(hkl: np.ndarray, inlier: np.ndarray
                        ) -> Dict[str, Any]:
    ids = np.where(inlier)[0]
    rows = []
    best = None
    for rule in EXTINCTION_RULES:
        forbidden = []
        for i in ids:
            h, k = int(hkl[i, 0]), int(hkl[i, 1])
            if rule["test"](h, k):
                forbidden.append(int(i))
        completeness = 1.0 if not forbidden else \
            1.0 - len(forbidden) / max(1, len(ids))
        row = {"id": rule["id"], "name": rule["name"],
               "completeness": completeness,
               "forbidden_observed": forbidden,
               "n_forbidden": len(forbidden)}
        rows.append(row)
        if best is None or (row["completeness"], -row["n_forbidden"]) > (
                best["completeness"], -best["n_forbidden"]):
            best = row
    p_row = next(r for r in rows if r["id"] == "p")
    detected = (best["id"] if best["id"] != "p" and best["completeness"] > 0.95
                and best["completeness"] - p_row["completeness"] > 0.10
                else None)
    return {"rules": rows, "detected": detected,
            "detected_name": next((r["name"] for r in rows
                                   if r["id"] == detected), None)}


# ---------------------------------------------------------------------------
# 候选生成：确定性枚举峰对 -> 2x2 基 -> 评估 -> 约化 -> 去重
# ---------------------------------------------------------------------------
def generate_candidates(
    q: np.ndarray,
    covs: np.ndarray,
    *,
    excluded: np.ndarray,
    locked: np.ndarray,
    outlier_budget: int,
    tol_mahal: float = TOL_MAHALANOBIS,
) -> Dict[str, Any]:
    active_ids = [int(i) for i in range(q.shape[0]) if not excluded[i]]
    pairs = list(itertools.combinations(active_ids, 2))[:MAX_PAIRS]

    raw: List[Dict[str, Any]] = []
    singular_log: List[Dict[str, Any]] = []
    for p_id, (a, b) in enumerate(pairs):
        B = np.column_stack([q[a], q[b]])
        det = float(np.linalg.det(B))
        cond = float(np.linalg.cond(B)) if np.isfinite(det) else np.inf
        if not np.isfinite(det) or abs(det) < DET_SINGULAR \
                or not np.isfinite(cond) or cond > BASIS_COND_LIMIT:
            if len(singular_log) < MAX_SINGULAR_LOG:
                singular_log.append({
                    "pair": [a, b],
                    "reason": "singular_or_ill_conditioned",
                    "det": None if not np.isfinite(det) else det,
                })
            continue
        ev = evaluate_basis(
            B, q, covs, excluded=excluded, locked=locked,
            outlier_budget=outlier_budget, tol_mahal=tol_mahal)
        if ev is None or ev["n_inlier"] < MIN_INLIERS or \
                ev["lock_violation"]:
            continue
        ev["pair"] = [a, b]
        ev["pair_id"] = p_id
        raw.append(ev)

    # 约化（不改变所张成晶格；hkl 经整数幺模 P 重新表达）
    survivors: List[Dict[str, Any]] = []
    for ev in raw:
        red = reduce_basis(ev["basis"])
        ev2 = evaluate_basis(
            red["red_basis"], q, covs, excluded=excluded, locked=locked,
            outlier_budget=outlier_budget, tol_mahal=tol_mahal)
        if ev2 is None or ev2["n_inlier"] < MIN_INLIERS or \
                ev2["lock_violation"]:
            continue
        ev2["pair"] = ev["pair"]
        ev2["pair_id"] = ev["pair_id"]
        ev2["reduction"] = red
        survivors.append(ev2)

    # 相同内点集合只保留最简、残差最小者
    by_key: Dict[str, Dict[str, Any]] = {}
    for ev in survivors:
        key = ev["inlier_key"]
        cur = by_key.get(key)
        ck = (-ev["n_inlier"], round(ev["weighted_rms"], 10),
              round(ev["complexity"], 10), ev["pair_id"])
        if cur is None or ck < (
                -cur["n_inlier"], round(cur["weighted_rms"], 10),
                round(cur["complexity"], 10), cur["pair_id"]):
            by_key[key] = ev
    deduped = list(by_key.values())

    # 稳定排序：索引比例↓、加权残差↓、复杂度↓、峰对 id↑
    deduped.sort(key=lambda e: (
        -round(e["indexed_ratio"], 10),
        round(e["weighted_rms"], 10),
        round(e["complexity"], 10),
        e["pair_id"],
    ))
    return {
        "candidates": deduped,
        "singular_log": singular_log,
        "n_pairs": len(pairs),
        "n_survived": len(survivors),
    }
