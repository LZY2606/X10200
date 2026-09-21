"""业务服务：候选生成、评分分项、同族、局部过期、分叉与冻结。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config, geometry
from .database import Database


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------
def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def jloads(text: str) -> Any:
    return json.loads(text)


def mat_to_list(mat: np.ndarray) -> List[List[float]]:
    return [[float(v) for v in row] for row in np.asarray(mat)]


def short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# 候选计算的数据结构
# ---------------------------------------------------------------------------
@dataclass
class ReflectionEval:
    observation_id: str
    peak_index: int
    hkl: Tuple[int, int, int]
    predicted: np.ndarray
    residual: np.ndarray
    residual_norm: float
    mahalanobis: float
    within_tol: bool
    mark: str
    is_overlap: bool
    role: str = "indexed"


@dataclass
class CandidateEval:
    basis: np.ndarray
    offset: np.ndarray
    anchors: Tuple[int, ...]
    reflections: List[ReflectionEval]
    indexed: List[ReflectionEval]
    outliers: List[ReflectionEval]
    excluded: List[ReflectionEval]
    indexed_count: int
    eligible_count: int
    indexed_fraction: float
    weighted_residual: float
    outlier_count: int
    outlier_budget: int
    outlier_budget_exceeded: bool
    complexity: float
    basis_condition: float


# ---------------------------------------------------------------------------
# 观察数据读取
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    id: str
    peak_index: int
    raw_point: np.ndarray
    calibrated_point: np.ndarray
    intensity: float
    covariance: np.ndarray
    cal_version_id: str
    is_overlap: bool
    mark: str


def load_observations(db: Database, analysis_id: str,
                      include_excluded: bool = True) -> List[Observation]:
    rows = db.query(
        "SELECT * FROM observations WHERE analysis_id=? ORDER BY peak_index",
        (analysis_id,))
    obs = []
    for row in rows:
        mark = row["mark"]
        if not include_excluded and mark == "excluded":
            continue
        obs.append(Observation(
            id=row["id"],
            peak_index=row["peak_index"],
            raw_point=np.array(jloads(row["raw_point_json"])),
            calibrated_point=np.array(jloads(row["calibrated_point_json"])),
            intensity=row["intensity"],
            covariance=np.array(jloads(row["covariance_json"])),
            cal_version_id=row["calibration_version_id"],
            is_overlap=bool(row["is_overlap"]),
            mark=mark,
        ))
    return obs


# ---------------------------------------------------------------------------
# 候选评估
# ---------------------------------------------------------------------------
def _eligible(obs: Observation) -> bool:
    return obs.mark != "excluded"


def evaluate_candidate(obs_list: Sequence[Observation],
                       basis: np.ndarray,
                       offset: np.ndarray,
                       index_tol: float,
                       outlier_budget: int,
                       anchors: Tuple[int, ...]) -> CandidateEval:
    """评估一个候选基：索引、残差、离群预算、评分分项。"""
    refls: List[ReflectionEval] = []
    excluded: List[ReflectionEval] = []

    for obs in obs_list:
        result = geometry.index_point(
            obs.calibrated_point - offset, basis, obs.covariance, index_tol)
        ref = ReflectionEval(
            observation_id=obs.id,
            peak_index=obs.peak_index,
            hkl=result.hkl,
            predicted=result.predicted + offset,
            residual=result.residual,
            residual_norm=result.residual_norm,
            mahalanobis=result.mahalanobis,
            within_tol=result.within_index_tol,
            mark=obs.mark,
            is_overlap=obs.is_overlap,
        )
        if obs.mark == "excluded":
            ref.role = "excluded"
            excluded.append(ref)
        refls.append(ref)

    eligible = [r for r in refls if r.role != "excluded"]

    # locked 反射强制接受（研究员钉住的对应）
    forced = [r for r in eligible if r.mark == "locked"]
    free = [r for r in eligible if r.mark != "locked"]

    in_tol = [r for r in free if r.within_tol]
    beyond = [r for r in free if not r.within_tol]

    # 稳定排序：残差小者优先，再按峰号，保证结果唯一
    in_tol.sort(key=lambda r: (round(r.residual_norm, 12), r.peak_index))
    beyond.sort(key=lambda r: (round(r.residual_norm, 12), r.peak_index))

    # 离群预算：超出容差的非锁定反射至多纳入 outlier_budget 个；
    # 研究员锁定的反射强制保留，单独计为 locked_outlier，不占预算。
    budget_outliers = beyond[:outlier_budget]
    budget_exceeded = len(beyond) > outlier_budget

    forced_outliers = [r for r in forced if not r.within_tol]
    all_outliers_set = {id(r): r for r in budget_outliers + forced_outliers}
    all_outliers = sorted(all_outliers_set.values(), key=lambda r: r.peak_index)

    indexed_normal = forced + in_tol
    for r in indexed_normal:
        r.role = "indexed"
    for r in budget_outliers:
        r.role = "outlier"
    for r in forced_outliers:
        r.role = "locked_outlier"
    accepted_ids = {id(r) for r in indexed_normal} | {id(r) for r in all_outliers}
    for r in eligible:
        if id(r) not in accepted_ids:
            r.role = "unindexed"

    indexed_set = [r for r in eligible if id(r) in accepted_ids]
    indexed_set.sort(key=lambda r: r.peak_index)
    scored = [r for r in indexed_normal if r.within_tol]
    if scored:
        weighted = float(np.sqrt(np.mean([r.mahalanobis ** 2
                                          for r in scored])))
    else:
        weighted = float("inf")

    eligible_count = len(eligible)
    indexed_count = len(indexed_set)
    fraction = indexed_count / eligible_count if eligible_count else 0.0

    # 复杂度：约化后正晶胞体积（数值稳定的伪行列式替代）
    reduced = geometry.reduce_basis(basis).reduced
    direct = np.linalg.inv(reduced).T
    complexity = abs(float(np.linalg.det(direct)))
    basis_condition = float(np.linalg.cond(basis))

    return CandidateEval(
        basis=basis,
        offset=offset,
        anchors=anchors,
        reflections=sorted(refls, key=lambda r: r.peak_index),
        indexed=sorted(indexed_set, key=lambda r: r.peak_index),
        outliers=all_outliers,
        excluded=excluded,
        indexed_count=indexed_count,
        eligible_count=eligible_count,
        indexed_fraction=fraction,
        weighted_residual=weighted,
        outlier_count=len(all_outliers),
        outlier_budget=outlier_budget,
        outlier_budget_exceeded=budget_exceeded,
        complexity=complexity,
        basis_condition=basis_condition,
    )


def _basis_from_anchors(points: np.ndarray) -> Optional[np.ndarray]:
    """三个独立倒易向量构造列基，近奇异（近共面）返回 None。"""
    basis = np.column_stack([points[1] - points[0],
                             points[2] - points[0],
                             points[0]])
    if geometry.is_singular(basis):
        return None
    # 条件数过大视为数值奇异基
    try:
        if np.linalg.cond(basis) > 1.0e10:
            return None
    except np.linalg.LinAlgError:
        return None
    return basis


def candidate_sort_key(c: CandidateEval) -> Tuple:
    """展示排序：高已索引率、低加权残差、低复杂度、低条件数、锚点字典序。

    注意不是总分，只是稳定的展示次序。"""
    wr = c.weighted_residual
    if not np.isfinite(wr):
        wr = 1.0e30
    return (-round(c.indexed_fraction, 12),
            round(wr, 12),
            round(c.complexity, 12),
            round(c.basis_condition, 12),
            c.anchors)



def generate_candidates(obs_list: Sequence[Observation],
                        index_tol: float,
                        outlier_budget: int,
                        max_candidates: int = config.MAX_CANDIDATES
                        ) -> List[CandidateEval]:
    """确定性候选枚举（格点向量命中过滤 + 受限三元组）。

    1. 原点候选 o 与非原点峰 p 形成候选倒易向量 d = p - o，统计全部峰对
       (a,b) 的投影 t = (b-a).d / |d|^2 接近整数的次数，过滤非格点向量；
    2. 按原点取命中最高的一批向量，组合成线性独立三元组构造基；
    3. 每个基做“加权偏移重拟合 + 重索引”，约化基数值去重；
       近似但超固定容差者保留（并列候选不合并）。
    """
    eligible = [o for o in obs_list if _eligible(o)]
    points = np.array([o.calibrated_point for o in eligible])
    n = len(eligible)

    centroid = points.mean(axis=0)
    origin_order = sorted(
        range(n),
        key=lambda i: (round(float(np.linalg.norm(points[i] - centroid)), 12),
                       eligible[i].peak_index))
    origin_limit = min(n, 12)

    diffs = points[:, None, :] - points[None, :, :]

    by_origin: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
    origin_candidates = origin_order[:origin_limit]
    for oi in origin_candidates:
        order = sorted(
            range(n),
            key=lambda j: (round(float(np.linalg.norm(
                points[j] - points[oi])), 12), eligible[j].peak_index))
        chosen = [j for j in order if j != oi][:40]
        scored: List[Tuple[int, int, np.ndarray]] = []
        for j in chosen:
            direction = points[j] - points[oi]
            norm2 = float(np.dot(direction, direction))
            if norm2 <= 1e-20:
                continue
            proj = (diffs @ direction) / norm2
            hits = int(np.sum(np.abs(proj - np.rint(proj)) <= index_tol))
            scored.append((hits, j, direction))
        scored.sort(key=lambda t: (-t[0], eligible[t[1]].peak_index))
        kept = [(j, d, h) for h, j, d in scored[:18]
                if h >= config.MIN_INDEXED * 2]
        if kept:
            by_origin[oi] = kept

    seen_reduced: List[np.ndarray] = []
    results: List[CandidateEval] = []
    point_by_id = {o.id: o for o in eligible}

    def try_basis(origin_idx: int, vec_indices: Tuple[int, ...],
                  basis0: np.ndarray) -> None:
        anchor_peaks = tuple(sorted(
            [eligible[origin_idx].peak_index] +
            [eligible[v].peak_index for v in vec_indices]))
        rough = evaluate_candidate(eligible, basis0, np.zeros(3),
                                   index_tol, outlier_budget, anchor_peaks)
        good = [r for r in rough.indexed if r.role == "indexed"]
        if len(good) < config.MIN_INDEXED:
            return
        pts = np.array([point_by_id[r.observation_id].calibrated_point
                        for r in good])
        covs = np.array([point_by_id[r.observation_id].covariance
                         for r in good])
        hkls = np.array([r.hkl for r in good])
        try:
            offset = geometry.fit_calibration_offset(pts, covs, basis0, hkls)
        except np.linalg.LinAlgError:
            return
        candidate = evaluate_candidate(obs_list, basis0, offset,
                                       index_tol, outlier_budget,
                                       anchor_peaks)
        if candidate.indexed_count < config.MIN_INDEXED:
            return
        reduced = geometry.reduce_basis(basis0).reduced
        for prev in seen_reduced:
            if np.max(np.abs(reduced - prev)) <= config.FAMILY_VECTOR_TOL:
                return
        seen_reduced.append(reduced)
        results.append(candidate)

    max_combos_per_origin = 500
    for oi in origin_candidates:
        vecs = by_origin.get(oi, [])
        combos = 0
        for (ja, da, _ha), (jb, dbv, _hb), (jc, dc, _hc) in combinations(vecs, 3):
            basis0 = np.column_stack([da, dbv, dc])
            if geometry.is_singular(basis0):
                continue
            if np.linalg.cond(basis0) > 1.0e10:
                continue
            try_basis(oi, (ja, jb, jc), basis0)
            combos += 1
            if combos >= max_combos_per_origin or len(results) >= max_candidates:
                break

    results.sort(key=candidate_sort_key)
    return results[:max_candidates]


def assign_domains(candidates: List[CandidateEval]) -> List[int]:
    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    sets = [set(r.observation_id for r in c.indexed) for c in candidates]
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            union_size = len(sets[i] | sets[j])
            jaccard = inter / union_size if union_size else 0.0
            if jaccard >= 0.5:
                union(i, j)

    roots: Dict[int, int] = {}
    labels = []
    next_label = 0
    for i in range(n):
        root = find(i)
        if root not in roots:
            roots[root] = next_label
            next_label += 1
        labels.append(roots[root])
    return labels


# ---------------------------------------------------------------------------
# 同族归并（允许的幺模基变换）
# ---------------------------------------------------------------------------
def group_families(candidates: List[CandidateEval]
                   ) -> List[Tuple[int, np.ndarray]]:
    """返回每个候选的 (族内代表下标, B_rep^-1 B_i 的幺模矩阵)。

    判定严格使用固定容差：约化基近似相同但超差者绝不合并。
    """
    n = len(candidates)
    reduced = [geometry.reduce_basis(c.basis).reduced for c in candidates]
    reps: List[int] = []
    labels: List[Tuple[int, np.ndarray]] = []
    for i in range(n):
        found = False
        for rep in reps:
            t = geometry.family_transform(reduced[rep], reduced[i])
            if t is not None:
                labels.append((rep, t))
                found = True
                break
        if not found:
            reps.append(i)
            labels.append((i, np.eye(3)))
    return labels
