"""核心数值算法测试：固定合成晶格、固定容差、稳定排序。"""
import numpy as np
import pytest

import lattice_core as core
import sample_data


def make_candidate(basis, seq=0):
    return core.Candidate(
        basis=np.asarray(basis, float), seed_basis=np.asarray(basis, float),
        reduction_steps=[], seq=seq,
        scores={"indexed_fraction": 0.5, "weighted_residual": 1.0,
                "complexity": abs(float(np.linalg.det(basis))),
                "outlier_used": 0, "outlier_budget": 1, "over_budget": False,
                "n_indexed": 1, "n_total": 2})


# ---- 等价基归族 ----
def test_equivalent_bases_same_family():
    B1 = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]])
    M = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])  # 单模矩阵 det=1
    B2 = B1 @ M
    c1, c2 = make_candidate(B1, 0), make_candidate(B2, 1)
    fams = core.group_families([c1, c2])
    assert len(fams) == 1
    assert np.array_equal(np.array(c2.family_transform), M)
    # 原候选与变换矩阵都保留
    assert fams[0][0] is c1 and fams[0][1] is c2


def test_near_identical_beyond_tolerance_not_merged():
    B1 = np.eye(3) * 0.25
    B2 = B1.copy()
    B2[0, 0] += 1e-3  # 超出 FAMILY_TOL 的近似相同晶胞
    c1, c2 = make_candidate(B1, 0), make_candidate(B2, 1)
    fams = core.group_families([c1, c2])
    assert len(fams) == 2


# ---- 双晶域 ----
def _sample_inputs():
    d = sample_data.build_sample()
    bias = np.array(d["bias"])
    qs = [core.correct_q(np.array(p["q"]), bias) for p in d["peaks"]]
    covs = [sample_data.cov6_to_matrix(p["cov"]) for p in d["peaks"]]
    return qs, covs, [False] * len(qs)


def test_two_domains_yield_multiple_families():
    qs, covs, locked = _sample_inputs()
    cands = core.generate_candidates(qs, covs, locked)
    fams = core.group_families(cands)
    assert len(cands) >= 2
    assert len(fams) >= 2
    # 每个候选只解释一个晶域：已索引比例显著小于 1
    assert cands[0].scores["indexed_fraction"] < 0.8
    complexities = sorted({round(c.scores["complexity"], 4) for c in cands})
    assert len(complexities) >= 2  # 两种不同晶胞体积


# ---- 缺失反射（系统消光）----
def test_missing_reflections_systematic_absences():
    rng = np.random.default_rng(7)
    B = np.array([[0.25, 0.001, 0.0], [0.0, 0.25, 0.001], [0.0, 0.0, 0.25]])
    qs, covs = [], []
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                if (h, k, l) == (0, 0, 0) or (h + k + l) % 2 != 0:
                    continue  # 体心系统消光
                q = B @ np.array([h, k, l], float)
                if 0.05 < np.linalg.norm(q) < 0.9:
                    qs.append(q + rng.normal(0, 0.001, 3))
                    covs.append(np.eye(3) * 1e-6)
    cands = core.generate_candidates(qs, covs, [False] * len(qs))
    assert cands
    assert cands[0].scores["indexed_fraction"] >= 0.95


# ---- 并列候选稳定排序 ----
def test_tied_candidates_stable_order():
    cands = []
    for seq in range(5):
        c = make_candidate(np.eye(3) * (0.25 + seq * 1e-9), seq)
        c.scores = {"indexed_fraction": 0.5, "weighted_residual": 1.0,
                    "complexity": 0.015625, "outlier_used": 0,
                    "outlier_budget": 1, "over_budget": False,
                    "n_indexed": 1, "n_total": 2}
        cands.append(c)
    for c in (cands, list(reversed(cands))):
        ordered = sorted(c, key=core._candidate_sort_key)
        assert [x.seq for x in ordered] == [0, 1, 2, 3, 4]


# ---- 奇异基 ----
def test_singular_basis_rejected():
    a = np.array([0.25, 0.0, 0.0])
    b = np.array([0.0, 0.25, 0.0])
    coplanar = np.column_stack([a, b, a + b])       # 共面 -> 奇异
    collinear = np.column_stack([a, 2 * a, b])      # 共线 -> 奇异
    assert not core.is_valid_triple(coplanar)
    assert not core.is_valid_triple(collinear)
    assert not core.is_valid_triple(np.zeros((3, 3)))
    good = np.column_stack([a, b, np.array([0.0, 0.0, 0.25])])
    assert core.is_valid_triple(good)


# ---- 离群预算 ----
def test_outlier_budget():
    B = np.eye(3) * 0.25
    hkls = [(h, k, l) for h in (1, 2, 3) for k in (1, 2, 3) for l in (1, 2, 3)]
    qs = [B @ np.array(hkl, float) for hkl in hkls]
    covs = [np.eye(3) * 1e-6] * len(qs)
    outliers = [np.array([0.13, 0.31, 0.07]), np.array([0.31, 0.07, 0.13]),
                np.array([0.07, 0.13, 0.31]), np.array([0.19, 0.41, 0.03]),
                np.array([0.03, 0.19, 0.41])]  # 5 个离群峰
    qs = qs + outliers
    covs = covs + [np.eye(3) * 1e-6] * len(outliers)
    cands = core.generate_candidates(qs, covs, [False] * len(qs))
    assert cands
    s = cands[0].scores
    assert s["outlier_used"] == 5
    assert s["outlier_budget"] == max(1, int(np.floor(0.15 * len(qs))))
    assert s["outlier_used"] > s["outlier_budget"]
    assert s["over_budget"]


# ---- 协方差权重 ----
def test_covariance_weights():
    B = np.eye(3) * 0.25
    q = np.array([0.26, 0.0, 0.0])  # 残差 (0.01,0,0)
    hkl_int = np.array([1, 0, 0])
    r = q - B @ hkl_int
    C_small = np.eye(3) * 1e-6
    C_large = np.eye(3) * 1e-2
    a_small = core.index_reflections(B, [q], [C_small], [False])[0]
    a_large = core.index_reflections(B, [q], [C_large], [False])[0]
    assert a_small.d2 == pytest.approx(float(r @ np.linalg.inv(C_small) @ r))
    assert a_large.d2 == pytest.approx(a_small.d2 / 1e4)
    # 大协方差 -> 不加权同样的残差被判为内点；小协方差 -> 离群
    assert a_small.is_outlier and not a_large.is_outlier
    # 各向异性：沿 x 的残差只被 C_xx 加权
    C_aniso = np.diag([1e-2, 1e-8, 1e-8])
    a = core.index_reflections(B, [q], [C_aniso], [False])[0]
    assert a.d2 == pytest.approx(0.01 ** 2 / 1e-2)


# ---- 约化过程记录 ----
def test_reduction_records_unimodular_steps():
    B0 = np.array([[0.25, 0.55, 0.1], [0.0, 0.05, 0.2], [0.0, 0.02, 0.27]])
    B_red, steps = core.reduce_basis(B0)
    assert steps, "应记录约化步骤"
    M_total = np.eye(3, dtype=int)
    for M in steps:
        M = np.array(M)
        assert abs(round(np.linalg.det(M))) == 1  # 允许的单模基变换
        M_total = M_total @ M
    assert np.allclose(B0 @ M_total, B_red)
    assert np.linalg.det(B_red) > 0
