import numpy as np

from app import geometry


DIRECT = np.array([
    [6.0, 0.3, 0.1],
    [0.0, 5.0, -0.2],
    [0.2, 0.0, 4.5],
])


def test_equivalent_bases_unimodular_reduce_to_same_cell():
    b = geometry.reciprocal_from_direct(DIRECT)
    # 允许的基变换（幺模）：交换两列 + 剪切
    p = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    s = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    b_equiv = b @ p @ s

    r1 = geometry.reduce_basis(b)
    r2 = geometry.reduce_basis(b_equiv)

    assert np.allclose(r1.reduced, r2.reduced, atol=1e-9)
    t = geometry.family_transform(b, b_equiv)
    assert t is not None
    assert geometry.is_unimodular(t)


def test_similar_but_beyond_tolerance_not_same_family():
    b = geometry.reciprocal_from_direct(DIRECT)
    b_near = b.copy()
    b_near[:, 0] *= 1.0 + 2e-5  # 超出固定同族容差
    assert geometry.family_transform(b, b_near, tol=1e-6) is None


def test_singular_basis_rejected():
    singular = np.column_stack([[1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    assert geometry.is_singular(singular)


def test_reduction_chain_reproduces_basis():
    b = geometry.reciprocal_from_direct(DIRECT)
    result = geometry.reduce_basis(b)
    composed = np.eye(3)
    for step in result.chain:
        composed = composed @ step.matrix
    assert np.allclose(b @ composed, result.reduced, atol=1e-9)
    assert np.allclose(composed, result.transform, atol=1e-12)


def test_covariance_weighting_anisotropic():
    rng = np.random.default_rng(7)
    residual = np.array([0.01, 0.0, 0.0])
    cov_iso = np.eye(3) * 0.01 ** 2
    cov_aniso = np.diag([0.02 ** 2, 0.005 ** 2, 0.01 ** 2])
    m_iso = geometry.whiten_residual(residual, cov_iso)
    m_aniso = geometry.whiten_residual(residual, cov_aniso)
    # x 方向误差椭球更长（0.02），同样几何残差的标准化权重更小
    assert m_iso == 1.0
    assert abs(m_aniso - 0.5) < 1e-9

    residual_tight = np.array([0.0, 0.01, 0.0])
    m_tight = geometry.whiten_residual(residual_tight, cov_aniso)
    assert m_tight == 2.0  # y 方向误差小（0.005），加权残差更大


def test_weighted_offset_fit_recovers_known_offset():
    b = geometry.reciprocal_from_direct(DIRECT)
    offset = np.array([0.05, -0.04, 0.02])
    hkls = np.array([[h, k, l] for h in (-1, 0, 1)
                     for k in (-1, 0, 1) for l in (-1, 0, 1)
                     if not (h == 0 and k == 0 and l == 0)], dtype=float)
    points = (b @ hkls.T).T + offset
    covs = np.tile(np.eye(3) * 1e-8, (len(hkls), 1, 1))
    fit = geometry.fit_calibration_offset(points, covs, b, hkls)
    assert np.allclose(fit, offset, atol=1e-6)
