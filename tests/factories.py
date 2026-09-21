"""确定性小晶格数据集工厂。"""
import numpy as np

from app import geometry
from app.models import AnalysisImportIn, CalibrationVersionIn, PeakIn


DIRECT_A = np.array([
    [6.0, 0.3, 0.1],
    [0.0, 5.0, -0.2],
    [0.2, 0.0, 4.5],
])
DIRECT_B = np.array([
    [4.8, 0.1, 0.0],
    [0.2, 5.4, 0.3],
    [0.0, -0.1, 6.2],
])


def make_lattice_peaks(direct, offset, h_range=range(-2, 3),
                       missing=(), extra=(), cov_scale=0.004, seed=1):
    rng = np.random.default_rng(seed)
    basis = geometry.reciprocal_from_direct(direct)
    peaks, truth = [], []
    cov = np.eye(3) * cov_scale ** 2
    idx = 0
    for h in h_range:
        for k in h_range:
            for ell in h_range:
                if h == 0 and k == 0 and ell == 0:
                    continue
                if (h, k, ell) in missing:
                    continue
                q = basis @ np.array([h, k, ell], dtype=float)
                noise = rng.multivariate_normal(np.zeros(3), cov)
                peaks.append(PeakIn(
                    peak_index=idx,
                    point=[float(v) for v in q + noise + offset],
                    intensity=100.0,
                    covariance=cov.tolist(),
                ))
                truth.append((h, k, ell))
                idx += 1
    for shift in extra:
        peaks.append(PeakIn(
            peak_index=idx, point=list(map(float, shift + offset)),
            intensity=20.0, covariance=cov.tolist()))
        idx += 1
    return peaks


def build_import(name="t", peaks=None, index_tol=0.25, outlier_budget=2):
    return AnalysisImportIn(
        name=name,
        calibration=CalibrationVersionIn(
            id="CAL-T1", label="test-cal", offset=[0.0, 0.0, 0.0]),
        peaks=peaks or [],
        index_tol=index_tol,
        outlier_budget=outlier_budget,
    )
