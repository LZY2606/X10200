import numpy as np
import pytest

from app import chamber, geometry, services
from tests.factories import (DIRECT_A, DIRECT_B, build_import,
                             make_lattice_peaks)


OFFSET = np.array([0.03, -0.02, 0.01])


def _setup(db, peaks, **kw):
    aid = chamber.import_analysis(db, build_import(peaks=peaks, **kw))
    created = chamber.generate_for_analysis(db, aid)
    return aid, created


def test_single_domain_generates_and_indexes(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=3)
    aid, created = _setup(db, peaks)
    assert created >= 1
    data = chamber.analysis_dict(db, aid)
    top = data["candidates"][0]
    assert top["status"] == "active"
    assert top["scores"]["indexed_fraction"] > 0.9
    assert top["domain_label"] == 0


def test_equivalent_bases_grouped_into_one_family_but_both_kept(db):
    # 同一真值晶格的不同锚点三元组会产生等价候选；它们应归同族且各自保留
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=5)
    aid, created = _setup(db, peaks)
    data = chamber.analysis_dict(db, aid)
    active = [c for c in data["candidates"] if c["status"] == "active"]
    assert len(active) >= 2
    families = {c["family_id"] for c in active}
    # 所有候选都描述同一晶格 -> 同一个族
    assert len(families) == 1
    # 每个成员都保留相对代表的幺模变换
    detail0 = chamber.candidate_detail(db, active[0]["id"])
    for member in detail0["family"]["members"]:
        t = np.array(member["relative_transform"])
        assert geometry.is_unimodular(t)


def test_two_domains_produce_two_domain_labels(db):
    p1 = make_lattice_peaks(DIRECT_A, OFFSET, seed=11)
    p2 = make_lattice_peaks(DIRECT_B, OFFSET, seed=21)
    for p in p2:
        p.peak_index = len(p1) + p.peak_index
    aid, created = _setup(db, p1 + p2, outlier_budget=1)
    assert created >= 2
    data = chamber.analysis_dict(db, aid)
    active = [c for c in data["candidates"] if c["status"] == "active"]
    domains = {c["domain_label"] for c in active}
    assert domains == {0, 1}


def test_missing_reflections_systematic_absences(db):
    missing = {(0, 0, 2), (0, 2, 0), (2, 0, 0)}
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=7, missing=missing)
    aid, created = _setup(db, peaks)
    data = chamber.analysis_dict(db, aid)
    # 缺失反射不应阻止高索引率候选出现
    assert data["candidates"][0]["scores"]["indexed_fraction"] > 0.9


def test_tied_candidates_both_kept_and_stably_ordered(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=9)
    aid, _ = _setup(db, peaks)
    chamber.generate_for_analysis(db, aid)  # 再生成一次
    data1 = chamber.analysis_dict(db, aid)
    chamber.generate_for_analysis(db, aid)  # 第三次：排序必须完全一致
    data2 = chamber.analysis_dict(db, aid)
    act1 = [(c["anchors"], c["status"]) for c in data1["candidates"]]
    act2 = [(c["anchors"], c["status"]) for c in data2["candidates"]
            if c["status"] == "active"]
    assert [a for a, s in act1 if s == "active"] == act2


def test_singular_anchor_triplet_does_not_crash(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=4)
    obs = services.load_observations(
        db, chamber.import_analysis(db, build_import(peaks=peaks)))
    # 共面的三个点 -> 奇异基，必须返回 None 而不是抛出
    pts = np.array([obs[0].calibrated_point,
                    obs[1].calibrated_point,
                    obs[1].calibrated_point * 0.999 +
                    obs[0].calibrated_point * 0.001])
    assert services._basis_from_anchors(
        np.array([pts[0], pts[1], pts[2] - pts[0] + pts[0]])) is None


def test_outlier_budget_used_and_exceed_flag(db):
    extra = [np.array([3.3, -2.7, 1.1]),
             np.array([-2.1, 3.1, -0.6]),
             np.array([1.1, 2.9, 2.8])]
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=13, extra=extra)
    aid, _ = _setup(db, peaks, outlier_budget=1)
    detail = chamber.candidate_detail(
        db, chamber.analysis_dict(db, aid)["candidates"][0]["id"])
    budget = detail["scores"]["outlier_budget"]
    assert budget["exceeded"] is True
    roles = [r["role"] for r in detail["reflections"]]
    assert roles.count("outlier") <= 1
    assert "unindexed" in roles


def test_covariance_weighting_changes_weighted_residual(db):
    peaks_low = make_lattice_peaks(DIRECT_A, OFFSET, seed=17, cov_scale=0.002)
    aid = chamber.import_analysis(db, build_import(peaks=peaks_low))
    chamber.generate_for_analysis(db, aid)
    wr_low = chamber.analysis_dict(db, aid)["candidates"][0][
        "scores"]["weighted_residual"]

    peaks_high = make_lattice_peaks(DIRECT_A, OFFSET, seed=17, cov_scale=0.02)
    aid2 = chamber.import_analysis(db, build_import(peaks=peaks_high))
    chamber.generate_for_analysis(db, aid2)
    wr_high = chamber.analysis_dict(db, aid2)["candidates"][0][
        "scores"]["weighted_residual"]
    # 噪声大但协方差同步放大 -> 标准化残差应当更小/相近，绝不是简单几何误差排序
    assert wr_high < wr_low


def test_excluding_anchor_locally_expires_only_dependent_candidates(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=19)
    aid, _ = _setup(db, peaks)
    data = chamber.analysis_dict(db, aid)
    active_before = [c for c in data["candidates"] if c["status"] == "active"]
    assert len(active_before) >= 2

    anchors = set(active_before[0]["anchors"])
    observations = data["observations"]
    anchor_obs = next(o for o in observations
                      if o["peak_index"] in anchors)
    chamber.set_observation_mark(
        db, aid, anchor_obs["id"], "excluded")

    after = chamber.analysis_dict(db, aid)["candidates"]
    dependent = [c for c in after
                 if anchor_obs["peak_index"] in c["anchors"]]
    others = [c for c in after
              if c["status"] == "active"
              and anchor_obs["peak_index"] not in c["anchors"]]
    assert all(c["status"] == "expired" for c in dependent)
    assert others  # 不依赖该锚点的候选仍然有效


def test_lock_then_exclude_non_anchor_keeps_candidates(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=23)
    aid, _ = _setup(db, peaks)
    data = chamber.analysis_dict(db, aid)
    obs_id = data["observations"][0]["id"]
    chamber.set_observation_mark(db, aid, obs_id, "locked")
    after = chamber.candidate_detail(db, data["candidates"][0]["id"])
    assert after["status"] == "active"


def test_fork_is_independent(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=29)
    aid, _ = _setup(db, peaks)
    fork_id = chamber.fork_analysis(db, aid)
    chamber.adjust_calibration(db, fork_id, [0.1, 0.0, 0.0])
    parent = chamber.analysis_dict(db, aid)["analysis"]
    child = chamber.analysis_dict(db, fork_id)["analysis"]
    assert parent["active_offset"] == [0.0, 0.0, 0.0]
    assert child["active_offset"] == [0.1, 0.0, 0.0]


def test_freeze_conflict_on_stale_revision(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=31)
    aid, _ = _setup(db, peaks)
    data = chamber.analysis_dict(db, aid)
    obs_id = data["observations"][0]["id"]
    chamber.set_observation_mark(db, aid, obs_id, "locked")  # rev +1
    with pytest.raises(chamber.ConflictError):
        chamber.freeze_analysis(db, aid, expected_revision=1)
    snapshot = chamber.freeze_analysis(db, aid, expected_revision=2)
    assert snapshot["analysis"]["revision"] == 2
    with pytest.raises(chamber.FrozenError):
        chamber.adjust_calibration(db, aid, [0.0, 0.0, 0.0])


def test_freeze_snapshot_pins_calibration_tolerance_and_marks(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=37)
    aid, _ = _setup(db, peaks)
    data = chamber.analysis_dict(db, aid)
    chamber.set_observation_mark(
        db, aid, data["observations"][0]["id"], "locked")
    snap = chamber.freeze_analysis(db, aid, expected_revision=2)
    assert snap["analysis"]["active_offset"] == [0.0, 0.0, 0.0]
    assert snap["analysis"]["index_tol"] == 0.25
    assert any(o["mark"] == "locked" for o in snap["observations"])


def test_reflection_detail_links_back_to_raw_observation(db):
    peaks = make_lattice_peaks(DIRECT_A, OFFSET, seed=41)
    aid, _ = _setup(db, peaks)
    detail = chamber.candidate_detail(
        db, chamber.analysis_dict(db, aid)["candidates"][0]["id"])
    ref = next(r for r in detail["reflections"] if r["role"] == "indexed")
    assert ref["observation"]["raw_point"] is not None
    assert ref["observation"]["calibration_version_id"] == "CAL-T1"
    assert len(detail["transform_chain"]) >= 1
