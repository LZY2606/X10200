"""用例编排：导入、生成、落库、局部过期、校准、分叉与冻结。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import geometry, services
from . import config as cfg
from .database import Database
from .models import AnalysisImportIn
from .services import (CandidateEval, Observation, ReflectionEval,
                       dumps, jloads, mat_to_list, short_id)


class ConflictError(Exception):
    """乐观并发冲突（如旧草案并发冻结）。"""


class FrozenError(Exception):
    """已冻结解释不允许修改。"""


# ---------------------------------------------------------------------------
# 内部读取
# ---------------------------------------------------------------------------
def _get_analysis(db: Database, analysis_id: str) -> Dict[str, Any]:
    row = db.query_one("SELECT * FROM analyses WHERE id=?", (analysis_id,))
    if row is None:
        raise KeyError(f"analysis {analysis_id} 不存在")
    return dict(row)


def _assert_draft(analysis: Dict[str, Any]) -> None:
    if analysis["status"] != "draft":
        raise FrozenError("解释已冻结，请先分叉到新草案")


def _expire_candidates(db: Database, candidate_ids: List[str],
                       reason: str) -> int:
    if not candidate_ids:
        return 0
    seq = db.next_seq()
    placeholders = ",".join("?" for _ in candidate_ids)
    db.execute(
        f"UPDATE candidates SET status='expired', expired_seq=?, expiry_reason=? "
        f"WHERE id IN ({placeholders}) AND status='active'",
        (seq, reason, *candidate_ids),
    )
    return len(candidate_ids)


# ---------------------------------------------------------------------------
# 导入（保存原始观测）
# ---------------------------------------------------------------------------
def import_analysis(db: Database, payload: AnalysisImportIn) -> str:
    with db.lock:
        seq = db.next_seq()
        db.execute(
            "INSERT INTO calibration_versions(id,label,offset_json,created_seq) "
            "VALUES (?,?,?,?)",
            (payload.calibration.id, payload.calibration.label,
             dumps(payload.calibration.offset), seq),
        )
        analysis_id = short_id("ana")
        db.execute(
            "INSERT INTO analyses(id,name,calibration_version_id,"
            "active_offset_json,index_tol,outlier_budget,revision,status,"
            "created_seq) VALUES (?,?,?,?,?,?,?,'draft',?)",
            (analysis_id, payload.name, payload.calibration.id,
             dumps(payload.calibration.offset), payload.index_tol,
             payload.outlier_budget, 1, seq),
        )
        for peak in payload.peaks:
            raw = np.array(peak.point, dtype=float)
            offset = np.array(payload.calibration.offset, dtype=float)
            obs_id = short_id("obs")
            db.execute(
                "INSERT INTO observations(id,analysis_id,peak_index,"
                "raw_point_json,calibrated_point_json,intensity,"
                "covariance_json,calibration_version_id,is_overlap,mark,"
                "created_seq) VALUES (?,?,?,?,?,?,?,?,?,'normal',?)",
                (obs_id, analysis_id, peak.peak_index,
                 dumps([float(v) for v in raw]),
                 dumps([float(v) for v in raw - offset]),
                 float(peak.intensity),
                 dumps(mat_to_list(np.array(peak.covariance, dtype=float))),
                 payload.calibration.id,
                 1 if peak.is_overlap else 0, seq),
            )
        db.log_event(analysis_id, "import", {
            "calibration_version": payload.calibration.id,
            "peak_count": len(payload.peaks),
            "index_tol": payload.index_tol,
            "outlier_budget": payload.outlier_budget,
        })
        return analysis_id


# ---------------------------------------------------------------------------
# 候选生成与落库
# ---------------------------------------------------------------------------
def _persist_reduction_chain(db: Database, candidate_id: str,
                             candidate: CandidateEval) -> None:
    red = geometry.reduce_basis(candidate.basis)
    for step_index, step in enumerate(red.chain):
        db.execute(
            "INSERT INTO transforms(candidate_id,step_index,operation,"
            "matrix_json,basis_before_json,basis_after_json,note) "
            "VALUES (?,?,?,?,?,?,?)",
            (candidate_id, step_index, step.operation,
             dumps(mat_to_list(step.matrix)),
             dumps(mat_to_list(step.basis_before)),
             dumps(mat_to_list(step.basis_after)),
             step.note),
        )
    db.execute(
        "INSERT INTO transforms(candidate_id,step_index,operation,"
        "matrix_json,basis_before_json,basis_after_json,note) "
        "VALUES (?,?,?,?,?,?,?)",
        (candidate_id, len(red.chain),
          "calibration offset (weighted least squares)",
          dumps(mat_to_list(np.eye(3))),
          dumps(mat_to_list(candidate.basis)),
          dumps(mat_to_list(candidate.basis)),
          dumps({"offset": mat_to_list(candidate.offset.reshape(3, 1))})),
    )


def _persist_candidate(db: Database, analysis_id: str, order: int,
                       revision: int, candidate: CandidateEval,
                       domain_label: int, family_id: str,
                       family_transform: List[List[float]]) -> str:
    seq = db.next_seq()
    cid = short_id("cand")
    reduced = geometry.reduce_basis(candidate.basis).reduced
    wr = candidate.weighted_residual
    if not np.isfinite(wr):
        wr = 1.0e30
    db.execute(
        "INSERT INTO candidates(id,analysis_id,seq_in_analysis,basis_json,"
        "reduced_basis_json,domain_label,family_id,status,indexed_count,"
        "eligible_count,outlier_count,outlier_budget,outlier_budget_exceeded,"
        "weighted_residual,complexity,basis_condition,"
        "anchor_peak_indexes_json,revision_created,created_seq) "
        "VALUES (?,?,?,?,?,?,?,'active',?,?,?,?,?,?,?,?,?,?,?)",
        (cid, analysis_id, order,
         dumps(mat_to_list(candidate.basis)),
         dumps(mat_to_list(reduced)),
         domain_label, family_id,
         candidate.indexed_count, candidate.eligible_count,
         candidate.outlier_count, candidate.outlier_budget,
         1 if candidate.outlier_budget_exceeded else 0,
         wr, candidate.complexity, candidate.basis_condition,
         dumps(list(candidate.anchors)), revision, seq),
    )

    for ref in candidate.reflections:
        db.execute(
            "INSERT INTO candidate_reflections(candidate_id,observation_id,"
            "hkl_json,predicted_json,residual_json,residual_norm,mahalanobis,"
            "role) VALUES (?,?,?,?,?,?,?,?)",
            (cid, ref.observation_id, dumps(list(ref.hkl)),
             dumps([float(v) for v in ref.predicted]),
             dumps([float(v) for v in ref.residual]),
             ref.residual_norm, ref.mahalanobis, ref.role),
        )
        db.execute(
            "INSERT OR IGNORE INTO candidate_dependencies"
            "(candidate_id,observation_id,is_anchor) VALUES (?,?,?)",
            (cid, ref.observation_id,
             1 if ref.peak_index in candidate.anchors else 0),
        )

    _persist_reduction_chain(db, cid, candidate)
    return cid


def generate_for_analysis(db: Database, analysis_id: str,
                          max_candidates: int = 120) -> int:
    with db.lock:
        analysis = _get_analysis(db, analysis_id)
        _assert_draft(analysis)
        obs_list = services.load_observations(db, analysis_id)
        candidates = services.generate_candidates(
            obs_list, analysis["index_tol"], analysis["outlier_budget"],
            max_candidates)
        domains = services.assign_domains(candidates)
        families = services.group_families(candidates)

        # 旧 active 候选标记 superseded（保留审计），expired 不动
        db.execute(
            "UPDATE candidates SET status='superseded' "
            "WHERE analysis_id=? AND status='active'", (analysis_id,))

        family_ids: Dict[int, str] = {}
        created = 0
        for order, (cand, domain, (rep_idx, rel_t)) in enumerate(
                zip(candidates, domains, families)):
            if rep_idx not in family_ids:
                fid = short_id("fam")
                family_ids[rep_idx] = fid
                db.execute(
                    "INSERT INTO families(id,analysis_id,"
                    "representative_candidate_id,created_seq) "
                    "VALUES (?,?,?,?)",
                    (fid, analysis_id, "", db.next_seq()))
            fid = family_ids[rep_idx]
            cid = _persist_candidate(
                db, analysis_id, order, analysis["revision"], cand,
                domain, fid, mat_to_list(rel_t))
            db.execute(
                "INSERT INTO family_members(family_id,candidate_id,"
                "relative_transform_json) VALUES (?,?,?)",
                (fid, cid, dumps(mat_to_list(rel_t))),
            )
            db.execute(
                "INSERT INTO transforms(family_id,step_index,operation,"
                "matrix_json,note) VALUES (?,?,?,?,?)",
                (fid, order,
                 f"member {cid}: B_member = B_representative @ T",
                 dumps(mat_to_list(rel_t)),
                 f"relative to family {fid}")),
            created += 1

        # 回填族代表（每组第一个落库者即代表）
        for rep_idx, fid in family_ids.items():
            rep_cid = db.query_one(
                "SELECT candidate_id FROM family_members WHERE family_id=? "
                "ORDER BY rowid LIMIT 1", (fid,))["candidate_id"]
            db.execute(
                "UPDATE families SET representative_candidate_id=? WHERE id=?",
                (rep_cid, fid))

        db.log_event(analysis_id, "generate", {
            "created": created,
            "revision": analysis["revision"],
        })
        return created


# ---------------------------------------------------------------------------
# 人工标记：锁定 / 排除 / 恢复（局部过期）
# ---------------------------------------------------------------------------
def set_observation_mark(db: Database, analysis_id: str,
                         observation_id: str, mark: str) -> None:
    if mark not in ("normal", "locked", "excluded"):
        raise ValueError("mark 只能是 normal/locked/excluded")
    with db.lock:
        analysis = _get_analysis(db, analysis_id)
        _assert_draft(analysis)
        obs = db.query_one(
            "SELECT * FROM observations WHERE id=? AND analysis_id=?",
            (observation_id, analysis_id))
        if obs is None:
            raise KeyError("observation 不存在")
        if obs["mark"] == mark:
            return

        # 排除参与构造的锚点：只过期真正依赖它的候选（局部过期）。
        # 普通锁定/恢复不改变观测集合，候选不过期，仅前端按最新标记展示。
        affected: List[str] = []
        if mark == "excluded":
            rows = db.query(
                "SELECT candidate_id FROM candidate_dependencies "
                "WHERE observation_id=? AND is_anchor=1", (observation_id,))
            affected = [r["candidate_id"] for r in rows]

        db.execute(
            "UPDATE observations SET mark=? WHERE id=?", (mark, observation_id))
        db.execute(
            "UPDATE analyses SET revision=revision+1 WHERE id=?", (analysis_id,))
        if affected:
            _expire_candidates(
                db, affected,
                f"anchor observation {observation_id} marked excluded")
        db.log_event(analysis_id, "mark_changed", {
            "observation_id": observation_id,
            "old": obs["mark"], "new": mark,
            "expired_candidates": affected,
        })


# ---------------------------------------------------------------------------
# 校准：调整点（全部候选过期）或在某候选下加权重拟合并应用
# ---------------------------------------------------------------------------
def adjust_calibration(db: Database, analysis_id: str,
                       offset: List[float]) -> None:
    if len(offset) != 3:
        raise ValueError("offset 必须是长度 3 的向量")
    with db.lock:
        analysis = _get_analysis(db, analysis_id)
        _assert_draft(analysis)
        old_offset = np.array(jloads(analysis["active_offset_json"]))
        new_offset = np.array(offset, dtype=float)
        delta = new_offset - old_offset

        rows = db.query(
            "SELECT id,raw_point_json FROM observations WHERE analysis_id=?",
            (analysis_id,))
        for row in rows:
            raw = np.array(jloads(row["raw_point_json"]))
            db.execute(
                "UPDATE observations SET calibrated_point_json=? WHERE id=?",
                (dumps([float(v) for v in raw - new_offset]), row["id"]))
        db.execute(
            "UPDATE analyses SET active_offset_json=?, revision=revision+1 "
            "WHERE id=?", (dumps([float(v) for v in new_offset]), analysis_id))

        affected = [r["id"] for r in db.query(
            "SELECT id FROM candidates WHERE analysis_id=? AND status='active'",
            (analysis_id,))]
        _expire_candidates(db, affected, "calibration adjustment changed offset")
        db.log_event(analysis_id, "calibration_adjusted", {
            "delta": [float(v) for v in delta],
            "new_offset": [float(v) for v in new_offset],
            "expired_candidates": affected,
        })


def refine_calibration(db: Database, analysis_id: str,
                       candidate_id: str) -> List[float]:
    with db.lock:
        analysis = _get_analysis(db, analysis_id)
        _assert_draft(analysis)
        crow = db.query_one(
            "SELECT * FROM candidates WHERE id=? AND analysis_id=?",
            (candidate_id, analysis_id))
        if crow is None or crow["status"] != "active":
            raise KeyError("候选不存在或已过期，不能用于校准重拟合")

        obs_list = services.load_observations(db, analysis_id)
        obs_by_id = {o.id: o for o in obs_list}
        refl_rows = db.query(
            "SELECT * FROM candidate_reflections WHERE candidate_id=? "
            "AND role='indexed'", (candidate_id,))
        points = np.array([obs_by_id[r["observation_id"]].calibrated_point
                           for r in refl_rows])
        covs = np.array([obs_by_id[r["observation_id"]].covariance
                         for r in refl_rows])
        hkls = np.array([jloads(r["hkl_json"]) for r in refl_rows], dtype=float)
        basis = np.array(jloads(crow["basis_json"]))
        current = np.array(jloads(analysis["active_offset_json"]))
        # 当前预测点含旧偏移；在“去偏移”坐标内求残差修正量
        correction = geometry.fit_calibration_offset(points, covs, basis, hkls)
        new_offset = current + correction
        adjust_calibration(db, analysis_id, [float(v) for v in new_offset])
        return [float(v) for v in new_offset]


# ---------------------------------------------------------------------------
# 分叉：保留冻结/旧草案，复制原始观测到新草案
# ---------------------------------------------------------------------------
def fork_analysis(db: Database, analysis_id: str,
                  new_name: Optional[str] = None) -> str:
    with db.lock:
        parent = _get_analysis(db, analysis_id)
        seq = db.next_seq()
        new_id = short_id("ana")
        db.execute(
            "INSERT INTO analyses(id,name,calibration_version_id,"
            "active_offset_json,index_tol,outlier_budget,revision,status,"
            "created_seq) VALUES (?,?,?,?,?,?,'draft',?)",
            (new_id, new_name or f"{parent['name']} (fork)",
             parent["calibration_version_id"],
             parent["active_offset_json"], parent["index_tol"],
             parent["outlier_budget"], 1, seq),
        )
        for row in db.query(
                "SELECT * FROM observations WHERE analysis_id=?",
                (analysis_id,)):
            db.execute(
                "INSERT INTO observations(id,analysis_id,peak_index,"
                "raw_point_json,calibrated_point_json,intensity,"
                "covariance_json,calibration_version_id,is_overlap,mark,"
                "created_seq) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (short_id("obs"), new_id, row["peak_index"],
                 row["raw_point_json"], row["calibrated_point_json"],
                 row["intensity"], row["covariance_json"],
                 row["calibration_version_id"], row["is_overlap"],
                 row["mark"], seq),
            )
        db.log_event(new_id, "forked", {"parent_analysis": analysis_id})
        db.log_event(analysis_id, "fork_created", {"child_analysis": new_id})
        return new_id


# ---------------------------------------------------------------------------
# 冻结：钉住校准、容差与人工标记；旧草案并发冻结必须冲突
# ---------------------------------------------------------------------------
def freeze_analysis(db: Database, analysis_id: str,
                    expected_revision: int) -> Dict[str, Any]:
    with db.lock:
        analysis = _get_analysis(db, analysis_id)
        if analysis["status"] == "frozen":
            raise FrozenError("该解释已经冻结")
        if analysis["revision"] != expected_revision:
            raise ConflictError(
                f"修订号冲突：expected={expected_revision}, "
                f"current={analysis['revision']}（旧草案并发冻结）")

        snapshot = _analysis_snapshot(db, analysis_id)
        freeze_id = short_id("frz")
        seq = db.next_seq()
        db.execute(
            "INSERT INTO freezes(id,analysis_id,revision,snapshot_json,"
            "created_seq) VALUES (?,?,?,?,?)",
            (freeze_id, analysis_id, analysis["revision"],
             dumps(snapshot), seq))
        db.execute(
            "UPDATE analyses SET status='frozen', frozen_at_seq=? WHERE id=?",
            (seq, analysis_id))
        db.log_event(analysis_id, "frozen", {
            "freeze_id": freeze_id, "revision": analysis["revision"]})
        return snapshot


# ---------------------------------------------------------------------------
# 快照与只读序列化
# ---------------------------------------------------------------------------
def _row_observation(row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "peak_index": row["peak_index"],
        "raw_point": jloads(row["raw_point_json"]),
        "calibrated_point": jloads(row["calibrated_point_json"]),
        "intensity": row["intensity"],
        "covariance": jloads(row["covariance_json"]),
        "calibration_version_id": row["calibration_version_id"],
        "is_overlap": bool(row["is_overlap"]),
        "mark": row["mark"],
    }


def _analysis_snapshot(db: Database, analysis_id: str) -> Dict[str, Any]:
    analysis = _get_analysis(db, analysis_id)
    observations = [_row_observation(r) for r in db.query(
        "SELECT * FROM observations WHERE analysis_id=? ORDER BY peak_index",
        (analysis_id,))]
    candidates = []
    for crow in db.query(
            "SELECT * FROM candidates WHERE analysis_id=? ORDER BY seq_in_analysis",
            (analysis_id,)):
        if crow["status"] != "active":
            continue
        candidates.append(_candidate_dict(db, dict(crow), light=True))
    return {
        "analysis": {
            "id": analysis["id"],
            "name": analysis["name"],
            "calibration_version_id": analysis["calibration_version_id"],
            "active_offset": jloads(analysis["active_offset_json"]),
            "index_tol": analysis["index_tol"],
            "outlier_budget": analysis["outlier_budget"],
            "revision": analysis["revision"],
        },
        "observations": observations,
        "candidates": candidates,
    }


def analysis_dict(db: Database, analysis_id: str) -> Dict[str, Any]:
    analysis = _get_analysis(db, analysis_id)
    observations = [_row_observation(r) for r in db.query(
        "SELECT * FROM observations WHERE analysis_id=? ORDER BY peak_index",
        (analysis_id,))]
    cand_rows = db.query(
        "SELECT * FROM candidates WHERE analysis_id=? ORDER BY seq_in_analysis",
        (analysis_id,))
    families: Dict[str, Dict[str, Any]] = {}
    candidates = []
    for crow in cand_rows:
        c = _candidate_dict(db, dict(crow), light=True)
        candidates.append(c)
        fid = crow["family_id"]
        if fid and crow["status"] == "active":
            families.setdefault(fid, {
                "id": fid,
                "domain_label": crow["domain_label"],
                "member_candidate_ids": [],
            })["member_candidate_ids"].append(crow["id"])
    events = [{"seq": r["seq"], "kind": r["kind"],
               "payload": jloads(r["payload_json"])}
              for r in db.query(
                  "SELECT * FROM event_log WHERE analysis_id=? ORDER BY seq",
                  (analysis_id,))]
    return {
        "analysis": {
            "id": analysis["id"],
            "name": analysis["name"],
            "calibration_version_id": analysis["calibration_version_id"],
            "active_offset": jloads(analysis["active_offset_json"]),
            "index_tol": analysis["index_tol"],
            "outlier_budget": analysis["outlier_budget"],
            "revision": analysis["revision"],
            "status": analysis["status"],
        },
        "observations": observations,
        "candidates": candidates,
        "families": sorted(families.values(), key=lambda f: f["id"]),
        "events": events,
    }


def _candidate_dict(db: Database, crow: Dict[str, Any],
                    light: bool = False) -> Dict[str, Any]:
    basis = np.array(jloads(crow["basis_json"]))
    reduced = np.array(jloads(crow["reduced_basis_json"]))
    out = {
        "id": crow["id"],
        "seq": crow["seq_in_analysis"],
        "status": crow["status"],
        "domain_label": crow["domain_label"],
        "family_id": crow["family_id"],
        "basis": mat_to_list(basis),
        "reduced_basis": mat_to_list(reduced),
        "cell": geometry.cell_parameters(reduced),
        "anchors": jloads(crow["anchor_peak_indexes_json"]),
        # 四项评分分项分别展示，不合成总分
        "scores": {
            "indexed_fraction": (
                crow["indexed_count"] / crow["eligible_count"]
                if crow["eligible_count"] else 0.0),
            "weighted_residual": crow["weighted_residual"],
            "complexity": crow["complexity"],
            "outlier_budget": {
                "used": crow["outlier_count"],
                "budget": crow["outlier_budget"],
                "exceeded": bool(crow["outlier_budget_exceeded"]),
            },
        },
        "indexed_count": crow["indexed_count"],
        "eligible_count": crow["eligible_count"],
        "basis_condition": crow["basis_condition"],
        "revision_created": crow["revision_created"],
        "expiry_reason": crow["expiry_reason"],
    }
    if not light:
        out["reflections"] = [
            {
                "observation_id": r["observation_id"],
                "hkl": jloads(r["hkl_json"]),
                "predicted": jloads(r["predicted_json"]),
                "residual": jloads(r["residual_json"]),
                "residual_norm": r["residual_norm"],
                "mahalanobis": r["mahalanobis"],
                "role": r["role"],
            }
            for r in db.query(
                "SELECT * FROM candidate_reflections WHERE candidate_id=? "
                "ORDER BY observation_id", (crow["id"],))
        ]
        out["transform_chain"] = [
            {
                "step_index": r["step_index"],
                "operation": r["operation"],
                "matrix": jloads(r["matrix_json"]),
                "basis_before": (jloads(r["basis_before_json"])
                                 if r["basis_before_json"] else None),
                "basis_after": (jloads(r["basis_after_json"])
                                if r["basis_after_json"] else None),
                "note": r["note"],
            }
            for r in db.query(
                "SELECT * FROM transforms WHERE candidate_id=? "
                "ORDER BY step_index", (crow["id"],))
        ]
    return out


def candidate_detail(db: Database, candidate_id: str) -> Dict[str, Any]:
    crow = db.query_one("SELECT * FROM candidates WHERE id=?", (candidate_id,))
    if crow is None:
        raise KeyError("候选不存在")
    result = _candidate_dict(db, dict(crow), light=False)

    # 同族其他成员与保留的相对变换
    members = db.query(
        "SELECT m.candidate_id, m.relative_transform_json, c.seq_in_analysis "
        "FROM family_members m JOIN candidates c ON c.id=m.candidate_id "
        "WHERE m.family_id=? ORDER BY c.seq_in_analysis",
        (crow["family_id"],))
    result["family"] = {
        "id": crow["family_id"],
        "members": [
            {"candidate_id": r["candidate_id"],
             "relative_transform": jloads(r["relative_transform_json"])}
            for r in members
        ],
    }

    # 观测追溯：每个 hkl 回到原始观测 + 校准版本
    obs_rows = {r["id"]: r for r in db.query(
        "SELECT * FROM observations WHERE analysis_id=?",
        (crow["analysis_id"],))}
    for ref in result["reflections"]:
        obs = obs_rows.get(ref["observation_id"])
        if obs is not None:
            ref["observation"] = _row_observation(obs)
    return result
