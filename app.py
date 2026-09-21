"""FastAPI entry point for the lattice candidate room (晶格候选室)."""
from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from typing import List, Optional

from pydantic import BaseModel

from db import ConflictError, Store
from lattice.synthetic import lattice_peaks

DB_PATH = os.environ.get("LATTICE_DB", "lattice.db")
store = Store(DB_PATH)
app = FastAPI(title="晶格候选室")


class PeakIn(BaseModel):
    q: List[float]
    intensity: float = 1.0
    cov: Optional[List[List[float]]] = None


class ImportBody(BaseModel):
    peaks: list[PeakIn]
    calibration_version: str


class AnalysisBody(BaseModel):
    name: str
    calibration: dict = {}
    tolerances: dict = {}


class MarkBody(BaseModel):
    observation_id: int
    locked: Optional[bool] = None
    excluded: Optional[bool] = None


class CalibrationBody(BaseModel):
    calibration: dict


class ForkBody(BaseModel):
    name: Optional[str] = None


class FreezeBody(BaseModel):
    expected_version: int


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "static", "index.html"))


@app.post("/api/observations/import")
def import_observations(body: ImportBody):
    n = store.import_observations([p.model_dump() for p in body.peaks],
                                  body.calibration_version)
    return {"imported": len(body.peaks), "last_id": n}


@app.get("/api/observations")
def observations():
    return {"observations": store.list_observations()}


@app.post("/api/analyses")
def create_analysis(body: AnalysisBody):
    analysis_id = store.create_analysis(body.name, body.calibration,
                                        body.tolerances)
    summary = store.run_analysis(analysis_id)
    return {"analysis_id": analysis_id, **summary}


@app.get("/api/analyses")
def analyses():
    return {"analyses": store.list_analyses()}


@app.get("/api/analyses/{analysis_id}")
def analysis_state(analysis_id: int):
    try:
        return store.get_analysis_state(analysis_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/analyses/{analysis_id}/marks")
def set_mark(analysis_id: int, body: MarkBody):
    try:
        return store.set_mark(analysis_id, body.observation_id,
                              body.locked, body.excluded)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/analyses/{analysis_id}/calibration")
def update_calibration(analysis_id: int, body: CalibrationBody):
    try:
        store.update_calibration(analysis_id, body.calibration)
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/analyses/{analysis_id}/fork")
def fork(analysis_id: int, body: ForkBody):
    try:
        new_id = store.fork_analysis(analysis_id, body.name)
        return {"analysis_id": new_id}
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/analyses/{analysis_id}/freeze")
def freeze(analysis_id: int, body: FreezeBody):
    try:
        return store.freeze(analysis_id, body.expected_version)
    except ConflictError as exc:
        raise HTTPException(409, str(exc))
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/analyses/{analysis_id}/freezes")
def freezes(analysis_id: int):
    return {"freezes": store.list_freezes(analysis_id)}


@app.get("/api/candidates/{candidate_id}")
def candidate(candidate_id: int):
    try:
        return store.get_candidate(candidate_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/demo/seed")
def demo_seed():
    """Deterministic two-domain synthetic dataset for the UI."""
    basis_a = np.array([[0.52, 0.0, 0.0], [0.0, 0.61, 0.0],
                        [0.0, 0.0, 0.71]])
    basis_b = np.array([[0.40, 0.0, 0.10], [0.0, 0.40, 0.0],
                        [0.0, 0.0, 0.45]])
    qs_a, covs_a, _ = lattice_peaks(basis_a, h_max=2, noise=0.002, seed=1)
    qs_b, covs_b, _ = lattice_peaks(basis_b, h_max=2, noise=0.002, seed=2)
    qs = np.vstack([qs_a, qs_b])
    covs = np.vstack([covs_a, covs_b])
    peaks = [{"q": q.tolist(), "intensity": 1.0, "cov": c.tolist()}
             for q, c in zip(qs, covs)]
    store.import_observations(peaks, "cal-2026.09")
    analysis_id = store.create_analysis("demo", {}, {})
    summary = store.run_analysis(analysis_id)
    return {"analysis_id": analysis_id, "peaks": len(peaks), **summary}
