"""FastAPI 路由层。"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import chamber, synthesis
from .database import Database
from .models import (AnalysisImportIn, CalibrationAdjustIn,
                     CalibrationRefineIn, ForkIn, FreezeIn, GenerateIn,
                     MarkIn)


def create_app(db: Database = None) -> FastAPI:
    app = FastAPI(title="晶格候选室")
    database = db or Database()
    app.state.db = database

    @app.exception_handler(chamber.ConflictError)
    async def conflict_handler(request: Request,
                               exc: chamber.ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409,
                            content={"error": "conflict",
                                     "detail": str(exc)})

    @app.exception_handler(chamber.FrozenError)
    async def frozen_handler(request: Request,
                             exc: chamber.FrozenError) -> JSONResponse:
        return JSONResponse(status_code=409,
                            content={"error": "frozen",
                                     "detail": str(exc)})

    static_dir = __file__.rsplit("/", 1)[0] + "/static"

    @app.get("/", response_class=FileResponse)
    async def index() -> FileResponse:
        return FileResponse(static_dir + "/index.html")

    @app.get("/api/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/demo")
    async def create_demo() -> Dict[str, Any]:
        data = synthesis.generate_synthetic_dataset()
        payload = AnalysisImportIn(
            name="合成双晶域演示",
            calibration=data["calibration_version"],
            peaks=[{"peak_index": p["peak_index"], "point": p["point"],
                    "intensity": p["intensity"],
                    "covariance": p["covariance"],
                    "is_overlap": p["is_overlap"]}
                   for p in data["peaks"]],
            index_tol=0.25,
            outlier_budget=2,
        )
        analysis_id = chamber.import_analysis(database, payload)
        created = chamber.generate_for_analysis(database, analysis_id)
        return {"analysis_id": analysis_id, "candidates_created": created,
                "true_offset": data["true_offset"]}

    @app.post("/api/analyses")
    async def import_analysis(payload: AnalysisImportIn) -> Dict[str, str]:
        analysis_id = chamber.import_analysis(database, payload)
        return {"analysis_id": analysis_id}

    @app.post("/api/analyses/{analysis_id}/generate")
    async def generate(analysis_id: str, payload: GenerateIn) -> Dict[str, int]:
        try:
            created = chamber.generate_for_analysis(
                database, analysis_id, payload.max_candidates)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"candidates_created": created}

    @app.get("/api/analyses/{analysis_id}")
    async def get_analysis(analysis_id: str) -> Dict[str, Any]:
        try:
            return chamber.analysis_dict(database, analysis_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/candidates/{candidate_id}")
    async def get_candidate(candidate_id: str) -> Dict[str, Any]:
        try:
            return chamber.candidate_detail(database, candidate_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/observations/{observation_id}/mark")
    async def set_mark(observation_id: str, payload: MarkIn,
                       analysis_id: str) -> Dict[str, str]:
        try:
            chamber.set_observation_mark(
                database, analysis_id, observation_id, payload.mark)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "ok"}

    @app.post("/api/analyses/{analysis_id}/calibration")
    async def calibrate(analysis_id: str,
                        payload: CalibrationAdjustIn) -> Dict[str, str]:
        chamber.adjust_calibration(
            database, analysis_id, payload.offset)
        return {"status": "ok"}

    @app.post("/api/analyses/{analysis_id}/calibration/refine")
    async def calibrate_refine(analysis_id: str,
                               payload: CalibrationRefineIn
                               ) -> Dict[str, Any]:
        try:
            offset = chamber.refine_calibration(
                database, analysis_id, payload.candidate_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"active_offset": offset}

    @app.post("/api/analyses/{analysis_id}/fork")
    async def fork(analysis_id: str, payload: ForkIn) -> Dict[str, str]:
        new_id = chamber.fork_analysis(
            database, analysis_id, payload.name)
        return {"analysis_id": new_id}

    @app.post("/api/analyses/{analysis_id}/freeze")
    async def freeze(analysis_id: str, payload: FreezeIn) -> Dict[str, Any]:
        try:
            return chamber.freeze_analysis(
                database, analysis_id, payload.expected_revision)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
