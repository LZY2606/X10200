"""Pydantic 请求模型。"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CalibrationVersionIn(BaseModel):
    id: str
    label: str
    offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class PeakIn(BaseModel):
    peak_index: int
    point: List[float]
    intensity: float
    covariance: List[List[float]]
    is_overlap: bool = False


class AnalysisImportIn(BaseModel):
    name: str
    calibration: CalibrationVersionIn
    peaks: List[PeakIn]
    index_tol: float = 0.25
    outlier_budget: int = 3


class MarkIn(BaseModel):
    mark: str  # normal | locked | excluded


class CalibrationAdjustIn(BaseModel):
    offset: List[float]


class CalibrationRefineIn(BaseModel):
    candidate_id: str


class ForkIn(BaseModel):
    name: Optional[str] = None


class FreezeIn(BaseModel):
    expected_revision: int


class GenerateIn(BaseModel):
    max_candidates: int = 120

