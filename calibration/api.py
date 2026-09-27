"""FastAPI router exposing the calibration service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .models import CalibrationConfig, CalibrationResult, FrameStatus, TaskStatus
from .service import CalibrationService


class ConfigIn(BaseModel):
    max_iterations: int = Field(default=50, ge=1)
    convergence_tolerance: float = Field(default=1e-10, gt=0)
    use_tangential: bool = True
    use_k3: bool = True

    def to_config(self) -> CalibrationConfig:
        return CalibrationConfig(
            max_iterations=self.max_iterations,
            convergence_tolerance=self.convergence_tolerance,
            use_tangential=self.use_tangential,
            use_k3=self.use_k3,
        )


class CreateTaskRequest(BaseModel):
    config: ConfigIn | None = None


class FrameIn(BaseModel):
    frame_id: str
    object_points: list[list[float]]
    image_points: list[list[float]]
    image_size: tuple[int, int]


class FrameResponse(BaseModel):
    frame_id: str
    status: FrameStatus


class ResultOut(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float]
    rms_error: float
    per_frame_error: dict[str, float]
    iterations: int
    converged: bool
    frame_ids: list[str]

    @classmethod
    def from_result(cls, r: CalibrationResult) -> "ResultOut":
        return cls(
            fx=r.fx, fy=r.fy, cx=r.cx, cy=r.cy,
            distortion=[float(v) for v in r.distortion],
            rms_error=r.rms_error,
            per_frame_error={k: float(v) for k, v in r.per_frame_error.items()},
            iterations=r.iterations,
            converged=r.converged,
            frame_ids=list(r.frame_ids),
        )


class TaskOut(BaseModel):
    task_id: str
    status: TaskStatus
    frame_count: int
    result: ResultOut | None


class PointsIn(BaseModel):
    points: list[list[float]]


class PointsOut(BaseModel):
    points: list[list[float]]


def create_app(db_path: str | Path = "calibration.db") -> FastAPI:
    """Build the FastAPI application backed by a local SQLite store."""
    app = FastAPI(title="Camera Calibration Service")
    service = CalibrationService(db_path)
    app.state.service = service

    @app.on_event("shutdown")
    def _shutdown() -> None:
        service.close()

    def _not_found(exc: KeyError) -> HTTPException:
        return HTTPException(status_code=404, detail=str(exc))

    @app.post("/tasks/{task_id}", response_model=TaskOut)
    def create_task(task_id: str, body: CreateTaskRequest | None = None) -> TaskOut:
        config = body.config.to_config() if body and body.config else None
        info = service.create_task(task_id, config)
        return TaskOut(
            task_id=info.task_id,
            status=info.status,
            frame_count=info.frame_count,
            result=ResultOut.from_result(info.result) if info.result else None,
        )

    @app.get("/tasks/{task_id}", response_model=TaskOut)
    def get_task(task_id: str) -> TaskOut:
        try:
            info = service.get_task(task_id)
        except KeyError as exc:
            raise _not_found(exc)
        return TaskOut(
            task_id=info.task_id,
            status=info.status,
            frame_count=info.frame_count,
            result=ResultOut.from_result(info.result) if info.result else None,
        )

    @app.post("/tasks/{task_id}/frames", response_model=FrameResponse)
    def submit_frame(task_id: str, frame: FrameIn) -> FrameResponse:
        try:
            status = service.submit_frame(
                task_id,
                frame.frame_id,
                np.asarray(frame.object_points, dtype=np.float64),
                np.asarray(frame.image_points, dtype=np.float64),
                frame.image_size,
            )
        except KeyError as exc:
            raise _not_found(exc)
        return FrameResponse(frame_id=frame.frame_id, status=status)

    @app.post("/tasks/{task_id}/frames/{frame_id}/abandon")
    def abandon_frame(task_id: str, frame_id: str) -> dict:
        try:
            ok = service.abandon_frame(task_id, frame_id)
        except KeyError as exc:
            raise _not_found(exc)
        if not ok:
            raise HTTPException(status_code=404, detail=f"unknown frame {frame_id!r}")
        return {"frame_id": frame_id, "status": FrameStatus.ABANDONED.value}

    @app.post("/tasks/{task_id}/calibrate", response_model=ResultOut)
    def calibrate(task_id: str, body: ConfigIn | None = None) -> ResultOut:
        config = body.to_config() if body else None
        try:
            result = service.calibrate(task_id, config)
        except KeyError as exc:
            raise _not_found(exc)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return ResultOut.from_result(result)

    @app.get("/tasks/{task_id}/result", response_model=ResultOut)
    def get_result(task_id: str) -> ResultOut:
        try:
            return ResultOut.from_result(service.get_result(task_id))
        except KeyError as exc:
            raise _not_found(exc)

    @app.post("/tasks/{task_id}/undistort-points", response_model=PointsOut)
    def undistort_points(task_id: str, body: PointsIn) -> PointsOut:
        try:
            pts = service.undistort_points(
                task_id, np.asarray(body.points, dtype=np.float64)
            )
        except KeyError as exc:
            raise _not_found(exc)
        return PointsOut(points=pts.tolist())

    @app.post("/tasks/{task_id}/distort-points", response_model=PointsOut)
    def distort_points(task_id: str, body: PointsIn) -> PointsOut:
        try:
            pts = service.distort_points(
                task_id, np.asarray(body.points, dtype=np.float64)
            )
        except KeyError as exc:
            raise _not_found(exc)
        return PointsOut(points=pts.tolist())

    return app
