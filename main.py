"""HTTP API for the camera calibration service.

Endpoints
---------
* ``POST /tasks`` — create a calibration task.
* ``POST /tasks/{task_id}/observations`` — append frames (any order;
  duplicates and corrupted frames are rejected/ignored).
* ``POST /tasks/{task_id}/calibrate`` — (re)calibrate from committed frames.
* ``GET  /tasks/{task_id}/result`` — last committed calibration result.
* ``POST /undistort/points`` — map distorted pixels to undistorted ones.
* ``POST /distort/points`` — forward-map undistorted pixels to distorted.

State is persisted under ``CALIBRATION_STORE`` (default ``./calibration_data``).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from calibration.models import (
    CameraIntrinsics,
    FrameObservation,
    SolverConfig,
    TaskConfig,
)
from calibration.service import CalibrationService
from calibration.solver import CalibrationError
from calibration.store import TaskExistsError, TaskNotFoundError
from calibration.distortion import distort_points, undistort_points

STORE_ROOT = os.environ.get("CALIBRATION_STORE", "./calibration_data")
service = CalibrationService(STORE_ROOT)
app = FastAPI(title="camera-calibration")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SolverConfigIn(BaseModel):
    max_iterations: int = Field(default=50, ge=1)
    convergence_tolerance: float = Field(default=1e-9, gt=0)
    initial_damping: float = Field(default=1e-3, gt=0)


class CreateTaskIn(BaseModel):
    task_id: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    solver: SolverConfigIn = SolverConfigIn()


class FrameIn(BaseModel):
    frame_id: str
    object_points: list[tuple[float, float, float]]
    image_points: list[tuple[float, float]]


class ObservationsIn(BaseModel):
    frames: list[FrameIn]


class PointsIn(BaseModel):
    intrinsics: dict[str, float]
    points: list[tuple[float, float]]


def _intrinsics_from_dict(data: dict[str, float]) -> CameraIntrinsics:
    try:
        return CameraIntrinsics(
            fx=float(data["fx"]), fy=float(data["fy"]),
            cx=float(data["cx"]), cy=float(data["cy"]),
            k1=float(data.get("k1", 0.0)), k2=float(data.get("k2", 0.0)),
            p1=float(data.get("p1", 0.0)), p2=float(data.get("p2", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"bad intrinsics: {exc}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/tasks", status_code=201)
def create_task(body: CreateTaskIn):
    config = TaskConfig(
        image_width=body.image_width,
        image_height=body.image_height,
        solver=SolverConfig(
            max_iterations=body.solver.max_iterations,
            convergence_tolerance=body.solver.convergence_tolerance,
            initial_damping=body.solver.initial_damping,
        ),
    )
    try:
        service.create_task(body.task_id, config)
    except TaskExistsError:
        raise HTTPException(status_code=409, detail="task already exists")
    return {"task_id": body.task_id}


@app.post("/tasks/{task_id}/observations")
def add_observations(task_id: str, body: ObservationsIn):
    frames = [
        FrameObservation(
            frame_id=f.frame_id,
            object_points=tuple(tuple(p) for p in f.object_points),
            image_points=tuple(tuple(p) for p in f.image_points),
        )
        for f in body.frames
    ]
    try:
        rejected = service.add_observations(task_id, frames)
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "accepted": len(frames) - len(rejected),
        "rejected": [{"frame_id": r.frame_id, "reason": r.reason} for r in rejected],
    }


@app.post("/tasks/{task_id}/calibrate")
def calibrate_task(task_id: str):
    try:
        result = service.calibrate(task_id)
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="task not found")
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _result_to_json(result)


@app.get("/tasks/{task_id}/result")
def get_result(task_id: str):
    try:
        result = service.get_result(task_id)
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="task not found")
    if result is None:
        raise HTTPException(status_code=404, detail="no calibration result yet")
    return _result_to_json(result)


@app.post("/undistort/points")
def undistort_points_api(body: PointsIn):
    intrinsics = _intrinsics_from_dict(body.intrinsics)
    mapped = undistort_points(body.points, intrinsics)
    return {"points": mapped.tolist()}


@app.post("/distort/points")
def distort_points_api(body: PointsIn):
    intrinsics = _intrinsics_from_dict(body.intrinsics)
    mapped = distort_points(body.points, intrinsics)
    return {"points": mapped.tolist()}


def _result_to_json(result) -> dict:
    intr = result.intrinsics
    return {
        "intrinsics": {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "k1": intr.k1, "k2": intr.k2, "p1": intr.p1, "p2": intr.p2,
        },
        "frame_errors": [
            {"frame_id": fe.frame_id, "mean_error": fe.mean_error,
             "max_error": fe.max_error}
            for fe in result.frame_errors
        ],
        "mean_reprojection_error": result.mean_reprojection_error,
        "iterations": result.iterations,
        "converged": result.converged,
        "used_frame_ids": list(result.used_frame_ids),
    }
