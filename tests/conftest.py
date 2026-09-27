"""Shared fixtures: synthetic calibration data with known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.models import FrameObservation, SolverConfig, TaskConfig
from calibration.solver import project_points

# Ground-truth intrinsics used to synthesise observations.
TRUE_INTRINSICS = np.array(
    [800.0, 810.0, 320.0, 240.0, -0.2, 0.05, 0.001, -0.002]
)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def board_points() -> np.ndarray:
    """9x6 planar board, 25 mm spacing, Z=0."""
    gx, gy = np.meshgrid(np.arange(9), np.arange(6))
    return np.column_stack(
        [gx.ravel() * 0.025, gy.ravel() * 0.025, np.zeros(gx.size)]
    )


def make_frame(frame_id: str, rng: np.random.Generator, noise: float = 0.05) -> FrameObservation:
    """One synthetic observation of the board from a random pose."""
    obj = board_points()
    rvec = rng.uniform(-0.4, 0.4, 3)
    tvec = np.array(
        [rng.uniform(-0.1, 0.1), rng.uniform(-0.08, 0.08), rng.uniform(0.4, 0.8)]
    )
    proj = project_points(obj, rvec, tvec, TRUE_INTRINSICS)
    if noise:
        proj = proj + rng.normal(0.0, noise, proj.shape)
    return FrameObservation(
        frame_id=frame_id,
        object_points=tuple(map(tuple, obj.tolist())),
        image_points=tuple(map(tuple, proj.tolist())),
    )


@pytest.fixture()
def synthetic_frames() -> list[FrameObservation]:
    """10 deterministic synthetic frames (seeded RNG)."""
    rng = np.random.default_rng(2026)
    return [make_frame(f"frame-{i:03d}", rng) for i in range(10)]


@pytest.fixture()
def task_config() -> TaskConfig:
    return TaskConfig(
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        solver=SolverConfig(max_iterations=100, convergence_tolerance=1e-12),
    )
