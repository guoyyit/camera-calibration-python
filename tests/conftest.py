"""Shared fixtures: synthetic calibration data with known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.models import FrameObservation
from calibration.solver import _rodrigues

# Ground-truth intrinsics used to synthesise observations.
TRUE_FX, TRUE_FY, TRUE_CX, TRUE_CY = 850.0, 855.0, 320.0, 240.0
TRUE_DIST = np.array([-0.18, 0.04, 0.0006, -0.0002, 0.0])
IMAGE_SIZE = (640, 480)


def make_board() -> np.ndarray:
    """9x6 planar grid with 25 mm spacing."""
    gx, gy = np.meshgrid(np.arange(9), np.arange(6))
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64) * 0.025


def project(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Project board points with the ground-truth model."""
    r = _rodrigues(np.asarray(rvec, dtype=np.float64))
    cam = np.column_stack([obj, np.zeros(len(obj))]) @ r.T + np.asarray(tvec)
    xu, yu = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    r2 = xu * xu + yu * yu
    k1, k2, p1, p2, k3 = TRUE_DIST
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xu * radial + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
    yd = yu * radial + p1 * (r2 + 2 * yu * yu) + 2 * p2 * xu * yu
    return np.column_stack([TRUE_FX * xd + TRUE_CX, TRUE_FY * yd + TRUE_CY])


def make_frames(n: int, seed: int = 0, prefix: str = "f") -> list[FrameObservation]:
    """Generate ``n`` deterministic synthetic frames."""
    rng = np.random.default_rng(seed)
    obj = make_board()
    frames = []
    for i in range(n):
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array([
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.04, 0.04),
            rng.uniform(0.4, 0.7),
        ])
        frames.append(
            FrameObservation(
                frame_id=f"{prefix}{i:03d}",
                object_points=obj,
                image_points=project(obj, rvec, tvec),
                image_size=IMAGE_SIZE,
            )
        )
    return frames


@pytest.fixture()
def frames() -> list[FrameObservation]:
    return make_frames(6, seed=0)


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "calib.db")
