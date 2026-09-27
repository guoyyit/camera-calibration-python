"""Solver accuracy, determinism and order-independence."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.models import SolverConfig
from calibration.solver import CalibrationError, calibrate

from .conftest import IMAGE_HEIGHT, IMAGE_WIDTH, TRUE_INTRINSICS


def test_recovers_ground_truth(synthetic_frames):
    result = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT,
                       SolverConfig(max_iterations=100, convergence_tolerance=1e-12))
    intr = result.intrinsics
    fx, fy, cx, cy, k1, k2, p1, p2 = TRUE_INTRINSICS
    assert abs(intr.fx - fx) < 5.0
    assert abs(intr.fy - fy) < 5.0
    assert abs(intr.cx - cx) < 3.0
    assert abs(intr.cy - cy) < 3.0
    assert abs(intr.k1 - k1) < 0.02
    assert abs(intr.k2 - k2) < 0.05
    assert abs(intr.p1 - p1) < 0.005
    assert abs(intr.p2 - p2) < 0.005
    # noise level is 0.05 px; RMS reprojection error should be comparable
    assert result.mean_reprojection_error < 0.15
    assert result.converged
    assert len(result.frame_errors) == len(synthetic_frames)
    assert result.used_frame_ids == tuple(
        f.frame_id for f in sorted(synthetic_frames, key=lambda f: f.frame_id)
    )


def test_deterministic_same_input(synthetic_frames):
    cfg = SolverConfig(max_iterations=100, convergence_tolerance=1e-12)
    r1 = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT, cfg)
    r2 = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT, cfg)
    assert r1 == r2  # exact equality, not just approximate


def test_order_independent(synthetic_frames):
    cfg = SolverConfig(max_iterations=100, convergence_tolerance=1e-12)
    rng = np.random.default_rng(99)
    r_ref = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT, cfg)
    for _ in range(3):
        shuffled = [synthetic_frames[i] for i in rng.permutation(len(synthetic_frames))]
        assert calibrate(shuffled, IMAGE_WIDTH, IMAGE_HEIGHT, cfg) == r_ref


def test_configurable_iterations_and_tolerance(synthetic_frames):
    loose = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT,
                      SolverConfig(max_iterations=1, convergence_tolerance=1e-3))
    tight = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT,
                      SolverConfig(max_iterations=100, convergence_tolerance=1e-12))
    assert loose.iterations <= 1
    assert tight.iterations > loose.iterations
    assert tight.mean_reprojection_error <= loose.mean_reprojection_error + 1e-12


def test_too_few_frames_rejected(synthetic_frames):
    with pytest.raises(CalibrationError):
        calibrate(synthetic_frames[:2], IMAGE_WIDTH, IMAGE_HEIGHT)


def test_per_frame_errors_present(synthetic_frames):
    result = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT)
    ids = [fe.frame_id for fe in result.frame_errors]
    assert ids == sorted(f.frame_id for f in synthetic_frames)
    for fe in result.frame_errors:
        assert fe.mean_error >= 0.0
        assert fe.max_error >= fe.mean_error
