"""Solver accuracy, determinism and configurability."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.models import CalibrationConfig, FrameObservation
from calibration.solver import calibrate
from tests.conftest import TRUE_CX, TRUE_CY, TRUE_DIST, TRUE_FX, TRUE_FY, make_frames


def test_recovers_ground_truth(frames):
    res = calibrate(frames)
    assert res.converged
    assert res.fx == pytest.approx(TRUE_FX, abs=1e-6)
    assert res.fy == pytest.approx(TRUE_FY, abs=1e-6)
    assert res.cx == pytest.approx(TRUE_CX, abs=1e-6)
    assert res.cy == pytest.approx(TRUE_CY, abs=1e-6)
    np.testing.assert_allclose(res.distortion, TRUE_DIST, atol=1e-6)
    assert res.rms_error < 1e-8


def test_per_frame_error_reported(frames):
    res = calibrate(frames)
    assert set(res.per_frame_error) == {f.frame_id for f in frames}
    for err in res.per_frame_error.values():
        assert err < 1e-8


def test_out_of_order_input_is_irrelevant(frames):
    ref = calibrate(frames)
    shuffled = [frames[i] for i in [3, 0, 5, 2, 4, 1]]
    res = calibrate(shuffled)
    assert res.fx == ref.fx and res.fy == ref.fy
    assert res.cx == ref.cx and res.cy == ref.cy
    np.testing.assert_array_equal(res.distortion, ref.distortion)
    assert res.per_frame_error == ref.per_frame_error
    assert res.rms_error == ref.rms_error


def test_deterministic_repeated_runs(frames):
    a = calibrate(frames)
    b = calibrate(frames)
    assert a.fx == b.fx and a.fy == b.fy
    assert a.cx == b.cx and a.cy == b.cy
    np.testing.assert_array_equal(a.distortion, b.distortion)
    assert a.iterations == b.iterations
    assert a.rms_error == b.rms_error


def test_configurable_iterations_and_tolerance(frames):
    loose = CalibrationConfig(max_iterations=2, convergence_tolerance=1e-3)
    tight = CalibrationConfig(max_iterations=200, convergence_tolerance=1e-14)
    res_loose = calibrate(frames, loose)
    res_tight = calibrate(frames, tight)
    assert res_loose.iterations <= 2
    assert res_tight.rms_error <= res_loose.rms_error + 1e-12
    assert res_tight.config.max_iterations == 200


def test_too_few_frames_rejected():
    frames = make_frames(1)
    with pytest.raises(ValueError, match="at least"):
        calibrate(frames)


def test_too_few_points_rejected():
    f = make_frames(2)
    bad = FrameObservation("bad", f[0].object_points[:3], f[0].image_points[:3], (640, 480))
    with pytest.raises(ValueError, match="points"):
        calibrate([bad, f[1]])
