"""Undistort / forward mapping and boundary-pixel policy."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.image_ops import distort_points, undistort_image, undistort_points
from calibration.models import CalibrationConfig, CalibrationResult


def _result(dist=None, fx=800.0, fy=810.0, cx=320.0, cy=240.0):
    return CalibrationResult(
        fx=fx, fy=fy, cx=cx, cy=cy,
        distortion=np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64),
        rms_error=0.0, per_frame_error={}, iterations=0, converged=True,
        frame_ids=[], config=CalibrationConfig(),
    )


def test_distort_undistort_round_trip():
    res = _result(dist=[-0.2, 0.05, 0.001, -0.0005, 0.0])
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [640, 480], size=(200, 2))
    rt = undistort_points(distort_points(pts, res), res)
    np.testing.assert_allclose(rt, pts, atol=1e-6)


def test_undistort_is_deterministic():
    res = _result(dist=[-0.2, 0.05, 0.001, -0.0005, 0.0])
    pts = np.array([[10.0, 20.0], [639.0, 479.0], [320.0, 240.0]])
    a = undistort_points(pts, res)
    b = undistort_points(pts, res)
    np.testing.assert_array_equal(a, b)


def test_principal_point_unmoved():
    res = _result(dist=[-0.2, 0.05, 0.001, -0.0005, 0.0])
    out = undistort_points(np.array([[320.0, 240.0]]), res)
    np.testing.assert_allclose(out[0], [320.0, 240.0], atol=1e-9)


def test_identity_mapping_reproduces_image():
    res = _result(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 255, (12, 16))
    out = undistort_image(img, res)
    np.testing.assert_allclose(out, img, atol=1e-9)


def test_out_of_bounds_pixels_get_border_value():
    # strong barrel distortion pushes corner sampling outside the image
    res = _result(dist=[0.5, 0.0, 0.0, 0.0, 0.0], fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    rng = np.random.default_rng(2)
    img = rng.uniform(0, 255, (8, 10))
    out = undistort_image(img, res, border_value=-1.0)
    assert out[0, -1] == -1.0
    assert out[-1, -1] == -1.0
    # origin maps to itself and stays intact
    assert out[0, 0] == img[0, 0]


def test_border_value_is_configurable():
    res = _result(dist=[0.5, 0.0, 0.0, 0.0, 0.0], fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    img = np.zeros((8, 10))
    out = undistort_image(img, res, border_value=123.0)
    assert out[0, -1] == 123.0


def test_color_image_supported():
    res = _result(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 255, (8, 10, 3))
    out = undistort_image(img, res)
    assert out.shape == (8, 10, 3)
    np.testing.assert_allclose(out, img, atol=1e-9)


def test_bilinear_interpolation_matches_manual_computation():
    res = _result(dist=[-0.05, 0.01, 0.0003, 0.0, 0.0])
    rng = np.random.default_rng(4)
    img = rng.uniform(0, 255, (20, 24))
    out = undistort_image(img, res)
    # verify one interior pixel against a manual bilinear sample
    src = distort_points(np.array([[7.0, 5.0]]), res)[0]
    sx, sy = src
    x0, y0 = int(np.floor(sx)), int(np.floor(sy))
    wx, wy = sx - x0, sy - y0
    expected = (
        img[y0, x0] * (1 - wx) * (1 - wy)
        + img[y0, x0 + 1] * wx * (1 - wy)
        + img[y0 + 1, x0] * (1 - wx) * wy
        + img[y0 + 1, x0 + 1] * wx * wy
    )
    np.testing.assert_allclose(out[5, 7], expected, atol=1e-9)
