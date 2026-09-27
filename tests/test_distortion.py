"""Forward/inverse mapping and boundary-pixel policy tests."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.distortion import (
    distort_points,
    undistort_image,
    undistort_points,
)
from calibration.models import BorderPolicy, CameraIntrinsics

INTRINSICS = CameraIntrinsics(
    fx=800.0, fy=810.0, cx=320.0, cy=240.0,
    k1=-0.2, k2=0.05, p1=0.001, p2=-0.002,
)
# Pincushion distortion pushes distorted points outwards, so undistorting
# border pixels samples outside the source image.
PINCUSHION = CameraIntrinsics(
    fx=800.0, fy=810.0, cx=320.0, cy=240.0,
    k1=0.3, k2=0.1, p1=0.0, p2=0.0,
)


def test_round_trip_points():
    rng = np.random.default_rng(0)
    pts = rng.uniform([10, 10], [630, 470], size=(500, 2))
    recovered = undistort_points(distort_points(pts, INTRINSICS), INTRINSICS)
    assert np.abs(recovered - pts).max() < 1e-6


def test_zero_distortion_is_identity():
    intr = CameraIntrinsics(800, 810, 320, 240, 0, 0, 0, 0)
    pts = [(0.0, 0.0), (320.5, 240.5), (639.0, 479.0)]
    assert np.allclose(distort_points(pts, intr), pts)
    assert np.allclose(undistort_points(pts, intr), pts)


def test_principal_point_is_fixed():
    # the principal point maps to itself in both directions
    src = np.array([[INTRINSICS.cx, INTRINSICS.cy]])
    assert np.allclose(distort_points(src, INTRINSICS), src, atol=1e-9)
    assert np.allclose(undistort_points(src, INTRINSICS), src, atol=1e-9)


def test_mapping_deterministic():
    pts = [(12.5, 400.25), (600.0, 30.0)]
    assert np.array_equal(
        distort_points(pts, INTRINSICS), distort_points(pts, INTRINSICS)
    )
    assert np.array_equal(
        undistort_points(pts, INTRINSICS), undistort_points(pts, INTRINSICS)
    )


def _gradient_image() -> np.ndarray:
    # no zero pixels, so SKIP-introduced zeros are unambiguous
    return (np.arange(480 * 640, dtype=np.uint32).reshape(480, 640) % 251 + 1).astype(
        np.uint8
    )


def test_skip_policy_zeroes_out_of_range_pixels():
    img = _gradient_image()
    out = undistort_image(img, PINCUSHION, BorderPolicy.SKIP)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert int((out == 0).sum()) > 100  # border band sampled outside
    # interior (near principal point) is unaffected by the policy
    assert out[240, 320] != 0


def test_clamp_policy_uses_border_pixels():
    img = _gradient_image()
    out_clamp = undistort_image(img, PINCUSHION, BorderPolicy.CLAMP)
    out_skip = undistort_image(img, PINCUSHION, BorderPolicy.SKIP)
    assert int((out_clamp == 0).sum()) == 0  # clamped to valid pixels
    # where SKIP has valid data both policies agree closely
    interior = out_skip != 0
    diff = np.abs(out_clamp.astype(int) - out_skip.astype(int))
    assert diff[interior].max() <= 1
    # clamped corners equal the nearest source border pixel
    assert out_clamp[0, 0] == img[0, 0] or out_clamp[0, 0] == img[0, 1]


def test_undistort_image_identity_without_distortion():
    img = _gradient_image()
    intr = CameraIntrinsics(800, 810, 320, 240, 0, 0, 0, 0)
    out = undistort_image(img, intr, BorderPolicy.SKIP)
    # identity map: every pixel samples itself exactly
    assert np.array_equal(out, img)


def test_color_image_supported():
    img = np.stack([_gradient_image()] * 3, axis=-1)
    out = undistort_image(img, PINCUSHION, BorderPolicy.CLAMP)
    assert out.shape == img.shape and out.dtype == img.dtype


def test_unknown_border_policy_rejected():
    with pytest.raises(ValueError):
        undistort_image(_gradient_image(), INTRINSICS, "mirror")


def test_undistort_image_deterministic():
    img = _gradient_image()
    a = undistort_image(img, PINCUSHION, BorderPolicy.CLAMP)
    b = undistort_image(img, PINCUSHION, BorderPolicy.CLAMP)
    assert np.array_equal(a, b)
