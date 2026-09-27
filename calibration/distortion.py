"""Distortion / undistortion mapping built on calibration results.

Two directions are provided:

* :func:`undistort_points` — iterative inverse mapping from distorted pixel
  coordinates to undistorted pixel coordinates.
* :func:`distort_points` — closed-form forward mapping from undistorted
  pixel coordinates to distorted pixel coordinates.
* :func:`undistort_image` — remap a whole image.  Pixels whose source
  coordinate falls outside the image are handled according to the
  configured :class:`~calibration.models.BorderPolicy`.

Iteration counts and tolerances of the inverse mapping are fixed constants
so results are reproducible for a given input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import BorderPolicy, CameraIntrinsics

# Fixed settings of the iterative inverse distortion: deterministic.
_UNDISTORT_ITERATIONS = 20
_UNDISTORT_TOL = 1e-12


def _distort_normalised(
    x: np.ndarray, y: np.ndarray, intrinsics: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the distortion model in normalised camera coordinates."""
    k1, k2, p1, p2 = intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


def distort_points(
    points: Sequence[tuple[float, float]],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Forward-map undistorted pixel coordinates to distorted ones.

    Parameters
    ----------
    points:
        ``(u, v)`` pixel coordinates of the undistorted image.
    intrinsics:
        Estimated camera intrinsics.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with distorted pixel coordinates.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = (pts[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (pts[:, 1] - intrinsics.cy) / intrinsics.fy
    x_d, y_d = _distort_normalised(x, y, intrinsics)
    return np.column_stack(
        [intrinsics.fx * x_d + intrinsics.cx, intrinsics.fy * y_d + intrinsics.cy]
    )


def undistort_points(
    points: Sequence[tuple[float, float]],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Map distorted pixel coordinates to undistorted ones (iterative).

    Uses fixed-point iteration on the distortion model with a fixed number
    of iterations and tolerance, so the output is reproducible.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x_d = (pts[:, 0] - intrinsics.cx) / intrinsics.fx
    y_d = (pts[:, 1] - intrinsics.cy) / intrinsics.fy
    x, y = x_d.copy(), y_d.copy()
    for _ in range(_UNDISTORT_ITERATIONS):
        x_dist, y_dist = _distort_normalised(x, y, intrinsics)
        x_new = x_d - (x_dist - x)
        y_new = y_d - (y_dist - y)
        delta = max(
            float(np.abs(x_new - x).max(initial=0.0)),
            float(np.abs(y_new - y).max(initial=0.0)),
        )
        x, y = x_new, y_new
        if delta < _UNDISTORT_TOL:
            break
    return np.column_stack(
        [intrinsics.fx * x + intrinsics.cx, intrinsics.fy * y + intrinsics.cy]
    )


def _bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample ``image`` at floating-point coordinates with bilinear weights.

    Coordinates must already be clamped to the valid range.
    """
    height, width = image.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = xs - x0
    wy = ys - y0

    def at(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return image[yy, xx].astype(np.float64)

    top = at(x0, y0) * (1.0 - wx)[..., None] + at(x1, y0) * wx[..., None] \
        if image.ndim == 3 else \
        at(x0, y0) * (1.0 - wx) + at(x1, y0) * wx
    bottom = at(x0, y1) * (1.0 - wx)[..., None] + at(x1, y1) * wx[..., None] \
        if image.ndim == 3 else \
        at(x0, y1) * (1.0 - wx) + at(x1, y1) * wx
    return top * (1.0 - wy)[..., None] + bottom * wy[..., None] \
        if image.ndim == 3 else top * (1.0 - wy) + bottom * wy


def undistort_image(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    border_policy: str = BorderPolicy.CLAMP,
) -> np.ndarray:
    """Return an undistorted copy of ``image``.

    For each destination pixel the corresponding distorted source location
    is computed and bilinearly interpolated.  Locations outside the source
    image are handled per ``border_policy``:

    * ``BorderPolicy.CLAMP`` — clamp to the nearest border pixel.
    * ``BorderPolicy.SKIP`` — leave the destination pixel at 0.
    """
    if border_policy not in (BorderPolicy.CLAMP, BorderPolicy.SKIP):
        raise ValueError(f"unknown border policy: {border_policy!r}")
    height, width = image.shape[:2]
    grid_u, grid_v = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    dst = np.column_stack([grid_u.ravel(), grid_v.ravel()])
    src = distort_points(dst, intrinsics)
    xs = src[:, 0]
    ys = src[:, 1]

    inside = (xs >= 0.0) & (xs <= width - 1) & (ys >= 0.0) & (ys <= height - 1)
    xs_c = np.clip(xs, 0.0, width - 1)
    ys_c = np.clip(ys, 0.0, height - 1)
    sampled = _bilinear_sample(image, xs_c, ys_c)

    if border_policy == BorderPolicy.SKIP:
        if image.ndim == 3:
            sampled = sampled * inside[:, None]
        else:
            sampled = sampled * inside

    if image.ndim == 3:
        out = sampled.reshape(height, width, image.shape[2])
    else:
        out = sampled.reshape(height, width)
    # Round to nearest (not truncate) so that an identity mapping reproduces
    # the source pixel exactly despite floating-point round-off.
    return np.clip(np.rint(out), 0, 255).astype(image.dtype)
